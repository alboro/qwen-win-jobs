from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import logging
import queue
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from qwen3_tts_win.core import (
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REFERENCE_PREFIX,
    DEFAULT_SHARED_DIR,
    DEFAULT_SPEAKER,
    DEFAULT_TEMP_DIR,
    PROJECT_ROOT,
    SynthesisRequest,
    create_voice_clone_prompt,
    ensure_project_runtime_dirs,
    estimate_audio_duration_seconds,
    find_reference_in_shared,
    find_reference_text_sidecar,
    load_qwen_model,
    prepare_reference_audio,
    resolve_path,
    resolve_model_for_task,
    resolve_shared_dir,
    strip_qwen_stress_marks,
    synthesize_to_file,
)

DEFAULT_JOBS_DIR = PROJECT_ROOT / ".data" / "jobs"
DEFAULT_JOB_OUTPUT_NAME = "audio.wav"
SUPPORTED_RESPONSE_FORMATS = {"wav"}
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8030
DEFAULT_JOB_RETENTION_HOURS = 24
DEFAULT_DOWNLOADED_JOB_RETENTION_HOURS = 6
DEFAULT_CLEANUP_INTERVAL_SECONDS = 900

AUDIO_MIME_TO_EXT = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
}


class CreateTTSJobRequest(BaseModel):
    input: str = Field(..., min_length=1)
    model: str | None = Field(default=None)
    task: str | None = Field(default=None)
    voice: str | None = Field(default=None)
    response_format: str | None = Field(default="wav")
    language: str | None = Field(default=None)
    speaker: str | None = Field(default=None)
    instruct: str | None = None
    instructions: str | None = None
    reference_text: str | None = None
    reference_audio_base64: str | None = None
    reference_audio_filename: str | None = None
    x_vector_only_mode: bool | None = None
    max_new_tokens: int | None = DEFAULT_MAX_NEW_TOKENS
    temperature: float | None = None
    top_p: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ServerSettings:
    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_SERVER_PORT
    shared_dir: Path = DEFAULT_SHARED_DIR
    jobs_dir: Path = DEFAULT_JOBS_DIR
    model: str = DEFAULT_MODEL
    task: str = "voice_clone"
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "auto"
    ffmpeg: str = "ffmpeg"
    language: str = DEFAULT_LANGUAGE
    speaker: str = DEFAULT_SPEAKER
    instruct: str | None = None
    max_chars: int = DEFAULT_MAX_CHARS
    max_new_tokens: int | None = DEFAULT_MAX_NEW_TOKENS
    job_retention_hours: int = DEFAULT_JOB_RETENTION_HOURS
    downloaded_job_retention_hours: int = DEFAULT_DOWNLOADED_JOB_RETENTION_HOURS
    cleanup_interval_seconds: int = DEFAULT_CLEANUP_INTERVAL_SECONDS


@dataclass(slots=True)
class LoadedServerModel:
    model_result: Any
    prompt_cache: dict[tuple[str, int, int, str, bool], Any] = field(default_factory=dict)


logger = logging.getLogger(__name__)


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON without leaving a half-written file after process/GPU crashes."""
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


class JobStore:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._load_existing_jobs()

    def _load_existing_jobs(self) -> None:
        for job_file in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(job_file.read_text(encoding="utf-8"))
                job_id = job["id"]
                status_value = job["status"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Skipping unreadable job file %s: %s", job_file, exc)
                continue
            if status_value in {"queued", "in_progress"}:
                now = utcnow_iso()
                job["status"] = "failed"
                job["error"] = "Server restarted before the job completed."
                job["failed_at"] = now
                job["updated_at"] = now
                write_json_atomic(job_file, job)
            self._jobs[job_id] = job

    def create_job(self, request: CreateTTSJobRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = utcnow_iso()
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        audio_path = job_dir / DEFAULT_JOB_OUTPUT_NAME

        job = {
            "id": job_id,
            "object": "tts.job",
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "failed_at": None,
            "model": request.model,
            "task": request.task,
            "voice": request.voice,
            "language": request.language,
            "response_format": request.response_format,
            "input_characters": len(request.input),
            "input_words": len(request.input.split()),
            "estimated_speech_seconds": estimate_audio_duration_seconds(request.input),
            "status_url": f"/v1/tts/jobs/{job_id}",
            "audio_url": f"/v1/tts/jobs/{job_id}/audio",
            "audio_ready": False,
            "audio_path": str(audio_path),
            "error": None,
            "metadata": request.metadata or {},
            "voice_source": "uploaded_reference" if request.reference_audio_base64 else "shared_voice",
            "reference_filename": None,
            "reference_text_source": "request" if request.reference_text else None,
            "download_count": 0,
            "first_downloaded_at": None,
            "last_downloaded_at": None,
            "chunks": None,
            "sample_rate": None,
            "x_vector_only_mode": request.x_vector_only_mode,
            "runtime_seconds": None,
            "model_load_seconds": None,
            "synthesis_seconds": None,
        }

        request_payload = request_to_payload(request)
        if request.reference_audio_base64:
            reference_path = decode_reference_audio_to_file(
                encoded=request.reference_audio_base64,
                filename=request.reference_audio_filename,
                job_dir=job_dir,
            )
            request_payload["reference_file"] = reference_path.name
            request_payload["reference_audio_base64"] = None
            job["reference_filename"] = reference_path.name

        write_json_atomic(job_dir / "request.json", request_payload)

        with self._lock:
            self._jobs[job_id] = job
            self._write_job(job)
        return self.public_view(job)

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.update(updates)
            job["updated_at"] = utcnow_iso()
            self._write_job(job)
            return dict(job)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def mark_downloaded(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            now = utcnow_iso()
            job["download_count"] = int(job.get("download_count") or 0) + 1
            job["first_downloaded_at"] = job.get("first_downloaded_at") or now
            job["last_downloaded_at"] = now
            job["updated_at"] = now
            self._write_job(job)
            return dict(job)

    def public_view(self, job: dict[str, Any]) -> dict[str, Any]:
        public_job = dict(job)
        public_job.pop("audio_path", None)
        return public_job

    def audio_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / DEFAULT_JOB_OUTPUT_NAME

    def uploaded_reference_path(self, job_id: str) -> Path | None:
        job = self.get_job(job_id)
        if not job or not job.get("reference_filename"):
            return None
        return self.job_dir(job_id) / str(job["reference_filename"])

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.log"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def cleanup_expired(
        self,
        *,
        job_retention: timedelta,
        downloaded_job_retention: timedelta,
    ) -> list[str]:
        expired: list[tuple[str, Path]] = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if not self._is_terminal(job):
                    continue
                if not self._is_expired(
                    job,
                    now=now,
                    job_retention=job_retention,
                    downloaded_job_retention=downloaded_job_retention,
                ):
                    continue
                expired.append((job_id, self.job_dir(job_id)))
                self._jobs.pop(job_id, None)

        removed_ids: list[str] = []
        for job_id, job_dir in expired:
            shutil.rmtree(job_dir, ignore_errors=True)
            removed_ids.append(job_id)
        return removed_ids

    def _write_job(self, job: dict[str, Any]) -> None:
        write_json_atomic(self.job_dir(job["id"]) / "job.json", job)

    @staticmethod
    def _is_terminal(job: dict[str, Any]) -> bool:
        return str(job.get("status") or "") in {"completed", "failed"}

    def _is_expired(
        self,
        job: dict[str, Any],
        *,
        now: datetime,
        job_retention: timedelta,
        downloaded_job_retention: timedelta,
    ) -> bool:
        downloaded_at = parse_iso_datetime(job.get("last_downloaded_at"))
        if downloaded_at is not None:
            return now >= downloaded_at + downloaded_job_retention

        terminal_at = (
            parse_iso_datetime(job.get("completed_at"))
            or parse_iso_datetime(job.get("failed_at"))
            or parse_iso_datetime(job.get("updated_at"))
        )
        if terminal_at is None:
            return False
        return now >= terminal_at + job_retention


class JobGarbageCollector:
    def __init__(self, settings: ServerSettings, store: JobStore):
        self.settings = settings
        self.store = store
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="qwen3-tts-job-cleaner", daemon=True)
        self._startup_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._startup_lock:
            if self._started:
                return
            self._thread.start()
            self._started = True

    def stop(self) -> None:
        self._stop_event.set()

    def sweep_now(self) -> list[str]:
        removed = self.store.cleanup_expired(
            job_retention=timedelta(hours=self.settings.job_retention_hours),
            downloaded_job_retention=timedelta(hours=self.settings.downloaded_job_retention_hours),
        )
        if removed:
            logger.info("Cleaned up %s expired job(s): %s", len(removed), ", ".join(removed))
        return removed

    def _run(self) -> None:
        self.sweep_now()
        while not self._stop_event.wait(self.settings.cleanup_interval_seconds):
            self.sweep_now()


class SynthesisWorker:
    def __init__(self, settings: ServerSettings, store: JobStore):
        self.settings = settings
        self.store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="qwen3-tts-job-worker", daemon=True)
        self._loaded_model: LoadedServerModel | None = None
        self._startup_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._startup_lock:
            if self._started:
                return
            self._thread.start()
            self._started = True

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._process_job(job_id)
            finally:
                self._queue.task_done()

    def _ensure_model(self) -> LoadedServerModel:
        if self._loaded_model is not None:
            return self._loaded_model
        model_result = load_qwen_model(
            model_name=self.settings.model,
            requested_device=self.settings.device,
            requested_dtype=self.settings.dtype,
            requested_attn_implementation=self.settings.attn_implementation,
        )
        self._loaded_model = LoadedServerModel(model_result=model_result)
        return self._loaded_model

    def _process_job(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return

        log_path = self.store.log_path(job_id)
        with log_path.open("a", encoding="utf-8") as log_file:
            try:
                started_at = utcnow_iso()
                self.store.update_job(job_id, status="in_progress", started_at=started_at)
                log_line(log_file, f"Job {job_id} started at {started_at}")

                with redirect_stdout(log_file), redirect_stderr(log_file):
                    loaded = self._ensure_model()
                model_result = loaded.model_result
                log_line(
                    log_file,
                    "Model ready: "
                    f"device={model_result.device}, dtype={model_result.dtype_name}, "
                    f"attn={model_result.attn_implementation}, load={model_result.load_seconds:.2f}s",
                )

                request_payload = self._load_request_payload(job_id)
                runtime_started = time.perf_counter()
                work_dir = Path(tempfile.mkdtemp(prefix=f"{job_id}_work_", dir=str(self.store.job_dir(job_id))))

                try:
                    synthesis_request = self._build_synthesis_request(
                        job_id=job_id,
                        payload=request_payload,
                        output_path=self.store.audio_path(job_id),
                        work_dir=work_dir,
                        loaded=loaded,
                        log_file=log_file,
                    )
                    with redirect_stdout(log_file), redirect_stderr(log_file):
                        result = synthesize_to_file(model_result, synthesis_request)
                finally:
                    shutil.rmtree(work_dir, ignore_errors=True)

                wall_seconds = time.perf_counter() - runtime_started
                completed_at = utcnow_iso()
                self.store.update_job(
                    job_id,
                    status="completed",
                    completed_at=completed_at,
                    audio_ready=self.store.audio_path(job_id).is_file(),
                    runtime_seconds=wall_seconds,
                    model_load_seconds=model_result.load_seconds,
                    synthesis_seconds=result.synthesis_seconds,
                    chunks=result.chunk_count,
                    sample_rate=result.sample_rate,
                    x_vector_only_mode=result.x_vector_only_mode,
                )
                log_line(log_file, f"Job {job_id} completed at {completed_at}")
            except Exception as exc:
                failed_at = utcnow_iso()
                self.store.update_job(
                    job_id,
                    status="failed",
                    failed_at=failed_at,
                    error=str(exc),
                    audio_ready=False,
                )
                log_line(log_file, f"Job {job_id} failed at {failed_at}: {exc}")

    def _load_request_payload(self, job_id: str) -> dict[str, Any]:
        request_path = self.store.job_dir(job_id) / "request.json"
        return json.loads(request_path.read_text(encoding="utf-8"))

    def _build_synthesis_request(
        self,
        *,
        job_id: str,
        payload: dict[str, Any],
        output_path: Path,
        work_dir: Path,
        loaded: LoadedServerModel,
        log_file,
    ) -> SynthesisRequest:
        task = str(payload.get("task") or self.settings.task)
        reference_audio: Path | None = None
        reference_text: str | None = payload.get("reference_text") or None
        voice_clone_prompt: Any | None = None
        x_vector_only_mode = payload.get("x_vector_only_mode")

        if task == "voice_clone":
            reference_audio, reference_text, reference_text_source = self._resolve_reference(
                job_id=job_id,
                payload=payload,
                work_dir=work_dir,
                existing_reference_text=reference_text,
            )
            self.store.update_job(job_id, reference_text_source=reference_text_source)
            log_line(log_file, f"Reference selected: {reference_audio}")
            log_line(log_file, f"Reference text source: {reference_text_source or 'missing'}")

            if x_vector_only_mode is None:
                x_vector_only_mode = not bool(reference_text)
            if x_vector_only_mode is False and not reference_text:
                raise ValueError("x_vector_only_mode=False requires reference_text.")
            voice_clone_prompt = self._get_voice_clone_prompt(
                loaded=loaded,
                reference_audio=reference_audio,
                reference_text=reference_text,
                x_vector_only_mode=bool(x_vector_only_mode),
            )

        return SynthesisRequest(
            text=str(payload["input"]).strip(),
            output_path=output_path,
            task=task,
            language=str(payload.get("language") or self.settings.language),
            reference_audio=reference_audio,
            reference_text=reference_text,
            x_vector_only_mode=x_vector_only_mode,
            voice_clone_prompt=voice_clone_prompt,
            speaker=str(payload.get("speaker") or self.settings.speaker),
            instruct=payload.get("instruct") or self.settings.instruct,
            max_chars=self.settings.max_chars,
            max_new_tokens=payload.get("max_new_tokens", self.settings.max_new_tokens),
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
        )

    def _resolve_reference(
        self,
        *,
        job_id: str,
        payload: dict[str, Any],
        work_dir: Path,
        existing_reference_text: str | None,
    ) -> tuple[Path, str | None, str | None]:
        uploaded_reference = self.store.uploaded_reference_path(job_id)
        reference_text = existing_reference_text
        reference_text_source = "request" if reference_text else None

        if uploaded_reference is not None:
            reference_path = uploaded_reference
        else:
            voice = str(payload.get("voice") or DEFAULT_REFERENCE_PREFIX)
            reference_path = find_reference_in_shared(self.settings.shared_dir, voice)
            if not reference_text:
                reference_text, reference_text_source = find_reference_text_sidecar(
                    reference_path,
                    shared_dir=self.settings.shared_dir,
                    prefix=voice,
                )

        prepared = prepare_reference_audio(reference_path, work_dir, self.settings.ffmpeg)
        return prepared, reference_text, reference_text_source

    def _get_voice_clone_prompt(
        self,
        *,
        loaded: LoadedServerModel,
        reference_audio: Path,
        reference_text: str | None,
        x_vector_only_mode: bool,
    ) -> Any:
        stat = reference_audio.stat()
        text_hash = hashlib.sha256((reference_text or "").encode("utf-8")).hexdigest()
        key = (
            str(reference_audio.resolve()),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            text_hash,
            bool(x_vector_only_mode),
        )
        if key not in loaded.prompt_cache:
            loaded.prompt_cache[key] = create_voice_clone_prompt(
                loaded.model_result.model,
                reference_audio=reference_audio,
                reference_text=reference_text,
                x_vector_only_mode=x_vector_only_mode,
            )
        return loaded.prompt_cache[key]


def parse_args(argv: list[str] | None = None) -> ServerSettings:
    parser = argparse.ArgumentParser(
        prog="qwen3-tts-server",
        description="Async Qwen3-TTS jobs server for Windows.",
    )
    parser.add_argument("--host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--shared-dir", default=str(DEFAULT_SHARED_DIR))
    parser.add_argument("--jobs-dir", default=str(DEFAULT_JOBS_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", choices=("voice_clone", "custom_voice", "voice_design"), default="voice_clone")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "sdpa", "eager", "flash_attention_2"),
        default="auto",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--job-retention-hours", type=int, default=DEFAULT_JOB_RETENTION_HOURS)
    parser.add_argument("--downloaded-job-retention-hours", type=int, default=DEFAULT_DOWNLOADED_JOB_RETENTION_HOURS)
    parser.add_argument("--cleanup-interval-seconds", type=int, default=DEFAULT_CLEANUP_INTERVAL_SECONDS)
    args = parser.parse_args(argv)
    resolved_model = resolve_model_for_task(args.model, args.task)

    return ServerSettings(
        host=args.host,
        port=args.port,
        shared_dir=resolve_shared_dir(args.shared_dir),
        jobs_dir=resolve_path(args.jobs_dir),
        model=resolved_model,
        task=args.task,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        ffmpeg=args.ffmpeg,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        max_chars=args.max_chars,
        max_new_tokens=args.max_new_tokens,
        job_retention_hours=max(args.job_retention_hours, 0),
        downloaded_job_retention_hours=max(args.downloaded_job_retention_hours, 0),
        cleanup_interval_seconds=max(args.cleanup_interval_seconds, 1),
    )


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    ensure_project_runtime_dirs()
    server_settings = settings or ServerSettings(
        shared_dir=resolve_shared_dir(str(DEFAULT_SHARED_DIR)),
        jobs_dir=DEFAULT_JOBS_DIR,
    )
    store = JobStore(server_settings.jobs_dir)
    worker = SynthesisWorker(server_settings, store)
    gc = JobGarbageCollector(server_settings, store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        gc.start()
        yield
        gc.stop()

    app = FastAPI(title="qwen3-tts-win", version="0.1.0", lifespan=lifespan)
    app.state.settings = server_settings
    app.state.store = store
    app.state.worker = worker
    app.state.gc = gc

    @app.get("/health")
    def health() -> dict[str, Any]:
        gc.sweep_now()
        return {
            "status": "ok",
            "model": server_settings.model,
            "task": server_settings.task,
            "device": server_settings.device,
            "dtype": server_settings.dtype,
            "attn_implementation": server_settings.attn_implementation,
            "shared_dir": str(server_settings.shared_dir),
            "jobs_dir": str(server_settings.jobs_dir),
            "temp_dir": str(DEFAULT_TEMP_DIR),
            "job_retention_hours": server_settings.job_retention_hours,
            "downloaded_job_retention_hours": server_settings.downloaded_job_retention_hours,
        }

    @app.post("/v1/tts/jobs", status_code=status.HTTP_202_ACCEPTED)
    def create_tts_job(request: CreateTTSJobRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        gc.sweep_now()
        normalized = normalize_request(request, server_settings)
        validate_request(normalized, server_settings)
        try:
            job = store.create_job(normalized)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        background_tasks.add_task(worker.submit, job["id"])
        return job

    @app.get("/v1/tts/jobs/{job_id}")
    def get_tts_job(job_id: str) -> dict[str, Any]:
        gc.sweep_now()
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
        return store.public_view(job)

    @app.get("/v1/tts/jobs/{job_id}/audio")
    def get_tts_job_audio(job_id: str) -> FileResponse:
        gc.sweep_now()
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

        audio_path = store.audio_path(job_id)
        if audio_path.is_file():
            store.mark_downloaded(job_id)
            return FileResponse(path=audio_path, media_type="audio/wav", filename=f"{job_id}.wav")

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Audio is not ready yet.",
                "status": job["status"],
                "error": job.get("error"),
            },
        )

    return app


def normalize_request(request: CreateTTSJobRequest, settings: ServerSettings) -> CreateTTSJobRequest:
    task = (request.task or settings.task).strip()
    instruct = request.instruct or request.instructions
    return CreateTTSJobRequest(
        input=strip_qwen_stress_marks(request.input.strip()) or "",
        model=resolve_model_for_task((request.model or settings.model).strip(), task),
        task=task,
        voice=(request.voice or DEFAULT_REFERENCE_PREFIX).strip() or DEFAULT_REFERENCE_PREFIX,
        response_format=(request.response_format or "wav").lower(),
        language=(request.language or settings.language).strip() or settings.language,
        speaker=(request.speaker or settings.speaker).strip() or settings.speaker,
        instruct=strip_qwen_stress_marks(instruct),
        instructions=None,
        reference_text=strip_qwen_stress_marks(" ".join(request.reference_text.split())) if request.reference_text else None,
        reference_audio_base64=request.reference_audio_base64,
        reference_audio_filename=request.reference_audio_filename,
        x_vector_only_mode=request.x_vector_only_mode,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        metadata=request.metadata,
    )


def validate_request(request: CreateTTSJobRequest, settings: ServerSettings) -> None:
    if not request.input:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="input must not be empty.")
    if request.model != settings.model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only model '{settings.model}' is currently loaded by this server.",
        )
    if request.task != settings.task:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only task '{settings.task}' is currently enabled by this server.",
        )
    if request.response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only response_format='wav' is supported.")
    effective_instruct = request.instruct or settings.instruct
    if effective_instruct and request.task == "voice_clone":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Qwen3-TTS Base voice_clone does not support instruct/instructions. Use custom_voice with a 1.7B CustomVoice model or voice_design.",
        )
    if effective_instruct and request.task == "custom_voice" and "0.6B-CustomVoice" in request.model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Qwen3-TTS 0.6B CustomVoice ignores instruct/instructions. Use Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice for style control.",
        )
    if request.task == "voice_clone" and not request.reference_audio_base64:
        try:
            reference_path = find_reference_in_shared(settings.shared_dir, request.voice)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if request.x_vector_only_mode is False and not request.reference_text:
            sidecar_text, _source = find_reference_text_sidecar(
                reference_path,
                shared_dir=settings.shared_dir,
                prefix=request.voice,
            )
            if not sidecar_text:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="x_vector_only_mode=false requires reference_text or a reference text sidecar.",
                )
    if request.task == "voice_clone" and request.reference_audio_base64 and request.x_vector_only_mode is False:
        if not request.reference_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded references need reference_text when x_vector_only_mode=false.",
            )


def request_to_payload(request: CreateTTSJobRequest) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    if hasattr(request, "dict"):
        return request.dict()
    return dict(vars(request))


def decode_reference_audio_to_file(encoded: str, filename: str | None, job_dir: Path) -> Path:
    mime_type, payload = split_data_uri(encoded)
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("reference_audio_base64 is not valid base64.") from exc

    suffix = infer_reference_suffix(filename, mime_type)
    reference_path = job_dir / f"reference{suffix}"
    reference_path.write_bytes(decoded)
    return reference_path


def split_data_uri(value: str) -> tuple[str | None, str]:
    normalized = value.strip()
    if normalized.startswith("data:"):
        if "," not in normalized:
            raise ValueError("Invalid data URI for reference audio.")
        header, payload = normalized.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or None
        return mime_type, payload.strip()
    return None, normalized


def infer_reference_suffix(filename: str | None, mime_type: str | None) -> str:
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix
    if mime_type and mime_type in AUDIO_MIME_TO_EXT:
        return AUDIO_MIME_TO_EXT[mime_type]
    return ".wav"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def log_line(handle, message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    handle.write(f"[{timestamp}] {message}\n")
    handle.flush()


def main(argv: list[str] | None = None) -> int:
    settings = parse_args(argv)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, access_log=False, log_level="warning")
    return 0


app = create_app()


if __name__ == "__main__":
    raise SystemExit(main())
