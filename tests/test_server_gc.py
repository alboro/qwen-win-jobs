from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if "uvicorn" not in sys.modules:
    sys.modules["uvicorn"] = types.SimpleNamespace(run=lambda *args, **kwargs: None)

if "fastapi" not in sys.modules:
    class _DummyFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()

        def get(self, *args, **kwargs):
            return lambda func: func

        def post(self, *args, **kwargs):
            return lambda func: func

    sys.modules["fastapi"] = types.SimpleNamespace(
        BackgroundTasks=object,
        FastAPI=_DummyFastAPI,
        HTTPException=Exception,
        status=types.SimpleNamespace(
            HTTP_202_ACCEPTED=202,
            HTTP_400_BAD_REQUEST=400,
            HTTP_404_NOT_FOUND=404,
            HTTP_409_CONFLICT=409,
        ),
    )

if "fastapi.responses" not in sys.modules:
    sys.modules["fastapi.responses"] = types.SimpleNamespace(FileResponse=object)

if "pydantic" not in sys.modules:
    class _DummyBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _dummy_field(default=None, **kwargs):
        return default

    sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=_DummyBaseModel, Field=_dummy_field)

from qwen3_tts_win.server import JobStore, ServerSettings, normalize_request, validate_request


class DummyRequest:
    def __init__(
        self,
        *,
        input: str,
        model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        task: str = "voice_clone",
        voice: str = "reference",
        response_format: str = "wav",
        language: str = "Russian",
        speaker: str = "Ryan",
        instruct: str | None = None,
        instructions: str | None = None,
        reference_text: str | None = None,
        reference_audio_base64: str | None = None,
        reference_audio_filename: str | None = None,
        x_vector_only_mode: bool | None = None,
        max_new_tokens: int | None = 2048,
        temperature: float | None = None,
        top_p: float | None = None,
        metadata: dict | None = None,
    ):
        self.input = input
        self.model = model
        self.task = task
        self.voice = voice
        self.response_format = response_format
        self.language = language
        self.speaker = speaker
        self.instruct = instruct
        self.instructions = instructions
        self.reference_text = reference_text
        self.reference_audio_base64 = reference_audio_base64
        self.reference_audio_filename = reference_audio_filename
        self.x_vector_only_mode = x_vector_only_mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.metadata = metadata


def iso_utc(hours_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


class TestJobStoreCleanup(unittest.TestCase):
    def test_normalize_request_inherits_server_voice_design_defaults(self):
        settings = ServerSettings(model="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", task="voice_design")
        normalized = normalize_request(DummyRequest(input="hello", model=None, task=None), settings)

        self.assertEqual(normalized.task, "voice_design")
        self.assertEqual(normalized.model, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

    def test_normalize_request_strips_qwen_stress_marks_from_text_fields(self):
        settings = ServerSettings(model="Qwen/Qwen3-TTS-12Hz-1.7B-Base", task="voice_clone")
        normalized = normalize_request(
            DummyRequest(
                input="Фра́нция",
                model=None,
                task=None,
                instruct="Скажи Царя́ спокойно",
                reference_text="То́мас Пэ́йн",
            ),
            settings,
        )

        self.assertEqual(normalized.input, "Франция")
        self.assertEqual(normalized.instruct, "Скажи Царя спокойно")
        self.assertEqual(normalized.reference_text, "Томас Пэйн")

    def test_normalize_request_accepts_openai_instructions_alias(self):
        settings = ServerSettings(model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", task="custom_voice")
        normalized = normalize_request(
            DummyRequest(
                input="hello",
                model=None,
                task=None,
                instructions="Read with a calm interrogative intonation.",
            ),
            settings,
        )

        self.assertEqual(normalized.instruct, "Read with a calm interrogative intonation.")
        self.assertIsNone(normalized.instructions)

    def test_validate_rejects_instruct_for_voice_clone(self):
        settings = ServerSettings(model="Qwen/Qwen3-TTS-12Hz-1.7B-Base", task="voice_clone")
        request = DummyRequest(
            input="hello",
            model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            task="voice_clone",
            instruct="Read dramatically.",
        )

        with self.assertRaises(Exception):
            validate_request(request, settings)

    def test_validate_rejects_instruct_for_06b_custom_voice(self):
        settings = ServerSettings(model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", task="custom_voice")
        request = DummyRequest(
            input="hello",
            model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            task="custom_voice",
            instruct="Read dramatically.",
        )

        with self.assertRaises(Exception):
            validate_request(request, settings)

    def test_mark_downloaded_updates_job_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            job = store.create_job(DummyRequest(input="hello"))
            store.update_job(job["id"], status="completed", completed_at=iso_utc(1), audio_ready=True)
            store.audio_path(job["id"]).write_bytes(b"RIFF")

            updated = store.mark_downloaded(job["id"])

            self.assertEqual(updated["download_count"], 1)
            self.assertIsNotNone(updated["first_downloaded_at"])
            self.assertIsNotNone(updated["last_downloaded_at"])

    def test_cleanup_removes_old_completed_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            job = store.create_job(DummyRequest(input="hello"))
            store.update_job(job["id"], status="completed", completed_at=iso_utc(30), audio_ready=True)
            store.audio_path(job["id"]).write_bytes(b"RIFF")

            removed = store.cleanup_expired(
                job_retention=timedelta(hours=24),
                downloaded_job_retention=timedelta(hours=6),
            )

            self.assertEqual(removed, [job["id"]])
            self.assertFalse(store.job_dir(job["id"]).exists())
            self.assertIsNone(store.get_job(job["id"]))


if __name__ == "__main__":
    unittest.main()
