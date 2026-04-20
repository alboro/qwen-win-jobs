from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_CUSTOM_VOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_LANGUAGE = "Russian"
DEFAULT_REFERENCE_PREFIX = "reference"
DEFAULT_SPEAKER = "Ryan"
DEFAULT_MAX_CHARS = 500
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_SILENCE_SECONDS = 0.18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_DIR = PROJECT_ROOT / "shared"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_DATA_DIR = PROJECT_ROOT / ".data"
DEFAULT_TEMP_DIR = PROJECT_ROOT / ".tmp"

REFERENCE_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".mkv",
    ".webm",
}


@dataclass(slots=True)
class ModelLoadResult:
    model: Any
    device: str
    dtype_name: str
    attn_implementation: str
    load_seconds: float


@dataclass(slots=True)
class SynthesisRequest:
    text: str
    output_path: Path
    task: str
    language: str = DEFAULT_LANGUAGE
    reference_audio: Path | None = None
    reference_text: str | None = None
    x_vector_only_mode: bool | None = None
    voice_clone_prompt: Any | None = None
    speaker: str = DEFAULT_SPEAKER
    instruct: str | None = None
    max_chars: int = DEFAULT_MAX_CHARS
    max_new_tokens: int | None = DEFAULT_MAX_NEW_TOKENS
    temperature: float | None = None
    top_p: float | None = None
    silence_seconds: float = DEFAULT_SILENCE_SECONDS


@dataclass(slots=True)
class SynthesisResult:
    output_path: Path
    chunk_count: int
    sample_rate: int
    synthesis_seconds: float
    x_vector_only_mode: bool | None = None


def ensure_project_runtime_dirs() -> None:
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_DATA_DIR / "huggingface").mkdir(parents=True, exist_ok=True)
    (DEFAULT_DATA_DIR / "cache").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(DEFAULT_DATA_DIR / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_DATA_DIR / "huggingface" / "hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_DATA_DIR / "cache"))
    os.environ.setdefault("TEMP", str(DEFAULT_TEMP_DIR))
    os.environ.setdefault("TMP", str(DEFAULT_TEMP_DIR))


def resolve_path(path_value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base_dir or Path.cwd()) / path).resolve()


def resolve_shared_dir(path_value: str | Path) -> Path:
    shared_dir = resolve_path(path_value)
    shared_dir.mkdir(parents=True, exist_ok=True)
    return shared_dir


def resolve_output_path(path: str | Path, *, overwrite: bool) -> Path:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    return output_path


def resolve_existing_file(value: str | Path, shared_dir: Path | None = None) -> Path | None:
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw]
    if shared_dir is not None and not raw.is_absolute():
        candidates.append(shared_dir / raw)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_text_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Input text file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input text file is empty: {path}")
    return text


def is_supported_reference_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in REFERENCE_EXTENSIONS


def find_reference_in_shared(shared_dir: Path, prefix: str) -> Path:
    prefix_normalized = prefix.lower()
    candidates = [
        candidate.resolve()
        for candidate in shared_dir.rglob("*")
        if is_supported_reference_file(candidate) and candidate.stem.lower().startswith(prefix_normalized)
    ]
    if not candidates:
        raise FileNotFoundError(f"No reference files found in {shared_dir} for prefix '{prefix}'.")
    return max(candidates, key=lambda item: (item.stat().st_mtime, item.name.lower()))


def resolve_reference_arg(
    reference_value: str | None,
    shared_dir: Path,
    default_prefix: str = DEFAULT_REFERENCE_PREFIX,
) -> tuple[Path, str, str]:
    if reference_value:
        resolved_file = resolve_existing_file(reference_value, shared_dir)
        if resolved_file is not None:
            return resolved_file, "explicit file", resolved_file.stem
        prefix = Path(reference_value).stem or reference_value
    else:
        prefix = default_prefix

    reference_path = find_reference_in_shared(shared_dir, prefix)
    return reference_path, f"newest shared match for prefix '{prefix}'", prefix


def read_reference_text_file(path: Path) -> str:
    text = read_text_file(path)
    return " ".join(text.split())


def find_reference_text_sidecar(
    reference_path: Path,
    *,
    shared_dir: Path | None = None,
    prefix: str | None = None,
) -> tuple[str | None, str | None]:
    candidates: list[Path] = [
        reference_path.with_suffix(".txt"),
        reference_path.with_name(f"{reference_path.name}.txt"),
    ]
    if prefix and shared_dir is not None:
        candidates.append(shared_dir / f"{prefix}.txt")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return read_reference_text_file(candidate), str(candidate)
    return None, None


def resolve_reference_text(
    *,
    explicit_text: str | None,
    text_file: str | None,
    reference_path: Path,
    shared_dir: Path | None,
    prefix: str | None,
) -> tuple[str | None, str | None]:
    if explicit_text and explicit_text.strip():
        return " ".join(explicit_text.split()), "explicit text"
    if text_file:
        resolved = resolve_existing_file(text_file, shared_dir)
        if resolved is None:
            raise FileNotFoundError(f"Reference text file not found: {text_file}")
        return read_reference_text_file(resolved), str(resolved)
    return find_reference_text_sidecar(reference_path, shared_dir=shared_dir, prefix=prefix)


def resolve_ffmpeg(ffmpeg_value: str) -> str | None:
    candidate = Path(ffmpeg_value)
    if candidate.is_file():
        return str(candidate.resolve())

    discovered = shutil.which(ffmpeg_value)
    if discovered:
        return discovered

    for fallback in iter_ffmpeg_fallbacks():
        if fallback.is_file():
            return str(fallback.resolve())
    return None


def iter_ffmpeg_fallbacks():
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_packages.is_dir():
            yield from winget_packages.glob("*FFmpeg*/*/bin/ffmpeg.exe")
        yield Path(local_app_data) / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe"

    yield Path("C:/ffmpeg/bin/ffmpeg.exe")
    yield Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe")


def require_ffmpeg(ffmpeg_value: str) -> str:
    ffmpeg_bin = resolve_ffmpeg(ffmpeg_value)
    if ffmpeg_bin:
        return ffmpeg_bin
    raise RuntimeError(
        "ffmpeg was not found. Install ffmpeg and add it to PATH, or pass --ffmpeg C:\\path\\to\\ffmpeg.exe."
    )


def run_command(command: list[str], description: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return
    details = (result.stderr or result.stdout or "No details returned.").strip()
    raise RuntimeError(f"{description} failed.\n{details}")


def convert_reference_to_wav(reference_path: Path, work_dir: Path, ffmpeg_bin: str) -> Path:
    if reference_path.suffix.lower() == ".wav":
        return reference_path

    converted_path = work_dir / f"{reference_path.stem}_reference.wav"
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(reference_path),
        "-vn",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(converted_path),
    ]
    run_command(command, "Reference conversion")
    return converted_path


def prepare_reference_audio(reference_path: Path, work_dir: Path, ffmpeg_value: str) -> Path:
    reference_path = reference_path.expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference_path}")
    if reference_path.suffix.lower() == ".wav":
        return reference_path
    return convert_reference_to_wav(reference_path, work_dir, require_ffmpeg(ffmpeg_value))


def estimate_audio_duration_seconds(text: str) -> float:
    words = len(text.split())
    return (words / 150.0) * 60.0


def split_text(text: str, max_chars: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    tokens = re.split(r"(\s*[.!?]+[\"')\]]*(?:\s+|$))", text)
    sentences: list[str] = []
    for index in range(0, len(tokens), 2):
        sentence = tokens[index] + (tokens[index + 1] if index + 1 < len(tokens) else "")
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)

    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_sentence(sentence, max_chars))
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    start = 0
    while start < len(sentence):
        hard_cut = min(start + max_chars, len(sentence))
        if hard_cut >= len(sentence):
            parts.append(sentence[start:].strip())
            break
        soft_cut = max(
            sentence.rfind(",", start, hard_cut),
            sentence.rfind(";", start, hard_cut),
            sentence.rfind(":", start, hard_cut),
            sentence.rfind(" ", start, hard_cut),
        )
        cut = soft_cut if soft_cut > start else hard_cut
        parts.append(sentence[start:cut].strip())
        start = cut
    return parts


def select_device(requested_device: str) -> str:
    import torch

    if requested_device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() returned False.")
        return "cuda:0"
    if requested_device == "cpu":
        return "cpu"
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() returned False.")
        return requested_device
    raise ValueError(f"Unsupported device: {requested_device}")


def resolve_dtype(dtype_name: str, device: str):
    import torch

    normalized = dtype_name.lower()
    if normalized == "auto":
        if device.startswith("cuda"):
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
            return ("bfloat16", torch.bfloat16) if is_bf16_supported() else ("float16", torch.float16)
        return "float32", torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return "float16", torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return "bfloat16", torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return "float32", torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_attn_implementation(attn_implementation: str, *, device: str, dtype_name: str) -> str:
    normalized = attn_implementation.lower()
    if normalized != "auto":
        return normalized
    if device.startswith("cuda") and dtype_name in {"float16", "bfloat16"}:
        if importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        return "sdpa"
    return "eager"


def load_qwen_model(
    *,
    model_name: str,
    requested_device: str = "auto",
    requested_dtype: str = "auto",
    requested_attn_implementation: str = "auto",
) -> ModelLoadResult:
    ensure_project_runtime_dirs()
    from qwen_tts import Qwen3TTSModel

    device = select_device(requested_device)
    dtype_name, dtype = resolve_dtype(requested_dtype, device)
    attn = resolve_attn_implementation(
        requested_attn_implementation,
        device=device,
        dtype_name=dtype_name,
    )

    started = time.perf_counter()
    kwargs = {
        "device_map": device,
        "dtype": dtype,
        "attn_implementation": attn,
    }
    try:
        model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
    except Exception:
        if attn == "eager":
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["attn_implementation"] = "eager"
        model = Qwen3TTSModel.from_pretrained(model_name, **fallback_kwargs)
        attn = "eager"

    return ModelLoadResult(
        model=model,
        device=device,
        dtype_name=dtype_name,
        attn_implementation=attn,
        load_seconds=time.perf_counter() - started,
    )


def build_generation_kwargs(request: SynthesisRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if request.max_new_tokens is not None:
        kwargs["max_new_tokens"] = request.max_new_tokens
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    return kwargs


def create_voice_clone_prompt(
    model: Any,
    *,
    reference_audio: Path,
    reference_text: str | None,
    x_vector_only_mode: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "ref_audio": str(reference_audio),
        "x_vector_only_mode": x_vector_only_mode,
    }
    if reference_text:
        kwargs["ref_text"] = reference_text
    return model.create_voice_clone_prompt(**kwargs)


def synthesize_to_file(loaded_model: ModelLoadResult, request: SynthesisRequest) -> SynthesisResult:
    import numpy as np
    import soundfile as sf

    started = time.perf_counter()
    chunks = split_text(request.text, request.max_chars)
    if not chunks:
        raise ValueError("Text must not be empty.")

    generated: list[Any] = []
    sample_rate: int | None = None
    generation_kwargs = build_generation_kwargs(request)
    task = request.task.lower()
    x_vector_only = request.x_vector_only_mode

    if task == "voice_clone":
        if request.reference_audio is None:
            raise ValueError("Voice clone task requires reference_audio.")
        if x_vector_only is None:
            x_vector_only = not bool(request.reference_text)
        if x_vector_only is False and not request.reference_text:
            raise ValueError("x_vector_only_mode=False requires reference_text.")

        prompt = request.voice_clone_prompt or create_voice_clone_prompt(
            loaded_model.model,
            reference_audio=request.reference_audio,
            reference_text=request.reference_text,
            x_vector_only_mode=bool(x_vector_only),
        )
        for chunk in chunks:
            wavs, sr = loaded_model.model.generate_voice_clone(
                text=chunk,
                language=request.language,
                voice_clone_prompt=prompt,
                **generation_kwargs,
            )
            generated.append(wavs[0])
            sample_rate = int(sr)
    elif task == "custom_voice":
        for chunk in chunks:
            wavs, sr = loaded_model.model.generate_custom_voice(
                text=chunk,
                language=request.language,
                speaker=request.speaker,
                instruct=request.instruct or "",
                **generation_kwargs,
            )
            generated.append(wavs[0])
            sample_rate = int(sr)
    else:
        raise ValueError(f"Unsupported task: {request.task}")

    if sample_rate is None:
        raise RuntimeError("Qwen3-TTS returned no audio.")

    audio = concatenate_audio(generated, sample_rate=sample_rate, silence_seconds=request.silence_seconds)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(request.output_path), audio, sample_rate)
    return SynthesisResult(
        output_path=request.output_path,
        chunk_count=len(chunks),
        sample_rate=sample_rate,
        synthesis_seconds=time.perf_counter() - started,
        x_vector_only_mode=x_vector_only,
    )


def concatenate_audio(chunks: list[Any], *, sample_rate: int, silence_seconds: float):
    import numpy as np

    arrays = [audio_to_numpy(chunk) for chunk in chunks]
    if len(arrays) == 1:
        return arrays[0]

    silence_len = max(int(sample_rate * silence_seconds), 0)
    pieces = []
    for index, array in enumerate(arrays):
        if index > 0 and silence_len > 0:
            if array.ndim == 1:
                silence_shape = (silence_len,)
            else:
                silence_shape = (silence_len, array.shape[1])
            pieces.append(np.zeros(silence_shape, dtype=array.dtype))
        pieces.append(array)
    return np.concatenate(pieces, axis=0)


def audio_to_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[0] < array.shape[1]:
        array = array.T
    return array.astype("float32", copy=False)


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S %Z")


def format_duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if hours >= 1:
        parts.append(f"{int(hours)}h")
    if minutes >= 1 or hours >= 1:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{secs:.1f}s")
    return " ".join(parts)


def cuda_summary() -> list[str]:
    lines: list[str] = []
    try:
        import torch

        lines.append(f"torch: {torch.__version__}")
        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                lines.append(f"cuda_device_{index}: {torch.cuda.get_device_name(index)}")
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
            lines.append(f"cuda_bf16_supported: {is_bf16_supported()}")
    except Exception as exc:
        lines.append(f"torch: ERROR ({exc})")
    return lines


def package_summary() -> list[str]:
    lines = [f"Python: {sys.version.split()[0]}"]
    packages = (
        ("qwen_tts", "qwen-tts"),
        ("soundfile", "soundfile"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
    )
    for module_name, distribution_name in packages:
        try:
            version = importlib.metadata.version(distribution_name)
            lines.append(f"{module_name}: {version}")
        except importlib.metadata.PackageNotFoundError:
            try:
                __import__(module_name)
                lines.append(f"{module_name}: OK")
            except Exception as exc:
                lines.append(f"{module_name}: ERROR ({exc})")
    return lines


def make_work_dir(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{output_path.stem}_work_", dir=str(output_path.parent)))
