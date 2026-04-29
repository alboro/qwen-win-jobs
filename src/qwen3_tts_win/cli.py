from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from qwen3_tts_win.core import (
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REFERENCE_PREFIX,
    DEFAULT_SHARED_DIR,
    DEFAULT_SPEAKER,
    DEFAULT_TEMP_DIR,
    cuda_summary,
    ensure_project_runtime_dirs,
    estimate_audio_duration_seconds,
    find_reference_in_shared,
    find_reference_text_sidecar,
    format_duration,
    format_timestamp,
    load_qwen_model,
    make_work_dir,
    package_summary,
    prepare_reference_audio,
    read_text_file,
    resolve_attn_implementation,
    resolve_dtype,
    resolve_ffmpeg,
    resolve_model_for_task,
    resolve_output_path,
    resolve_path,
    resolve_reference_arg,
    resolve_reference_text,
    resolve_shared_dir,
    select_device,
    strip_qwen_stress_marks,
    synthesize_to_file,
    SynthesisRequest,
)


@dataclass(slots=True)
class ResolvedRun:
    text: str
    output_path: Path
    text_source_label: str
    reference_path: Path | None
    reference_source_label: str | None
    reference_prefix: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3-tts",
        description="Windows-first Qwen3-TTS 0.6B CLI for Russian synthesis.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Voice clone: INPUT_FILE OUTPUT [REFERENCE_OR_PREFIX]. "
            "With --text: TEXT OUTPUT [REFERENCE_OR_PREFIX]. "
            "With --file: OUTPUT [REFERENCE_OR_PREFIX]. "
            "Custom voice / voice design: INPUT_FILE OUTPUT, or --text TEXT OUTPUT."
        ),
    )
    parser.add_argument("--file", dest="input_file", help="Read text from a UTF-8 file.")
    parser.add_argument("--text", action="store_true", help="Treat the first positional argument as literal text.")
    parser.add_argument("--shared-dir", default=str(DEFAULT_SHARED_DIR))
    parser.add_argument("--reference-prefix", default=DEFAULT_REFERENCE_PREFIX)
    parser.add_argument("--ref-text", default=None, help="Transcript of the reference audio.")
    parser.add_argument("--ref-text-file", default=None, help="UTF-8 file with the reference audio transcript.")
    parser.add_argument(
        "--require-ref-text",
        action="store_true",
        help="Fail instead of using x_vector_only_mode when no reference transcript is found.",
    )
    parser.add_argument(
        "--x-vector-only-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help="Voice clone prompt mode. auto uses x-vector only when no reference text is available.",
    )
    parser.add_argument("--task", choices=("voice_clone", "custom_voice", "voice_design"), default="voice_clone")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER)
    parser.add_argument(
        "--instruct",
        default=None,
        help="Optional natural-language instruction for custom_voice or voice_design.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "sdpa", "eager", "flash_attention_2"),
        default="auto",
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--doctor", action="store_true", help="Print runtime diagnostics and exit.")
    return parser


def resolve_cli_inputs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    shared_dir: Path,
) -> ResolvedRun:
    if args.input_file and args.text:
        parser.error("Use either --file or --text, not both.")

    wants_reference = args.task == "voice_clone"

    if args.input_file:
        allowed_lengths = (1, 2) if wants_reference else (1,)
        if len(args.inputs) not in allowed_lengths:
            parser.error("With --file pass OUTPUT [REFERENCE_OR_PREFIX] for voice_clone, or OUTPUT for custom_voice/voice_design.")
        input_path = resolve_path(args.input_file)
        text = read_text_file(input_path)
        output_path = Path(args.inputs[0])
        reference_value = args.inputs[1] if wants_reference and len(args.inputs) == 2 else None
        return with_optional_reference(text, output_path, str(input_path), reference_value, args, shared_dir)

    if args.text:
        allowed_lengths = (2, 3) if wants_reference else (2,)
        if len(args.inputs) not in allowed_lengths:
            parser.error("With --text pass TEXT OUTPUT [REFERENCE_OR_PREFIX] for voice_clone, or TEXT OUTPUT for custom_voice/voice_design.")
        text = args.inputs[0].strip()
        if not text:
            parser.error("Text must not be empty.")
        output_path = Path(args.inputs[1])
        reference_value = args.inputs[2] if wants_reference and len(args.inputs) == 3 else None
        return with_optional_reference(text, output_path, "<inline text>", reference_value, args, shared_dir)

    allowed_lengths = (2, 3) if wants_reference else (2,)
    if len(args.inputs) not in allowed_lengths:
        parser.error("Usage: INPUT_FILE OUTPUT [REFERENCE_OR_PREFIX]")

    input_path = resolve_path(args.inputs[0])
    text = read_text_file(input_path)
    output_path = Path(args.inputs[1])
    reference_value = args.inputs[2] if wants_reference and len(args.inputs) == 3 else None
    return with_optional_reference(text, output_path, str(input_path), reference_value, args, shared_dir)


def with_optional_reference(
    text: str,
    output_path: Path,
    text_source_label: str,
    reference_value: str | None,
    args: argparse.Namespace,
    shared_dir: Path,
) -> ResolvedRun:
    if args.task != "voice_clone":
        return ResolvedRun(
            text=text,
            output_path=output_path,
            text_source_label=text_source_label,
            reference_path=None,
            reference_source_label=None,
            reference_prefix=None,
        )

    reference_path, label, prefix = resolve_reference_arg(
        reference_value=reference_value,
        shared_dir=shared_dir,
        default_prefix=args.reference_prefix,
    )
    return ResolvedRun(
        text=text,
        output_path=output_path,
        text_source_label=text_source_label,
        reference_path=reference_path,
        reference_source_label=label,
        reference_prefix=prefix,
    )


def parse_x_vector_only_mode(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "on"


def run_doctor(args: argparse.Namespace, shared_dir: Path) -> int:
    ensure_project_runtime_dirs()
    print(f"project_root: {Path(__file__).resolve().parents[2]}")
    print(f"shared_dir: {shared_dir}")
    print(f"temp_dir: {DEFAULT_TEMP_DIR}")

    ffmpeg_bin = resolve_ffmpeg(args.ffmpeg)
    print(f"ffmpeg: {'OK (' + ffmpeg_bin + ')' if ffmpeg_bin else 'MISSING'}")

    try:
        reference_path = find_reference_in_shared(shared_dir, args.reference_prefix)
        print(f"default_reference: {reference_path}")
        ref_text, source = find_reference_text_sidecar(
            reference_path,
            shared_dir=shared_dir,
            prefix=args.reference_prefix,
        )
        if ref_text:
            print(f"default_reference_text: OK ({source})")
        else:
            print("default_reference_text: MISSING (will use x_vector_only_mode=True)")
    except Exception as exc:
        print(f"default_reference: ERROR ({exc})")

    for line in package_summary():
        print(line)
    for line in cuda_summary():
        print(line)

    try:
        device = select_device(args.device)
        dtype_name, _dtype = resolve_dtype(args.dtype, device)
        attn = resolve_attn_implementation(args.attn_implementation, device=device, dtype_name=dtype_name)
        print(f"selected_device: {device}")
        print(f"selected_dtype: {dtype_name}")
        print(f"selected_attn_implementation: {attn}")
    except Exception as exc:
        print(f"selection: ERROR ({exc})")

    print(f"default_model: {resolve_model_for_task(args.model, args.task)}")
    return 0


def print_run_summary(
    *,
    started_at: datetime,
    args: argparse.Namespace,
    resolved: ResolvedRun,
    output_path: Path,
    reference_text_source: str | None,
    reference_text: str | None,
) -> None:
    word_count = len(resolved.text.split())
    effective_model = resolve_model_for_task(args.model, args.task)
    print(f"Start: {format_timestamp(started_at)}")
    print(f"Task: {args.task}")
    print(f"Model: {effective_model}")
    print(f"Input: {resolved.text_source_label}")
    print(f"Output: {output_path}")
    print(f"Language: {args.language}")
    print(f"Text size: {len(resolved.text)} chars, {word_count} words")
    print(f"Estimated speech: {format_duration(estimate_audio_duration_seconds(resolved.text))}")
    print(f"Chunk max chars: {args.max_chars}")
    if args.task == "voice_clone":
        print(f"Reference: {resolved.reference_path}")
        print(f"Reference source: {resolved.reference_source_label}")
        print(f"Reference text: {reference_text_source or 'missing'}")
        if not reference_text:
            print("Reference mode: x_vector_only_mode=True unless --x-vector-only-mode off is used")
    else:
        print(f"Speaker: {args.speaker}")
        if args.instruct:
            print(f"Instruct: {args.instruct}")


def print_finish_summary(
    *,
    started_at: datetime,
    model_load_seconds: float,
    result_seconds: float,
    wall_seconds: float,
    chunk_count: int,
    sample_rate: int,
    x_vector_only_mode: bool | None,
) -> None:
    finished_at = datetime.now().astimezone()
    print(f"Finish: {format_timestamp(finished_at)}")
    print(f"Started at {format_timestamp(started_at)}, finished at {format_timestamp(finished_at)}.")
    print(f"Wall time: {format_duration(wall_seconds)}")
    print(f"Model load: {format_duration(model_load_seconds)}")
    print(f"Synthesis: {format_duration(result_seconds)}")
    print(f"Chunks: {chunk_count}")
    print(f"Sample rate: {sample_rate}")
    if x_vector_only_mode is not None:
        print(f"x_vector_only_mode: {x_vector_only_mode}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    shared_dir = resolve_shared_dir(args.shared_dir)

    if args.doctor:
        return run_doctor(args, shared_dir)

    try:
        ensure_project_runtime_dirs()
        args.model = resolve_model_for_task(args.model, args.task)
        resolved = resolve_cli_inputs(args, parser, shared_dir)
        output_path = resolve_output_path(resolved.output_path, overwrite=args.overwrite)
        text_for_tts = strip_qwen_stress_marks(resolved.text) or ""

        reference_text: str | None = None
        reference_text_source: str | None = None
        if args.task == "voice_clone":
            assert resolved.reference_path is not None
            reference_text, reference_text_source = resolve_reference_text(
                explicit_text=args.ref_text,
                text_file=args.ref_text_file,
                reference_path=resolved.reference_path,
                shared_dir=shared_dir,
                prefix=resolved.reference_prefix,
            )
            if args.require_ref_text and not reference_text:
                raise ValueError("Reference text is required but was not provided or found as a sidecar.")
            reference_text = strip_qwen_stress_marks(reference_text)

        started_at = datetime.now().astimezone()
        wall_started = time.perf_counter()
        print_run_summary(
            started_at=started_at,
            args=args,
            resolved=resolved,
            output_path=output_path,
            reference_text_source=reference_text_source,
            reference_text=reference_text,
        )

        work_dir = make_work_dir(output_path)
        try:
            reference_audio: Path | None = None
            if args.task == "voice_clone":
                assert resolved.reference_path is not None
                reference_audio = prepare_reference_audio(resolved.reference_path, work_dir, args.ffmpeg)

            print(f"Loading model on device={args.device}, dtype={args.dtype}, attn={args.attn_implementation}")
            loaded_model = load_qwen_model(
                model_name=args.model,
                requested_device=args.device,
                requested_dtype=args.dtype,
                requested_attn_implementation=args.attn_implementation,
            )
            print(
                "Model ready: "
                f"device={loaded_model.device}, dtype={loaded_model.dtype_name}, "
                f"attn={loaded_model.attn_implementation}, "
                f"load={format_duration(loaded_model.load_seconds)}"
            )

            request = SynthesisRequest(
                text=text_for_tts,
                output_path=output_path,
                task=args.task,
                language=args.language,
                reference_audio=reference_audio,
                reference_text=reference_text,
                x_vector_only_mode=parse_x_vector_only_mode(args.x_vector_only_mode),
                speaker=args.speaker,
                instruct=strip_qwen_stress_marks(args.instruct),
                max_chars=args.max_chars,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            result = synthesize_to_file(loaded_model, request)
            wall_seconds = time.perf_counter() - wall_started
            print(f"Saved: {output_path}")
            print_finish_summary(
                started_at=started_at,
                model_load_seconds=loaded_model.load_seconds,
                result_seconds=result.synthesis_seconds,
                wall_seconds=wall_seconds,
                chunk_count=result.chunk_count,
                sample_rate=result.sample_rate,
                x_vector_only_mode=result.x_vector_only_mode,
            )
            return 0
        finally:
            if args.keep_temp:
                print(f"Temporary files kept at: {work_dir}")
            else:
                shutil.rmtree(work_dir, ignore_errors=True)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
