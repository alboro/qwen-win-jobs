# qwen3-tts-win

Windows-first Qwen3-TTS 0.6B toolkit for local Russian speech synthesis on CUDA.

This project mirrors the shape of `tts_win`: a file-first CLI plus a small async
FastAPI jobs server with polling and WAV download endpoints. The default model is
`Qwen/Qwen3-TTS-12Hz-0.6B-Base`, which is the 0.6B voice-clone checkpoint.

## Hardware Target

- Windows 10 or 11
- NVIDIA GeForce RTX 3070 Ti or similar CUDA GPU
- CUDA-capable PyTorch wheel
- ffmpeg for non-WAV references

The bootstrap script installs CUDA PyTorch first, then installs this project and
`qwen-tts`. Runtime defaults are chosen for this GPU class:

- `device=auto` resolves to `cuda:0` when CUDA is available
- `dtype=auto` resolves to `float16` on RTX 3070 Ti
- `attn_implementation=auto` uses `flash_attention_2` only if it is installed,
  otherwise it uses PyTorch SDPA

## Bootstrap

```cmd
scripts\bootstrap_windows.cmd
qwen3-tts.cmd --doctor
```

The first real synthesis downloads Qwen3-TTS weights into `.data\huggingface`.

## Reference Audio

Put voice references into `shared\`. The newest file matching the requested
prefix wins:

- `shared\reference.wav`
- `shared\reference_long.m4a`
- `shared\alla_2026-04-20.flac`

Qwen3-TTS Base works best when the reference audio transcript is provided. Use
one of these:

- CLI: `--ref-text "exact transcript of the reference audio"`
- CLI: `--ref-text-file shared\reference.txt`
- API: `reference_text`
- Sidecar auto-discovery: `reference.txt`, `reference_long.txt`, or
  `reference_long.m4a.txt`

If no transcript is available, the toolkit uses `x_vector_only_mode=True`. That
keeps synthesis usable, but voice cloning quality can be lower.

## CLI

```cmd
qwen3-tts.cmd text.txt .\output\speech.wav reference --ref-text-file .\shared\reference.txt
```

Inline text:

```cmd
qwen3-tts.cmd --text "Privet. Eto lokalnyy test sinteza." .\output\hello.wav reference
```

Force CUDA and float16:

```cmd
qwen3-tts.cmd text.txt .\output\speech.wav reference --device cuda --dtype float16
```

Use a custom-voice checkpoint instead of voice clone:

```cmd
qwen3-tts.cmd text.txt .\output\speech.wav --task custom_voice --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --speaker Ryan
```

Use the 1.7B CustomVoice checkpoint when you need `instruct`/`instructions`
style control:

```cmd
qwen3-tts-server.cmd --host 0.0.0.0 --port 8030 --device cuda --dtype float16 --task custom_voice --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --language Russian --speaker Ryan
```

## Async Server

```cmd
qwen3-tts-server.cmd --host 127.0.0.1 --port 8030
```

Health check:

```cmd
curl http://127.0.0.1:8030/health
```

Create a job:

```json
POST /v1/tts/jobs
{
  "input": "Privet. Eto test russkogo sinteza.",
  "voice": "reference",
  "reference_text": "Transcript of the reference audio.",
  "response_format": "wav"
}
```

Poll status:

```cmd
curl http://127.0.0.1:8030/v1/tts/jobs/<job_id>
```

Download audio:

```cmd
curl http://127.0.0.1:8030/v1/tts/jobs/<job_id>/audio --output result.wav
```

## API Notes

- `voice` is a local shared reference prefix, not a hosted voice name.
- `instructions` is accepted as an OpenAI-style alias for Qwen `instruct`.
- Qwen Base `voice_clone` does not support `instruct`/`instructions`.
- Qwen `0.6B-CustomVoice` ignores `instruct`/`instructions`; use
  `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` for style control.
- `reference_audio_base64` overrides shared reference lookup for that request.
- `reference_text` is strongly recommended for Qwen3-TTS Base voice clone.
- `response_format` is currently `wav` only.
- The server uses one worker thread to keep GPU generation serialized.
- Jobs are persisted under `.data\jobs`.
- Terminal jobs are cleaned up automatically.

## Responsible Use

Use only voices and recordings you have the right to use.
