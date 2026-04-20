$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $ScriptDir "qwen3-tts.cmd") @args
exit $LASTEXITCODE
