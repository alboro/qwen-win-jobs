$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $ScriptDir "qwen3-tts-server.cmd") @args
exit $LASTEXITCODE
