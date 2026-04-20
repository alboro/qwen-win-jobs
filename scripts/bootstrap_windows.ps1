param(
    [string]$PythonVersion = "3.12",
    [string]$PythonExeOverride = "",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$TorchVersion = "2.8.0",
    [switch]$InstallFlashAttention
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$PipCacheDir = Join-Path $ProjectRoot ".pip-cache"
$TempDir = Join-Path $ProjectRoot ".tmp"

Write-Host "Project root: $ProjectRoot"

New-Item -ItemType Directory -Force -Path $PipCacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot ".data\huggingface") | Out-Null

$env:PIP_CACHE_DIR = $PipCacheDir
$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:HF_HOME = Join-Path $ProjectRoot ".data\huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ProjectRoot ".data\huggingface\hub"
$env:XDG_CACHE_HOME = Join-Path $ProjectRoot ".data\cache"

function Resolve-BasePython {
    param(
        [string]$PythonVersion,
        [string]$PythonExeOverride
    )

    if ($PythonExeOverride) {
        if (-not (Test-Path -LiteralPath $PythonExeOverride)) {
            throw "PythonExeOverride does not exist: $PythonExeOverride"
        }
        return $PythonExeOverride
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        $resolved = (& py -$PythonVersion -c "import sys; print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0) {
            return $resolved.Trim()
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $resolved = (& python -c "import sys; print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0) {
            return $resolved.Trim()
        }
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "No Python executable found. Install Python $PythonVersion or pass -PythonExeOverride."
}

$BasePython = Resolve-BasePython -PythonVersion $PythonVersion -PythonExeOverride $PythonExeOverride
Write-Host "Base Python: $BasePython"

if (-not (Test-Path -LiteralPath $VenvPath)) {
    Write-Host "Creating virtual environment..."
    & $BasePython -m venv $VenvPath
}

Write-Host "Upgrading pip/setuptools/wheel..."
& $PythonExe -m pip install --upgrade --no-cache-dir pip setuptools wheel

Write-Host "Installing PyTorch + torchaudio CUDA wheels..."
& $PythonExe -m pip install --no-cache-dir "torch==$TorchVersion" "torchaudio==$TorchVersion" --index-url $TorchIndexUrl

Write-Host "Installing qwen-tts..."
& $PythonExe -m pip install --upgrade --no-cache-dir qwen-tts

if ($InstallFlashAttention) {
    Write-Host "Installing flash-attn. This is optional and may fail on Windows toolchains."
    $env:MAX_JOBS = "4"
    & $PythonExe -m pip install --upgrade --no-cache-dir flash-attn --no-build-isolation
}

Write-Host "Installing project in editable mode..."
& $PythonExe -m pip install --no-cache-dir -e $ProjectRoot

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($null -eq $ffmpeg) {
    Write-Warning "ffmpeg was not found in PATH. Install ffmpeg before using non-WAV references."
} else {
    Write-Host "ffmpeg: $($ffmpeg.Source)"
}

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Next commands:"
Write-Host "  .\qwen3-tts.cmd --doctor"
Write-Host "  .\qwen3-tts-server.cmd --host 127.0.0.1 --port 8030"
Write-Host "  .\qwen3-tts.cmd text.txt .\output\hello.wav reference --ref-text-file .\shared\reference.txt"
