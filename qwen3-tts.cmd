@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "PROJECT_ROOT=%~dp0"
set "VENV_DIR=%PROJECT_ROOT%.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "VENV_CFG=%VENV_DIR%\pyvenv.cfg"
set "VENV_SITE_PACKAGES=%VENV_DIR%\Lib\site-packages"

if not exist "%PYTHON_EXE%" (
  echo Virtual environment not found at ".venv". Run "scripts\bootstrap_windows.cmd" first.
  exit /b 1
)

set "BASE_PYTHON="
if exist "%VENV_CFG%" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%VENV_CFG%") do (
    set "CFG_KEY=%%~A"
    set "CFG_VALUE=%%~B"
    call :trim_var CFG_KEY
    call :trim_var CFG_VALUE
    if /I "!CFG_KEY!"=="executable" set "BASE_PYTHON=!CFG_VALUE!"
    if /I "!CFG_KEY!"=="home" if not defined BASE_PYTHON set "BASE_PYTHON=!CFG_VALUE!\python.exe"
  )
)

set "HF_HOME=%PROJECT_ROOT%.data\huggingface"
set "HUGGINGFACE_HUB_CACHE=%PROJECT_ROOT%.data\huggingface\hub"
set "XDG_CACHE_HOME=%PROJECT_ROOT%.data\cache"
set "TEMP=%PROJECT_ROOT%.tmp"
set "TMP=%PROJECT_ROOT%.tmp"

if not exist "%PROJECT_ROOT%.data" mkdir "%PROJECT_ROOT%.data"
if not exist "%PROJECT_ROOT%.tmp" mkdir "%PROJECT_ROOT%.tmp"

if defined BASE_PYTHON if exist "!BASE_PYTHON!" if exist "%VENV_SITE_PACKAGES%" (
  if defined PYTHONPATH (
    set "PYTHONPATH=%PROJECT_ROOT%src;%VENV_SITE_PACKAGES%;%PYTHONPATH%"
  ) else (
    set "PYTHONPATH=%PROJECT_ROOT%src;%VENV_SITE_PACKAGES%"
  )
  set "VIRTUAL_ENV=%VENV_DIR%"
  "!BASE_PYTHON!" -m qwen3_tts_win %*
) else (
  "%PYTHON_EXE%" -m qwen3_tts_win %*
)
exit /b %ERRORLEVEL%

:trim_var
setlocal EnableDelayedExpansion
set "VALUE=!%~1!"
for /f "tokens=* delims= " %%Z in ("!VALUE!") do set "VALUE=%%Z"
:trim_loop
if defined VALUE if "!VALUE:~-1!"==" " (
  set "VALUE=!VALUE:~0,-1!"
  goto trim_loop
)
endlocal & set "%~1=%VALUE%"
goto :eof
