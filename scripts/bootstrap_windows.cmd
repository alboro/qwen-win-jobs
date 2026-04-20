@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap_windows.ps1" %*
exit /b %ERRORLEVEL%
