@echo off
REM Convenience shim so `play sling` works from a Windows shell.
REM Prefers the repo's own .venv, because that is where requirements.txt is
REM meant to be installed; falls back to whatever `python` is on PATH.
setlocal
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\python.exe" (
    "%HERE%.venv\Scripts\python.exe" "%HERE%play.py" %*
) else (
    python "%HERE%play.py" %*
)
endlocal
