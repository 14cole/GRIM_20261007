@echo off
setlocal DisableDelayedExpansion
set "GRIM_BACKEND=%~dp0GRIM_Backend"
if not exist "%GRIM_BACKEND%\python_env\python.exe" (
    echo Unzip embedded Python into GRIM_Backend\python_env and supply local wheels first.
    pause
    exit /b 1
)
"%GRIM_BACKEND%\python_env\python.exe" -I -B "%GRIM_BACKEND%\release_validation.py" --install %*
set "GRIM_RESULT=%ERRORLEVEL%"
pause
exit /b %GRIM_RESULT%
