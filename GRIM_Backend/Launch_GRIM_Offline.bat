@echo off
setlocal DisableDelayedExpansion
set "GRIM_BACKEND=%~dp0"
if not exist "%GRIM_BACKEND%\python_env\python.exe" (
    echo Missing GRIM_Backend\python_env\python.exe. Run Setup_GRIM_Offline.bat first.
    pause
    exit /b 1
)
"%GRIM_BACKEND%\python_env\python.exe" -I -B "%GRIM_BACKEND%\setup_offline.py" --launch %*
set "GRIM_RESULT=%ERRORLEVEL%"
if not "%GRIM_RESULT%"=="0" pause
exit /b %GRIM_RESULT%
