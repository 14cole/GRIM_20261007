@echo off
setlocal DisableDelayedExpansion
set "GRIM_BACKEND=%~dp0"
if not exist "%GRIM_BACKEND%\python_env\python.exe" (
    echo Unzip your embedded Python for Windows x64 into GRIM_Backend\python_env first.
    echo Expected: "%GRIM_BACKEND%\python_env\python.exe"
    pause
    exit /b 1
)
"%GRIM_BACKEND%\python_env\python.exe" -I -B "%GRIM_BACKEND%\setup_offline.py"
set "GRIM_RESULT=%ERRORLEVEL%"
if not "%GRIM_RESULT%"=="0" echo Setup failed. Review the message above, then rerun this file.
pause
exit /b %GRIM_RESULT%
