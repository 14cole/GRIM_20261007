@echo off
setlocal DisableDelayedExpansion
set "GRIM_BACKEND=%~dp0GRIM_Backend"
if not exist "%GRIM_BACKEND%\python_env\python.exe" (
    echo Unzip your embedded Python for Windows x64 into GRIM_Backend\python_env first.
    echo Expected: "%GRIM_BACKEND%\python_env\python.exe"
    pause
    exit /b 1
)
echo GRIM setup will download matching wheels and create the application environment.
echo An internet connection is required. Keep this window open until setup finishes.
"%GRIM_BACKEND%\python_env\python.exe" -I -B "%GRIM_BACKEND%\setup_online.py"
set "GRIM_RESULT=%ERRORLEVEL%"
if not "%GRIM_RESULT%"=="0" echo Setup failed. Review the message above, then rerun this file.
pause
exit /b %GRIM_RESULT%
