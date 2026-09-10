@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3 -m venv .venv
if not errorlevel 1 goto install
python -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Setup complete. Double-click launch_windows.bat to start.
pause
exit /b 0
:failed
echo.
echo Setup failed. Install Python 3.10 or newer with Tcl/Tk and pip enabled.
echo Check your network connection and the error above, then retry.
pause
exit /b 1
