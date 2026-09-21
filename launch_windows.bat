@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
call install_windows.bat
if errorlevel 1 exit /b 1
:run
".venv\Scripts\python.exe" -c "from tkinterdnd2 import TkinterDnD; assert hasattr(TkinterDnD, 'require')" >nul 2>&1
if not errorlevel 1 goto launch
call install_windows.bat
if errorlevel 1 exit /b 1
:launch
".venv\Scripts\python.exe" gui.py
if errorlevel 1 pause
