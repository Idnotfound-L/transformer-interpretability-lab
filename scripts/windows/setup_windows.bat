@echo off
cd /d "%~dp0..\.."
setlocal
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 python -m venv .venv
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -e .
if errorlevel 1 goto fail
echo.
echo Setup complete. For Hugging Face support, run setup_huggingface_windows.bat next.
echo For TinyGPT, run train_windows.bat and run_dashboard_windows.bat.
pause
exit /b 0
:fail
echo Setup failed. Read the error above and share it for troubleshooting.
pause
exit /b 1
