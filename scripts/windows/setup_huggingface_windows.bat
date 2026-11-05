@echo off
cd /d "%~dp0..\.."
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_windows.bat first to create the base environment.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade "transformers>=4.53.0" "accelerate>=0.30.0" "safetensors>=0.4.3" "sentencepiece>=0.2.0"
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -e .
echo.
echo Hugging Face dependencies installed. You can now run run_hf_research_windows.bat or run_hf_dashboard_windows.bat.
pause
exit /b 0
:fail
echo Hugging Face dependency installation failed. Share the error output for troubleshooting.
pause
exit /b 1
