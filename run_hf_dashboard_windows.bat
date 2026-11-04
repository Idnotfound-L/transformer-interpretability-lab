@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
  echo Streamlit not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\streamlit.exe run dashboard\hf_app.py
pause
