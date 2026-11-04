@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe scripts\run_hf_research.py --model-id HuggingFaceTB/SmolLM3-3B-Base --device auto --layers 2 --max-heads 4 --neuron-top-k 2
pause
