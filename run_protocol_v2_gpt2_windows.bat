@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
set SEED=%1
if "%SEED%"=="" set SEED=7
.venv\Scripts\python.exe scripts\run_sequence_copy_study_v2.py --model-id gpt2 --device auto --dtype auto --discovery-cases 96 --evaluation-cases 128 --control-cases 128 --patch-cases 12 --layers 8 --batch-size 2 --seed %SEED%
if errorlevel 1 (
  echo Protocol V2 run failed. Read the traceback above.
  pause
  exit /b 1
)
echo Protocol V2 GPT-2 seed %SEED% completed.
pause
