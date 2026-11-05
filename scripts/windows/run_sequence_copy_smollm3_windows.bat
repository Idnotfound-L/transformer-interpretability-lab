@echo off
cd /d "%~dp0..\.."
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_windows.bat and setup_huggingface_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe scripts\run_sequence_copy_study.py --model-id HuggingFaceTB/SmolLM3-3B-Base --device cuda --dtype bfloat16 --discovery-cases 96 --evaluation-cases 128 --patch-cases 12 --layers 8 --batch-size 4
if errorlevel 1 goto fail
echo.
echo SmolLM3 sequence-copying study complete. Check artifacts\sequence_copy.
pause
exit /b 0
:fail
echo The research run failed. Confirm CUDA memory and Hugging Face connectivity.
pause
exit /b 1
