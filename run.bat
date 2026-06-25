
@echo off
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  py -3 -m venv .venv
)

REM Check if core packages are installed
.venv\Scripts\python -c "import fastapi, torch" 2>nul
if errorlevel 1 (
  echo Dependencies missing or incomplete. Running staged installer...
  call install_deps.bat
  if errorlevel 1 (
    echo.
    echo Install failed. See SETUP.md for manual steps and network troubleshooting.
    pause
    exit /b 1
  )
)

if not exist weights\tumor_classifier.pt (
  echo Model weights missing. Running lab setup ^(synthetic data + training^)...
  .venv\Scripts\python scripts\setup_lab.py
  if errorlevel 1 (
    echo Training failed. Check errors above.
    pause
    exit /b 1
  )
)

echo.
echo Starting MRI Tumor Detection Assistant on http://127.0.0.1:8780
echo Press Ctrl+C to stop.
echo.
.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8780
>>>>>>> 95eb9d2250568a0e2a5d1de777831210e935debd
