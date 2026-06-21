<<<<<<< HEAD
@echo off
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  py -3 -m venv .venv
)

echo.
echo === Upgrading pip ===
.venv\Scripts\python -m pip install --upgrade pip --default-timeout 600 --retries 10

echo.
echo === Stage 1/3: Web API packages ===
.venv\Scripts\pip install -r requirements-web.txt --default-timeout 600 --retries 10
if errorlevel 1 goto fail

echo.
echo === Stage 2/3: PyTorch CPU (~200 MB, may take several minutes) ===
.venv\Scripts\pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --default-timeout 600 --retries 10
if errorlevel 1 (
  echo Retrying with latest CPU wheels...
  .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --default-timeout 600 --retries 10
  if errorlevel 1 goto fail
)

echo.
echo === Stage 3/3: OpenCV ===
.venv\Scripts\pip install -r requirements-ml.txt --default-timeout 600 --retries 10
if errorlevel 1 goto fail

echo.
echo === Verifying imports ===
.venv\Scripts\python scripts\verify_deps.py
if errorlevel 1 goto fail

echo.
echo Done. Run:  .\run.bat
exit /b 0

:fail
echo.
echo Install failed. Try a more stable network or see SETUP.md
pause
exit /b 1
=======
@echo off
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  py -3 -m venv .venv
)

echo.
echo === Upgrading pip ===
.venv\Scripts\python -m pip install --upgrade pip --default-timeout 600 --retries 10

echo.
echo === Stage 1/3: Web API packages ===
.venv\Scripts\pip install -r requirements-web.txt --default-timeout 600 --retries 10
if errorlevel 1 goto fail

echo.
echo === Stage 2/3: PyTorch CPU (~200 MB, may take several minutes) ===
.venv\Scripts\pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --default-timeout 600 --retries 10
if errorlevel 1 (
  echo Retrying with latest CPU wheels...
  .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --default-timeout 600 --retries 10
  if errorlevel 1 goto fail
)

echo.
echo === Stage 3/3: OpenCV ===
.venv\Scripts\pip install -r requirements-ml.txt --default-timeout 600 --retries 10
if errorlevel 1 goto fail

echo.
echo === Verifying imports ===
.venv\Scripts\python scripts\verify_deps.py
if errorlevel 1 goto fail

echo.
echo Done. Run:  .\run.bat
exit /b 0

:fail
echo.
echo Install failed. Try a more stable network or see SETUP.md
pause
exit /b 1
>>>>>>> 95eb9d2250568a0e2a5d1de777831210e935debd
