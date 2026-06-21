<<<<<<< HEAD
@echo off
cd /d "%~dp0"
echo Installing MRI Tumor Detection dependencies (staged, long timeout)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_deps.ps1"
if errorlevel 1 (
  echo.
  echo Install failed. See SETUP.md for offline / retry options.
  pause
  exit /b 1
)
echo.
echo Install complete.
pause
=======
@echo off
cd /d "%~dp0"
echo Installing MRI Tumor Detection dependencies (staged, long timeout)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_deps.ps1"
if errorlevel 1 (
  echo.
  echo Install failed. See SETUP.md for offline / retry options.
  pause
  exit /b 1
)
echo.
echo Install complete.
pause
>>>>>>> 95eb9d2250568a0e2a5d1de777831210e935debd
