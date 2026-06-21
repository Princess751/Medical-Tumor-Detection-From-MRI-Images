# Staged dependency install for slow or unreliable PyPI connections.
# Run from project root:  powershell -ExecutionPolicy Bypass -File scripts\install_deps.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Pip = Join-Path $Root ".venv\Scripts\pip.exe"
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Creating virtual environment..."
    py -3 -m venv (Join-Path $Root ".venv")
}

$pipArgs = @("--default-timeout", "600", "--retries", "10")

Write-Host ""
Write-Host "=== Upgrading pip ==="
& $Python -m pip install --upgrade pip @pipArgs

Write-Host ""
Write-Host "=== Stage 1/3: Web API packages (small) ==="
& $Pip install -r (Join-Path $Root "requirements-web.txt") @pipArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Stage 2/3: PyTorch CPU (~200 MB, may take several minutes) ==="
Write-Host "Using PyTorch CPU wheel index (download.pytorch.org)..."
& $Pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu @pipArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pinned CPU install failed; retrying latest CPU wheels..."
    & $Pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu @pipArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "=== Stage 3/3: OpenCV ==="
& $Pip install -r (Join-Path $Root "requirements-ml.txt") @pipArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Verifying imports ==="
& $Python (Join-Path $Root "scripts\verify_deps.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Done. Start the app with:  .\run.bat"
Write-Host "Or:  .venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8780"
