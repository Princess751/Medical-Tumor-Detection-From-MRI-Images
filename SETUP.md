# Setup troubleshooting (Windows)

## Common mistakes

### 1. Do not run `python app/main.py` directly (without deps)

FastAPI is a **web server**. Start it with **uvicorn**:

```powershell
cd C:\Users\Owner\Projects\mri-tumor-detection
.\.venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8780
```

Or double-click **`run.bat`** — or in PowerShell: **`.\run.bat`**

### 2. Always use the project virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Then install into **that** venv, not the Windows Store global Python.

### 3. `pip install -r requirements.txt` times out on torch

PyTorch is **~200 MB**. On slow Wi‑Fi, the default 15s pip timeout fails.

**Fix — use the staged installer:**

```powershell
cd C:\Users\Owner\Projects\mri-tumor-detection
.\.venv\Scripts\activate
.\install_deps.bat
```

Alternative (PowerShell): `powershell -ExecutionPolicy Bypass -File scripts\install_deps.ps1`

The installer:
- Uses **600s timeout** and **10 retries**
- Installs small packages first (FastAPI, SQLAlchemy, …)
- Installs **PyTorch CPU** from `download.pytorch.org` (often more reliable than PyPI for torch)
- Installs OpenCV last

### 4. Manual install (if the script still fails)

```powershell
.\.venv\Scripts\activate
python -m pip install --upgrade pip --default-timeout 600 --retries 10
pip install -r requirements-web.txt --default-timeout 600 --retries 10
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --default-timeout 600 --retries 10
pip install opencv-python-headless==4.10.0.84 --default-timeout 600 --retries 10
```

If torch still fails, try a stable network (phone hotspot), or download the wheel manually from [pytorch.org](https://pytorch.org/get-started/locally/) and:

```powershell
pip install C:\path\to\torch-....whl
```

### 5. Verify installation

```powershell
.\.venv\Scripts\python -c "import fastapi, torch, cv2; print('OK', torch.__version__)"
```

### 6. Train model + start app

```powershell
.\.venv\Scripts\python scripts\setup_lab.py
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8780
```

Open **http://127.0.0.1:8780** in your browser.

## What each file is for

| File | Purpose |
|------|---------|
| `run.bat` | One-click: venv → install → train → serve |
| `scripts/install_deps.ps1` | Staged pip install for slow networks |
| `requirements-web.txt` | FastAPI, SQLAlchemy, Pillow, NumPy |
| `requirements-ml.txt` | OpenCV (after torch) |
| `scripts/setup_lab.py` | Synthetic data + training |
| `app/main.py` | API code — **start via uvicorn**, not as a plain script |
