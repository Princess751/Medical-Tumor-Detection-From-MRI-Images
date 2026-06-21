import base64
import json
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.database import Base, engine, get_db
from app.models import AuditEvent, Study
from ml.inference import TumorDetector

app = FastAPI(title="MRI Tumor Detection Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = REPO / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

detector = TumorDetector()
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    try:
        detector.load()
    except FileNotFoundError:
        pass  # UI shows setup instructions until weights exist


def log_audit(db: Session, actor: str, action: str, detail: str | None = None):
    db.add(AuditEvent(actor=actor, action=action, detail=detail))
    db.commit()


@app.get("/", response_class=HTMLResponse)
def index():
    return (static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    weights_ok = detector.weights_path.exists()
    return {
        "status": "ok" if weights_ok else "needs_training",
        "model_loaded": detector.model is not None,
        "weights_path": str(detector.weights_path),
        "device": str(detector.device),
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    radiologist_id: str = Form(...),
    study_ref: str = Form(...),
    include_heatmap: bool = Form(True),
    db: Session = Depends(get_db),
):
    if not radiologist_id.strip() or not study_ref.strip():
        raise HTTPException(status_code=400, detail="radiologist_id and study_ref are required")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 20 MB limit")

    try:
        result = detector.predict(data, include_heatmap=include_heatmap)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    study = Study(
        study_ref=study_ref.strip(),
        radiologist_id=radiologist_id.strip(),
        filename=file.filename or "unknown",
        predicted_class=result.predicted_class,
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        tumor_detected=1 if result.tumor_detected else 0,
        result_json=json.dumps(result.to_dict(), ensure_ascii=False),
        model_version=result.model_version,
    )
    db.add(study)
    log_audit(
        db,
        actor=radiologist_id.strip(),
        action="mri_analyzed",
        detail=json.dumps({"study_ref": study_ref, "class": result.predicted_class}),
    )
    db.commit()
    db.refresh(study)

    payload = result.to_dict()
    payload["study_id"] = study.id
    if result.heatmap_png:
        payload["heatmap_base64"] = base64.b64encode(result.heatmap_png).decode("ascii")
    return payload


@app.get("/api/studies")
def list_studies(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.execute(select(Study).order_by(Study.created_at.desc()).limit(limit)).scalars().all()
    return [
        {
            "id": r.id,
            "study_ref": r.study_ref,
            "radiologist_id": r.radiologist_id,
            "filename": r.filename,
            "predicted_class": r.predicted_class,
            "predicted_label": r.predicted_label,
            "confidence": r.confidence,
            "tumor_detected": bool(r.tumor_detected),
            "model_version": r.model_version,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


@app.get("/api/audit")
def audit_log(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.execute(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)).scalars().all()
    return [
        {
            "id": r.id,
            "actor": r.actor,
            "action": r.action,
            "detail": r.detail,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


if __name__ == "__main__":
    import uvicorn

    print("Starting server at http://127.0.0.1:8780")
    print("Tip: run.bat also trains the model if weights are missing.")
    uvicorn.run(app, host="127.0.0.1", port=8780)
