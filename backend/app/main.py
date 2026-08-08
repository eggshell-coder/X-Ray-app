"""
app/main.py — FastAPI backend for chest X-ray disease classification (GATv2).

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /api/health         → model load status
    POST /api/predict        → multipart image upload, returns class probabilities
    GET  /                   → serves the frontend (if built into ../frontend_dist)
"""

from __future__ import annotations
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.model_service import model_service

ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff",
    "application/dicom", "application/octet-stream"
}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

app = FastAPI(
    title="CXR-GNN — Chest X-ray Disease Classifier",
    description="GATv2 superpixel-graph classifier for chest X-ray images.",
    version="1.1.0",
)

# In production, replace "*" with your actual frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model() -> None:
    model_service.load()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if model_service.is_ready else "model_not_loaded",
        "device": str(model_service.device),
        "classes": list(model_service.idx2class.values()) if model_service.is_ready else [],
        "input_validator_ready": model_service.input_validator.is_ready,
        "require_dicom": model_service.input_validator.require_dicom,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if not model_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not loaded. Place best_gatv2.pt in backend/checkpoints/ and restart the server.",
        )

    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.content_type not in ALLOWED_CONTENT_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type} ({ext})")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 15 MB).")

    try:
        result = model_service.predict(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return JSONResponse(content=result)


# ── Serve the frontend (React dist or fallback) ────────────────────────────────
_frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

if os.path.isdir(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
elif os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
