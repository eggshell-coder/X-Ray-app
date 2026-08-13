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
import json
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.model_service import model_service
from rag.service import explain_result, answer_followup, chat_assistant


class ExplanationRequest(BaseModel):
    prediction: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    review_needed: bool = False
    question: str | None = Field(default=None, max_length=500)
    focus_regions: list[str] = Field(default_factory=list, max_length=10)
    image_base64: str | None = Field(default=None, max_length=8_000_000)


class ChatRequest(BaseModel):
    prediction: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    focus_regions: list[str] = Field(default_factory=list, max_length=10)
    question: str = Field(min_length=1, max_length=500)


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(description="Message author")
    content: str = Field(max_length=500)


class ConversationRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500, description="Current user message")
    conversation_history: list[ConversationMessage] = Field(default_factory=list, max_length=50, description="Previous messages in conversation")


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


@app.post("/api/explain")
def explain(request: ExplanationRequest) -> dict:
    try:
        answer = explain_result(request.prediction, request.confidence, request.review_needed, request.question, request.focus_regions)
        return {"status": "ok", "answer": answer}
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    try:
        return {"status": "ok", "answer": answer_followup(request.prediction, request.confidence, request.focus_regions, request.question)}
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/chat-assistant")
def chat_with_assistant(request: ConversationRequest) -> dict:
    """Multi-turn conversational assistant for chest X-ray medical questions.
    
    Maintains conversation history and responds only within medical domain restrictions.
    Can ask clarifying questions and engage in natural dialogue.
    """
    try:
        # Convert Pydantic models to dicts for the chat function
        history = [msg.model_dump() for msg in request.conversation_history] if request.conversation_history else []
        result = chat_assistant(request.message, history)
        return {"status": "ok", **result}
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


# ── Serve the frontend (React dist or fallback) ────────────────────────────────
_frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

if os.path.isdir(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
elif os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
