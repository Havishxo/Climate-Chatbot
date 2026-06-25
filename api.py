"""
api.py — FastAPI REST backend for ClimateBot.

Endpoints:
  POST /chat              — main conversational endpoint
  POST /upload            — file upload + ingestion
  GET  /live/co2          — real-time CO2 data
  GET  /live/temperature  — real-time temperature anomaly
  POST /footprint         — carbon footprint calculation
  POST /extract           — entity extraction
  POST /summarise         — document summarisation
  GET  /stats             — knowledge base statistics
  GET  /health            — health check
"""

import os
import sys
import logging
import tempfile
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))

from app.agent import agent_respond, ConversationMemory
from app.tools import get_live_co2, get_temperature_anomaly, calculate_carbon_footprint
from app.entity_extraction import extract_entities, summarise_document
from app.multimodal import process_upload
from app.rag import get_vectorstore_stats, ingest_text
from app.config import USE_IBM, USE_ANTHROPIC, GRANITE_CHAT_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClimateBot API",
    description=(
        "REST API for the Climate Awareness Chatbot. "
        "Powered by IBM Granite, RAG, Agentic AI, and multimodal inputs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session memory store (keyed by session_id) ─────────────────────────────────
_sessions: Dict[str, ConversationMemory] = {}

def _get_memory(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(max_turns=12)
    return _sessions[session_id]


# ── Request / Response models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User's question")
    session_id: str = Field(default="default", description="Session identifier for memory")
    uploaded_text: str = Field(default="", description="Text extracted from uploaded file")

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    entities: Dict[str, Any]
    context_used: bool
    session_id: str

class FootprintRequest(BaseModel):
    car_km_year: float = Field(default=10000, ge=0)
    flights_short: int = Field(default=2, ge=0)
    flights_long: int = Field(default=1, ge=0)
    electricity_kwh_month: float = Field(default=250, ge=0)
    natural_gas_kwh_month: float = Field(default=100, ge=0)
    diet_type: str = Field(default="average")
    goods_spend_monthly_usd: float = Field(default=300, ge=0)

class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)

class SummariseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    max_words: int = Field(default=200, ge=50, le=1000)
    style: str = Field(default="factual")  # factual | accessible | bullet


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    stats = get_vectorstore_stats()
    return {
        "status": "ok",
        "llm_backend": "IBM Granite" if USE_IBM else ("Anthropic" if USE_ANTHROPIC else "MockLLM"),
        "model": GRANITE_CHAT_MODEL if USE_IBM else "claude-sonnet-4-6",
        "vector_store_chunks": stats["document_count"],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main conversational endpoint with RAG + Agentic AI."""
    memory = _get_memory(req.session_id)
    try:
        result = agent_respond(
            query=req.message,
            memory=memory,
            uploaded_text=req.uploaded_text,
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            tool_calls=result.get("tool_calls", []),
            entities=result.get("entities", {}),
            context_used=result.get("context_used", False),
            session_id=req.session_id,
        )
    except Exception as exc:
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF, image, or CSV for ingestion into the knowledge base.
    Returns extracted text and processing metadata.
    """
    allowed_exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".csv", ".tsv", ".txt", ".md"}
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_exts)}",
        )

    # Save to temp file
    try:
        suffix = ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = process_upload(tmp_path, original_name=file.filename or "")
        os.unlink(tmp_path)

        return {
            "filename": file.filename,
            "type": result["type"],
            "chunks_added": result["chunks_added"],
            "description": result["description"],
            "entities": result.get("entities", {}),
            "text_preview": result.get("text_extracted", "")[:500],
            "error": result.get("error"),
        }
    except Exception as exc:
        logger.error("Upload error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/live/co2")
async def live_co2():
    """Fetch real-time atmospheric CO₂ concentration."""
    return get_live_co2()


@app.get("/live/temperature")
async def live_temperature():
    """Fetch real-time global surface temperature anomaly."""
    return get_temperature_anomaly()


@app.post("/footprint")
async def footprint(req: FootprintRequest):
    """Calculate annual personal carbon footprint in tonnes CO₂e."""
    return calculate_carbon_footprint(
        car_km_year=req.car_km_year,
        flights_short=req.flights_short,
        flights_long=req.flights_long,
        electricity_kwh_month=req.electricity_kwh_month,
        natural_gas_kwh_month=req.natural_gas_kwh_month,
        diet_type=req.diet_type,
        goods_spend_monthly_usd=req.goods_spend_monthly_usd,
    )


@app.post("/extract")
async def extract(req: ExtractRequest):
    """Extract named entities from climate text."""
    return extract_entities(req.text)


@app.post("/summarise")
async def summarise(req: SummariseRequest):
    """Summarise a climate document using map-reduce strategy for long texts."""
    return summarise_document(
        text=req.text,
        max_summary_words=req.max_words,
        style=req.style,
    )


@app.get("/stats")
async def stats():
    """Knowledge base and system statistics."""
    kb = get_vectorstore_stats()
    return {
        "knowledge_base": kb,
        "sessions_active": len(_sessions),
        "llm_backend": "IBM Granite" if USE_IBM else ("Anthropic" if USE_ANTHROPIC else "MockLLM"),
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    if session_id in _sessions:
        _sessions[session_id].clear()
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}
