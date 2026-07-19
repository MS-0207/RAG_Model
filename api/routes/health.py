from datetime import datetime
from pathlib import Path
from api.config import settings
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["System"])


# -------------------------
# Response Models
# -------------------------


class HomeResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    service: str
    message: str
    timestamp: str


class ReadinessResponse(BaseModel):
    status: str
    vector_store_available: bool
    raw_directory_available: bool
    processed_directory_available: bool


class VersionResponse(BaseModel):
    embed_model: str
    vector_database: str
    version: str
    status: str
    app: str
    llm: str


# -------------------------
# Endpoints
# -------------------------


@router.get("/", response_model=HomeResponse)
def home() -> HomeResponse:
    return HomeResponse(message="First RAG API")


# /health/live answers: Is the API process running?
# /health/ready answers: Is the RAG service actually ready to handle requests?


@router.get("/health/live", response_model=HealthResponse)
def liveness() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="RAG API",
        message="API is running",
        timestamp=datetime.now().isoformat(),
    )


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    vector_store_available = (
        Path(settings.vector_store_dir, "index.faiss").exists()
        and Path(settings.vector_store_dir, "index.pkl").exists()
    )

    raw_directory_available = settings.raw_dir.exists()
    processed_directory_available = settings.processed_dir.exists()

    is_ready = (
        vector_store_available
        and raw_directory_available
        and processed_directory_available
    )

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        vector_store_available=vector_store_available,
        raw_directory_available=raw_directory_available,
        processed_directory_available=processed_directory_available,
    )


@router.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(
        status="version",
        app="rag_model",
        embed_model="openai_embedding_model",
        llm="openai",
        vector_database="FAISS",
        version="python.ie",
    )
