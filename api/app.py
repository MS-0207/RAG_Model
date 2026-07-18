from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.exception import NoDocumentsFoundError
from api.exception_handler import (
    file_not_found_handler,
    global_exception_handler,
    no_documents_found_handler,
)
from api.middleware import RequestLoggingMiddleware
from api.routes.documents import router as documents_router
from api.routes.feedback import router as feedback_router
from api.routes.health import router as health_router
from api.routes.rag import router as rag_router
from Database import models
from Database.connection import Base, engine


app = FastAPI(title="RAG FAST_API")


# Request logging and request ID middleware
app.add_middleware(RequestLoggingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip middleware
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)


# Exception handlers
app.add_exception_handler(
    Exception,
    global_exception_handler,
)

app.add_exception_handler(
    FileNotFoundError,
    file_not_found_handler,
)

app.add_exception_handler(
    NoDocumentsFoundError,
    no_documents_found_handler,
)


# Routers
app.include_router(health_router)
app.include_router(rag_router)
app.include_router(documents_router)
app.include_router(feedback_router)


# Create missing database tables

Base.metadata.create_all(bind=engine)