from fastapi import FastAPI
from api.routes.health import router as health_router
from api.routes.rag import router as rag_router
from api.routes.documents import router as documents_router
from api.routes.feedback import router as feedback_router
from api.exception_handler import global_exception_handler
from api.exception_handler import file_not_found_handler
from api.exception import NoDocumentsFoundError
from api.exception_handler import no_documents_found_handler
from api.middleware import RequestLoggingMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import FastAPI
from Database.connection import Base, engine
from Database import models

app = FastAPI(title="RAG API")
app.add_middleware(RequestLoggingMiddleware)
#-----------------------------------------
# HOW TO READ: create handler add_exception_handler
# exception -- this is error name
# global_exception_handler-- this is handler name
# if exception error comes run global_exception_handler:
#------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG FAST_API")

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

app.add_exception_handler(
    Exception,
    global_exception_handler
)

app.add_exception_handler(
    FileNotFoundError,
    file_not_found_handler
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)
#-----------------------------------------
# HOW TO READ: create handler add_exception_handler
# NoDocumentsFoundError -- this is exception object name
# no_documents_found_handler--- this is handler name
# if this exception error comes run global_exception_handler:
#------------------------------------------

app.add_exception_handler(
    NoDocumentsFoundError,
    no_documents_found_handler,
)

app.include_router(health_router)
app.include_router(rag_router)
app.include_router(documents_router)
app.include_router(feedback_router)


app.include_router(health_router)
app.include_router(rag_router)
app.include_router(documents_router)
app.include_router(feedback_router)


Base.metadata.create_all(bind=engine)