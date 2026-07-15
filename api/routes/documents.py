from typing import Any
from fastapi import APIRouter, Path
from pydantic import BaseModel

from ingest.delete_document import delete_document_from_storage
from ingest.document_loader import get_all_documents
from ingest.get_all_document import get_document_information
from ingest.pipeline import run_ingestion_pipeline


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

# -------------------------
# Response Models
# -------------------------

class IngestResponse(BaseModel):
    status: str
    message: str

class DocumentListResponse(BaseModel):
    total_documents: int
    documents: list[Any]


class DocumentResponse(BaseModel):
    document: Any


class DeleteDocumentResponse(BaseModel):
    status: str
    message: str

# -------------------------
# Endpoints
# -------------------------

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=200,
)
def ingest() -> IngestResponse:
    run_ingestion_pipeline()

    return IngestResponse(
        status="success",
        message="Documents ingested successfully",
    )

@router.get(
    "",
    response_model=DocumentListResponse,
)
def get_documents() -> DocumentListResponse:
    documents = get_all_documents()

    return DocumentListResponse(
        total_documents=len(documents),
        documents=documents,
    )

@router.get(
    "/{document_name}",
    response_model=DocumentResponse,
)
def get_document(
    document_name: str = Path(
        ...,
        min_length=1,
        description="Name of the document to retrieve",
    ),
) -> DocumentResponse:
    document = get_document_information(document_name)

    return DocumentResponse(
        document=document,
    )


@router.delete(
    "/{document_name}",
    response_model=DeleteDocumentResponse,
)
def delete_document(
    document_name: str = Path(
        ...,
        min_length=1,
        description="Name of the document to delete",
    ),
) -> DeleteDocumentResponse:
    delete_document_from_storage(document_name)

    return DeleteDocumentResponse(
        status="success",
        message=f"Document '{document_name}' deleted successfully",
    )