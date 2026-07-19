from typing import Any
from fastapi import APIRouter, Path
from pydantic import BaseModel
from api.tasks import run_ingestion_task
from ingest.delete_document import delete_document_from_storage
from ingest.document_loader import get_all_documents
from ingest.get_all_document import get_document_information


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

class IngestResponse(BaseModel):
    status: str
    message: str
    task_id: str


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: list[Any]


class DocumentResponse(BaseModel):
    document: Any


class DeleteDocumentResponse(BaseModel):
    status: str
    message: str


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=202,
)
def ingest_documents() -> IngestResponse:
    print(">>> /documents/ingest endpoint called", flush=True)

    task = run_ingestion_task.delay()

    print(f">>> Celery task queued: {task.id}", flush=True)

    return IngestResponse(
        status="queued",
        message="Document ingestion started in the background.",
        task_id=task.id,
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
    document_name: str = Path(..., min_length=1),
) -> DocumentResponse:
    return DocumentResponse(
        document=get_document_information(document_name),
    )


@router.delete(
    "/{document_name}",
    response_model=DeleteDocumentResponse,
)
def delete_document(
    document_name: str = Path(..., min_length=1),
) -> DeleteDocumentResponse:
    delete_document_from_storage(document_name)

    return DeleteDocumentResponse(
        status="success",
        message=f"Document '{document_name}' deleted successfully",
    )
