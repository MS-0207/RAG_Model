from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from ingest.pipeline import run_ingestion_pipeline
from api.app import app
from api.config import settings

client = TestClient(app)


@patch("api.routes.documents.run_ingestion_pipeline")
def test_ingest_endpoint(mock_run_ingestion_pipeline):
    mock_run_ingestion_pipeline.return_value = {
        "status": "success",
        "documents_loaded": 2,
        "chunks_created": 6,
    }

    response = client.post("/documents/ingest")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["message"] == "Documents ingested successfully"

    mock_run_ingestion_pipeline.assert_called_once_with()


@patch("ingest.pipeline.run_embedding_pipeline")
@patch("ingest.pipeline.save_chunks")
@patch("ingest.pipeline.chunk_documents")
@patch("ingest.pipeline.load_all_documents")
def test_run_ingestion_pipeline(
    mock_load_all_documents,
    mock_chunk_documents,
    mock_save_chunks,
    mock_run_embedding_pipeline,
):


    documents = [
        Document(
            page_content="Document one",
            metadata={"source": "file1.pdf"},
        ),
        Document(
            page_content="Document two",
            metadata={"source": "file2.pdf"},
        ),
    ]

    chunks = [
        Document(
            page_content="Chunk one",
            metadata={"source": "file1.pdf"},
        ),
        Document(
            page_content="Chunk two",
            metadata={"source": "file1.pdf"},
        ),
        Document(
            page_content="Chunk three",
            metadata={"source": "file2.pdf"},
        ),
    ]

    mock_load_all_documents.return_value = documents
    mock_chunk_documents.return_value = chunks
    mock_run_embedding_pipeline.return_value = {
        "status": "success",
        "chunks_embedded": 3,
        "vector_store_dir": str(settings.vector_store_dir),
    }

    result = run_ingestion_pipeline()

    mock_load_all_documents.assert_called_once_with(
        settings.raw_dir
    )

    mock_chunk_documents.assert_called_once_with(
        documents
    )

    mock_save_chunks.assert_called_once_with(
        chunks,
        settings.processed_dir,
    )

    mock_run_embedding_pipeline.assert_called_once_with(
        settings.processed_dir,
        settings.vector_store_dir,
    )

    assert result["status"] == "success"
    assert result["documents_loaded"] == 2
    assert result["chunks_created"] == 3


@patch("ingest.pipeline.load_all_documents")
def test_ingestion_no_documents(mock_load_all_documents):
    from api.exception import NoDocumentsFoundError
    from ingest.pipeline import run_ingestion_pipeline

    mock_load_all_documents.side_effect = NoDocumentsFoundError(
        "No supported documents found."
    )

    try:
        run_ingestion_pipeline()
        assert False, "NoDocumentsFoundError was not raised"
    except NoDocumentsFoundError as exc:
        assert str(exc) == "No supported documents found."


@patch("api.routes.documents.run_ingestion_pipeline")
def test_ingest_no_documents_response(
    mock_run_ingestion_pipeline,
):
    from api.exception import NoDocumentsFoundError

    mock_run_ingestion_pipeline.side_effect = (
        NoDocumentsFoundError(
            "No supported documents found."
        )
    )

    response = client.post("/documents/ingest")

    assert response.status_code == 404
    assert response.json()["success"] is False
    assert response.json()["error"] == (
        "No supported documents found."
    )
    assert response.json()["path"] == "/documents/ingest"