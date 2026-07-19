from fastapi.testclient import TestClient
from api.app import app
from unittest.mock import patch

client = TestClient(app)


@patch("api.routes.documents.get_all_documents")
def test_get_documents(mock_get_all_documents):
    mock_get_all_documents.return_value = [
        {"name": "file1.pdf"},
        {"name": "file2.txt"},
    ]

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["total_documents"] == 2
    assert len(response.json()["documents"]) == 2


@patch("api.routes.documents.get_document_information")
def test_get_document(mock_get_document_information):
    mock_get_document_information.return_value = {
        "document_name": "sample.pdf",
        "chunks": 12,
        "size": "2 MB",
    }

    response = client.get("/documents/sample.pdf")

    assert response.status_code == 200
    assert response.json()["document"]["document_name"] == "sample.pdf"
    assert response.json()["document"]["chunks"] == 12

    mock_get_document_information.assert_called_once_with("sample.pdf")


@patch("api.routes.documents.delete_document_from_storage")
def test_delete_document(mock_delete_document):
    mock_delete_document.return_value = None

    response = client.delete("/documents/sample.pdf")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "deleted successfully" in response.json()["message"]

    mock_delete_document.assert_called_once_with("sample.pdf")



@patch("api.routes.documents.run_ingestion_task.delay")
def test_ingest_documents(mock_delay):
    mock_delay.return_value.id = "test-task-123"

    response = client.post("/documents/ingest")

    assert response.status_code == 202
    assert response.json() == {
        "status": "queued",
        "message": "Document ingestion started in the background.",
        "task_id": "test-task-123",
    }
    mock_delay.assert_called_once_with()
