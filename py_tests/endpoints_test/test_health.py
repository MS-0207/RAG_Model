from fastapi.testclient import TestClient
from api.app import app
client = TestClient(app)

def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "First RAG API"

def test_liveness():
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_readiness():
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] in ["ready", "not_ready"]

def test_version():
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["app"] == "rag_model"