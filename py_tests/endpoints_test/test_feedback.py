from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_save_feedback():
    response = client.post(
        "/feedback",
        json={
            "query": "What is RAG?",
            "rating": 5,
            "comment": "Helpful response",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["message"] == "Feedback saved successfully"


def test_feedback_invalid_rating():
    response = client.post(
        "/feedback",
        json={
            "query": "What is RAG?",
            "rating": 8,
            "comment": "Invalid rating",
        },
    )

    assert response.status_code == 422


def test_feedback_empty_query():
    response = client.post(
        "/feedback",
        json={
            "query": "",
            "rating": 4,
            "comment": "Empty query",
        },
    )

    assert response.status_code == 422
