from unittest.mock import patch
from api.exception import NoDocumentsFoundError


# ============================================================
# 1. Ingest → Query → Verify Answer
# ============================================================


def test_ingest_and_query(client):
    response = client.post(
        "/ask",
        json={"query": "What is self-attention?"},
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["answer"]
    assert "self-attention" in body["answer"].lower()
    assert body["grounding_verdict"] == "SUPPORTED"
    assert body["confidence"] == "HIGH"
    assert body["unsupported_claims"] == []


# ============================================================
# 2. Unknown Question (No Hallucination)
# ============================================================


def test_unknown_question(client):
    response = client.post(
        "/ask",
        json={"query": "What is the capital of Mars?"},
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["answer"]
    assert body["confidence"] == "LOW"
    assert body["grounding_verdict"] == "HALLUCINATED"
    assert "The capital of Mars" in body["unsupported_claims"]


# ============================================================
# 3. Multiple Document Retrieval
# ============================================================

# def test_correct_document_retrieved(client):
#     response = client.post(
#         "/ask",
#         json={"query": " What is The Language Modeling Head"},
#     )
#
#     assert response.status_code == 200, response.text
#
#     body = response.json()
#
#     assert "transformer.pdf" in body["sources"]
#     # assert "python.pdf" not in body["sources"]


# ============================================================
# 4. Citation Test
# ============================================================


def test_sources_returned(client):
    response = client.post(
        "/ask",
        json={"query": "What is self-attention"},
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert isinstance(body["sources"], list)
    assert len(body["sources"]) > 0


# ============================================================
# 5. Grounding Test
# ============================================================


def test_grounding_score(client):
    response = client.post(
        "/ask",
        json={"query": "What is self-attention"},
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["grounding_verdict"] == "SUPPORTED"
    assert body["confidence"] in ("HIGH", "MEDIUM")
    assert isinstance(body["unsupported_claims"], list)


# ============================================================
# 6. Cache Test
# ============================================================


def test_cache(client):
    response1 = client.post(
        "/ask",
        json={"query": "what is Parallelizing multi-head attention"},
    )

    response2 = client.post(
        "/ask",
        json={"query": "what is Parallelizing multi-head attention"},
    )

    assert response1.status_code == 200, response1.text
    assert response2.status_code == 200, response2.text

    body1 = response1.json()
    body2 = response2.json()

    assert body1["cache_hit"] is True
    assert body2["cache_hit"] is True


# ============================================================
# 7. Empty Vector Store
# ============================================================


@patch("api.services.main.run_rag_pipeline")
def test_no_documents(mock_pipeline, client):
    mock_pipeline.side_effect = NoDocumentsFoundError()

    response = client.post(
        "/ask",
        json={"query": "What is self-attention"},
    )

    assert response.status_code == 404

    body = response.json()

    assert body["detail"] == "No documents found."


# ============================================================
# 8. Large Document
# ============================================================
#
# def test_large_document(client):
#     response = client.post(
#         "/ask",
#         json={"query": "Summarize the uploaded manual."},
#     )
#
#     assert response.status_code == 200, response.text
#
#     body = response.json()
#
#     assert len(body["answer"]) > 0
