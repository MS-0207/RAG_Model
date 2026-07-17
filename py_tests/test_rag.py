from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.app import app
from api.config import settings
client = TestClient(app)


def create_mock_document(
    content: str,
    source: str,
) -> MagicMock:
    document = MagicMock()
    document.page_content = content
    document.metadata = {"source": source}
    return document


@patch("api.routes.rag.run_rag_pipeline")
def test_ask(mock_run_rag_pipeline):
    mock_run_rag_pipeline.return_value = {
        "query": "What is RAG?",
        "answer": "RAG combines retrieval with generation.",
        "grounded": True,
    }

    response = client.post(
        "/ask",
        json={
            "query": "What is RAG?",
        },
        headers={
            "x-api-key": settings.api_key,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "RAG combines retrieval with generation."
    )

    mock_run_rag_pipeline.assert_called_once_with(
        "What is RAG?"
    )


@patch("api.routes.rag.hybrid_retrieval")
def test_retrieve(mock_hybrid_retrieval):
    mock_hybrid_retrieval.return_value = [
        create_mock_document(
            content="RAG retrieves relevant information.",
            source="rag.pdf",
        ),
        create_mock_document(
            content="The retrieved context is sent to an LLM.",
            source="llm.pdf",
        ),
    ]

    response = client.post(
        "/retrieve",
        json={
            "query": "What is RAG?",
        },
    )

    assert response.status_code == 200
    assert response.json()["query"] == "What is RAG?"
    assert response.json()["total_documents"] == 2
    assert len(response.json()["documents"]) == 2
    assert response.json()["documents"][0]["content"] == (
        "RAG retrieves relevant information."
    )

    mock_hybrid_retrieval.assert_called_once_with(
        query="What is RAG?",
        bm25_top_k=10,
        vector_top_k=10,
        fetch_k=30,
        lambda_mult=0.5,
    )


@patch("api.routes.rag.cross_encoder_rerank")
@patch("api.routes.rag.hybrid_retrieval")
def test_rerank(
    mock_hybrid_retrieval,
    mock_cross_encoder_rerank,
):
    retrieved_document = create_mock_document(
        content="Retrieved document",
        source="retrieved.pdf",
    )

    reranked_document = create_mock_document(
        content="Highest-ranked document",
        source="ranked.pdf",
    )

    mock_hybrid_retrieval.return_value = [
        retrieved_document
    ]
    mock_cross_encoder_rerank.return_value = [
        reranked_document
    ]

    response = client.post(
        "/rerank",
        json={
            "query": "Explain RAG",
        },
    )

    assert response.status_code == 200
    assert response.json()["total_documents"] == 1
    assert response.json()["documents"][0]["content"] == (
        "Highest-ranked document"
    )

    mock_cross_encoder_rerank.assert_called_once_with(
        query="Explain RAG",
        docs=[retrieved_document],
        top_k=3,
    )


@patch("api.routes.rag.generate_answer")
@patch("api.routes.rag.cross_encoder_rerank")
@patch("api.routes.rag.hybrid_retrieval")
def test_generate(
    mock_hybrid_retrieval,
    mock_cross_encoder_rerank,
    mock_generate_answer,
):
    retrieved_document = create_mock_document(
        content="Retrieved context",
        source="context.pdf",
    )

    ranked_document = create_mock_document(
        content="Ranked context",
        source="ranked.pdf",
    )

    mock_hybrid_retrieval.return_value = [
        retrieved_document
    ]
    mock_cross_encoder_rerank.return_value = [
        ranked_document
    ]
    mock_generate_answer.return_value = {
        "answer": "Generated RAG answer",
    }

    response = client.post(
        "/generate",
        json={
            "query": "Explain RAG",
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "Generated RAG answer"
    )

    mock_generate_answer.assert_called_once_with(
        query="Explain RAG",
        top_docs=[ranked_document],
    )


@patch("api.routes.rag.check_grounding")
@patch("api.routes.rag.generate_answer")
@patch("api.routes.rag.cross_encoder_rerank")
@patch("api.routes.rag.hybrid_retrieval")
def test_grounding(
    mock_hybrid_retrieval,
    mock_cross_encoder_rerank,
    mock_generate_answer,
    mock_check_grounding,
):
    retrieved_document = create_mock_document(
        content="Retrieved context",
        source="context.pdf",
    )

    ranked_document = create_mock_document(
        content="Ranked context",
        source="ranked.pdf",
    )

    mock_hybrid_retrieval.return_value = [
        retrieved_document
    ]
    mock_cross_encoder_rerank.return_value = [
        ranked_document
    ]
    mock_generate_answer.return_value = {
        "answer": "Generated answer",
    }
    mock_check_grounding.return_value = {
        "grounded": True,
        "score": 0.95,
    }

    response = client.post(
        "/grounding",
        json={
            "query": "Explain RAG",
        },
    )

    assert response.status_code == 200
    assert response.json()["grounded"] is True
    assert response.json()["score"] == 0.95

    mock_check_grounding.assert_called_once_with(
        query="Explain RAG",
        answer="Generated answer",
        top_docs=[ranked_document],
    )


def test_ask_empty_query():
    response = client.post(
        "/ask",
        json={
            "query": "",
        },
        headers={
            "x-api-key": settings.api_key,
        },
    )

    assert response.status_code == 422


def test_ask_invalid_api_key():
    response = client.post(
        "/ask",
        json={
            "query": "What is RAG?",
        },
        headers={
            "x-api-key": "invalid-key",
        },
    )

    assert response.status_code == 401