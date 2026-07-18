from unittest.mock import MagicMock, patch
from api.services.main import run_rag_pipeline
from unittest.mock import patch
from redis.exceptions import ConnectionError
from Database.redis_cache import clear_rag_cache
from Database.redis_cache import save_cached_response
from unittest.mock import MagicMock, patch
from redis.exceptions import ConnectionError
from api.services.main import run_rag_pipeline
from unittest.mock import patch
from redis.exceptions import ConnectionError
from Database.redis_cache import get_cached_response


@patch("api.services.main.get_all_docs_from_faiss")
@patch("api.services.main.get_cached_response")
def test_run_rag_pipeline_cache_hit(
    mock_get_cached_response,
    mock_get_all_docs,
):
    # Arrange
    mock_get_cached_response.return_value = {
        "query": "What is RAG?",
        "answer": "RAG combines retrieval with generation.",
    }

    mock_db = MagicMock()

    # Act
    response = run_rag_pipeline(
        query="What is RAG?",
        db=mock_db,
    )

    # Assert
    assert response["answer"] == (
        "RAG combines retrieval with generation."
    )
    assert response["cache_hit"] is True

    mock_get_cached_response.assert_called_once_with(
        "What is RAG?"
    )

    # The RAG pipeline must not continue after a cache hit
    mock_get_all_docs.assert_not_called()


@patch("api.services.main.save_cached_response")
@patch("api.services.main.build_final_response")
@patch("api.services.main.check_grounding")
@patch("api.services.main.generate_answer")
@patch("api.services.main.cross_encoder_rerank")
@patch("api.services.main.merge_and_deduplicate")
@patch("api.services.main.vector_mmr_retrieval")
@patch("api.services.main.bm25_retrieval")
@patch("api.services.main.get_all_docs_from_faiss")
@patch("api.services.main.get_cached_response")
def test_run_rag_pipeline_cache_miss(
    mock_get_cached_response,
    mock_get_all_docs,
    mock_bm25,
    mock_vector,
    mock_merge,
    mock_rerank,
    mock_generate,
    mock_grounding,
    mock_build_response,
    mock_save_cache,
):
    # ----------------------------
    # Arrange
    # ----------------------------
    mock_get_cached_response.return_value = None

    docs = [MagicMock(), MagicMock()]
    top_docs = [MagicMock()]

    mock_get_all_docs.return_value = docs
    mock_bm25.return_value = docs
    mock_vector.return_value = docs
    mock_merge.return_value = docs
    mock_rerank.return_value = top_docs

    mock_generate.return_value = {
        "answer": "This is a generated answer."
    }

    mock_grounding.return_value = {
        "grounded": True,
        "score": 0.95,
    }

    mock_build_response.return_value = {
        "query": "What is RAG?",
        "answer": "This is a generated answer.",
    }

    mock_db = MagicMock()

    # ----------------------------
    # Act
    # ----------------------------
    response = run_rag_pipeline(
        query="What is RAG?",
        db=mock_db,
    )

    # ----------------------------
    # Assert
    # ----------------------------
    assert response["cache_hit"] is False
    assert response["answer"] == "This is a generated answer."

    mock_get_cached_response.assert_called_once_with(
        "What is RAG?"
    )

    mock_get_all_docs.assert_called_once_with(mock_db)

    mock_bm25.assert_called_once()

    mock_vector.assert_called_once()

    mock_merge.assert_called_once()

    mock_rerank.assert_called_once()

    mock_generate.assert_called_once()

    mock_grounding.assert_called_once()

    mock_build_response.assert_called_once()

    mock_save_cache.assert_called_once_with(
        query="What is RAG?",
        response=response,
    )


@patch("Database.redis_cache.redis_client.get")
def test_get_cached_response_returns_none_when_redis_fails(
    mock_redis_get,
):
    mock_redis_get.side_effect = ConnectionError(
        "Redis unavailable"
    )

    result = get_cached_response("What is RAG?")

    assert result is None


@patch("Database.redis_cache.redis_client.setex")
def test_save_cached_response_does_not_crash_when_redis_fails(
    mock_setex,
):
    mock_setex.side_effect = ConnectionError(
        "Redis unavailable"
    )

    response = {
        "query": "What is RAG?",
        "answer": "RAG combines retrieval and generation.",
    }

    result = save_cached_response(
        query="What is RAG?",
        response=response,
    )

    assert result is None
    mock_setex.assert_called_once()


@patch("Database.redis_cache.redis_client.scan_iter")
def test_clear_rag_cache_returns_zero_when_redis_fails(
    mock_scan_iter,
):
    mock_scan_iter.side_effect = ConnectionError(
        "Redis unavailable"
    )

    result = clear_rag_cache()

    assert result == 0