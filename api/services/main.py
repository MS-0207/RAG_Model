from langchain_community.vectorstores import FAISS
from Database.redis_cache import (
    get_cached_response,
    save_cached_response,
)
from LLM.answer_generator import check_grounding, generate_answer
from reranking.cross_encoder_rank import cross_encoder_rerank
from retrieval.BM25 import (
    bm25_retrieval,
    get_all_docs_from_faiss,
    merge_and_deduplicate,
    vector_mmr_retrieval,
)
from Response.response import build_final_response
from utils.logger import get_logger

logger = get_logger(__name__)


def run_rag_pipeline(
    query: str,
    db: FAISS,
) -> dict:
    # --------------------------------------------------
    # Step 1: Check Redis cache
    # --------------------------------------------------
    cached_response = get_cached_response(query)

    if cached_response is not None:
        logger.info(
            "Returning cached RAG response for query: %s",
            query,
        )

        cached_response["cache_hit"] = True
        return cached_response

    logger.info(
        "Cache miss. Running RAG pipeline for query: %s",
        query,
    )

    # --------------------------------------------------
    # Step 2: Get all documents from FAISS
    # --------------------------------------------------
    all_docs = get_all_docs_from_faiss(db)

    # --------------------------------------------------
    # Step 3: BM25 retrieval
    # --------------------------------------------------
    bm25_docs = bm25_retrieval(
        query=query,
        docs=all_docs,
        top_k=10,
    )
    # --------------------------------------------------
    # Step 4: Vector MMR retrieval
    # --------------------------------------------------
    vector_docs = vector_mmr_retrieval(
        query=query,
        db=db,
        top_k=10,
        fetch_k=30,
        lambda_mult=0.5,
    )

    # --------------------------------------------------
    # Step 5: Merge and remove duplicates
    # --------------------------------------------------
    retrieved_docs = merge_and_deduplicate(
        bm25_docs=bm25_docs,
        vector_docs=vector_docs,
    )

    logger.info(
        "Retrieved %d unique documents",
        len(retrieved_docs),
    )

    # --------------------------------------------------
    # Step 6: Cross-encoder reranking
    # --------------------------------------------------
    logger.info("Ranking documents with cross encoder")

    top_docs = cross_encoder_rerank(
        query=query,
        docs=retrieved_docs,
        top_k=3,
    )

    # --------------------------------------------------
    # Step 7: Generate answer
    # --------------------------------------------------
    answer_result = generate_answer(
        query=query,
        top_docs=top_docs,
    )

    # --------------------------------------------------
    # Step 8: Grounding check
    # --------------------------------------------------
    grounding_result = check_grounding(
        query=query,
        answer=answer_result["answer"],
        top_docs=top_docs,
    )

    # --------------------------------------------------
    # Step 9: Extract sources and build final response
    # --------------------------------------------------
    sources = list(
        dict.fromkeys(
            doc.metadata.get("source") for doc in top_docs if doc.metadata.get("source")
        )
    )

    final_response = build_final_response(
        query=query,
        answer_result=answer_result,
        grounding_result=grounding_result,
        top_docs=top_docs,
    )

    final_response["sources"] = sources
    final_response["cache_hit"] = False

    # --------------------------------------------------
    # Step 10: Save final response in Redis
    # --------------------------------------------------
    save_cached_response(
        query=query,
        response=final_response,
    )

    logger.info("Final RAG response generated and saved to Redis")

    return final_response
