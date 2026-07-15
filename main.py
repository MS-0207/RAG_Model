from langchain_community.vectorstores import FAISS

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
    all_docs = get_all_docs_from_faiss(db)

    bm25_docs = bm25_retrieval(
        query=query,
        docs=all_docs,
        top_k=10,
    )

    vector_docs = vector_mmr_retrieval(
        query=query,
        db=db,
        top_k=10,
        fetch_k=30,
        lambda_mult=0.5,
    )

    retrieved_docs = merge_and_deduplicate(
        bm25_docs=bm25_docs,
        vector_docs=vector_docs,
    )

    logger.info("Ranking documents with cross encoder")

    top_docs = cross_encoder_rerank(
        query=query,
        docs=retrieved_docs,
        top_k=3,
    )

    answer_result = generate_answer(
        query=query,
        top_docs=top_docs,
    )

    grounding_result = check_grounding(
        query=query,
        answer=answer_result["answer"],
        top_docs=top_docs,
    )

    final_response = build_final_response(
        query=query,
        answer_result=answer_result,
        grounding_result=grounding_result,
        top_docs=top_docs,
    )

    logger.info("Final RAG response generated")

    return final_response