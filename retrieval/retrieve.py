from LLM.answer_generator import check_grounding, generate_answer
from Response.response import build_final_response
from reranking.cross_encoder_rank import cross_encoder_rerank
from retrieval.BM25 import hybrid_retrieval
from utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    query = input("Enter your query: ")

    logger.info("Loading vector store...")
    retrieved_docs = hybrid_retrieval(
        query=query,
        bm25_top_k=10,
        vector_top_k=10,
        fetch_k=30,
        lambda_mult=0.5,
    )

    logger.info("Ranking by cross encoder...")
    responses = cross_encoder_rerank(
        query=query,
        docs=retrieved_docs,
        top_k=3,
    )

    logger.info("Generating answer...")
    answer_result = generate_answer(
        query=query,
        top_docs=responses,
    )

    logger.info("Checking grounding...")
    grounding_result = check_grounding(
        query=query,
        answer=answer_result["answer"],
        top_docs=responses,
    )

    logger.info("Building final response...")
    final_response = build_final_response(
        query=query,
        answer_result=answer_result,
        grounding_result=grounding_result,
        top_docs=responses,
    )
