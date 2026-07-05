from retrieval.BM25 import hybrid_retrieval
from reranking.cross_encoder_rank import cross_encoder_rerank
from LLM.answer_generator import generate_answer
from LLM.answer_generator import check_grounding
from Response.response import build_final_response

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
        lambda_mult=0.5
    )

    logger.info("Ranking by cross encoder...")
    responses = cross_encoder_rerank(
        query = query,
        docs = retrieved_docs,
        top_k = 3
    )

    logger.info("Retrieving ground truth...")
    generate = generate_answer(
        query= query,
        top_docs=responses
    )

    logger.info("Computing ground truth...")
    check_grounding = check_grounding(
        query  =query,
        answer = generate['answer'],
        top_docs = responses
    )

    logger.info("Final response...")
    final_response = build_final_response(
        query=query,
        answer_result=generate,
        grounding_result=check_grounding,
        top_docs=responses
    )
    print(f"print final response {final_response}")