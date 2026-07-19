from typing import Any
from fastapi import APIRouter, Depends
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from LLM.answer_generator import check_grounding, generate_answer
from api.services.main import run_rag_pipeline
from reranking.cross_encoder_rank import cross_encoder_rerank
from retrieval.BM25 import (bm25_retrieval,get_all_docs_from_faiss,merge_and_deduplicate,vector_mmr_retrieval,)
from utils.logger import get_logger
from fastapi import Request
from api.dependencies import get_vector_store, verify_api_key

logger = get_logger(__name__)
router = APIRouter(tags=["RAG"])


# --------------------------------------------------
# Request models
# --------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question submitted to the RAG pipeline",
    )


# --------------------------------------------------
# Response models
# --------------------------------------------------


class RetrievedDocument(BaseModel):
    content: str
    metadata: dict[str, Any]


class RetrievalResponse(BaseModel):
    query: str
    total_documents: int
    documents: list[RetrievedDocument]


# --------------------------------------------------
# Shared retrieval helper
# --------------------------------------------------

def retrieve_documents(
    query: str,
    db: FAISS,
) -> list[Document]:
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

    return merge_and_deduplicate(
        bm25_docs=bm25_docs,
        vector_docs=vector_docs,
    )

def build_retrieval_response(
    query: str,
    documents: list[Document],
) -> RetrievalResponse:
    return RetrievalResponse(
        query=query,
        total_documents=len(documents),
        documents=[
            RetrievedDocument(
                content=document.page_content,
                metadata=document.metadata,
            )
            for document in documents
        ],
    )


# --------------------------------------------------
# Endpoints
# --------------------------------------------------


@router.post(
    "/ask",
    response_model=dict[str, Any],
)
def ask(
    http_request: Request,
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),
    db: FAISS = Depends(get_vector_store),
):
    request_id = getattr(
        http_request.state,
        "request_id",
        "unknown",
    )

    logger.info(
        "Processing request_id=%s",
        request_id,
    )

    return run_rag_pipeline(
        query=request.query,
        db=db,
    )

@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
)
def retrieve(
    request: QueryRequest,
    db: FAISS = Depends(get_vector_store),
) -> RetrievalResponse:

    retrieved_docs = retrieve_documents(
        query=request.query,
        db=db,
    )

    return build_retrieval_response(
        query=request.query,
        documents=retrieved_docs,
    )

@router.post(
    "/rerank",
    response_model=RetrievalResponse,
)
def rerank(
    request: QueryRequest,
    db: FAISS = Depends(get_vector_store),) -> RetrievalResponse:

    retrieved_docs = retrieve_documents(
        query=request.query,
        db=db,
    )

    reranked_docs = cross_encoder_rerank(
        query=request.query,
        docs=retrieved_docs,
        top_k=3,
    )

    return build_retrieval_response(
        query=request.query,
        documents=reranked_docs,
    )


@router.post(
    "/generate",
    response_model=dict[str, Any],
)
def generate(
    request: QueryRequest,
    db: FAISS = Depends(get_vector_store),
) -> dict[str, Any]:

    retrieved_docs = retrieve_documents(
        query=request.query,
        db=db,
    )

    top_docs = cross_encoder_rerank(
        query=request.query,
        docs=retrieved_docs,
        top_k=3,
    )

    return generate_answer(
        query=request.query,
        top_docs=top_docs,
    )


@router.post(
    "/grounding",
    response_model=dict[str, Any],
)
def grounding(
    request: QueryRequest,
    db: FAISS = Depends(get_vector_store),
) -> dict[str, Any]:

    retrieved_docs = retrieve_documents(
        query=request.query,
        db=db,
    )

    top_docs = cross_encoder_rerank(
        query=request.query,
        docs=retrieved_docs,
        top_k=3,
    )

    answer_result = generate_answer(
        query=request.query,
        top_docs=top_docs,
    )

    return check_grounding(
        query=request.query,
        answer=answer_result["answer"],
        top_docs=top_docs,
    )
