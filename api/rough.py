from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS

from api.dependencies import get_vector_store
from retrieval.BM25 import (
    get_all_docs_from_faiss,
    bm25_retrieval,
    vector_mmr_retrieval,
    merge_and_deduplicate,
)

router = APIRouter(tags=["RAG"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class RetrievedDocument(BaseModel):
    content: str
    metadata: dict[str, Any]


class RetrievalResponse(BaseModel):
    query: str
    total_documents: int
    documents: list[RetrievedDocument]


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
)
def retrieve(
    request: QueryRequest,
    db: FAISS = Depends(get_vector_store),
) -> RetrievalResponse:
    all_docs = get_all_docs_from_faiss(db)

    bm25_docs = bm25_retrieval(
        query=request.query,
        docs=all_docs,
        top_k=10,
    )

    vector_docs = vector_mmr_retrieval(
        query=request.query,
        db=db,
        top_k=10,
        fetch_k=30,
        lambda_mult=0.5,
    )

    retrieved_docs = merge_and_deduplicate(
        bm25_docs=bm25_docs,
        vector_docs=vector_docs,
    )

    return RetrievalResponse(
        query=request.query,
        total_documents=len(retrieved_docs),
        documents=[
            RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in retrieved_docs
        ],
    )
