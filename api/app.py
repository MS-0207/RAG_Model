from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from datetime import datetime
import time

from main import run_rag_pipeline
from api.dependencies import verify_api_key
from retrieval.BM25 import hybrid_retrieval
from reranking.cross_encoder_rank import cross_encoder_rerank
from LLM.answer_generator import generate_answer, check_grounding
from ingest.pipeline import run_ingestion_pipeline
from ingest.document_loader import get_all_documents
from ingest.get_all_document import get_document_information
from ingest.delete_document import delete_document_from_storage

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    query: str


class FeedbackRequest(BaseModel):
    query: str
    rating: int
    comment: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    print(
        f"{request.method} {request.url.path} "
        f"completed in {process_time:.4f}s "
        f"with status {response.status_code}"
    )

    return response


@app.get("/")
def home():
    return {"message": "first rag api"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "RAG API",
        "message": "API is running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/version")
def version():
    return {
        "status": "version",
        "app": "rag_model",
        "embed_model": "openai_embedding_model",
        "llm": "openai",
        "vector_database": "FAISS"
    }


@app.post("/ask")
def ask(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        return run_rag_pipeline(request.query)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Vector database not found. Please run /documents/ingest first."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    retrieved_docs = hybrid_retrieval(
        query=request.query,
        bm25_top_k=10,
        vector_top_k=10,
        fetch_k=30,
        lambda_mult=0.5
    )

    return {
        "query": request.query,
        "total_documents": len(retrieved_docs),
        "documents": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in retrieved_docs
        ]
    }


@app.post("/rerank")
def rerank(request: QueryRequest):
    retrieved_docs = hybrid_retrieval(
        query=request.query,
        bm25_top_k=10,
        vector_top_k=10,
        fetch_k=30,
        lambda_mult=0.5
    )

    reranked_docs = cross_encoder_rerank(
        query=request.query,
        docs=retrieved_docs,
        top_k=3
    )

    return {
        "query": request.query,
        "total_documents": len(reranked_docs),
        "documents": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in reranked_docs
        ]
    }


@app.post("/generate")
def generate(request: QueryRequest):
    retrieved_docs = hybrid_retrieval(
        query=request.query,
        bm25_top_k=10,
        vector_top_k=10,
        fetch_k=30,
        lambda_mult=0.5
    )

    top_docs = cross_encoder_rerank(
        query=request.query,
        docs=retrieved_docs,
        top_k=3
    )

    return generate_answer(
        query=request.query,
        top_docs=top_docs
    )


@app.post("/grounding")
def grounding(request: QueryRequest):
    retrieved_docs = hybrid_retrieval(
        query=request.query,
        bm25_top_k=10,
        vector_top_k=10,
        fetch_k=30,
        lambda_mult=0.5
    )

    top_docs = cross_encoder_rerank(
        query=request.query,
        docs=retrieved_docs,
        top_k=3
    )

    answer_result = generate_answer(
        query=request.query,
        top_docs=top_docs
    )

    return check_grounding(
        query=request.query,
        answer=answer_result["answer"],
        top_docs=top_docs
    )

@app.post("/documents/ingest")
def ingest():
    run_ingestion_pipeline()
    return {"status": "success", "message": "Documents ingested successfully"}

@app.get("/documents")
def get_documents():
    documents = get_all_documents()
    return {
        "total_documents": len(documents),
        "documents": documents
    }

@app.get("/documents/{document_name}")
def get_document(document_name: str):
    return get_document_information(document_name)

@app.delete("/documents/{document_name}")
def delete_document(document_name: str):
    return delete_document_from_storage(document_name)

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    return {"status": "saved"}