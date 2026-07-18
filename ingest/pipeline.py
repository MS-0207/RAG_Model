from Database.redis_cache import clear_rag_cache
from api.config import settings
from ingest.chunker import chunk_documents
from ingest.generate import run_embedding_pipeline
from ingest.loader import load_all_documents
from ingest.save_chunks import save_chunks
from utils.logger import get_logger

logger = get_logger(__name__)

def run_ingestion_pipeline() -> dict:

    logger.info("Loading documents...")
    docs = load_all_documents(settings.raw_dir)
    logger.info("Loaded %d documents", len(docs))

    logger.info("Chunking documents...")
    chunks = chunk_documents(docs)
    logger.info("Created %d chunks", len(chunks))

    logger.info("Saving chunks...")
    save_chunks(
        chunks,
        settings.processed_dir,
    )
    logger.info("Saved %d chunks", len(chunks))

    logger.info("Creating embeddings and FAISS vector store...")
    embedding_result = run_embedding_pipeline(
        settings.processed_dir,
        settings.vector_store_dir,
    )

    # Clear cached RAG answers only after vector-store creation succeeds.
    deleted_keys = clear_rag_cache()

    logger.info(
        "Cleared %d cached RAG responses after ingestion",
        deleted_keys,
    )

    return {
        "status": "success",
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "embedding_result": embedding_result,
        "cached_responses_cleared": deleted_keys,
    }