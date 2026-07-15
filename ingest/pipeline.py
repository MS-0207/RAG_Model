from ingest.loader import load_all_documents
from ingest.chunker import chunk_documents
from ingest.generate import run_embedding_pipeline
from ingest.save_chunks import save_chunks
# from utils.config import RAW_DIR, PROCESSED_DIR
# from utils.config import RAW_DIR
from utils.logger import get_logger
from utils.config import settings


logger = get_logger(__name__)

def run_ingestion_pipeline():

    logger.info("Loading documents...")
    docs = load_all_documents(settings.raw_dir)
    logger.info("Loaded %d documents",len(docs))

    logger.info("Chunking documents...")
    chunks = chunk_documents(docs)
    logger.info("Created %d chunks", len(chunks))

    logger.info("saving chunks...")
    save_chunks(chunks, settings.processed_dir)
    logger.info("Loaded total %d chunks", {len(chunks)})

    logger.info("Creating embeddings and FAISS vector store...")
    embedding_result = run_embedding_pipeline(
        settings.processed_dir,
        settings.vector_store_dir
    )

    return {
        "status": "success",
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "embedding_result": embedding_result
  }

#
# if __name__ == "__main__":
#     print(run_ingestion_pipeline())