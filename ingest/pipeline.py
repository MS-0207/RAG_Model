from ingest.loader import load_all_documents
from ingest.chunker import chunk_documents
from ingest.generate import run_embedding_pipeline
from ingest.save_chunks import save_chunks

RAW_DIR = r"C:\Users\msdha\PycharmProjects\RAG_Project\RAG\data\raw"
PROCESSED_DIR = r"C:\Users\msdha\PycharmProjects\RAG_Project\RAG\data\processed"

def run_ingestion_pipeline():

    print("📥 Loading documents...")
    docs = load_all_documents(RAW_DIR)
    print(f"✅ Loaded {len(docs)} documents")

    print('chunking documents...')
    chunks = chunk_documents(docs)
    print(f"✅ chunks {len(chunks)} ")

    print('saving documents...')
    save_chunks(chunks, PROCESSED_DIR)
    print(f"Total chunks: {len(chunks)}")

    print('saving embeddings...')
    run_embedding_pipeline()

    print("Vector store built successfully.")

if __name__ == "__main__":
    run_ingestion_pipeline()
