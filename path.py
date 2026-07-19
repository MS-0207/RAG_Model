import os

BASE_DIR = r"C:\Users\msdha\PyCharmMiscProject/RAG"

FOLDERS = [
    "data/raw",
    "data/processed",
    "ingest",
    "embeddings/cache",
    "retrieval",
    "reranking",
    "chain",
    "hallucination",
    "api",
    "utils",
    "notebooks",
]

FILES = {
    "main.py": "",
    "requirements.txt": "",
    "README.md": "",
    # Ingest
    "ingest/loader.py": "# Load documents from various formats\n",
    "ingest/chunker.py": "# Split documents into chunks\n",
    "ingest/pipeline.py": "# End-to-end document ingestion pipeline\n",
    # Embeddings
    "embeddings/generate.py": "# Generate and store embeddings\n",
    # Retrieval
    "retrieval/retriever.py": "# Retrieve relevant chunks from vector store\n",
    # Reranking
    "reranking/reranker.py": "# Apply MMR or LLM-based reranking\n",
    # Chain
    "chain/build_rag_chain.py": "# Connect retriever and LLM\n",
    "chain/memory.py": "# Optional chat memory for conversations\n",
    # Hallucination
    "hallucination/relevance_check.py": "# Check answer against context for hallucination\n",
    # API
    "api/app.py": "# FastAPI or Flask app entrypoint\n",
    "api/routes.py": "# API routes for /query, etc.\n",
    # Utils
    "utils/config.py": "# Configuration settings\n",
    "utils/logger.py": "# Logging setup\n",
    "utils/helpers.py": "# Utility functions\n",
}


def create_structure():
    for folder in FOLDERS:
        path = os.path.join(BASE_DIR, folder)
        os.makedirs(path, exist_ok=True)
        print(f"📁 Created folder: {path}")

    for file_path, content in FILES.items():
        full_path = os.path.join(BASE_DIR, file_path)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"📄 Created file: {full_path}")


if __name__ == "__main__":
    create_structure()
