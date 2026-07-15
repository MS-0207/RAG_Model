from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

def load_chunks_from_txt(processed_dir):
    documents = []
    for file_path in Path(processed_dir).glob("*.txt"):
        # print(f"value of fp {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
    return documents

def embed_and_store(docs: list[Document]
                    ,vector_store_dir):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    vector_store_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    db.save_local(vector_store_dir)

    logger.info(
        "Vector store saved to %s",
        vector_store_dir
    )

def run_embedding_pipeline(processed_dir: Path,vector_store_dir:Path):
    logger.info(
        "Loading processed chunks..."
    )

    docs = load_chunks_from_txt(processed_dir)

    if not docs:
        raise ValueError(
            f"No valid chunks found in {processed_dir}"
        )

    logger.info("Loaded %d chunks", len(docs))

    embed_and_store(docs,vector_store_dir)

    return {
        "status": "success",
        "chunks_embedded": len(docs),
        "vector_store_dir": str(vector_store_dir),
    }


