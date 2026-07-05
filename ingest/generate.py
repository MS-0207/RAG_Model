PROCESSED_DIR = r"C:\Users\msdha\PycharmProjects\RAG_Project\RAG\data\processed"
VECTOR_STORE_DIR = r"C:\Users\msdha\PycharmProjects\RAG_Project\RAG\embeddings\cache"


import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def load_chunks_from_txt(processed_dir=PROCESSED_DIR):
    documents = []
    for file_path in Path(processed_dir).glob("*.txt"):
        # print(f"value of fp {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
    return documents

def embed_and_store(docs):
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_STORE_DIR)
    print(f"✅ Vector store saved to: {VECTOR_STORE_DIR}")

def run_embedding_pipeline():
    docs = load_chunks_from_txt()
    print(docs)

    if not docs:
        print("❌ No valid documents found.")
        return

    print(f"✅ Loaded {len(docs)} chunks")


