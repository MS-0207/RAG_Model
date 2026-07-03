# Generate and store embeddings
# rag_project/embeddings/generate.py

import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS  # or Chroma
from langchain_openai import OpenAIEmbeddings

# 🔐 Set your API key directly here
PROCESSED_DIR = r"C:\Users\msdha\PyCharmMiscProject\new\data\processed"
VECTOR_STORE_DIR = r"C:\Users\msdha\PyCharmMiscProject\new\embeddings\cache"

def load_chunks_from_txt(processed_dir=PROCESSED_DIR):
    documents = []
    for file_path in Path(processed_dir).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
            else:
                print(f"⚠️ Skipping empty file: {file_path.name}")
    return documents

def embed_and_store(docs):
    embeddings = OpenAIEmbeddings(openai_api_key="")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_STORE_DIR)
    print(f"✅ Vector store saved to: {VECTOR_STORE_DIR}")

def run_embedding_pipeline():
    print("📄 Loading text chunks from processed directory...")
    docs = load_chunks_from_txt()
    print("value of docs")


    if not docs:
        print("❌ No valid documents found in processed folder.")
        return

    print(f"✅ Loaded {len(docs)} non-empty chunks")
    print("🔢 Generating embeddings and storing in FAISS...")
    embed_and_store(docs)
    print("🎉 Embedding complete!")

if __name__ == "__main__":
    run_embedding_pipeline()
