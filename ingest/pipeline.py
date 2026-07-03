# End-to-end document ingestion pipeline
import os
import pathlib
from dataclasses import field
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from numpy.lib.recfunctions import unstructured_to_structured
from ingest.loader import load_all_documents
from ingest.chunker import chunk_documents

RAW_DIR = r"C:\Users\msdha\PyCharmMiscProject\new\data\raw"
PROCESSED_DIR = r"C:\Users\msdha\PyCharmMiscProject\new\data\processed"

def save_chunks(chunks, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        file_path = os.path.join(processed_dir, f"chunk_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chunk.page_content)

def run_ingestion_pipeline():
    print("📥 Loading documents...")
    docs = load_all_documents(RAW_DIR)
    print(f"✅ Loaded {len(docs)} documents")
    print(type(docs))
    print(docs)
    print(docs[0])
    print(docs[0].page_content)
    if not docs:
        print("No documents found")
        return
    print(docs[0])
    print(docs[0].page_content)
    chunks = chunk_documents(docs)
    print(type(chunks))
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        print(f"First chunk: {chunks[0]}")
        print(f"First chunk content: {chunks[0].page_content}")
    save_chunks(chunks, PROCESSED_DIR)
    print(f"Total chunks: {len(chunks)}")

    first_chunk_file = os.path.join(PROCESSED_DIR, "chunk_0.txt")
    with open(first_chunk_file, "r", encoding="utf-8") as f:
        content = f.read()

    print("Content of chunk_0.txt:")
    print(content)

if __name__ == "__main__":
    run_ingestion_pipeline()