# Load documents from various formats

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
import os

def load_single_file(file_path: Path):
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".txt":
            loader = TextLoader(str(file_path))
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(str(file_path))
        else:
            print(f"⚠️ Skipping unsupported file type: {file_path}")
            return []

        return loader.load()

    except Exception as e:
        print(f"❌ Failed to load {file_path.name}: {e}")
        return []

def load_all_documents(directory: str):
    all_docs = []
    for file_path in Path(directory).rglob("*"):
        if file_path.is_file():
            docs = load_single_file(file_path)
            all_docs.extend(docs)
    return all_docs
