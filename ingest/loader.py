from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

from api.exception import NoDocumentsFoundError
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_single_file(file_path: Path) -> List[Document]:
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning("Skipping unsupported file type: %s", file_path)
        return []

    try:
        loader: PyPDFLoader | TextLoader | UnstructuredWordDocumentLoader

        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            loader = UnstructuredWordDocumentLoader(str(file_path))

        documents = loader.load()
        logger.info("Loaded file: %s", file_path.name)
        return documents

    except Exception:
        logger.exception("Failed to load file: %s", file_path)
        raise


def load_all_documents(directory: str | Path) -> List[Document]:
    directory_path = Path(directory)

    if not directory_path.exists():
        raise NoDocumentsFoundError(
            f"Raw document directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise NoDocumentsFoundError(
            f"Configured raw document path is not a directory: {directory_path}"
        )

    supported_files = [
        file_path
        for file_path in directory_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not supported_files:
        raise NoDocumentsFoundError(
            f"No supported documents found in: {directory_path}"
        )

    all_documents: List[Document] = []

    for file_path in supported_files:
        all_documents.extend(load_single_file(file_path))

    if not all_documents:
        raise NoDocumentsFoundError(
            "Supported files were found, but no document content could be loaded."
        )

    logger.info(
        "Loaded %d document pages from %d files",
        len(all_documents),
        len(supported_files),
    )

    return all_documents
