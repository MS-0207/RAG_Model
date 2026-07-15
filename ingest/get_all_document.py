from pathlib import Path
from datetime import datetime
from utils.logger import get_logger
from utils.config import settings


def get_document_information(document_name: str):

    file_path = settings.RAW_DIR / document_name

    if not file_path.exists():
        return {
            "status": "error",
            "message": "Document not found"
        }

    return {
        "name": file_path.name,
        "extension": file_path.suffix,
        "size_bytes": file_path.stat().st_size,
        "last_modified": datetime.fromtimestamp(
            file_path.stat().st_mtime
        ).isoformat()
    }