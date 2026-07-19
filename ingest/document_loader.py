from datetime import datetime
from api.config import settings


def get_all_documents():
    documents = []

    for file in settings.raw_dir.rglob("*"):
        if file.is_file():
            documents.append(
                {
                    "name": file.name,
                    "extension": file.suffix,
                    "size_bytes": file.stat().st_size,
                    "last_modified": datetime.fromtimestamp(
                        file.stat().st_mtime
                    ).isoformat(),
                }
            )

    return documents
