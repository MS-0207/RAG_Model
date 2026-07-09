from pathlib import Path
from datetime import datetime

RAW_DIR = Path(r"C:\Users\msdha\PycharmProjects\RAG_Project\RAG\data\raw")


def get_all_documents():

    documents = []

    for file in RAW_DIR.rglob("*"):

        if file.is_file():

            documents.append({
                "name": file.name,
                "extension": file.suffix,
                "size_bytes": file.stat().st_size,
                "last_modified": datetime.fromtimestamp(
                    file.stat().st_mtime
                ).isoformat()
            })

    return documents