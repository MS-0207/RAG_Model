from pathlib import Path

RAW_DIR = Path(r"C:\Users\msdha\PycharmProjects\RAG_Project\RAG\data\raw")


def delete_document_from_storage(document_name: str):

    file_path = RAW_DIR / document_name

    if not file_path.exists():
        return {
            "status": "error",
            "message": f"{document_name} not found."
        }

    file_path.unlink()

    return {
        "status": "success",
        "message": f"{document_name} deleted successfully."
    }