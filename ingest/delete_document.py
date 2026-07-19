from api.config import settings


def delete_document_from_storage(document_name: str):
    file_path = settings.raw_dir / document_name

    if not file_path.exists():
        return {"status": "error", "message": f"{document_name} not found."}

    file_path.unlink()

    return {"status": "success", "message": f"{document_name} deleted successfully."}
