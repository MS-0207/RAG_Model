from fastapi import Request
from fastapi.responses import JSONResponse
from api.exception import NoDocumentsFoundError

async def global_exception_handler(
    request: Request,
    exc: Exception
):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server not of the dock error",
            "path": request.url.path
        }
    )

async def file_not_found_handler(
    request: Request,
    exc: Exception
):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": str(exc),
            "path": request.url.path
        }
    )

async def no_documents_found_handler(
    request: Request,
    exc: NoDocumentsFoundError,
):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": str(exc),
            "path": request.url.path,
        },
    )