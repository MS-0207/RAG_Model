import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from utils.logger import get_logger

logger = get_logger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.perf_counter()

        logger.info(
            "Request started | request_id=%s | method=%s | path=%s",
            request_id,
            request.method,
            request.url.path,
        )

        try:
            response = await call_next(request)

        except Exception:
            logger.exception(
                "Request failed | request_id=%s | method=%s | path=%s",
                request_id,
                request.method,
                request.url.path,
            )
            raise

        process_time = time.perf_counter() - start_time

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        logger.info(
            "Request completed | request_id=%s | method=%s | path=%s "
            "| status_code=%d | process_time=%.4f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            process_time,
        )

        return response