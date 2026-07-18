from celery import Celery
from api.config import settings

celery_app = Celery(
    "rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["api.tasks"],
)