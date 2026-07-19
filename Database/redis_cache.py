import hashlib
import json
import logging

import redis
from redis.exceptions import RedisError

from api.config import settings

logger = logging.getLogger(__name__)

CACHE_PREFIX = "rag:"

redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    decode_responses=True,
)


def build_cache_key(query: str) -> str:
    normalized_query = " ".join(query.lower().split())
    query_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()

    return f"{CACHE_PREFIX}{query_hash}"


def get_cached_response(query: str):
    """
    Returns cached response or None.
    If Redis is unavailable, gracefully fall back.
    """
    try:
        key = build_cache_key(query)
        cached = redis_client.get(key)

        if cached is None:
            return None

        return json.loads(cached)

    except RedisError as exc:
        logger.warning("Redis unavailable while reading cache: %s", exc)
        return None


def save_cached_response(query: str, response: dict):
    """
    Saves response into Redis.
    If Redis is unavailable, continue normally.
    """
    try:
        key = build_cache_key(query)

        redis_client.setex(
            key,
            settings.redis_ttl,
            json.dumps(response),
        )

    except RedisError as exc:
        logger.warning("Redis unavailable while saving cache: %s", exc)


def clear_rag_cache() -> int:
    """
    Deletes all RAG cache entries.
    Returns number deleted.
    """
    try:
        keys = list(redis_client.scan_iter(f"{CACHE_PREFIX}*"))

        if not keys:
            return 0

        return redis_client.delete(*keys)

    except RedisError as exc:
        logger.warning("Redis unavailable while clearing cache: %s", exc)
        return 0
