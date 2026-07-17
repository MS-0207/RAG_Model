import hashlib
import json
from typing import Any

from Database.redis_connection import redis_client
from api.config import settings

CACHE_PREFIX = "rag:"


def build_cache_key(query: str) -> str:
    normalized_query = " ".join(query.lower().split())
    query_hash = hashlib.sha256(
        normalized_query.encode("utf-8")
    ).hexdigest()

    return f"{CACHE_PREFIX}{query_hash}"


def get_cached_response(query: str) -> dict[str, Any] | None:
    cache_key = build_cache_key(query)
    cached_value = redis_client.get(cache_key)

    if cached_value is None:
        return None

    return json.loads(cached_value)


def save_cached_response(
    query: str,
    response: dict[str, Any],
) -> None:
    cache_key = build_cache_key(query)

    redis_client.setex(
        cache_key,
        settings.redis_cache_ttl_seconds,
        json.dumps(response),
    )


def clear_rag_cache() -> int:
    keys = list(redis_client.scan_iter(f"{CACHE_PREFIX}*"))

    if not keys:
        return 0

    return redis_client.delete(*keys)