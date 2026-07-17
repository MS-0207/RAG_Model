from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str
    openai_api_key: str
    database_url: str

    raw_dir: Path
    processed_dir: Path
    vector_store_dir: Path

    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_cache_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        frozen=True,
    )


settings = Settings()