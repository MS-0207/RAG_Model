# Configuration settings

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# # API_KEY = os.getenv("API_KEY")
# # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # RAW_DIR = Path(os.getenv("RAW_DIR"))
# # PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR"))
# # VECTOR_STORE_DIR =Path(os.getenv("VECTOR_STORE_DIR"))

class Settings(BaseSettings):

    # ----------------------------
    # API
    # ----------------------------
    api_key: str

    # ----------------------------
    # OpenAI
    # ----------------------------
    openai_api_key: str

    # ----------------------------
    # Directories
    # ----------------------------
    raw_dir: Path
    processed_dir: Path
    vector_store_dir: Path

    # ----------------------------
    # RAG
    # ----------------------------
    # chunk_size: int = 1000
    # chunk_overlap: int = 200
    #
    # retrieval_top_k: int = 10
    # rerank_top_k: int = 3

    # ----------------------------
    # Environment
    # ----------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

settings = Settings()
