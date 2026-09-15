from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://rag:rag-local-only@127.0.0.1:55436/rag"
    search_url: str = "http://127.0.0.1:19200"
    search_index: str = "enterprise-rag-s1"
    ollama_url: str = "http://127.0.0.1:11436"
    embed_model: str = "bge-m3:567m"
    chat_model: str = "qwen2.5:7b-instruct"
    embed_dimension: int = 1024
    storage_dir: Path = Path(".runtime/files")
    demo_mode: bool = False
    demo_password: str = ""
    cookie_secure: bool = False
    allowed_origins: list[str] = ["http://127.0.0.1:8000"]
    max_upload_bytes: int = 10 * 1024 * 1024
    session_hours: int = 8
    chunk_chars: int = 700
    chunk_overlap: int = 100
    top_k: int = 4
    min_similarity: float = 0.35
    model_timeout_seconds: float = 180
    chunk_strategy: str = "structure"
    parser_version: str = "docling-2.127-office-xml-v2"
    retrieval_mode: str = "dense"
    context_token_budget: int = 5000


@lru_cache
def settings() -> Settings:
    return Settings()
