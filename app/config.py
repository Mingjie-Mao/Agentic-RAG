from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://rag:rag-local-only@127.0.0.1:55436/rag"
    search_url: str = "http://127.0.0.1:19200"
    search_index: str = "agentic-rag-s1"
    ollama_url: str = "http://127.0.0.1:11436"
    embed_model: str = "bge-m3:567m"
    chat_model: str = "qwen2.5:7b-instruct"
    agent_policy_model: str = "qwen2.5:7b-instruct"
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
    # Recorded per version. "document_header" embeds and indexes each chunk with its
    # document's title, source and date; "web_v1" drops web page furniture before
    # chunking. Both default off so the frozen Chinese corpus reproduces unchanged.
    chunk_context: str = "none"
    boilerplate_filter: str = "none"
    parser_version: str = "docling-2.127-office-xml-v2"
    retrieval_mode: str = "hybrid"
    rewrite_mode: str = "rule"
    rerank_mode: str = "off"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = 30
    rerank_batch: int = 8
    rerank_max_tokens: int = 512
    context_token_budget: int = 5000
    # A question that names a publication gets part of its evidence budget retrieved
    # from that publication's documents alone. No-op for documents without a source.
    source_routing: bool = True
    source_clause_queries: bool = False
    document_quota: int = 2
    # Reorder each routed lane and the global pool with the cross-encoder before
    # admission. Off by default: it has only been measured on retrieval so far.
    passage_rerank: bool = False
    passage_rerank_depth: int = 24
    # With reranking on: every passage of the top N articles of each lane is scored,
    # not only the passages that happened to rank in the lane.
    passage_expand_documents: int = 0
    semantic_shadow_enabled: bool = False
    verdict_protocol: Literal["legacy", "structured"] = "legacy"
    verdict_span_mode: Literal["free", "constrained"] = "constrained"
    agent_lease_seconds: int = 300
    memory_url: str = ""
    # JSON: {"tenant_id:user_id": "bearer-token"}. Kept out of API responses/logs.
    memory_tokens_json: str = "{}"
    memory_timeout_seconds: float = 20
    # Optional public trial. Listed users become read-only and share a hard daily
    # request budget enforced in PostgreSQL before model work starts.
    trial_mode: bool = False
    trial_usernames: str = ""
    trial_daily_limit: int = 10


@lru_cache
def settings() -> Settings:
    return Settings()
