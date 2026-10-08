from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
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
    # Optional separate Ollama endpoint for the dynamic Agent's action choice only, for
    # a policy model the pinned runtime cannot serve. Empty: same endpoint as generation.
    agent_policy_url: str = ""
    # Structured coverage judge for the semantic evidence evaluator (shadow/experiments).
    # Its calls are accounted separately from the policy and the answer generator.
    semantic_judge_model: str = "qwen2.5:7b-instruct"
    semantic_judge_url: str = ""
    # Record semantic coverage beside the lexical proxies in the hybrid Agent. Shadow only.
    semantic_coverage_shadow: bool = False
    # Let the semantic evaluator decide coverage, stopping and evidence order in the
    # hybrid Agent (experiment arm; fixed before the unseen benchmark was written).
    semantic_coverage_control: bool = False
    # V2 post-completion observation has its own budget and never writes task state.
    semantic_slot_shadow_enabled: bool = False
    semantic_slot_shadow_seconds: float = Field(default=90, gt=0, le=300)
    # Diagnostic observations run outside the request, with one worker and a
    # bounded queue. Sampling also limits model contention with primary tasks.
    semantic_slot_shadow_sample_rate: float = Field(default=0.1, ge=0, le=1)
    semantic_slot_control_enabled: bool = False
    semantic_slot_gate_path: str = ""
    semantic_slot_calibration_path: str = ""
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
    # Tenants whose corpus has no competing rules to reconcile. The cross-document
    # conflict check is an extra model call per multi-document answer; on the news
    # corpus it ran on 86 of 150 answers and was accepted 5 times.
    conflict_check_disabled_tenants: list[str] = ["multihop"]
    source_clause_queries: bool = False
    source_focus_queries: bool = False
    source_facet_queries: bool = False
    source_facet_document_queries: bool = False
    article_first_lanes: bool = False
    source_query_plan: bool = False
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
    answer_contract_enabled: bool = False
    source_facts_enabled: bool = False
    # Candidates remain opt-in until paired quality, refusal and cost gates pass.
    passage_window_enabled: bool = False
    passage_window_extra: int = Field(default=2, ge=0, le=4)
    passage_scan_limit: int = Field(default=64, ge=8, le=256)
    source_facts_protocol: Literal["strict", "partial"] = "strict"
    source_facts_fallback_enabled: bool = False
    focused_generation_enabled: bool = False
    claim_consistency_enabled: bool = False
    adaptive_routing_enabled: bool = True
    agent_lease_seconds: int = 300
    agent_task_timeout_seconds: float = Field(default=600, gt=0)
    agent_policy_max_calls: int = Field(default=12, ge=1)
    agent_judge_max_calls: int = Field(default=8, ge=1)
    agent_judge_token_budget: int = Field(default=24000, ge=1)
    agent_generation_max_calls: int = Field(default=12, ge=1)
    task_contract_enabled: bool = True
    answer_quality_enabled: bool = False
    answer_extractive_enabled: bool = False
    answer_semantic_audit_enabled: bool = False
    answer_literal_judgment_enabled: bool = False
    answer_semantic_audit_gate_path: str = ""
    # Separate from historical coverage/slot scorers and their frozen gates.
    answer_comparison_focus_enabled: bool = False
    answer_composition_model: str = "qwen3.5:9b"
    answer_composition_url: str = "http://127.0.0.1:11437"
    answer_composition_reasoning: Literal["off", "low", "medium", "high"] = "off"
    answer_composition_max_tokens: int = Field(default=900, ge=200, le=2400)
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


def conflict_check_enabled(tenant_id) -> bool:
    return tenant_id not in settings().conflict_check_disabled_tenants
