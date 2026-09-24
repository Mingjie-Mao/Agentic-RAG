"""Permission-scoped retrieval shared by single-turn RAG and Agent tools."""

import time
from dataclasses import dataclass

from sqlalchemy import select

from app.chunking import token_count
from app.models import Tenant
from app.security import readable_documents, require_chunk
from app.task_analysis import multi_source_intent, needs_document_diversity


def _boilerplate_only(text: str) -> bool:
    remainder = text.replace("本项目自建的虚构资料，仅用于开发与演示，不代表任何真实公司的制度。", "")
    remainder = remainder.replace("#", "").strip()
    return "本项目自建的虚构资料" in text and len(remainder) < 18


def foreign_tenant_mentioned(db, user, query: str) -> bool:
    """Reject an explicit tenant switch before unrelated local evidence reaches a model."""
    if db is None or not hasattr(db, "execute") or not getattr(user, "tenant_id", None):
        return False
    tenants = db.execute(select(Tenant.id, Tenant.name)).all()
    return any(
        tenant_id != user.tenant_id and len(name.strip()) >= 2 and name.strip() in query
        for tenant_id, name in tenants
    )


@dataclass
class RetrievalResult:
    evidence: list[dict]
    candidates: list[dict]
    readable_documents: int
    searchable_documents: int
    context_tokens: int
    embed_ms: float
    retrieval_ms: float


def retrieve_authorized(
    db,
    user,
    query,
    *,
    cfg,
    models,
    search,
    top_k=None,
    readable_documents_fn=readable_documents,
    require_chunk_fn=require_chunk,
):
    """Retrieve evidence without generating an answer or writing history.

    Identity and ACL scope come from the server. Every returned chunk is checked
    against the business database before it is exposed to either a model or caller.
    """
    documents = readable_documents_fn(db, user)
    versions = [document.active_version_id for document in documents if document.active_version_id]
    evidence, candidates = [], []
    embed_ms = retrieval_ms = 0.0
    context_tokens = 0
    limit = min(max(top_k or cfg.top_k, 1), 8)
    if foreign_tenant_mentioned(db, user, query):
        return RetrievalResult(
            evidence, candidates, len(documents), len(versions), 0, embed_ms, retrieval_ms
        )
    if not versions:
        return RetrievalResult(
            evidence, candidates, len(documents), 0, 0, embed_ms, retrieval_ms
        )

    started = time.monotonic()
    vector = models.embed([query])[0] if cfg.retrieval_mode != "bm25" else None
    embed_ms = (time.monotonic() - started) * 1000
    started = time.monotonic()
    diversify = needs_document_diversity(query) or multi_source_intent(query)
    search_limit = min(limit * 3, 24) if diversify else limit
    if cfg.retrieval_mode == "hybrid":
        hits = search.retrieve_hybrid(query, vector, user.tenant_id, versions, search_limit)
    elif vector is not None:
        hits = search.retrieve(vector, user.tenant_id, versions, search_limit)
    else:
        hits = search.retrieve_bm25(query, user.tenant_id, versions, search_limit)
    retrieval_ms = (time.monotonic() - started) * 1000

    admitted_by_document = {}
    for rank, hit in enumerate(hits, 1):
        chunk, version, document = require_chunk_fn(db, user, hit["chunk_id"], active_only=True)
        candidate = {
            "chunk_id": chunk.id,
            "rank": rank,
            "title": document.title,
            "document_id": document.id,
            "version_id": version.id,
            "locator_label": chunk.locator.get("label", ""),
            "score": round(hit.get("cosine_similarity", hit.get("score", 0)), 4),
            "bm25_rank": hit.get("bm25_rank"),
            "dense_rank": hit.get("dense_rank"),
            "fusion_score": round(hit["score"], 6) if cfg.retrieval_mode == "hybrid" else None,
            "admitted": False,
            "excluded_because": None,
        }
        candidates.append(candidate)
        if cfg.retrieval_mode == "dense" and hit["cosine_similarity"] < cfg.min_similarity:
            candidate["excluded_because"] = "below_min_similarity"
            continue
        if _boilerplate_only(chunk.text):
            candidate["excluded_because"] = "boilerplate_only"
            continue
        if len(evidence) >= limit:
            candidate["excluded_because"] = "top_k_full"
            continue
        if diversify and admitted_by_document.get(document.id, 0) >= 2:
            candidate["excluded_because"] = "document_quota"
            continue
        cost = token_count(chunk.text) + token_count(document.title) + 100
        if context_tokens + cost > cfg.context_token_budget:
            candidate["excluded_because"] = "context_budget_exhausted"
            continue
        context_tokens += cost
        candidate["admitted"] = True
        candidate["evidence_id"] = f"E{len(evidence) + 1}"
        evidence.append(
            {
                "id": candidate["evidence_id"],
                "chunk_id": chunk.id,
                "document_id": document.id,
                "version_id": version.id,
                "title": document.title,
                "text": chunk.text,
                "locator": chunk.locator,
                "metadata": document.metadata_json,
            }
        )
        admitted_by_document[document.id] = admitted_by_document.get(document.id, 0) + 1

    for item in evidence:
        require_chunk_fn(db, user, item["chunk_id"], active_only=True)
    return RetrievalResult(
        evidence,
        candidates,
        len(documents),
        len(versions),
        context_tokens,
        embed_ms,
        retrieval_ms,
    )
