import re
import time

from fastapi import HTTPException

from app.clients import Models, Search
from app.config import settings
from app.models import Answer
from app.security import readable_documents, require_chunk


def normalize_quote(text):
    return re.sub(r"\s+", "", text)


def validate_claims(generated, evidence):
    by_id = {item["id"]: item for item in evidence}
    if not generated.answerable:
        return [], "insufficient_evidence"
    if not generated.claims:
        return [], "verification_failed"
    checked = []
    for claim in generated.claims:
        if len(claim.evidence_ids) != len(claim.quotes):
            return [], "verification_failed"
        for evidence_id, quote in zip(claim.evidence_ids, claim.quotes, strict=True):
            source = by_id.get(evidence_id)
            normalized = normalize_quote(quote)
            if (
                source is None
                or len(normalized) < 4
                or normalized not in normalize_quote(source["text"])
            ):
                return [], "verification_failed"
        # Mechanical checks catch fabricated identifiers/quotes, not semantic entailment.
        checked.append(
            {
                "text": claim.text,
                "evidence_ids": [by_id[key]["chunk_id"] for key in claim.evidence_ids],
                "quotes": claim.quotes,
            }
        )
    return checked, "answered"


def answer_question(db, user, question):
    started = time.monotonic()
    cfg = settings()
    documents = readable_documents(db, user)
    versions = [d.active_version_id for d in documents if d.active_version_id]
    evidence, candidates, usage = [], [], {}
    embed_ms = retrieval_ms = generation_ms = 0
    context_tokens = 0
    if versions:
        models = Models()
        t = time.monotonic()
        vector = models.embed([question])[0] if cfg.retrieval_mode == "dense" else None
        embed_ms = (time.monotonic() - t) * 1000
        t = time.monotonic()
        hits = (
            Search().retrieve(vector, user.tenant_id, versions, cfg.top_k)
            if vector is not None
            else Search().retrieve_bm25(question, user.tenant_id, versions, cfg.top_k)
        )
        retrieval_ms = (time.monotonic() - t) * 1000
        for hit in hits:
            chunk, version, document = require_chunk(db, user, hit["chunk_id"], active_only=True)
            candidates.append(
                {
                    "chunk_id": chunk.id,
                    "title": document.title,
                    "score": round(hit.get("cosine_similarity", hit.get("score", 0)), 4),
                }
            )
            if cfg.retrieval_mode == "dense" and hit["cosine_similarity"] < cfg.min_similarity:
                continue
            from app.chunking import token_count

            cost = token_count(chunk.text) + token_count(document.title) + 100
            if context_tokens + cost > cfg.context_token_budget:
                continue
            context_tokens += cost
            evidence.append(
                {
                    "id": f"E{len(evidence) + 1}",
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "version_id": version.id,
                    "title": document.title,
                    "text": chunk.text,
                    "locator": chunk.locator,
                    "metadata": document.metadata_json,
                }
            )
        # Recheck all evidence immediately before the model receives it.
        for item in evidence:
            require_chunk(db, user, item["chunk_id"], active_only=True)
        if evidence:
            t = time.monotonic()
            generated, usage = models.generate(question, evidence)
            generation_ms = (time.monotonic() - t) * 1000
            claims, status = validate_claims(generated, evidence)
            if status == "answered" and usage.get("answer_status") == "conflict":
                status = "conflict"
        else:
            claims, status = [], "insufficient_evidence"
    else:
        claims, status = [], "insufficient_evidence"
    # No cached or historical content is provided to the model in this S1 path.
    for item in candidates:
        require_chunk(db, user, item["chunk_id"], active_only=True)
    used = {key for claim in claims for key in claim["evidence_ids"]}
    citations = [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "version_id": item["version_id"],
            "title": item["title"],
            "locator": item["locator"],
            "text": item["text"],
            "preview_url": f"/api/evidence/{item['chunk_id']}",
            "original_url": f"/api/versions/{item['version_id']}/original",
        }
        for item in evidence
        if item["chunk_id"] in used
    ]
    payload = {
        "status": status,
        "claims": claims,
        "citations": citations,
        "message": (
            "资料存在冲突，以下规定无法同时成立；在确认适用关系前不能只采用其中一份。"
            if status == "conflict"
            else ""
        )
        if claims
        else (
            "模型的引用未通过检查，本次未返回答案。请调整问题后重试。"
            if status == "verification_failed"
            else "在你有权访问的资料中，未找到足够依据。"
        ),
        "trace": {
            "method": cfg.retrieval_mode,
            "prompt_version": "grounded-v4-source-spans",
            "top_k": cfg.top_k,
            "min_similarity": cfg.min_similarity,
            "context_token_budget": cfg.context_token_budget,
            "context_tokens_estimate": context_tokens,
            "token_counter": "cl100k_base",
            "embedding_model": cfg.embed_model,
            "generation_model": cfg.chat_model,
            "candidates": candidates,
            "embed_ms": round(embed_ms, 1),
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "total_ms": round((time.monotonic() - started) * 1000, 1),
        },
        "usage": usage,
    }
    # Trace contains authorized candidate titles too, so all candidates are dependencies.
    row = Answer(
        user_id=user.id,
        tenant_id=user.tenant_id,
        question=question,
        payload=payload,
        evidence_chunk_ids=[c["chunk_id"] for c in candidates],
    )
    db.add(row)
    db.commit()
    return {"id": row.id, "question": question, **payload}


def visible_answer(db, user, row):
    if row.user_id != user.id or row.tenant_id != user.tenant_id:
        raise HTTPException(404, "记录不存在")
    try:
        for chunk_id in row.evidence_chunk_ids:
            require_chunk(db, user, chunk_id)
    except HTTPException:
        return {
            "id": row.id,
            "question": "相关记录已隐藏",
            "status": "access_changed",
            "message": "引用资料已删除或访问权限已变化，这条历史记录已隐藏。",
            "claims": [],
            "citations": [],
        }
    return {"id": row.id, "question": row.question, **row.payload}
