import re
import time

from fastapi import HTTPException

from app.clients import Models, Search
from app.config import settings
from app.models import Answer
from app.rewrite import rewrite_query
from app.retrieval import retrieve_authorized
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


_CJK = re.compile(r"[\u4e00-\u9fff]+")


def _bigrams(text):
    joined = "".join(_CJK.findall(text or ""))
    return {joined[i : i + 2] for i in range(max(0, len(joined) - 1))}


def shadow_scores(question, evidence, claims):
    """Record relevance and entailment without acting on them.

    Calibration on 34 machine-labelled cases put the rule, embedding and cross-encoder
    scorers within overlapping confidence intervals, so none of them has earned the
    right to withhold an answer. Recording them now means a later decision can be made
    on production traffic rather than on 34 cases.
    """
    material = "\n".join(item["text"] for item in evidence[:4])
    question_grams = _bigrams(question)
    relevance = (
        len(question_grams & _bigrams(material)) / len(question_grams) if question_grams else None
    )
    supported = []
    for claim in claims:
        grams = _bigrams(claim["text"])
        supported.append(round(len(grams & _bigrams(material)) / len(grams), 4) if grams else None)
    return {
        "relevance": round(relevance, 4) if relevance is not None else None,
        "claim_support": supported,
        "scorer": "bigram-overlap-v1",
        "mode": "shadow",
        "action": "recorded only; no answer is withheld on these scores",
        "calibration": "artifacts/a-q1-calibration.json",
    }


MESSAGES = {
    "verification_failed": "模型的引用未通过检查，本次未返回答案。请调整问题后重试。",
    "no_readable_documents": "你当前没有可访问的资料。上传资料，或请管理者把已有资料分享到你所在的组。",
    "documents_processing": "你的资料还在处理中，全部处理完成后才能检索。可以在资料库查看进度。",
}


def answer_question(db, user, question, history=None):
    started = time.monotonic()
    cfg = settings()
    # Rewriting produces a retrieval query and nothing else: it is not stored, never
    # crosses a session, and never becomes evidence.
    query, rewrite_trace = rewrite_query(question, history, mode=cfg.rewrite_mode)
    models = Models()
    # Short fact questions are especially vulnerable to one off-topic chunk occupying
    # a quarter of the context. Six candidates fixed observed misses while the token
    # budget still provides the hard upper bound; longer questions keep the frozen default.
    retrieval_top_k = 6 if len(question) <= 30 else cfg.top_k
    found = retrieve_authorized(
        db,
        user,
        query,
        cfg=cfg,
        models=models,
        search=Search(),
        readable_documents_fn=readable_documents,
        require_chunk_fn=require_chunk,
        top_k=retrieval_top_k,
    )
    evidence, candidates, usage = found.evidence, found.candidates, {}
    embed_ms, retrieval_ms, generation_ms = found.embed_ms, found.retrieval_ms, 0
    context_tokens = found.context_tokens
    if found.searchable_documents:
        if evidence:
            t = time.monotonic()
            generated, usage = models.generate(question, evidence)
            generation_ms = (time.monotonic() - t) * 1000
            claims, status = validate_claims(generated, evidence)
            if status == "answered" and usage.get("answer_status") == "conflict":
                status = "conflict"
        else:
            claims, status = [], "insufficient_evidence"
    elif not found.readable_documents:
        # Nothing readable at all is a different situation from "searched and found nothing".
        claims, status = [], "no_readable_documents"
    else:
        claims, status = [], "documents_processing"
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
        else MESSAGES.get(status, "在你有权访问的资料中，未找到足够依据。"),
        "trace": {
            "method": cfg.retrieval_mode,
            "scope": {
                "readable_documents": found.readable_documents,
                "searchable_documents": found.searchable_documents,
            },
            "original_question": question,
            "retrieval_query": query,
            "query_rewritten": rewrite_trace["rewritten"],
            "rewrite": rewrite_trace,
            "shadow_scores": shadow_scores(question, evidence, claims) if evidence else None,
            "prompt_version": "grounded-v5-conflict-gated",
            "top_k": retrieval_top_k,
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
