import re
import time

from fastapi import HTTPException

from app.clients import DependencyError, Models, Search
from app.config import settings
from app.fact_integrity import missing_slots, required_slots
from app.models import Answer
from app.rewrite import rewrite_query
from app.retrieval import retrieve_authorized
from app.security import readable_documents, require_chunk
from app.task_analysis import judgment_intent, multi_source_intent


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


def shadow_scores(question, evidence, claims, *, semantic=False):
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
    result = {
        "relevance": round(relevance, 4) if relevance is not None else None,
        "claim_support": supported,
        "scorer": "bigram-overlap-v1",
        "mode": "shadow",
        "action": "recorded only; no answer is withheld on these scores",
        "calibration": "artifacts/a-q1-calibration.json",
    }
    if semantic and claims:
        try:
            from app.semantic_shadow import score_claims

            result["semantic"] = score_claims(evidence, claims)
        except Exception:
            # Observation must never change whether the grounded answer is returned.
            result["semantic"] = {"mode": "shadow", "status": "unavailable"}
    return result


def answer_verdict(models, question, claims, status):
    """Expose the yes/no answer of a judgment question as its own field.

    The system answers "do both reports say X?" in prose, and a reader looking for a
    verdict cannot find one. This adds the verdict without touching the claims: it is
    read out of the validated claims only, it cites the claim it came from, and it is
    `unclear` when the claims do not settle the question. A failure here returns no
    verdict rather than failing an answer that already passed citation validation.
    """
    if not claims or status not in {"answered", "conflict"} or not judgment_intent(question):
        return None, {}
    try:
        verdict, usage = models.decide_verdict(question, claims)
    except DependencyError:
        return None, {}
    from app.verdict import StructuredVerdict, resolve_verdict

    if isinstance(verdict, StructuredVerdict):
        return resolve_verdict(
            verdict, question, claims, constrained=settings().verdict_span_mode == "constrained"
        ), usage
    index = verdict.claim_index if 1 <= verdict.claim_index <= len(claims) else None
    value = verdict.verdict if index or verdict.verdict == "unclear" else "unclear"
    return {
        "value": value,
        "claim_index": index,
        "evidence_ids": claims[index - 1]["evidence_ids"] if index else [],
        "method": "claims_only_classifier",
        "scope": "只读取已通过引用校验的 claims，不接触原文，也不引入新事实",
    }, usage


def merge_verdict_usage(usage, verdict_usage):
    if verdict_usage:
        usage["verdict_prompt_tokens"] = verdict_usage.get("prompt_tokens", 0)
        usage["verdict_completion_tokens"] = verdict_usage.get("completion_tokens", 0)
        for key in ("model_duration_ms", "prompt_eval_ms", "completion_eval_ms"):
            usage[f"verdict_{key}"] = verdict_usage.get(key, 0)
    return usage


def repair_exact_values(models, question, evidence, claims, usage):
    """One focused cited repair when a uniquely labelled exact value was omitted.

    Keep the original claims and accept only a new claim containing the missing value
    and citing its authorized source chunk. A failed optional repair cannot erase a
    previously validated answer.
    """
    missing = missing_slots(required_slots(question, evidence), claims)
    usage["exact_value_slots_missing_first_pass"] = len(missing)
    if not missing or not claims:
        return claims, usage
    focused_ids = {slot["chunk_id"] for slot in missing}
    focused = [row for row in evidence if row["chunk_id"] in focused_ids]
    labels = {
        "event_id": "事件标识符",
        "event_id_field": "事件去重字段",
        "rollback_threshold": "回滚触发阈值",
        "rollback_target": "回滚目标版本",
    }
    fields = "；".join(dict.fromkeys(
        f"{labels.get(slot['field'], slot['field'])}：{slot['value']}" for slot in missing
    ))
    try:
        generated, extra = models.generate(
            f"{question}\n仅补充这些遗漏字段的原文值：{fields}。逐项引用原文。",
            focused,
            acceptance_items_override=[labels.get(slot["field"], slot["field"]) for slot in missing],
            check_conflict=False,
            max_output_tokens=220,
        )
        added, status = validate_claims(generated, focused)
    except DependencyError:
        usage["exact_value_repair_status"] = "unavailable"
        return claims, usage
    for key in ("prompt_tokens", "completion_tokens", "model_duration_ms", "model_load_ms", "prompt_eval_ms", "completion_eval_ms"):
        usage[key] = (usage.get(key) or 0) + (extra.get(key) or 0)
    if status != "answered":
        usage["exact_value_repair_status"] = status
        return claims, usage
    seen = {claim["text"] for claim in claims}
    accepted = []
    for claim in added:
        if claim["text"] in seen:
            continue
        if any(
            slot["chunk_id"] in claim["evidence_ids"]
            and not missing_slots([slot], [claim])
            for slot in missing
        ):
            accepted.append(claim)
            seen.add(claim["text"])
    usage["exact_value_repair_status"] = "added" if accepted else "no_supported_value"
    usage["exact_value_repair_claims"] = len(accepted)
    return claims + accepted, usage


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
    # A question that names several sources needs evidence from several documents, and
    # four chunks are routinely spent inside one of them: the external run found all
    # gold documents for only a third of its multi-hop questions. Such questions get the
    # full budget and a per-document quota. The Chinese path keeps the budget its frozen
    # evaluations were measured with; raising that is a separate decision with its own
    # re-run, not a side effect of this one.
    retrieval_top_k = (
        8 if multi_source_intent(question) else 6 if len(question) <= 30 else cfg.top_k
    )
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
    evidence, candidates, usage, verdict = found.evidence, found.candidates, {}, None
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
            if status == "answered":
                claims, usage = repair_exact_values(models, question, evidence, claims, usage)
            verdict, verdict_usage = answer_verdict(models, question, claims, status)
            usage = merge_verdict_usage(usage, verdict_usage)
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
        "verdict": verdict,
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
            "shadow_scores": (
                shadow_scores(question, evidence, claims, semantic=cfg.semantic_shadow_enabled)
                if evidence else None
            ),
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
