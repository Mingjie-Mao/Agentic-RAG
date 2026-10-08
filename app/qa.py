import re
import time

from fastapi import HTTPException

from app.clients import DependencyError, Models, Search
from app.config import conflict_check_enabled, settings
from app.answer_contract import claim_scope_issues, quantitative_support_issues
from app.fact_integrity import missing_slots, required_slots
from app.models import Answer
from app.rewrite import rewrite_query
from app.retrieval import retrieve_authorized
from app.security import readable_documents, require_chunk
from app.task_analysis import judgment_intent, multi_source_intent


def normalize_quote(text):
    return re.sub(r"\s+", "", text)


# Embedded-post placeholders and consent prompts are page furniture, not reporting.
_BOILERPLATE = re.compile(
    r"[^.!?。！？]*(?:cannot be displayed in your browser|enable javascript|view original content on|"
    r"not responsible for the content of external sites|skip \w+ post|end of \w+ post|allow \w+ content\?|"
    r"contains content provided by \w+|ask for your permission before anything is loaded|may be using cookies|"
    r"cookie policy, external|choose .accept and continue|本(?:网)?站使用\s*cookie)[^.!?。！？]*[.!?。！？]?",
    re.I,
)


def boilerplate_quote(quote):
    """True when nothing substantive remains after removing embed/consent sentences."""
    rest = _BOILERPLATE.sub("", quote)
    return rest != quote and len(re.sub(r"[\W_]+", "", rest)) < 15


def furniture_reference(claim_text, quote, source):
    """Code for a literal quote that is page furniture rather than support for this claim."""
    if boilerplate_quote(quote):
        return "boilerplate_reference"
    from app.task_contract import _VALUE

    # Bare section labels do not support a numeric fact merely by residing in a
    # related document. Entity/attribute headings that occur in the claim, and
    # factual headlines, are still allowed.
    normalized = normalize_quote(quote)
    headings = {
        normalize_quote(re.sub(r"^#{1,6}\s+", "", line.strip()))
        for line in source["text"].splitlines()
        if re.match(r"^#{1,6}\s+", line.strip())
    }
    if (
        _VALUE.search(claim_text)
        and normalized in headings
        and not _VALUE.search(quote)
        and normalized not in normalize_quote(claim_text)
    ):
        return "unrelated_heading_reference"
    return None


def claim_validation_issues(generated, evidence):
    """Return claim-local failure codes; never expose an unvalidated draft as an answer."""
    by_id = {item["id"]: item for item in evidence}
    if not generated.answerable:
        return []
    if not generated.claims:
        return [{"claim_index": None, "codes": ["empty_answer"]}]
    issues = []
    for index, claim in enumerate(generated.claims):
        codes = []
        if len(claim.evidence_ids) != len(claim.quotes):
            codes.append("citation_arity")
        for evidence_id, quote in zip(claim.evidence_ids, claim.quotes):
            source = by_id.get(evidence_id)
            normalized = normalize_quote(quote)
            if source is None:
                codes.append("unknown_evidence")
            elif len(normalized) < 4 or normalized not in normalize_quote(source["text"]):
                codes.append("nonliteral_quote")
            elif settings().answer_quality_enabled and (code := furniture_reference(claim.text, quote, source)):
                codes.append(code)
        if settings().answer_contract_enabled and claim_scope_issues(claim.text, claim.quotes):
            codes.extend(claim_scope_issues(claim.text, claim.quotes))
        if settings().answer_quality_enabled:
            from app.answer_contract import publication_binding_issues
            codes.extend(publication_binding_issues(claim.text, claim.evidence_ids, evidence, claim.quotes))
        codes.extend(quantitative_support_issues(claim.text, claim.quotes,
                source_titles=[by_id[key].get('title', '') for key in claim.evidence_ids if key in by_id]))
        if codes:
            issues.append({"claim_index": index, "codes": sorted(set(codes))})
    return issues


def validate_claims(generated, evidence):
    by_id = {item["id"]: item for item in evidence}
    if not generated.answerable:
        return [], "insufficient_evidence"
    if claim_validation_issues(generated, evidence):
        return [], "verification_failed"
    checked = []
    for claim in generated.claims:
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


def answer_verdict(models, question, claims, status, *, facts=None, verified_verdict=None, evidence=None):
    """Expose the yes/no answer of a judgment question as its own field.

    The system answers "do both reports say X?" in prose, and a reader looking for a
    verdict cannot find one. This adds the verdict without touching the claims: it is
    read out of the validated claims only, it cites the claim it came from, and it is
    `unclear` when the claims do not settle the question. A failure here returns no
    verdict rather than failing an answer that already passed citation validation.
    """
    if not claims or status not in {"answered", "conflict"} or not judgment_intent(question):
        return None, {}
    if verified_verdict is not None:
        return verified_verdict, {}
    if facts is not None:
        from app.source_facts import fact_verdict
        return fact_verdict(question, facts, claims), {}
    verdict_claims = claims
    if evidence:
        by_chunk = {row["chunk_id"]: row for row in evidence}
        verdict_claims = [
            {**claim, "sources": [
                {"publication": (by_chunk[ref].get("metadata") or {}).get("source"),
                 "title": by_chunk[ref].get("title")}
                for ref in dict.fromkeys(claim["evidence_ids"]) if ref in by_chunk
            ]}
            for claim in claims
        ]
    try:
        cfg = settings()
        proof_groups = []
        if (cfg.answer_quality_enabled and evidence and cfg.verdict_protocol == "structured"
                and cfg.verdict_span_mode == "constrained"):
            from app.answer_contract import verdict_proof_groups
            proof_groups = verdict_proof_groups(question, evidence, claims)
        verdict, usage = models.decide_verdict(
            question, verdict_claims, **({"proof_groups": proof_groups} if proof_groups else {})
        )
    except DependencyError as exc:
        return None, {
            **exc.usage,
            "verdict_status": "unavailable",
            "verdict_failure_stage": exc.stage,
            "verdict_error_kind": type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__,
        }
    from app.verdict import StructuredVerdict, resolve_verdict

    if isinstance(verdict, StructuredVerdict):
        result = resolve_verdict(
            verdict, question, claims, constrained=settings().verdict_span_mode == "constrained"
        )
        if settings().claim_consistency_enabled:
            from app.claim_consistency import reconcile_verdict
            result = reconcile_verdict(question, claims, result)
        if settings().answer_quality_enabled and evidence:
            from app.answer_contract import bind_verdict_source_groups
            result = bind_verdict_source_groups(question, result, evidence, claims)
        return result, usage
    index = verdict.claim_index if 1 <= verdict.claim_index <= len(claims) else None
    value = verdict.verdict if index or verdict.verdict == "unclear" else "unclear"
    result = {
        "value": value,
        "claim_index": index,
        "evidence_ids": claims[index - 1]["evidence_ids"] if index else [],
        "method": "claims_only_classifier",
        "scope": "只读取已通过引用校验的 claims，不接触原文，也不引入新事实",
    }
    if settings().claim_consistency_enabled:
        from app.claim_consistency import reconcile_verdict
        result = reconcile_verdict(question, claims, result)
    if settings().answer_quality_enabled and evidence:
        from app.answer_contract import bind_verdict_source_groups
        result = bind_verdict_source_groups(question, result, evidence, claims)
    return result, usage


def merge_verdict_usage(usage, verdict_usage):
    if verdict_usage:
        for key in ("prompt_tokens", "completion_tokens"):
            if key in verdict_usage:
                usage[f"verdict_{key}"] = verdict_usage[key]
        for key in ("model_duration_ms", "prompt_eval_ms", "completion_eval_ms"):
            if key in verdict_usage:
                usage[f"verdict_{key}"] = verdict_usage[key]
        for key in ("verdict_status", "verdict_failure_stage", "verdict_error_kind"):
            if key in verdict_usage:
                usage[key] = verdict_usage[key]
    return usage


def repair_exact_values(models, question, evidence, claims, usage, *, facts=None):
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
        repair_query=f"{question}\n仅补充这些遗漏字段的原文值：{fields}。逐项引用原文。"
        options={'acceptance_items_override':[labels.get(slot['field'],slot['field']) for slot in missing],
                 'check_conflict':False,'max_output_tokens':220}
        if settings().answer_quality_enabled:
            from app.answer_quality import recover_answer
            generated,extra=recover_answer(models,repair_query,focused,options=options)
        else:
            generated,extra=models.generate(repair_query,focused,**options)
        added, status = validate_claims(generated, focused)
        if extra.get('quality_validation_failed'):
            added,status=[],'verification_failed'
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
    if facts is not None:
        from app.source_facts import public_facts
        facts.extend(public_facts(generated.facts, accepted))
    return claims + accepted, usage


MESSAGES = {
    "verification_failed": "模型的引用未通过检查，本次未返回答案。请调整问题后重试。",
    "no_readable_documents": "你当前没有可访问的资料。上传资料，或请管理者把已有资料分享到你所在的组。",
    "documents_processing": "你的资料还在处理中，全部处理完成后才能检索。可以在资料库查看进度。",
}


def answer_question(db, user, question, history=None, *, benchmark_run_token=None):
    from app.execution_budget import ExecutionBudget, current_budget, use_budget
    if current_budget():
        return _answer_question(db, user, question, history, benchmark_run_token=benchmark_run_token)
    budget = ExecutionBudget(settings())
    try:
        with use_budget(budget):
            result = _answer_question(db, user, question, history, benchmark_run_token=benchmark_run_token)
    except Exception as exc:
        exc.execution_budget = budget.snapshot()
        raise
    if settings().semantic_slot_shadow_enabled:
        from app.slot_shadow import schedule_completed
        schedule_completed("answer", result["id"], user.id)
    return result


def _answer_question(db, user, question, history=None, *, benchmark_run_token=None):
    started = time.monotonic()
    cfg = settings()
    # Rewriting produces a retrieval query and nothing else: it is not stored, never
    # crosses a session, and never becomes evidence.
    query, rewrite_trace = rewrite_query(question, history, mode=cfg.rewrite_mode)
    models = Models()
    contract, contract_trace, allow_historical = None, None, False
    if cfg.task_contract_enabled:
        from app.task_contract import build_contract
        contract = build_contract(question)
        if contract.intent == "conditional":
            query = contract.slots[0].query
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
    if contract and contract.intent != "ordinary":
        from agent.controller import _evidence_from_refs
        from agent.tools import KnowledgeTools
        from app.task_contract_runtime import prepare_contract
        tools = KnowledgeTools(db, user, models=models)
        calls = 0
        def call(name, arguments):
            nonlocal calls
            if calls >= 5:
                return None
            calls += 1
            return tools.call(name, arguments)
        evidence, contract_trace = prepare_contract(contract, evidence,
            search=lambda query: call("search_documents", {"query": query, "top_k": 8}),
            load=lambda refs, historical: _evidence_from_refs(db, user, refs, allow_historical=historical, limit=None),
            versions=lambda doc: call("get_document_version", {"document_id": doc, "limit": 20}),
            compare=lambda doc, old, new: call("compare_versions", {"document_id": doc, "from_version_id": old, "to_version_id": new}),
            open_version=lambda doc, vid: call("open_document", {"document_id": doc, "version_id": vid, "limit": 8}))
        allow_historical = contract.intent == "history" and contract.version_chain_complete
        known = {row["chunk_id"] for row in candidates}
        for row in evidence:
            if row["chunk_id"] not in known:
                candidates.append({"chunk_id": row["chunk_id"], "document_id": row["document_id"],
                                   "title": row["title"], "admitted": True, "lane": "task_contract"})
    elif contract:
        # Ordinary first-search recovery is shared with the established planner.
        from agent.planner import recovery_query, retrieval_quality
        quality = retrieval_quality(query, [r["chunk_id"] for r in evidence],
                                    [{"snippet": r["text"], "title": r["title"]} for r in evidence])
        retry = recovery_query(query, query) if quality in {"empty", "irrelevant"} else None
        if retry and retry != query and found.searchable_documents:
            recovered = retrieve_authorized(db, user, retry, cfg=cfg, models=models, search=Search(),
                readable_documents_fn=readable_documents, require_chunk_fn=require_chunk, top_k=retrieval_top_k)
            evidence = recovered.evidence
            candidates += [r for r in recovered.candidates if r["chunk_id"] not in {c["chunk_id"] for c in candidates}]
            contract_trace = {"recovery_attempted": True, "query": retry}
    passage_trace = None
    if cfg.passage_window_enabled and evidence and not (contract and contract.intent != "ordinary"):
        from app.passages import prepare_passages
        evidence, passage_trace = prepare_passages(db, user, question, evidence, cfg)
        known = {row['chunk_id'] for row in candidates}
        for row in evidence:
            if row['chunk_id'] not in known:
                candidates.append({'chunk_id': row['chunk_id'], 'document_id': row['document_id'],
                                   'title': row['title'], 'admitted': True, 'lane': 'sentence_window'})
    facts = []
    scope_trace=None
    if cfg.answer_quality_enabled and evidence and not allow_historical:
        from app.evidence_scope import attach_scope
        evidence, scope_trace = attach_scope(db,user,question,evidence)
    embed_ms, retrieval_ms, generation_ms = found.embed_ms, found.retrieval_ms, 0
    context_tokens = passage_trace['context_tokens'] if passage_trace else found.context_tokens
    if found.searchable_documents:
        if evidence:
            t = time.monotonic()
            # Only a tenant with the check switched off changes the call at all.
            options = {} if conflict_check_enabled(user.tenant_id) else {"check_conflict": False}
            if contract:
                from app.task_contract import contract_generate
                generated, usage = contract_generate(models, question, evidence, contract=contract, options=options)
                usage["contract_preparation"] = contract_trace
            else:
                if cfg.answer_quality_enabled:
                    from app.answer_quality import recover_answer
                    generated, usage = recover_answer(models, question, evidence, options=options)
                else:
                    generated, usage = models.generate(question, evidence, **options)
            if passage_trace is not None:
                usage['passage_selection'] = passage_trace
            if scope_trace is not None:
                usage['scope_preparation']=scope_trace
            generation_ms = (time.monotonic() - t) * 1000
            usage['generation_context_chunk_ids']=[item['chunk_id'] for item in evidence]
            claims, status = validate_claims(generated, evidence)
            if usage.get('quality_validation_failed'):
                claims,status=[],'verification_failed'
            facts.extend(generated.facts)
            if status == "answered" and usage.get("answer_status") == "conflict":
                status = "conflict"
            if status == "answered" and not (contract and contract.intent != "ordinary"):
                claims, usage = repair_exact_values(models, question, evidence, claims, usage,
                                                    facts=facts if cfg.source_facts_enabled else None)
            from app.source_facts import public_facts
            facts = public_facts(facts, claims)
            if contract and contract.intent != "ordinary":
                verdict, verdict_usage = None, {}
            else:
                verdict, verdict_usage = answer_verdict(models, question, claims, status,
                                                        evidence=evidence,
                                                        verified_verdict=usage.get('verified_verdict') or usage.get('evidence_scope',{}).get('verdict'),
                                                        facts=facts if cfg.source_facts_enabled
                                                        and not usage.get('extraction_fallback') else None)
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
        require_chunk(db, user, item["chunk_id"], active_only=not (
            allow_historical or (contract and contract.intent == 'history')))
    candidate_ids={item['chunk_id'] for item in candidates}
    for item in evidence:
        if item['chunk_id'] not in candidate_ids:
            require_chunk(db,user,item['chunk_id'],active_only=not (
                allow_historical or (contract and contract.intent=='history')))
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
        **({"facts": facts} if cfg.source_facts_enabled else {}),
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
            "generation_context_chunk_ids": [item["chunk_id"] for item in evidence],
            "candidates": candidates,
            "embed_ms": round(embed_ms, 1),
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "total_ms": round((time.monotonic() - started) * 1000, 1),
        },
        "usage": usage,
    }
    from app.execution_budget import current_budget
    if current_budget():
        current_budget().check()
        payload["usage"]["execution_budget"] = current_budget().snapshot()
    if benchmark_run_token:
        payload["trace"]["benchmark_run_token"] = benchmark_run_token
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
