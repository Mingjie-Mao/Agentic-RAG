"""Three-layer, hash-bound benchmark assessment; paths are diagnostics only.

Model review is explicitly allowed by the user's benchmark protocol. It is never
represented as human gold, and missing observations/reviews are never zero scores.
"""
import json
import statistics
from copy import deepcopy

from scripts.benchmark_runtime import percentile
from scripts.research_review import digest, paired_ci

METHODS = ("rag", "workflow", "dynamic", "hybrid")
VERDICTS = {"supported", "partial", "unsupported", "contradicted", "unclear"}
CHECKS = ("no_critical_error", "version_correct", "condition_correct", "citations_supported", "abstention_correct")
ANSWER_REVIEW_PROTOCOL = "enterprise-benchmark-review-v3-ai-assisted"
SCORING_RULES = "benchmarks/enterprise_rag/v1/scoring-rules-v3-ai-assisted.json"
EVALUATION_POLICY = "benchmarks/enterprise_rag/v1/evaluation-policy-v2-ai-assisted.json"
REVIEW_RUBRICS = {
    "enterprise-benchmark-review-v2": "benchmarks/enterprise_rag/v1/scoring-rules-v2.json",
    ANSWER_REVIEW_PROTOCOL: SCORING_RULES,
}


def review_evidence(rows, *, blind_implementation=False):
    if rows is None:
        return None
    projected = [{k: deepcopy(v) for k, v in row.items() if k in {
        'id', 'chunk_id', 'document_id', 'version_id', 'source_sha256', 'text', 'title', 'locator',
        'effective_from', 'is_active', 'authorized', 'active', 'metadata'}} for row in rows]
    for row in projected:
        if blind_implementation and isinstance(row.get('title'), str) and row['title'].endswith(' · 当前版本'):
            row['title'] = row['title'].removesuffix(' · 当前版本')
        if isinstance(row.get('metadata'), dict):
            row['metadata'] = {k: v for k, v in row['metadata'].items() if k in {
                'publication', 'publisher', 'author', 'published_at', 'published_date', 'date',
                'effective_from', 'title', 'url'}}
    return projected


def review_answer(payload, *, blind_message=False):
    """Keep answer content and provenance, excluding implementation metadata."""
    fields = {'status', 'claims', 'verdict', 'citations', 'message', 'facts'}
    answer = {k: deepcopy(v) for k, v in payload.items() if k in fields}
    if isinstance(answer.get('verdict'), dict):
        answer['verdict'] = {k: v for k, v in answer['verdict'].items()
                             if k in {'value', 'claim_index', 'evidence_ids'}}
    answer['claims'] = [{k: v for k, v in claim.items() if k in {'text', 'evidence_ids', 'quotes'}}
                        for claim in answer.get('claims', [])]
    answer['citations'] = review_evidence(answer.get('citations', []), blind_implementation=blind_message)
    # This fixed product diagnostic reveals the method in an otherwise blinded
    # abstention. Preserve the unredacted original in the coordinator/raw record.
    if blind_message and answer.get('message') == 'Agent 找到了资料，但最终答案未通过引用校验。':
        answer['message'] = '找到了资料，但最终答案未通过引用校验。'
    return answer


def conjunction(values):
    """A definite failure dominates; otherwise uncertainty remains pending."""
    values = list(values)
    return False if False in values else None if None in values else True


def semantic_success(verdicts):
    return conjunction(None if v == 'unclear' else v == 'supported' for v in verdicts)


def semantic_ratio(verdicts):
    values = list(verdicts)
    return None if 'unclear' in values else ratio(sum(v == 'supported' for v in values), len(values))


def measured_token_total(usage):
    """Require complete counters; an explicit measured zero is still valid."""
    if any(usage.get('unknown_' + k, 0) for k in ('prompt_tokens', 'completion_tokens')):
        return None
    budget = usage.get('execution_budget', {})
    if budget.get('calls') is not None:
        calls = budget['calls']
        records = budget.get('model_call_records', [])
        if any(any(type(r.get(k)) is not int or r[k] < 0 for k in ('prompt_tokens', 'completion_tokens'))
               for r in records):
            return None
        values = [r.get(k) for r in calls.values() for k in ('prompt_tokens', 'completion_tokens')]
    elif 'total_prompt_tokens' in usage:
        values = [usage.get(k) for k in ('total_prompt_tokens', 'total_completion_tokens')]
    elif 'prompt_tokens' in usage or 'completion_tokens' in usage:
        values = [usage.get(k) for k in ('prompt_tokens', 'completion_tokens')]
    else:
        return None
    if any(type(v) is not int or v < 0 for v in values):
        return None
    return sum(values)


def task_review_packet(suite, spec, root):
    from pathlib import Path
    files = {v["path"]: digest((Path(root) / v["path"]).read_text())
             for d in spec["documents"] for v in d["versions"]}
    items = [{"id": t["id"], "task": t, "required_checks": ["question", "facts", "evidence", "scope", "fairness"]}
             for t in suite["tasks"]]
    return {"protocol": "enterprise-benchmark-review-v1", "kind": "task", "items": items,
            "source_sha256": digest(items), "corpus_sha256": digest({"spec": spec, "files": files}),
            "corpus_spec": spec,
            "reviewer_policy": "named GPT or human, explicit provenance; not necessarily independent", "reviews": []}


def validate_reviews(packet, reviewed):
    if any(reviewed.get(k) != packet.get(k) for k in ("protocol", "kind", "source_sha256", "corpus_sha256")):
        raise ValueError("review input/corpus changed")
    if reviewed.get("items") != packet["items"] or digest(packet["items"]) != packet["source_sha256"]:
        raise ValueError("review items changed")
    if reviewed.get("corpus_spec") != packet.get("corpus_spec"):
        raise ValueError("review corpus permissions/spec changed")
    if packet['protocol'] in REVIEW_RUBRICS and any(
            reviewed.get(k) != packet.get(k) for k in ('rubric_version', 'rubric_sha256')):
        raise ValueError('review scoring rules changed')
    rows = reviewed.get("reviews", [])
    if len({r.get("id") for r in rows}) != len(rows):
        raise ValueError("duplicate review IDs")
    if {r.get("id") for r in rows} != {r["id"] for r in packet["items"]}:
        raise ValueError("review incomplete")
    for r in rows:
        if r.get("reviewer_type") not in {"model", "human"} or any(
                not isinstance(r.get(k), str) or not r[k].strip() for k in ('reviewer', 'reason')):
            raise ValueError("named reviewer, type and reason required")
        if not isinstance(r.get("independent"), bool):
            raise ValueError("review independence must be explicitly recorded")
        if r["reviewer_type"] == "model" and (not isinstance(r.get('model'), str) or not r['model'].strip()):
            raise ValueError("GPT review model identity required")
    return {r["id"]: r for r in rows}


def require_task_review(suite, spec, root):
    from pathlib import Path
    packet = task_review_packet(suite, spec, root)
    path = Path(root) / suite["reviewed_packet"]
    if not path.exists():
        raise ValueError("task annotation review pending (GPT review accepted)")
    rows = validate_reviews(packet, json.loads(path.read_text()))
    for item in packet["items"]:
        row = rows[item["id"]]
        if row.get("label") != "approved" or any(row.get("checks", {}).get(k) is not True for k in item["required_checks"]):
            raise ValueError("task facts/evidence/scope/fairness review not approved")


def provisional_score(task, payload, trace, steps, latency_ms, **kwargs):
    from scripts.unseen_v2_scoring import provisional_score as old_score
    final_scope_only = task.get("security_kind") == "version_scope"
    scored_task = {**task, "forbidden": []} if final_scope_only else task
    row = old_score(scored_task, payload, trace, steps, latency_ms, **kwargs)
    if final_scope_only:
        from scripts.run_agent_hard_benchmark import fact_matches, normalized
        text = normalized(" ".join(c.get("text", "") for c in payload.get("claims", [])))
        row["forbidden_present"] = {v: fact_matches(v, text) for v in task["forbidden"]}
        row["version_scope_violation"] = any(row["forbidden_present"].values())
    # Literal matches and historical path requirements cannot qualify a strict pass.
    row["literal_answer_diagnostic"] = row.pop("answer_correct")
    row["legacy_policy_diagnostic"] = row.pop("task_success")
    row.update(answer_correct=None, task_success=None, strict_task_success=None,
               quality_basis="pending_GPT_fact_citation_scope_review")
    return row


def answer_review_packet(raw, suite, *, protocol=ANSWER_REVIEW_PROTOCOL):
    if protocol not in {'enterprise-benchmark-review-v1', *REVIEW_RUBRICS}:
        raise ValueError('unknown answer review protocol')
    tasks = {t["id"]: t for t in suite["tasks"]}
    items = []
    for method, rows in raw["results"].items():
        for row in rows:
            if row.get("skipped"):
                continue
            # Opaque identity; do not expose method names or the policy trace to
            # answer graders. Retrieval review sees actual evidence, not gold alone.
            public_task = {k: v for k, v in tasks[row["task_id"]].items() if k not in
                           {"expected_behavior", "required_tools", "conditional_transition", "max_steps"}}
            # RAG trace/usage and Agent accounting identify the method even when
            # its name is removed. Keep only the actual user-facing answer.
            public_answer = {k: v for k, v in row["answer_payload"].items() if k in
                             {"status", "claims", "verdict", "citations", "message", "facts"}}
            if protocol in REVIEW_RUBRICS:
                public_answer = review_answer(row['answer_payload'], blind_message=protocol == ANSWER_REVIEW_PROTOCOL)
                public_task = {k: deepcopy(v) for k, v in public_task.items() if k in {
                    'id', 'question', 'goal', 'category', 'required_facts', 'gold_answer', 'gold_documents',
                    'gold_evidence', 'answerable', 'expected_status', 'temporal_scope', 'requires_version',
                    'annotation_status', 'forbidden', 'security_kind', 'state_change', 'scenario_events'}}
            items.append({"id": digest([method, row["task_id"], row.get("run_id")]),
                          "task": public_task, "answer": public_answer,
                          "retrieved_evidence": review_evidence(row.get('retrieved_evidence'),
                              blind_implementation=protocol == ANSWER_REVIEW_PROTOCOL)
                          if protocol in REVIEW_RUBRICS else row.get('retrieved_evidence'),
                          "required_checks": list(CHECKS)})
    items.sort(key=lambda r: r["id"])
    packet = {"protocol": protocol, "kind": "answer", "items": items,
            "source_sha256": digest(items), "corpus_sha256": raw["corpus_sha256"], "reviews": [],
            "reviewer_policy": "blinded GPT factual assessment, explicit model provenance"}
    if protocol in REVIEW_RUBRICS:
        from pathlib import Path
        import hashlib
        rubric = Path(__file__).resolve().parents[1] / REVIEW_RUBRICS[protocol]
        packet.update(rubric_version=json.loads(rubric.read_text())['version'],
                      rubric_sha256=hashlib.sha256(rubric.read_bytes()).hexdigest())
    return packet


def ratio(n, d):
    return n / d if d else None


def assess(task, row, review=None):
    docs = set(task["gold_documents"])
    evidence = row.get("retrieved_evidence")
    retrieved = list(dict.fromkeys(p["document_id"] for p in evidence or []))
    first = row.get("first_retrieval_documents")
    retrieval = {"gold_documents": len(docs),
                 "gold_documents_retrieved": len(docs & set(retrieved)) if evidence is not None else None,
                 "gold_document_recall": ratio(len(docs & set(retrieved)), len(docs)) if evidence is not None else None,
                 "recall_at_k": {str(k): ratio(len(docs & set(first[:k])), len(docs)) if first is not None else None
                                 for k in (1, 5, 10)},
                 "gold_fact_recall": None, "evidence_coverage": None}
    policy = {k: row.get(k) for k in ("steps", "latency_ms", "usage", "execution_error_kind", "scenario_events")}
    policy.update({k: None for k in ("unnecessary_tool_calls", "early_stop", "late_stop", "recovery_after_failed_retrieval")})
    trace = row.get("tool_trace", [])
    calls = [digest([r.get("tool"), r.get("arguments")]) for r in trace]
    policy["repeated_tool_calls"] = len(calls) - len(set(calls)) if row.get("trace_complete") else None
    policy["failed_tool_calls"] = sum(r.get("status") in {"error", "failed", "denied"} for r in trace) if row.get("trace_complete") else None
    policy["invalid_tool_calls"] = sum(r.get("status") == "invalid" or
        r.get("error", (r.get("result") or {}).get("error_code")) in {"unknown_tool", "invalid_arguments"}
        for r in trace) if row.get("trace_complete") else None
    queries = [digest(r.get("arguments", {}).get("query", "")) for r in trace if r.get("tool") == "search_documents"]
    policy["repeated_queries"] = len(queries) - len(set(queries)) if row.get("trace_complete") else None
    query_failures = [True if r.get("status") in {"error", "failed", "denied"} else
                      (count == 0 if (count := r.get("evidence_count", (r.get("result") or {}).get("evidence_count"))) is not None else None)
                      for r in trace if r.get("tool") == "search_documents"]
    policy["failed_queries"] = sum(query_failures) if row.get("trace_complete") and all(v is not None for v in query_failures) else None
    policy["tool_calls"] = len(trace) if row.get("trace_complete") else None
    policy["expected_behavior"] = task["expected_behavior"]
    answer = {"strict_task_success": None, "fact_recall": None, "fact_precision": None,
              "citation_correctness": None, "faithfulness": None, "answer_completeness": None}
    if review is None:
        return {"retrieval": retrieval, "policy": policy, "answer": answer}
    required = {f["id"] for f in task["required_facts"]}
    verdicts = review.get("fact_verdicts", {})
    if set(verdicts) != required or any(v not in VERDICTS for v in verdicts.values()):
        raise ValueError("every required fact needs a semantic verdict")
    if any(k not in review or (review[k] is not None and type(review[k]) is not bool) for k in CHECKS):
        raise ValueError("all strict checks need explicit boolean or pending decisions")
    cited = review.get("citation_verdicts", [])
    claims = review.get("claim_verdicts", [])
    payload = row["answer_payload"]
    if len(cited) != len(payload.get("citations", [])) or any(v not in VERDICTS for v in cited):
        raise ValueError("every actual citation needs a support verdict")
    if len(claims) != len(payload.get("claims", [])) or any(v not in VERDICTS for v in claims):
        raise ValueError("every actual claim needs a factual/support verdict")
    status = payload.get("status")
    expected = task["expected_status"]
    status_ok = status in (expected if isinstance(expected, list) else [expected])
    execution_ok = row.get("execution_error_kind") is None or (
        task.get("state_change") and row.get("execution_error_kind") == "authorization_change")
    exercised = all(row.get("scenario_events", {}).get(e) is True for e in task.get("scenario_events", []))
    safe = not any(row.get("forbidden_present", {}).values())
    # Forbidden branch mention in retrieved input is legal; final condition review
    # determines whether a branch was incorrectly asserted.
    if task["category"] == "conditional_planning":
        safe = True
    complete = semantic_success(verdicts.values())
    # Reuse the production binding checks in addition to GPT's semantic review.
    # A correct value elsewhere in context cannot support a wrongly bound quote.
    binding_ok = True
    if payload.get('claims'):
        from app.clients import GeneratedAnswer
        from app.qa import claim_validation_issues, validate_claims
        try:
            generated = GeneratedAnswer(answerable=True, claims=payload['claims'])
            sources = [dict(c, id=c['chunk_id']) for c in payload.get('citations', [])]
            if review.get('scoring_protocol') == ANSWER_REVIEW_PROTOCOL:
                # The current production guard also applies evolving number/date
                # rules. Those are semantic audit dimensions, not forged binding.
                # Reuse its ID/arity/literal checks; never replay a newer semantic
                # guard as an unrecorded veto on a frozen historical answer.
                mechanical = {'empty_answer', 'citation_arity', 'unknown_evidence', 'nonliteral_quote'}
                binding_ok = not any(mechanical.intersection(issue['codes'])
                                     for issue in claim_validation_issues(generated, sources))
            else:
                binding_ok = validate_claims(generated, sources)[1] == 'answered'
        except (ValueError, KeyError, TypeError):
            binding_ok = False
    answer['citation_binding_valid'] = binding_ok
    success = conjunction([complete, *[review[k] for k in CHECKS], status_ok, execution_ok, exercised, safe,
                           semantic_success(cited), semantic_success(claims), binding_ok,
                           bool(cited) or not task['answerable']])
    atomic = review.get("answer_facts")
    precision = None
    if atomic is not None:
        if review.get("atomic_facts_complete") is not True or not isinstance(atomic, list):
            raise ValueError("atomic answer facts must be exhaustively reviewed")
        if any(not r.get("text") or r.get("verdict") not in VERDICTS or type(r.get("claim_index")) is not int
               or not 0 <= r["claim_index"] < len(claims) for r in atomic):
            raise ValueError("atomic fact must bind to a real claim")
        if {r["claim_index"] for r in atomic} != set(range(len(claims))):
            raise ValueError("atomic answer facts must cover every claim")
        precision = semantic_ratio(r['verdict'] for r in atomic)
        success = conjunction([success, semantic_success(r['verdict'] for r in atomic)])
    answer.update(strict_task_success=success, fact_recall=semantic_ratio(verdicts.values()),
                  fact_precision=precision,
                  citation_correctness=semantic_ratio(cited),
                  faithfulness=semantic_ratio(claims),
                  answer_completeness=semantic_ratio(verdicts.values()))
    answer["strict_checks"] = {k: review[k] for k in CHECKS}
    policy["task_success"] = success
    retrieved_facts = review.get("retrieved_fact_verdicts")
    if retrieved_facts is not None:
        if evidence is None or set(retrieved_facts) != required or any(v not in VERDICTS for v in retrieved_facts.values()):
            raise ValueError("retrieved fact assessment must bind every fact to observed evidence")
        retrieval["gold_fact_recall"] = semantic_ratio(retrieved_facts.values())
        retrieval["evidence_coverage"] = retrieval["gold_fact_recall"]
    for k in ("unnecessary_tool_calls", "early_stop", "late_stop", "recovery_after_failed_retrieval"):
        value = review.get("policy", {}).get(k)
        if value is not None and ((k == "unnecessary_tool_calls" and (type(value) is not int or value < 0))
                                  or (k != "unnecessary_tool_calls" and type(value) is not bool)):
            raise ValueError("invalid policy diagnostic")
        policy[k] = value
    return {"retrieval": retrieval, "policy": policy, "answer": answer}


def report(raw, suite, reviewed=None, *, baseline_method="workflow", pending_task_ids=()):
    tasks = {t["id"]: t for t in suite["tasks"]}
    if set(raw["results"]) != set(suite["arms"]):
        raise ValueError("all registered methods required")
    pending = set(pending_task_ids)
    if pending and suite["split"] != "dev":
        raise ValueError("pending annotation overrides are Dev-only")
    if not pending <= tasks.keys() or baseline_method not in raw["results"]:
        raise ValueError("pending task or baseline comparator is not registered")
    reviews = validate_reviews(answer_review_packet(raw, suite, protocol=reviewed['protocol']), reviewed) if reviewed else {}
    by_method, details = {}, {}
    for method, rows in raw["results"].items():
        if len(rows) != len(tasks) or {r["task_id"] for r in rows} != set(tasks) or any(r.get("skipped") for r in rows):
            raise ValueError("same complete task denominator required; no skipped tasks")
        details[method] = {}
        for row in rows:
            rid = digest([method, row["task_id"], row.get("run_id")])
            review = reviews.get(rid)
            if review is not None:
                review = {**review, 'scoring_protocol': reviewed['protocol']}
            details[method][row["task_id"]] = assess(tasks[row["task_id"]], row, review)
            if row["task_id"] in pending or row.get('not_observed'):
                result = details[method][row["task_id"]]
                result["annotation_status"] = "pending" if row["task_id"] in pending else "not_observed"
                for key in ("strict_task_success", "fact_recall", "fact_precision",
                            "citation_correctness", "faithfulness", "answer_completeness"):
                    result["answer"][key] = None
                result["policy"]["task_success"] = None
                result['retrieval']['gold_fact_recall'] = None
                result['retrieval']['evidence_coverage'] = None
        scored = list(details[method].values())
        def mean(layer, field):
            observed = [r[layer][field] for r in scored if r[layer].get(field) is not None]
            return {"value": statistics.mean(observed) if observed else None, "applicable": len(observed), "total": len(rows)}
        latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        measured_tokens = [v for r in rows if (v := measured_token_total(r.get('usage') or {})) is not None]
        by_method[method] = {"tasks": len(rows), "strict_task_success_rate": mean("answer", "strict_task_success"),
                             "retrieval": {k: mean("retrieval", k) for k in ("gold_document_recall", "gold_fact_recall", "evidence_coverage")},
                             "answer": {k: mean("answer", k) for k in ("fact_recall", "fact_precision", "citation_correctness", "faithfulness", "answer_completeness")},
                             "policy": {k: mean("policy", k) for k in ("steps", "tool_calls", "unnecessary_tool_calls", "failed_tool_calls", "invalid_tool_calls", "repeated_tool_calls", "failed_queries", "repeated_queries", "early_stop", "late_stop", "recovery_after_failed_retrieval")},
                             "total_tokens": {"sum": sum(measured_tokens) if len(measured_tokens) == len(rows) and rows else None,
                                              "observed_sum": sum(measured_tokens) if measured_tokens else None,
                                              "mean": statistics.mean(measured_tokens) if measured_tokens else None,
                                              "observed": len(measured_tokens), "total": len(rows)},
                             "latency_ms": {"p50": statistics.median(latencies) if latencies else None,
                                            "p95": percentile(latencies, .95), "observed": len(latencies), "total": len(rows)},
                             "execution_failures": sum(r.get("status") == "execution_failed" for r in rows),
                             "forbidden_output_tasks": sum(any(r.get("forbidden_present", {}).values()) for r in rows
                                                           if tasks[r["task_id"]]["category"] != "conditional_planning"),
                             "acl_canary_output_tasks": sum(any(r.get("forbidden_present", {}).values()) for r in rows
                                                           if tasks[r["task_id"]].get("forbidden_kind") == "acl_content_leak"),
                             "version_scope_violations": sum(r.get("version_scope_violation") is True for r in rows),
                             "unexercised_security_events": sum(any(r.get("scenario_events", {}).get(e) is not True for e in
                                                           tasks[r["task_id"]].get("scenario_events", [])) for r in rows)}
        by_method[method]["by_category"] = {}
        by_method[method]["retrieval"]["recall_at_k"] = {}
        for k in ("1", "5", "10"):
            observed = [r["retrieval"]["recall_at_k"][k] for r in scored if r["retrieval"]["recall_at_k"][k] is not None]
            by_method[method]["retrieval"]["recall_at_k"][k] = {"value": statistics.mean(observed) if observed else None,
                                                              "applicable": len(observed), "total": len(rows)}
        for category in sorted({t["category"] for t in tasks.values()}):
            selected = [r["answer"]["strict_task_success"] for tid, r in details[method].items() if tasks[tid]["category"] == category]
            by_method[method]["by_category"][category] = {"n": len(selected), "strict_task_success_rate":
                statistics.mean(selected) if all(v is not None for v in selected) else None}
    paired = {}
    if reviewed:
        baseline = details[baseline_method]
        for method in suite["arms"]:
            if method == baseline_method:
                continue
            observed = [tid for tid in sorted(tasks) if
                        details[method][tid]["answer"]["strict_task_success"] is not None
                        and baseline[tid]["answer"]["strict_task_success"] is not None]
            delta = [int(details[method][tid]["answer"]["strict_task_success"]) - int(baseline[tid]["answer"]["strict_task_success"]) for tid in observed]
            groups = [tasks[tid].get("family_id", tid) for tid in observed]
            paired[method] = {"baseline": baseline_method, "n": len(delta), "total": len(tasks),
                              "pending_pairs": len(tasks) - len(delta), "independent_groups": len(set(groups)),
                              "difference": statistics.mean(delta) if delta else None,
                              "ci95": paired_ci(delta, groups=groups) if delta else None,
                              "bootstrap_unit": "task_family"}
    registration = raw.get("registration", {})
    frozen_verified = False
    frozen_rules_verified = False
    if registration.get("suite_sha256") == digest(suite) and raw.get("freeze"):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        freeze_path = root / raw["freeze"]
        if freeze_path.exists():
            frozen = json.loads(freeze_path.read_text())
            frozen_verified = digest(frozen) == registration.get("freeze_sha256")
            frozen_rules_verified = bool(reviewed) and (
                frozen.get('inputs', {}).get(SCORING_RULES) == reviewed.get('rubric_sha256')
                and digest(frozen.get('indexed_corpus')) == raw.get('corpus_sha256'))
    complete_review = all(d['answer']['strict_task_success'] is not None for rows in details.values() for d in rows.values())
    review_protocol_current = bool(reviewed) and reviewed['protocol'] == ANSWER_REVIEW_PROTOCOL
    reviewer_policy_met = bool(reviews) and all(r['reviewer_type'] == 'human' or
        (r['reviewer_type'] == 'model' and (review_protocol_current or
         str(r.get('model', '')).lower().startswith('gpt-')))
        for r in reviews.values())
    full_core = suite.get('evaluation_protocol') == 'enterprise-benchmark-v1' and len(tasks) == 70
    return {"package": suite["version"], "split": suite["split"], "headline_eligible": suite["split"] == "core" and full_core and bool(reviewed) and frozen_verified and frozen_rules_verified and complete_review and review_protocol_current and reviewer_policy_met,
            "scoring_rules_version": reviewed.get('rubric_version') if reviewed else None,
            "frozen_scoring_rules_verified": frozen_rules_verified,
            "current_review_protocol": review_protocol_current, "headline_reviewer_policy_met": reviewer_policy_met,
            "frozen_registration_verified": frozen_verified,
            "status": ("AI_assisted_reviewed" if review_protocol_current else
                       "GPT_or_human_reviewed" if reviewer_policy_met else "model_reference_reviewed")
                      if reviewed else "provisional_no_accuracy_claim", "summary": by_method,
            "paired": paired, "details": details, "raw_sha256": digest(raw),
            "review_provenance": [{k: r.get(k) for k in ("reviewer", "reviewer_type", "model", "independent")} for r in reviews.values()]}
