import pytest

from app.semantic_evidence import SlotEvidenceEvaluator, Thresholds, aggregate_slots


P = [{"chunk_id": "a", "text": "甲服务的配额为 20 次。", "title": "甲", "version_id": "v1", "document_id": "d1"},
     {"chunk_id": "b", "text": "乙服务的配额为 90 次。", "title": "乙", "version_id": "v2", "document_id": "d2"}]
ITEMS = [{"id": "s1", "text": "甲服务配额"}]


def make(rows=None, scores=None, **kwargs):
    calls = []
    def judge(payload):
        calls.append(payload)
        return {"output": {"judgments": rows or []}, "prompt_tokens": 13, "completion_tokens": 7}
    def scorer(_q, texts):
        return scores or [5]*len(texts)
    return SlotEvidenceEvaluator(judge=judge, relevance_scorer=scorer, authorize=lambda _p: True,
                                 thresholds=Thresholds(keep=-2, high=2), **kwargs), calls


def row(sid="s1", ref="E1", quote=P[0]["text"], status="supported", confidence=0):
    return {"subgoal_id": sid, "status": status, "support_refs": [ref], "quote_spans": [{"ref": ref, "quote": quote}],
            "missing_slots": [], "reason_code": "value_present", "confidence": confidence}


def test_actual_quoted_support_not_an_uncited_high_score():
    evaluator, _ = make([row(ref="E2", quote=P[1]["text"])], scores=[6, 0])
    judgment = evaluator.coverage("q", ITEMS, P)[0]
    assert judgment.status == "partial"
    assert judgment.reason_code == "quoted_support_low_relevance"


def test_low_self_confidence_never_changes_completion_rule():
    evaluator, _ = make([row()])
    judgments = evaluator.coverage("q", ITEMS, P)
    assert judgments[0].confidence == 0
    assert aggregate_slots(ITEMS, judgments)["complete"] == ["s1"]


@pytest.mark.parametrize("change", [
    {"support_refs": ["E999"]}, {"quote_spans": [{"ref": "E1", "quote": "不存在的原文"}]},
    {"quote_spans": []}, {"support_refs": ["E1", "E999"]},
    {"evaluation_validity": "unknown"},
    {"confidence": 1.5},
])
def test_invalid_or_ambiguous_judgment_is_unknown_not_missing_fact(change):
    evaluator, _ = make([{**row(), **change}])
    judgments = evaluator.coverage("q", ITEMS, P)
    assert judgments[0].evaluation_validity == "unknown"
    assert aggregate_slots(ITEMS, judgments)["unknown"] == ["s1"]
    assert not aggregate_slots(ITEMS, judgments)["ready"]


def test_transport_schema_failure_has_cost_and_unknown():
    evaluator, _ = make([{"invalid": True}])
    judgments = evaluator.coverage("q", ITEMS, P)
    assert judgments[0].evaluation_validity == "unknown"
    assert evaluator.usage.attempted == evaluator.usage.failed == 1
    assert evaluator.usage.prompt_tokens == 13


def test_fair_round_robin_reserves_context_for_other_slot():
    evaluator, calls = make([row(), row(sid="s2", ref="E2", quote=P[1]["text"])], max_passages=2)
    evaluator._relevance_scorer = lambda q, _texts: [9, 8] if q == "first" else [-9, 3]
    evaluator.coverage("q", [{"id": "s1", "text": "first"}, {"id": "s2", "text": "second"}], P)
    assert [p["text"] for p in calls[0]["evidence"]] == [p["text"] for p in P]


def test_context_omission_cannot_complete_uncited_slot():
    evaluator, _ = make([row()], max_passages=1)
    evaluator._relevance_scorer = lambda q, _texts: [9, -9] if q == "first" else [-9, 3]
    judgments = evaluator.coverage("q", [{"id": "s1", "text": "first"}, {"id": "s2", "text": "second"}], P)
    assert judgments[1].reason_code == "context_omitted"


def test_cache_uses_content_and_version_and_reauthorizes_before_reuse():
    evaluator, calls = make([row()])
    first = evaluator.coverage("q", ITEMS, P)
    assert evaluator.coverage("q", ITEMS, P) == first
    assert len(calls) == 1 and evaluator.cache_hits == 1
    evaluator.authorize = lambda _p: False
    with pytest.raises(PermissionError):
        evaluator.coverage("q", ITEMS, P)
    evaluator.authorize = lambda _p: True
    evaluator.coverage("q", ITEMS, [{**P[0], "version_id": "v-new"}, P[1]])
    assert len(calls) == 2


def test_technical_unknown_is_not_cached_but_valid_low_relevance_is():
    evaluator, calls = make([])
    evaluator.coverage("q", ITEMS, P)
    evaluator.coverage("q", ITEMS, P)
    assert len(calls) == 2


def test_versions_are_not_conflicts_without_same_scope_proof():
    conflict = {**row(status="contradicted"), "support_refs": ["E1", "E2"],
                "quote_spans": [{"ref": "E1", "quote": P[0]["text"]}, {"ref": "E2", "quote": P[1]["text"]}]}
    evaluator, _ = make([conflict])
    assert evaluator.coverage("q", ITEMS, P)[0].reason_code == "conflict_scope_unproven"
    scope = {"subject": "甲服务", "attribute": "quota", "period": "2026-09"}
    judged = evaluator.coverage("q", ITEMS, [{**p, "scope": scope} for p in P])[0]
    assert judged.status == "contradicted" and judged.evaluation_validity == "evaluated"


def test_claim_without_own_citations_cannot_borrow_another_claims_support():
    evaluator, calls = make([row()])
    judgments = evaluator.coverage("q", [{**ITEMS[0], "purpose": "claim", "allowed_refs": []}], P)
    assert judgments[0].evaluation_validity == "unknown" and not calls


def test_explicit_scope_mismatch_cannot_be_supported():
    evaluator, _ = make([row()])
    judgments = evaluator.coverage("q", [{**ITEMS[0], "version_id": "v2"}], P)
    assert judgments[0].reason_code == "scope_mismatch"


def test_inactive_conditional_slots_are_not_completion_requirements():
    evaluator, _ = make([row()])
    judgments = evaluator.coverage("q", ITEMS, P)
    report = aggregate_slots(ITEMS + [{"id": "false_branch", "active": False}], judgments)
    assert report["ready"] and report["inactive"] == ["false_branch"]


def test_literal_short_circuit_is_single_attribute_only_and_never_claim_faithfulness():
    evaluator, calls = make([])
    evidence = [{"chunk_id": "a", "version_id": "v1", "document_id": "d1", "title": "生产数据库",
                 "text": "生产数据库 RPO 为 15 分钟，RTO 为 60 分钟。"}]
    item = {"id": "rpo", "text": "生产数据库的RPO是多少？"}
    result = evaluator.coverage("q", [item], evidence)[0]
    assert result.source == "literal_binding" and result.status == "supported" and calls == []
    assert SlotEvidenceEvaluator._literal_short_circuit({**item, "text": "生产数据库的RPO是多少并说明灾备流程"}, evidence) is None
    assert SlotEvidenceEvaluator._literal_short_circuit({**item, "purpose": "claim"}, evidence) is None
    changed = [{**evidence[0], "text": "生产数据库 RPO 不是 15 分钟。"}]
    assert SlotEvidenceEvaluator._literal_short_circuit(item, changed) is None


def test_authorization_requires_positive_confirmation_not_a_missing_return():
    evaluator, calls = make([row()])
    evaluator.authorize = lambda _p: None
    with pytest.raises(PermissionError):
        evaluator.coverage("q", ITEMS, P)
    assert calls == []


def test_oversized_context_is_unknown_before_server_can_silently_truncate():
    evaluator, calls = make([row()])
    passages = [{**P[0], "text": "原文很长。" * 3000}]
    judged = evaluator.coverage("q", ITEMS, passages)[0]
    assert judged.evaluation_validity == "unknown"
    assert judged.reason_code == "context_budget_exceeded"
    assert calls == [] and evaluator.usage.attempted == 0


def test_revocation_during_judge_invalidates_verdict_and_does_not_cache():
    permitted = True
    def judge(_payload):
        nonlocal permitted
        permitted = False
        return {"output": {"judgments": [row()]}}
    evaluator = SlotEvidenceEvaluator(authorize=lambda _p: permitted, judge=judge,
                                      relevance_scorer=lambda q, texts: [5]*len(texts))
    with pytest.raises(PermissionError):
        evaluator.coverage("q", ITEMS, P)
    assert evaluator._cache == {} and evaluator.usage.attempted == 1


def test_revocation_during_ranking_invalidates_relevance_cache():
    permitted = True
    def score(_q, texts):
        nonlocal permitted
        permitted = False
        return [5]*len(texts)
    evaluator = SlotEvidenceEvaluator(authorize=lambda _p: permitted, relevance_scorer=score)
    with pytest.raises(PermissionError):
        evaluator.relevance(ITEMS, P)
    assert evaluator._relevance_cache == {}


def test_cache_restores_matching_diagnostic_context_and_literal_clears_it():
    evaluator, calls = make([row()])
    evaluator.coverage("q", ITEMS, [P[0]])
    original = evaluator.last_context
    evaluator.coverage("different", ITEMS, P)
    evaluator.coverage("q", ITEMS, [P[0]])
    assert len(calls) == 2 and evaluator.last_context == original
    assert evaluator.last_raw_judgments[0]["subgoal_id"] == "s1"
    evidence = [{**P[0], "text": "甲服务 RPO 为 15 分钟。"}]
    result = evaluator.coverage("q", [{"id": "rpo", "text": "甲服务的RPO是多少？"}], evidence)[0]
    assert result.source == "literal_binding"
    assert evaluator.last_context == {} and evaluator.last_raw_judgments == []


@pytest.mark.parametrize("change", [{"allowed_refs": []}, {"subject": "乙服务"},
                                     {"version_id": "another-version"}])
def test_literal_cannot_bypass_slot_scope_or_own_reference_restrictions(change):
    item = {"id": "rpo", "text": "甲服务的RPO是多少？", **change}
    evidence = [{**P[0], "text": "甲服务 RPO 为 15 分钟。"}]
    assert SlotEvidenceEvaluator._literal_short_circuit(item, evidence) is None
