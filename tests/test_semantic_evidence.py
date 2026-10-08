from app.semantic_evidence import (
    EvidenceJudgment,
    SemanticEvidenceEvaluator,
    Thresholds,
    aggregate_coverage,
)

ITEMS = [{"id": "s1", "text": "回滚触发条件"}, {"id": "s2", "text": "回滚目标版本"}]
PASSAGES = [
    {"chunk_id": "c1", "title": "发布手册", "text": "错误率连续 5 分钟超过 2% 时回滚。"},
    {"chunk_id": "c2", "title": "会议纪要", "text": "纪要在会后 24 小时内归档。"},
]


def scorer(table):
    def score(query, texts):
        return [table.get((query, text.split("\n", 1)[1]), -9.0) for text in texts]
    return score


def judge_returning(rows, calls):
    def judge(payload):
        calls.append(payload)
        return {"output": {"judgments": rows}, "prompt_tokens": 10, "completion_tokens": 5}
    return judge


def test_relevance_labels_rank_but_every_pair_is_kept():
    evaluator = SemanticEvidenceEvaluator(
        relevance_scorer=scorer({("回滚触发条件", PASSAGES[0]["text"]): 3.0}),
        thresholds=Thresholds(keep=-2.0, high=1.0),
    )
    rows = evaluator.relevance(ITEMS, PASSAGES)
    assert len(rows) == 4  # nothing is dropped
    assert {(r.subgoal_id, r.chunk_id): r.label for r in rows}[("s1", "c1")] == "relevant"
    assert {(r.subgoal_id, r.chunk_id): r.label for r in rows}[("s2", "c2")] == "irrelevant"


def test_a_subgoal_with_no_relevant_passage_is_unsupported_without_a_model_call():
    calls = []
    evaluator = SemanticEvidenceEvaluator(
        relevance_scorer=scorer({}), judge=judge_returning([], calls), thresholds=Thresholds(keep=-2.0)
    )
    judged = evaluator.coverage("q", ITEMS, PASSAGES)
    assert calls == [] and {j.status for j in judged} == {"unsupported"}
    assert evaluator.usage.skipped_by_cascade == 1


def test_relevant_but_half_answered_is_partial_not_covered():
    calls = []
    rows = [
        {"subgoal_id": "s1", "status": "supported", "support_refs": ["E1"], "missing_slots": [],
         "reason_code": "value_present", "confidence": 0.9},
        {"subgoal_id": "s2", "status": "unsupported", "support_refs": [], "missing_slots": ["目标版本"],
         "reason_code": "topic_only", "confidence": 0.8},
    ]
    evaluator = SemanticEvidenceEvaluator(
        relevance_scorer=scorer({("回滚触发条件", PASSAGES[0]["text"]): 3.0,
                                 ("回滚目标版本", PASSAGES[0]["text"]): 2.5}),
        judge=judge_returning(rows, calls), thresholds=Thresholds(keep=-2.0, high=1.0),
    )
    judged = {j.subgoal_id: j for j in evaluator.coverage("回滚触发条件和目标版本是什么？", ITEMS, PASSAGES)}
    assert len(calls) == 1  # one batched call for all pending subgoals
    assert judged["s1"].status == "supported" and judged["s1"].support_refs == ["c1"]
    assert judged["s2"].status == "unsupported"  # highly relevant, still not covered
    assert [e["id"] for e in calls[0]["evidence"]] == ["E1"]  # no ids leak, labels only


def test_supported_without_a_cited_passage_is_not_supported():
    rows = [{"subgoal_id": "s1", "status": "supported", "support_refs": ["E9"], "missing_slots": [],
             "reason_code": "value_present", "confidence": 0.95}]
    evaluator = SemanticEvidenceEvaluator(
        relevance_scorer=scorer({("回滚触发条件", PASSAGES[0]["text"]): 3.0}),
        judge=judge_returning(rows, []), thresholds=Thresholds(keep=-2.0),
    )
    judged = {j.subgoal_id: j for j in evaluator.coverage("q", ITEMS[:1], PASSAGES)}
    assert judged["s1"].status == "unsupported" and judged["s1"].reason_code == "no_cited_support"


def test_an_unavailable_judge_never_reads_as_coverage():
    def broken(_payload):
        raise RuntimeError("down")
    evaluator = SemanticEvidenceEvaluator(
        relevance_scorer=scorer({("回滚触发条件", PASSAGES[0]["text"]): 3.0}),
        judge=broken, thresholds=Thresholds(keep=-2.0),
    )
    judged = evaluator.coverage("q", ITEMS[:1], PASSAGES)
    assert judged[0].status == "unsupported" and judged[0].source == "judge_unavailable"


def test_sufficiency_is_code_and_low_confidence_support_is_partial():
    judgments = [
        EvidenceJudgment(subgoal_id="s1", status="supported", support_refs=["c1"], confidence=0.9),
        EvidenceJudgment(subgoal_id="s2", status="supported", support_refs=["c1"], confidence=0.3),
        EvidenceJudgment(subgoal_id="s3", status="contradicted", support_refs=["c1", "c2"], confidence=0.8),
    ]
    items = [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}, {"id": "s4"}]
    report = aggregate_coverage(items, judgments, min_supported_confidence=0.6)
    assert report.complete == ["s1"] and report.partial == ["s2"]
    assert report.contradicted == ["s3"] and report.missing == ["s4"]


def test_judge_support_without_a_highly_relevant_passage_is_only_partial():
    rows = [{"subgoal_id": "s1", "status": "supported", "support_refs": ["E1"], "missing_slots": [],
             "reason_code": "value_present", "confidence": 1.0}]
    evaluator = SemanticEvidenceEvaluator(
        relevance_scorer=scorer({("回滚触发条件", PASSAGES[0]["text"]): 0.5}),
        judge=judge_returning(rows, []), thresholds=Thresholds(keep=-2.0, high=1.0),
    )
    judged = evaluator.coverage("q", ITEMS[:1], PASSAGES)[0]
    assert judged.status == "partial" and judged.reason_code == "low_relevance_support"


def test_judge_transport_and_schema_failures_are_attempted_failed_with_tokens():
    for result in (None, {"output": {"judgments": "invalid"}, "prompt_tokens": 19, "completion_tokens": 4}):
        def failing(_payload):
            if result is None:
                raise RuntimeError("unavailable")
            return result
        evaluator = SemanticEvidenceEvaluator(
            relevance_scorer=scorer({("回滚触发条件", PASSAGES[0]["text"]): 3.0}),
            judge=failing, thresholds=Thresholds(keep=-2.0))
        evaluator.coverage("q", ITEMS[:1], PASSAGES)
        assert (evaluator.usage.attempted, evaluator.usage.succeeded, evaluator.usage.failed) == (1, 0, 1)
        assert evaluator.usage.prompt_tokens == (19 if result else 0)
