"""Component runs retain completed work and costs across interruption."""
import copy
import json

import pytest

from app.execution_budget import current_budget
from app.semantic_evidence import EvidenceJudgment, JudgeUsage, RelevanceJudgment, Thresholds
from scripts.predict_semantic_slots import predict
from scripts.research_review import packet
from scripts.semantic_slot_eval import distinct_inputs, select_thresholds


def inputs():
    base = {"kind": "coverage", "corpus": "s3", "question": "配额是多少？", "qid": "q",
            "slot": {"id": "s1", "text": "配额是多少？"}, "family_id": "d", "split": "test",
            "evidence": [{"chunk_id": "c", "text": "配额为20次。", "document_id": "d", "version_id": "v"}]}
    return packet([{**base, "id": "first"}, {**base, "id": "duplicate"},
                   {**base, "id": "ranking", "kind": "relevance"}], "unit fixture only")


@pytest.fixture
def runtime(monkeypatch):
    from app import semantic_evidence
    import paired_semantic_replay
    import run_unseen_benchmark
    calls = []
    class Evaluator:
        def __init__(self, **kwargs):
            self.thresholds = kwargs.get("thresholds", Thresholds())
            self.usage = JudgeUsage()
        def relevance(self, items, passages):
            calls.append("ranking")
            return [RelevanceJudgment(subgoal_id=items[0]["id"], chunk_id=p["chunk_id"], score=5,
                                      label="relevant") for p in passages]
        def coverage(self, question, items, passages, relevance):
            calls.append("judge")
            with current_budget().call("judge", reserve_tokens=10) as usage:
                usage.update(prompt_tokens=4, completion_tokens=2)
            self.usage.attempted += 1
            return [EvidenceJudgment(subgoal_id=items[0]["id"], status="supported")]
    monkeypatch.setattr(semantic_evidence, "SlotEvidenceEvaluator", Evaluator)
    monkeypatch.setattr(semantic_evidence, "SemanticEvidenceEvaluator", Evaluator)
    monkeypatch.setattr(run_unseen_benchmark, "configuration", lambda: {"models": {"judge": "test"}})
    monkeypatch.setattr(paired_semantic_replay, "code_inputs", lambda: {"test": "frozen"})
    return calls


def test_relevance_calibration_prepass_never_calls_a_judge(runtime, tmp_path):
    output = predict(inputs(), relevance_only=True, checkpoint=tmp_path / "checkpoint.json")
    assert runtime == ["ranking"]
    assert output["selected_complete"] and not output["complete"]
    assert set(output["rows"]) == {"ranking"}
    assert output["rows"]["ranking"]["execution_budget"]["calls"] == {}


def test_identical_frozen_inputs_reuse_verdict_without_duplicating_costs(runtime, tmp_path):
    path = tmp_path / "checkpoint.json"
    output = predict(inputs(), checkpoint=path)
    assert output["complete"] and output["reused_rows"] == 1
    duplicate = output["rows"]["duplicate"]
    assert duplicate["reused_from"] == "first" and duplicate["execution_budget"]["calls"] == {}
    assert output["rows"]["first"]["execution_budget"]["calls"]["judge"]["attempted"] == 2
    assert runtime.count("judge") == 2
    runtime.clear()
    assert predict(inputs(), checkpoint=path, resume=True)["rows"] == output["rows"]
    assert runtime == []


def test_changed_models_cannot_resume_completed_predictions(runtime, tmp_path, monkeypatch):
    import run_unseen_benchmark
    path = tmp_path / "checkpoint.json"
    predict(inputs(), checkpoint=path)
    monkeypatch.setattr(run_unseen_benchmark, "configuration", lambda: {"models": {"judge": "changed"}})
    with pytest.raises(ValueError, match="hashes differ"):
        predict(inputs(), checkpoint=path, resume=True)


def test_interrupted_attempt_is_unknown_and_its_reserved_cost_is_not_refunded(runtime, tmp_path):
    path = tmp_path / "checkpoint.json"
    output = predict(inputs(), checkpoint=path)
    saved = json.loads(path.read_text())
    first = saved["completed"].pop("components:first")
    budget = first["execution_budget"]
    budget["calls"]["judge"].update(attempted=3)
    saved["pending"]["components:first"] = {"execution_budget": budget}
    path.write_text(json.dumps(saved))
    runtime.clear()
    resumed = predict(inputs(), checkpoint=path, resume=True)
    result = resumed["rows"]["first"]
    assert result["slot_v2"] == "unknown" and result["error"] == "interrupted_prediction"
    assert result["execution_budget"]["calls"]["judge"]["attempted"] == 3
    assert result["execution_budget"]["calls"]["judge"]["failed"] == 1
    assert resumed["rows"]["ranking"] == output["rows"]["ranking"] and runtime == []


def test_repeated_claims_do_not_inflate_component_sample_size_or_intervals():
    source = inputs()
    labels = {r["id"]: "supported" for r in source["items"]}
    assert len(distinct_inputs(source, labels)) == 2
    labels["duplicate"] = "partial"
    with pytest.raises(ValueError, match="labels disagree"):
        distinct_inputs(source, labels)


def test_identical_inputs_across_splits_are_rejected():
    source = inputs()
    source["items"][1]["split"] = "dev"
    with pytest.raises(ValueError, match="crosses source"):
        distinct_inputs(source, {r["id"]: "supported" for r in source["items"]})


def test_calibration_requires_finite_scores_and_ordered_cascade_thresholds():
    source = inputs()
    source["items"] = [source["items"][-1]]
    source["items"][0]["split"] = "dev"
    second = copy.deepcopy(source["items"][0])
    second.update(id="second", question="另一个问题", qid="second", family_id="second")
    second["evidence"][0].update(chunk_id="second-chunk", document_id="second-doc")
    source["items"].append(second)
    labels = {r["id"]: "relevant" for r in source["items"]}
    for value in (None, float("nan"), True):
        with pytest.raises(ValueError, match="finite relevance"):
            select_thresholds(source, labels, {"ranking": {"score": value}, "second": {"score": 1}})
    cuts = select_thresholds(source, labels, {"ranking": {"score": 1}, "second": {"score": 3}})
    assert cuts["thresholds"]["s3"]["high"] >= cuts["thresholds"]["s3"]["keep"]


def test_review_delivery_removes_duplicate_work_without_creating_labels():
    from scripts.prepare_slot_review_delivery import delivery
    source = inputs()
    expanded, pilot = delivery(source)
    assert len(expanded["items"]) == 2 and expanded["reviews"] == []
    assert expanded["deduplication"]["representative_by_id"]["duplicate"] == "first"
    assert pilot["items"] == [] and pilot["reviews"] == []  # All fixture rows are test.
    source["reviews"] = [{"id": "first", "label": "supported"}]
    with pytest.raises(ValueError, match="unreviewed"):
        delivery(source)


def test_annotation_pilot_never_borrows_test_inputs_to_fill_a_quota():
    from scripts.prepare_slot_review_delivery import delivery
    source = inputs()
    source["items"][2]["split"] = "dev"
    source["items"][2].update(family_id="dev-family", qid="dev-query")
    source["items"][2]["evidence"] = [{**source["items"][2]["evidence"][0],
                                      "chunk_id": "dev-chunk", "document_id": "dev-doc"}]
    from scripts.research_review import digest
    source["source_sha256"] = digest(source["items"])
    expanded, pilot = delivery(source)
    assert len(pilot["items"]) == 1 and pilot["items"][0]["id"] == "ranking"
    assert expanded["items"][0]["split"] == "test"


def test_coverage_and_relevance_cannot_split_the_same_source_family():
    from scripts.research_review import validate_component_partitions
    source = inputs()
    source["items"][2].update(split="dev", family_id="artificial-dev", qid="different-id")
    with pytest.raises(ValueError, match="query/document family"):
        validate_component_partitions(source["items"])
