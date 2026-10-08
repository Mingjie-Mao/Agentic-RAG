import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from scripts.research_review import packet, validated_labels, group_splits, wilson, paired_ci
from scripts.paired_semantic_replay import cost_gate, require_gate
from scripts.unseen_v2_scoring import provisional_score


def reviewed(source, label="supported"):
    value = copy.deepcopy(source)
    value["reviews"] = [{"id": r["id"], "label": label, "reviewer": "external-reviewer", "reviewer_type": "human",
                         "independent": True, "reason": "Read the quoted source and scope."} for r in source["items"]]
    return value


def test_no_independent_labels_means_no_gate():
    source = packet([{"id": "x", "kind": "coverage"}], "test")
    with pytest.raises(ValueError, match="incomplete"):
        validated_labels(source, source)
    review = reviewed(source)
    review["reviews"][0]["reviewer_type"] = "assistant"
    with pytest.raises(ValueError, match="human"):
        validated_labels(source, review)


def test_review_cannot_change_inputs_or_skip_disputes():
    source = packet([{"id": "x", "kind": "coverage", "evidence": "source"}], "test")
    review = reviewed(source)
    review["items"][0]["evidence"] = "modified"
    with pytest.raises(ValueError, match="changed"):
        validated_labels(source, review)
    review = reviewed(source)
    review["reviews"].append({**review["reviews"][0], "reviewer": "second", "label": "partial"})
    with pytest.raises(ValueError, match="adjudication"):
        validated_labels(source, review)
    review["adjudications"] = [{**review["reviews"][0], "reviewer": "third"}]
    assert validated_labels(source, review) == {"x": "supported"}


def test_variants_and_source_version_families_never_cross_splits():
    rows = [{"qid": "a", "corpus": "s3", "families": ["d1"]},
            {"qid": "a", "corpus": "s3", "families": ["d2"]},
            {"qid": "b", "corpus": "s3", "families": ["d2"]}]
    grouped = group_splits(rows)
    assert len({r["family_id"] for r in grouped}) == len({r["split"] for r in grouped}) == 1


def test_intervals_do_not_invent_certainty_from_small_or_single_family_samples():
    assert wilson(5, 5)[0] < .95
    assert paired_ci([1, 1], groups=["same", "same"]) == [None, None]


def costs(tid="a", judge=0, latency=100, tokens=100):
    return {"task_id": tid, "latency_ms": latency, "status": "answered", "forbidden_present": {},
            "usage": {"execution_budget": {"calls": {"generation": {"prompt_tokens": tokens},
                                                      "judge": {"attempted": judge}}}}}


def test_cost_gate_counts_attempts_and_missing_telemetry_and_p95_tail():
    baseline = [costs("a"), costs("b")]
    sem = [costs("a", judge=1), costs("b")]
    assert cost_gate(baseline, sem)["passed"]
    sem[1]["latency_ms"] = 1000
    assert not cost_gate(baseline, sem)["checks"]["p95"]
    sem[1]["latency_ms"] = None
    assert not cost_gate(baseline, sem)["checks"]["complete_telemetry"]


def test_unreviewed_or_fabricated_gate_is_rejected(tmp_path):
    import json
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"version": "semantic-slot-v2-gate", "passed": True}))
    with pytest.raises(ValueError, match="bound"):
        require_gate(path, "semantic-slot-v2-gate")


def test_one_retrieval_can_satisfy_latent_link_without_forced_tool_sequence():
    task = {"id": "x", "category": "latent_link", "expected_status": "answered", "facts": ["department"],
            "forbidden": [], "expected_evidence_documents": ["first", "second"],
            "conditional_transition": {"observe_document": "first", "then_document": "second"}}
    payload = {"status": "answered", "claims": [{"text": "department"}],
               "citations": [{"document_id": "first"}, {"document_id": "second"}]}
    result = provisional_score(task, payload, [], 1, 100)
    assert result["answer_correct"] and "conditional_transition" not in result["capabilities"]
    assert result["quality_provisional"]


def test_condition_checks_final_claims_not_authorized_source_containing_inactive_rule():
    task = {"id": "x", "category": "conditional_planning", "expected_status": "answered", "facts": ["观察"],
            "forbidden": ["B-99"]}
    payload = {"status": "answered", "claims": [{"text": "继续观察"}], "citations": [{"text": "若超过阈值回滚 B-99"}]}
    result = provisional_score(task, payload, [], 1, 100)
    assert result["answer_correct"] and not result["condition_violation"]


def test_new_holdout_has_70_tasks_seven_balanced_classes_and_refuses_unreviewed_freeze():
    import json
    from scripts.unseen_v2_scoring import reviewed_suite
    root = Path(__file__).resolve().parents[1]
    suite = json.loads((root / "fixtures/unseen_v2/tasks.json").read_text())
    assert len(suite["tasks"]) == 70
    assert len(suite["categories"]) == 7 and set(suite["categories"].values()) == {10}
    assert len(suite["safety_tasks"]) == 2
    assert not (root / "fixtures/unseen_v2/freeze.json").exists()
    with pytest.raises(ValueError, match="independent"):
        reviewed_suite(suite, root)


def test_zero_model_calls_are_recorded_cost_not_missing_telemetry():
    rows = [costs("a"), costs("b")]
    for r in rows:
        r["usage"]["execution_budget"]["calls"] = {}
    result = cost_gate(rows, rows)
    assert result["checks"]["complete_telemetry"]
    assert result["ratios"]["tokens"] == 1


def test_formal_quality_uses_reviewed_completeness_and_condition_not_string_diagnostics():
    from scripts.unseen_v2_scoring import answer_packet, quality_gate
    tasks = [{"id": tid, "category": category, "goal": "test question", "facts": ["value"],
              "required_slots": [{"id": "value"}], "expected_evidence_documents": ["doc"]}
             for tid, category in (("a", "conditional_planning"), ("b", "latent_link"))]
    suite = {"tasks": tasks}
    raw = {"results": {arm: [{"task_id": t["id"], "run_id": arm+t["id"], "category": t["category"],
          "status": "answered", "answer_correct": True, "condition_violation": True,
          "answer_payload": {"status": "answered", "claims": [{"text": "value"}]},
          "forbidden_present": {}, "citation_documents": ["doc"]} for t in tasks]
          for arm in ("workflow", "hybrid")}}
    source = answer_packet(raw, suite)
    review = reviewed(source, label="correct")
    for arm, rows in raw["results"].items():
        from scripts.research_review import digest
        for r in rows:
            rid = digest([arm, r["task_id"], r["run_id"]])
            label = next(v for v in review["reviews"] if v["id"] == rid)
            good = arm == "hybrid"
            label.update(label="correct" if good else "incorrect", slot_verdicts={"value": "supported" if good else "partial"},
                         citation_valid=True, condition_compliant=True, version_scope_valid=True, status_valid=True)
    gate = quality_gate(raw, suite, review)
    assert gate["passed"] and gate["checks"]["condition_no_regression"]
    assert gate["by_category"]["workflow"]["latent_link"]["supported_fact_slots"] == 0
    assert gate["by_category"]["hybrid"]["latent_link"]["supported_fact_slots"] == 1
    assert gate["reports"]["answer_correct"]["difference"] == 1
