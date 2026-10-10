"""Offline delivery proxies must preserve source identity and missing denominators."""

from copy import deepcopy
import json

import pytest

from scripts.multihop_retrieval_eval import document_id
from scripts.retrieval_quality_metrics import (
    diagnose,
    pair_profiles,
    phase_coverage,
    summarize,
    supporting_facts,
)


def native_task(task_id="native", **updates):
    return {"id": task_id, "category": "multi_hop", "gold_evidence": [
        {"id": "ge-a", "document_id": "a", "source_sha256": "sha-a",
         "quote": "服务首次响应时限为三十分钟。"},
        {"id": "ge-b", "document_id": "b", "source_sha256": "sha-b",
         "quote": "负责人每周轮换一次。"},
    ], **updates}


def external_task():
    return {"id": "external", "category": "comparison", "upstream_gold_evidence": [
        {"title": "Synthetic Article", "fact": "The first response limit is thirty minutes."},
    ], "gold_evidence": [
        {"id": "article", "document_id": document_id("Synthetic Article"),
         "source_sha256": "article-sha", "quote": "Entire unrelated source span"},
    ]}


def evidence(facts, **updates):
    return [{"chunk_id": f"chunk-{i}", "document_id": fact["document_id"],
             "source_sha256": fact["source_sha256"], "text": fact["text"], "rank": i,
             **updates} for i, fact in enumerate(facts, 1)]


def phases(rows):
    return {name: rows for name in ("candidates", "admitted", "context")}


@pytest.mark.parametrize("updates", [
    {"document_id": "wrong"}, {"source_sha256": "old-version"},
    {"source_sha256": None}, {"document_id": None},
])
def test_wrong_source_or_version_cannot_satisfy_reference(updates):
    facts = supporting_facts(native_task())
    coverage = phase_coverage(facts, evidence(facts, **updates))
    assert coverage["total"] == 2
    assert coverage["delivered"] == 0
    assert coverage["matched_reference_ids"] == []
    assert coverage["all_delivered"] is False


def test_missing_telemetry_is_null_and_empty_telemetry_is_measured():
    facts = supporting_facts(native_task())
    assert phase_coverage(facts, None) is None
    assert phase_coverage([], None) is None
    assert phase_coverage(facts, [])["delivered"] == 0


def test_duplicate_chunks_do_not_inflate_partial_reference_coverage():
    facts = supporting_facts(native_task())
    rows = evidence(facts[:1])
    coverage = phase_coverage(facts, rows * 5)
    assert coverage["delivered"] == 1
    assert coverage["matched_reference_ids"] == ["ge-a"]
    assert coverage["first_supporting_ranks"] == {"ge-a": 1}


def test_external_uses_upstream_word4_proxy_and_actual_source_hash():
    facts = supporting_facts(external_task())
    assert len(facts) == 1
    assert facts[0]["basis"] == "upstream_word4_proxy"
    assert facts[0]["source_sha256"] == "article-sha"
    assert facts[0]["text"] == external_task()["upstream_gold_evidence"][0]["fact"]
    assert phase_coverage(facts, evidence(facts))["all_delivered"] is True
    assert phase_coverage(facts, evidence(facts, text="Entire unrelated source span"))["delivered"] == 0


def test_native_character_proxy_normalizes_case_and_whitespace():
    task = native_task(gold_evidence=[{"id": "ge", "document_id": "a",
                                    "source_sha256": "sha-a", "quote": "ABCD 服务首 次响应"}])
    facts = supporting_facts(task)
    assert facts[0]["basis"] == "source_span_char4_proxy"
    assert phase_coverage(facts, evidence(facts, text="abcd服务首次响应"))["all_delivered"]
    assert phase_coverage(facts, evidence(facts, text=""))["delivered"] == 0


def test_native_source_span_proxy_has_the_declared_half_threshold():
    task = native_task(gold_evidence=[{"id": "ge", "document_id": "a",
                                    "source_sha256": "sha-a", "quote": "abcdefgh"}])
    facts = supporting_facts(task)
    assert phase_coverage(facts, evidence(facts, text="abcde"))["delivered"] == 0
    assert phase_coverage(facts, evidence(facts, text="abcdef"))["delivered"] == 1


def test_word_proxy_unions_chunks_and_reports_threshold_completion_rank():
    facts = [{"id": "f", "document_id": "a", "source_sha256": "sha-a",
              "basis": "upstream_word4_proxy", "text": "one two three four five six seven eight"}]
    rows = evidence(facts, text="one two three four", rank=2)
    rows += evidence(facts, text="five six seven eight", rank=5)
    assert phase_coverage(facts, rows)["delivered"] == 0  # two of five 4-grams
    rows += evidence(facts, text="two three four five", rank=7)
    coverage = phase_coverage(facts, rows)
    assert coverage["delivered"] == 1
    assert coverage["first_supporting_rank"] == 7
    assert coverage["first_supporting_ranks"] == {"f": 7}


def test_supporting_rank_falls_back_to_retrieval_rank_without_inventing_positions():
    facts = supporting_facts(native_task())
    rows = evidence(facts, rank=None, retrieval_rank=8)
    assert phase_coverage(facts, rows)["first_supporting_rank"] == 8
    assert phase_coverage(facts, evidence(facts, rank=None))["first_supporting_rank"] is None


@pytest.mark.parametrize("candidate,admitted,context,cause", [
    (0, 0, 0, "candidate_missing"), (2, 1, 1, "admission_loss"),
    (2, 2, 1, "context_loss"), (2, 2, 2, "proxy_complete"),
])
def test_diagnoses_each_phase_loss(candidate, admitted, context, cause):
    task = native_task()
    rows = evidence(supporting_facts(task))
    result = diagnose(task, {"candidates": rows[:candidate], "admitted": rows[:admitted],
                             "context": rows[:context]})
    assert result["status"] == "measured"
    assert result["eligible"] is True
    assert result["cause"] == cause
    assert result["semantic_recall"] is None
    assert result["strict_task_success"] is None


def test_partial_candidate_coverage_and_later_loss_are_both_retained():
    task = native_task()
    rows = evidence(supporting_facts(task))
    result = diagnose(task, {"candidates": rows[:1], "admitted": [], "context": []})
    assert result["cause"] == "candidate_missing"
    assert result["losses"] == {"candidate_missing": 1, "admission_loss": 1, "context_loss": 0}


def test_unavailable_phase_and_retrieval_failure_remain_errors():
    task = native_task()
    rows = evidence(supporting_facts(task))
    result = diagnose(task, {"candidates": rows, "admitted": None, "context": []})
    assert result["status"] == "error"
    assert result["cause"] is None
    assert result["phases"]["admitted"] is None
    assert result["phases"]["candidates"]["delivered"] == 2
    assert diagnose(task, None)["status"] == "error"


def test_missing_telemetry_keeps_known_phase_denominators_in_summary():
    task = native_task()
    row = diagnose(task, {"candidates": evidence(supporting_facts(task))})
    summary = summarize([row])["by_basis"]["source_span_char4_proxy"]
    assert summary["phases"]["candidates"]["reference_denominator"] == 2
    assert summary["phases"]["context"]["reference_denominator"] == 0
    assert summary["phases"]["context"]["proxy_coverage"] is None
    assert summary["phases"]["context"]["missing_tasks"] == 1


def test_external_temporal_category_is_eligible_when_no_local_history_is_required():
    task = external_task()
    task.update(category="temporal_version", requires_version=False)
    assert diagnose(task, phases([]))["status"] == "measured"


def test_missing_gold_source_hash_cannot_match_missing_evidence_source_hash():
    task = native_task()
    del task["gold_evidence"][0]["source_sha256"]
    assert diagnose(task, phases([]))["reason"] == "invalid_references"


@pytest.mark.parametrize("task,reason", [
    (native_task(requires_version=True), "requires_version"),
    (native_task(gold_evidence=[], category="null_insufficient"), "no_references"),
])
def test_history_and_no_references_are_explicitly_not_applicable(task, reason):
    result = diagnose(task, phases([]))
    assert result["status"] == "not_applicable"
    assert result["cause"] == "not_applicable"
    assert result["eligible"] is False
    assert result["reason"] == reason
    assert all(value is None for value in result["phases"].values())
    assert phase_coverage([], [])["status"] == "not_applicable"
    assert phase_coverage([], [])["delivered"] is None


@pytest.mark.parametrize("change", ["missing", "ambiguous"])
def test_external_reference_without_unique_source_binding_is_an_error(change):
    task = external_task()
    if change == "missing":
        task["gold_evidence"] = []
    else:
        task["gold_evidence"] += [{**task["gold_evidence"][0], "source_sha256": "other"}]
    with pytest.raises(ValueError, match="source binding"):
        supporting_facts(task)
    assert diagnose(task, phases([]))["status"] == "error"


def test_duplicate_gold_facts_are_deduplicated_and_duplicate_ids_rejected():
    task = external_task()
    task["upstream_gold_evidence"] *= 2
    assert len(supporting_facts(task)) == 1
    task = native_task()
    task["gold_evidence"][1]["id"] = "ge-a"
    with pytest.raises(ValueError, match="duplicate reference"):
        supporting_facts(task)


def test_summaries_keep_basis_category_and_missing_denominators_separate():
    native, external = native_task(), external_task()
    rows = [diagnose(native, phases(evidence(supporting_facts(native)))),
            diagnose(native_task("failed"), None),
            diagnose(native_task("history", requires_version=True), phases([])),
            diagnose(native_task("null", gold_evidence=[], category="null_insufficient"), phases([])),
            diagnose(external, phases([]))]
    result = summarize(rows)
    assert result["tasks"] == 5
    assert result["eligible_tasks"] == 3
    assert result["measured_tasks"] == 2
    assert result["missing_tasks"] == 1
    assert result["not_applicable_tasks"] == 2
    assert result["semantic_recall"] is result["strict_task_success"] is None
    assert "proxy_coverage" not in result
    groups = result["by_basis"]["source_span_char4_proxy"]["by_category"]
    group = groups["multi_hop"]
    assert group["eligible_tasks"] == 2
    assert group["phases"]["context"]["measured_tasks"] == 1
    assert group["phases"]["context"]["missing_tasks"] == 1
    assert group["phases"]["context"]["reference_denominator"] == 2
    assert group["phases"]["context"]["proxy_coverage"] == 1.0
    assert result["by_basis"]["upstream_word4_proxy"]["by_category"]["comparison"][
        "phases"]["context"]["proxy_coverage"] == 0.0
    assert summarize([])["measured_tasks"] == 0


def test_pairing_uses_ids_and_reference_fingerprints_and_retains_failed_pairs():
    task = native_task()
    good = diagnose(task, phases(evidence(supporting_facts(task))))
    bad = diagnose(task, None)
    null = diagnose(native_task("null", gold_evidence=[]), phases([]))
    result = pair_profiles([good, null], [null, bad])
    assert result["total_pairs"] == 2
    assert result["eligible_pairs"] == 1
    assert result["complete_pairs"] == 0
    assert result["missing_pairs"] == 1
    assert result["not_applicable_pairs"] == 1
    assert len(result["pairs"]) == 2
    assert result["phases"]["context"]["missing_pairs"] == 1
    assert pair_profiles([good], [deepcopy(good)])["complete_pairs"] == 1


@pytest.mark.parametrize("change", ["duplicate", "task_set", "text", "source", "category", "eligibility"])
def test_pairing_rejects_duplicates_or_mismatched_tasks_and_references(change):
    task = native_task()
    left = [diagnose(task, phases([]))]
    right = deepcopy(left)
    if change == "duplicate":
        right *= 2
    elif change == "task_set":
        right[0]["task_id"] = "other"
    elif change == "eligibility":
        right[0]["eligible"] = False
    else:
        changed = deepcopy(task)
        if change == "text":
            changed["gold_evidence"][0]["quote"] += "changed"
        elif change == "source":
            changed["gold_evidence"][0]["source_sha256"] = "changed"
        else:
            changed["category"] = "changed"
        right = [diagnose(changed, phases([]))]
    with pytest.raises(ValueError):
        pair_profiles(left, right)


def test_diagnostics_summaries_and_pairs_never_contain_source_question_or_answer():
    task = external_task()
    task.update(question="PRIVATE QUESTION", gold_answer="PRIVATE ANSWER")
    rows = [diagnose(task, phases(evidence(supporting_facts(task))))]
    output = json.dumps({"rows": rows, "summary": summarize(rows), "pairs": pair_profiles(rows, rows)})
    for secret in ("Synthetic Article", "The first response limit", "Entire unrelated source span",
                   "PRIVATE QUESTION", "PRIVATE ANSWER"):
        assert secret not in output


def test_complete_question_atom_ignores_unrequested_disclaimer():
    atom = {'id': 'atom', 'document_id': 'a', 'source_sha256': 'sha-a',
            'basis': 'source_atom_exact_proxy', 'text': '生产数据库的 RPO 为 10 分钟。'}
    assert phase_coverage([atom], evidence([atom]))['delivered'] == 1
    assert phase_coverage([atom], evidence([atom], text='其他服务的 RPO 为 10 分钟。'))['delivered'] == 0
    assert phase_coverage([atom], evidence([atom], source_sha256='wrong'))['delivered'] == 0


def test_atom_requires_negation_and_qualifier_even_if_number_matches():
    atom = {'id': 'atom', 'document_id': 'a', 'source_sha256': 'sha-a',
            'basis': 'source_atom_exact_proxy', 'text': '不含外部故障时，生产服务响应上限为 30 分钟。'}
    assert phase_coverage([atom], evidence([atom], text='生产服务响应上限为 30 分钟。'))['delivered'] == 0


def test_scope_and_value_blocks_must_both_be_in_matching_source_chunks():
    atom = {'id': 'atom', 'document_id': 'a', 'source_sha256': 'sha-a',
            'basis': 'source_atom_exact_proxy',
            'text': '本文件适用于第四季度。\n\n## 窗口\n\n窗口为 08:00 至 18:00。'}
    rows = evidence([atom], text='窗口为 08:00 至 18:00。', rank=2)
    assert phase_coverage([atom], rows)['delivered'] == 0
    rows += evidence([atom], text='本文件适用于第四季度。', rank=5)
    actual = phase_coverage([atom], rows)
    assert actual['delivered'] == 1 and actual['first_supporting_rank'] == 5
