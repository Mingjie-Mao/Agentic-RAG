"""Behavior checks for the isolated, authorized-evidence business policy."""

from copy import deepcopy
from collections import Counter
import json

import pytest

from app.chunking import token_count
from app.evidence_selection import (
    SelectionResult,
    _Pool,
    _caps,
    replace_weak_evidence,
    select_complementary,
    selection_facets,
)


QUESTION = "What is the timeout; what is the retry limit?"


def row(cid, body, rank=1, doc="a", **extra):
    return dict(chunk_id=cid, document_id=doc, version_id=doc + "-v1",
                source_sha256=doc + "-sha", title="Alpha", text=body,
                rank=rank, lane_type="global", lane_sha256="global", **extra)


def ids(result):
    assert isinstance(result, SelectionResult)
    return [item["chunk_id"] for item in result.evidence]


def test_complementary_facets_beat_duplicate_background():
    candidates = [row("a1", "The timeout is 30 seconds.", 1),
                  row("a2", "The timeout is 30 seconds.", 2),
                  row("b1", "The retry limit is 3 attempts.", 3, "b")]
    assert ids(select_complementary(QUESTION, candidates, limit=2)) == ["a1", "b1"]


def test_explicit_two_facets_are_complementary():
    question = "What is the timeout, and what is the retry limit?"
    assert len(selection_facets(question)) == 2
    candidates = [row("a1", "The timeout is 30 seconds.", 1),
                  row("a2", "The timeout is 30 seconds.", 2),
                  row("b1", "The retry limit is 3 attempts.", 3, "b")]
    result = select_complementary(question, candidates, limit=2)
    assert ids(result) == ["a1", "b1"]
    assert result.trace["objective"][0] == 2


def test_facets_choose_longer_and_subgoals_win_ties(monkeypatch):
    monkeypatch.setattr("app.evidence_selection.subgoals", lambda q: ["first", "second"])
    monkeypatch.setattr("app.evidence_selection.question_slots", lambda q: ["other", "else"])
    assert selection_facets("q") == ["first", "second"]
    monkeypatch.setattr("app.evidence_selection.question_slots", lambda q: list(map(str, range(9))))
    assert selection_facets("q") == list(map(str, range(6)))


@pytest.mark.parametrize("question", ["What is the timeout?", "", "the and is"])
def test_single_facet_and_no_lexical_signal_fall_back_to_rank(question):
    candidates = [row("b", "Unrelated scenery mountains.", 2, "b"),
                  row("a", "Unrelated scenery mountains.", 1)]
    assert ids(select_complementary(question, candidates, limit=1)) == ["a"]


def test_document_quota_blocks_third_critical_chunk_and_none_relaxes():
    candidates = [row("a1", "The timeout is 30 seconds.", 1),
                  row("a2", "The timeout is 40 seconds and handles connect.", 2),
                  row("a3", "The retry limit is 3 attempts.", 3)]
    strict = select_complementary("", candidates)
    assert ids(strict) == ["a1", "a2"]
    assert ids(select_complementary("", candidates, document_quota=None)) == ["a1", "a2", "a3"]


def test_source_quota_is_independent_and_global_is_uncapped():
    candidates = [dict(row(str(i), "timeout retry values", i, str(i)),
                       lane_type="source", lane_sha256="lane-secret") for i in range(1, 4)]
    assert len(select_complementary("", candidates, source_quota=1).evidence) == 1
    assert len(select_complementary("", candidates, source_quota=2).evidence) == 2
    assert len(select_complementary("", [dict(c, lane_type="global") for c in candidates],
                                   source_quota=1).evidence) == 3


def test_duplicate_real_global_alternative_rescues_full_source_lane():
    a = dict(row("a", "Timeout value is 30 seconds.", 1), lane_type="source", lane_sha256="lane")
    b = dict(row("b", "Retry limit is 3 attempts.", 2, "b"),
             lane_type="source", lane_sha256="lane")
    result = select_complementary(QUESTION, [a, b, dict(b, rank=4, lane_type="global")],
                                  source_quota=1, limit=2)
    assert ids(result) == ["a", "b"]
    assert result.evidence[1]["lane_type"] == "global"
    assert len(select_complementary(QUESTION, [a, b], source_quota=1).evidence) == 1


def test_stable_ids_and_distinct_documents_and_versions_are_not_content_deduped():
    candidates = [row("c", "same words", 1, "c"), row("a", "same words", 1),
                  row("b", "same words", 1, "b")]
    candidates.append(dict(candidates[0]))
    assert ids(select_complementary("", candidates)) == ["a", "b", "c"]
    versions = [row("v1", "same words"), dict(row("v2", "same words"), version_id="a-v2")]
    assert ids(select_complementary("", versions)) == ["v1", "v2"]


def test_oversize_skipped_and_cost_includes_title_and_overhead():
    small = row("small", "The timeout is 30 seconds.", 2)
    cost = token_count(small["text"]) + token_count(small["title"]) + 100
    large = row("large", "large " * 300, 1)
    assert ids(select_complementary(QUESTION, [large, small], token_budget=cost)) == ["small"]
    assert select_complementary(QUESTION, [small], token_budget=cost - 1).context_tokens == 0
    assert select_complementary(QUESTION, [small], token_budget=cost).context_tokens == cost


def test_selection_and_replacement_do_not_mutate_inputs():
    candidates = [row("a", "Timeout is 30 seconds."), row("b", "Retry limit is 3 attempts.", 2, "b")]
    snapshot = deepcopy(candidates)
    result = select_complementary(QUESTION, candidates, limit=1)
    seed = deepcopy(result.evidence)
    replacement = replace_weak_evidence(QUESTION, candidates, seed, limit=2)
    assert candidates == snapshot and seed == result.evidence
    result.evidence[0]["text"] = "changed"
    replacement.evidence[0]["title"] = "changed"
    assert candidates == snapshot


@pytest.mark.parametrize("field,value", [("document_id", "different"), ("version_id", "v2"),
                                         ("source_sha256", "another-sha"), ("text", "different text")])
def test_conflicting_duplicate_identity_raises_sanitized_error(field, value):
    original = row("private-id", "private text")
    with pytest.raises(ValueError) as error:
        select_complementary("private question", [original, dict(original, **{field: value})])
    assert "private" not in str(error.value)


@pytest.mark.parametrize("reason", ["boilerplate_only", "below_min_similarity",
                                    "outside_requested_publication_dates", "tenant_scope"])
def test_hard_exclusions_survive_relaxation(reason):
    blocked = row("blocked", "The timeout is 30 seconds.", excluded_because=reason)
    assert ids(select_complementary(QUESTION, [blocked], document_quota=None)) == []
    with pytest.raises(ValueError):
        select_complementary(QUESTION, [blocked], required_ids=["blocked"])


@pytest.mark.parametrize("reason", ["document_quota", "source_quota", "top_k_full",
                                    "context_budget_exhausted"])
def test_soft_exclusions_are_reconsidered_under_actual_caps(reason):
    candidate = row("a", "The timeout is 30 seconds.", excluded_because=reason)
    assert ids(select_complementary(QUESTION, [candidate])) == ["a"]


def test_required_chunks_reserved_and_never_swapped_out():
    candidates = [row("required", "Unrelated but authorized material.", 9),
                  row("a", "Timeout is 30 seconds.", 1, "b"),
                  row("b", "Retry limit is 3 attempts.", 2, "c")]
    seed = select_complementary(QUESTION, candidates, limit=2, required_ids=["required"])
    assert "required" in ids(seed)
    result = replace_weak_evidence(QUESTION, candidates, seed.evidence, limit=2,
                                   required_ids=["required"])
    assert "required" in ids(result)


def test_required_lane_assignment_reserves_scarce_real_lane():
    a = dict(row("a", "A material", 1), lane_type="source", lane_sha256="lane")
    b = dict(row("b", "B material", 2, "b"), lane_type="source", lane_sha256="lane")
    result = select_complementary("", [a, b, dict(a, rank=3, lane_type="global")],
                                  required_ids=["a", "b"], source_quota=1)
    assert set(ids(result)) == {"a", "b"}
    assert next(r for r in result.evidence if r["chunk_id"] == "a")["lane_type"] == "global"


@pytest.mark.parametrize("required,caps", [(["missing"], {}), (["a", "b"], {"limit": 1}),
                                           (["a"], {"token_budget": 1}),
                                           (["a", "b"], {"document_quota": 1})])
def test_missing_or_infeasible_required_chunks_raise(required, caps):
    candidates = [row("a", "Timeout is 30 seconds."), row("b", "Retry limit is 3 attempts.", 2)]
    with pytest.raises(ValueError):
        select_complementary(QUESTION, candidates, required_ids=required, **caps)


@pytest.mark.parametrize("caps", [{"limit": 0}, {"limit": 9}, {"token_budget": 0},
                                   {"token_budget": 5001}, {"document_quota": 0},
                                   {"source_quota": -1}, {"limit": True}])
def test_caps_validated(caps):
    with pytest.raises(ValueError):
        select_complementary("", [], **caps)


def test_replacement_improves_actual_one_for_one_under_document_cap():
    # Greedy consumes both slots in doc a, preventing its third row. Replacing
    # duplicate a2 with b1 frees a slot but improves lexical coverage itself.
    candidates = [row("a1", "Timeout is 30 seconds.", 1),
                  row("a2", "Timeout is 30 seconds.", 2),
                  row("b1", "Retry limit is 3 attempts.", 3, "b")]
    seed = candidates[:2]
    result = replace_weak_evidence(QUESTION, candidates, seed, limit=2, max_replacements=1)
    assert ids(result) == ["a1", "b1"]
    assert result.trace["accepted_replacements"] == 1
    assert result.trace["changes"][0]["removed_ids"] == ["a2"]
    assert result.trace["objective_after"] > result.trace["objective_before"]


def test_replacement_improves_the_actual_strict_greedy_seed():
    question = "What is alpha, and what is bravo, and what is charlie, and what is delta?"
    candidates = [dict(row("a1", "alpha bravo value: 1.", 1), title="Configuration"),
                  dict(row("a2", "alpha charlie value: 2.", 2), title="Configuration"),
                  dict(row("a3", "bravo delta value: 3.", 3), title="Configuration")]
    seed = select_complementary(question, candidates, limit=2)
    assert ids(seed) == ["a1", "a2"]
    result = replace_weak_evidence(question, candidates, seed.evidence, limit=2)
    assert ids(result) == ["a2", "a3"]
    assert result.trace["objective_before"][0] == 3
    assert result.trace["objective_after"][0] == 4
    assert result.trace["accepted_replacements"] == 1
    repeated = replace_weak_evidence(question, candidates, result.evidence, limit=2)
    assert ids(repeated) == ids(result) and repeated.trace["accepted_replacements"] == 0


def test_content_redundancy_breaks_equal_new_coverage_and_terms():
    candidates = [row("a", "timeout value is 30 seconds.", 1),
                  row("b", "timeout value is 30 seconds.", 2, "b"),
                  row("c", "timeout configuration expires after half a minute.", 3, "c")]
    assert ids(select_complementary("What is the timeout?", candidates, limit=2)) == ["a", "c"]


def test_replacement_rejects_conflicting_or_infeasible_seed_without_mutation():
    candidate = row("a", "timeout value is 30 seconds.")
    with pytest.raises(ValueError, match="Conflicting seed identity"):
        replace_weak_evidence(QUESTION, [candidate], [dict(candidate, text="unauthorized")])
    with pytest.raises(ValueError, match="Infeasible seed evidence"):
        replace_weak_evidence(QUESTION, [candidate], [candidate], token_budget=1)
    with pytest.raises(ValueError, match="Invalid seed evidence"):
        replace_weak_evidence(QUESTION, [candidate], [candidate, candidate])


@pytest.mark.parametrize("maximum", [-1, 9, True])
def test_replacement_count_must_be_bounded(maximum):
    with pytest.raises(ValueError, match="Invalid max_replacements"):
        replace_weak_evidence("", [], [], max_replacements=maximum)


def test_replacement_respects_source_and_budget_caps_and_no_improvement_stops():
    a = dict(row("a", "Timeout is 30 seconds.", 1), lane_type="source", lane_sha256="lane")
    b = dict(row("b", "Retry limit is 3 attempts.", 2, "b"),
             lane_type="source", lane_sha256="lane")
    result = replace_weak_evidence(QUESTION, [a, b], [a], source_quota=1, required_ids=["a"])
    assert ids(result) == ["a"] and result.trace["accepted_replacements"] == 0
    cost = token_count(a["text"]) + token_count(a["title"]) + 100
    result = replace_weak_evidence(QUESTION, [a, row("b", "Retry limit is 3 attempts.", 2, "b")],
                                   [a], token_budget=cost, required_ids=["a"])
    assert result.context_tokens <= cost
    again = replace_weak_evidence(QUESTION, [a, b], result.evidence, source_quota=1,
                                  required_ids=["a"])
    assert again.trace["accepted_replacements"] == 0


def test_replacement_append_is_bounded_and_zero_stops():
    candidates = [row("a", "Timeout is 30 seconds.", 1),
                  row("b", "Retry limit is 3 attempts.", 2, "b")]
    assert ids(replace_weak_evidence(QUESTION, candidates, [], max_replacements=0)) == []
    result = replace_weak_evidence(QUESTION, candidates, [], max_replacements=1)
    assert len(result.evidence) == 1 and result.trace["accepted_replacements"] == 1
    result = replace_weak_evidence(QUESTION, candidates, [], max_replacements=8)
    assert len(result.evidence) == 2 and result.trace["accepted_replacements"] == 2


def test_trace_contains_ids_hashes_and_counts_but_no_source_material():
    candidate = dict(row("a", "Secret timeout text is 30 seconds.", doc="private-document"),
                     title="Private title", lane_type="source", lane_sha256="lane-secret")
    result = select_complementary(QUESTION, [candidate])
    trace = json.dumps(result.trace)
    for secret in [QUESTION, candidate["text"], candidate["title"], candidate["document_id"],
                   candidate["source_sha256"], candidate["lane_sha256"]]:
        assert secret not in trace
    assert result.trace["coverage_type"] == "lexical_candidate_only"


def test_empty_selection_is_stable():
    assert select_complementary("", []) == select_complementary("", [])
    assert ids(replace_weak_evidence("", [], [])) == []


def test_equivalent_lane_options_keep_earliest_rank_and_stable_equal_rank():
    original = dict(row("a", "Authorized text.", 3), lane_type="source", lane_sha256="lane")
    earlier = dict(original, rank=1, rerank_score=0.9)
    equal = dict(earlier, rerank_score=0.8)
    distinct = dict(original, lane_sha256="other-lane")
    global_option = dict(original, lane_type="global")
    pool = _Pool("", [original, earlier, equal, distinct, global_option], _caps(8, 5000, 2, 1))
    assert len(pool.options["a"]) == 3
    assert pool.options["a"][0] == earlier
    assert {(option["lane_type"], option["lane_sha256"]) for option in pool.options["a"]} == {
        ("source", "lane"), ("source", "other-lane"), ("global", "lane"),
    }


@pytest.mark.parametrize("repetitions", [1, 7])
def test_infeasible_many_lane_assignment_explores_each_failed_state_once(monkeypatch, repetitions):
    # Eight chunks compete for seven lanes. Branch-count instrumentation fails
    # fast without wall-clock thresholds, even if equivalent options repeat.
    candidates = [dict(row(str(chunk), "Authorized text.", chunk * 100 + lane, str(chunk)),
                       lane_type="source", lane_sha256=f"lane-{lane}")
                  for chunk in range(8) for lane in range(7) for _ in range(repetitions)]
    pool = _Pool("", candidates, _caps(8, 5000, 2, 1))
    accesses = 0

    class BoundedCounter(Counter):
        def __getitem__(self, key):
            nonlocal accesses
            accesses += 1
            assert accesses <= 2500, "Equivalent failed lane states were expanded repeatedly"
            return super().__getitem__(key)

    monkeypatch.setattr("app.evidence_selection.Counter", BoundedCounter)
    assigned, reason = pool.assign(list(map(str, range(8))))
    assert assigned is None and reason == "source_quota"
    assert accesses <= 2500


def test_many_repeated_options_preserve_feasible_global_fallback():
    candidates = [dict(row(str(chunk), "Authorized text.", chunk * 100 + lane, str(chunk)),
                       lane_type="source", lane_sha256=f"lane-{lane}")
                  for chunk in range(8) for lane in range(7) for _ in range(7)]
    candidates.append(dict(candidates[-1], lane_type="global", rank=999))
    result = select_complementary("", candidates, source_quota=1,
                                  required_ids=list(map(str, range(8))))
    assert ids(result) == list(map(str, range(8)))
    assert result.evidence[-1]["lane_type"] == "global"
    assert [item["lane_sha256"] for item in result.evidence[:-1]] == [f"lane-{i}" for i in range(7)]
