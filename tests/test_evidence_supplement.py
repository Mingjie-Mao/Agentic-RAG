"""The supplement experiment shares a literal bridge, never a task oracle."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from app.execution_budget import ExecutionBudget
from agent.evidence_supplement import run_supplement


QUESTION = "What is order A-1 carrier phone?"


def row(cid="seed", text="Order A-1 carrier Acme. Phone unknown.", **extra):
    return dict(chunk_id=cid, document_id=cid, version_id=cid + "-v1",
                source_sha256=cid + "-sha", text=text, title="Orders",
                parent_rank=1, lane_type="global", lane_sha256="global", **extra)


def plan():
    return {"steps": [
        {"id": "s1", "purpose": "Find carrier", "query": "A-1 carrier",
         "extract": {"name": "carrier", "kind": "entity", "description": "A-1 carrier"}},
        {"id": "s2", "purpose": "Find phone", "query": "{s1.carrier} phone",
         "depends_on": ["s1"]},
    ]}


class Models:
    def __init__(self, planning=None, entity="Acme", fail=None):
        self.planning = plan() if planning is None else planning
        self.entity, self.fail, self.bodies = entity, fail, []

    def _post(self, path, body, **kwargs):
        self.bodies.append(deepcopy(body))
        if self.fail == len(self.bodies):
            raise RuntimeError("private backend failure")
        value = (self.planning if isinstance(self.planning, str) else json.dumps(self.planning))
        return {"message": {"content": value if len(self.bodies) == 1 else self.entity}}


def budget(**limits):
    return ExecutionBudget(SimpleNamespace(agent_task_timeout_seconds=180,
        agent_policy_max_calls=limits.get("policy", 2), agent_judge_max_calls=0,
        agent_generation_max_calls=0, agent_judge_token_budget=0))


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr("agent.evidence_supplement.evidence_coverage",
                        lambda facets, rows, question: {f: [] for f in facets})
    def invoke(**kwargs):
        seen = []
        def search(args):
            seen.append(args)
            return [row("fresh", "Acme phone is 555.", rerank_score=None)]
        model = kwargs.pop("models", Models())
        allocation = kwargs.pop("budget", budget())
        result = run_supplement(kwargs.pop("question", QUESTION), kwargs.pop("seed", [row()]),
            kwargs.pop("candidates", [row()]), models=model, budget=allocation,
            reauthorize=kwargs.pop("reauthorize", deepcopy),
            search=kwargs.pop("search", search),
            validate_scope=kwargs.pop("validate_scope", lambda original, query: True), **kwargs)
        return result, model, allocation, seen
    return invoke


def test_gate_false_carries_baseline_without_calls(run, monkeypatch):
    monkeypatch.setattr("agent.evidence_supplement.evidence_coverage",
                        lambda facets, rows, question: {f: ["seed"] for f in facets})
    result, model, allocation, seen = run()
    assert result.trace["status"] == "gate_false"
    assert not model.bodies and not seen and not allocation.state["calls"]
    assert result.arms["control"]["evidence"][0]["chunk_id"] == "seed"


def test_valid_pair_has_shared_two_policy_calls_and_required_source(run):
    result, model, allocation, seen = run()
    assert result.trace["gates"]["missing_facets"]
    assert [args.query for args in seen] == [QUESTION + "\nphone", QUESTION + "\nAcme phone"]
    assert all(args.top_k == 8 for args in seen)
    assert [b["options"]["num_predict"] for b in model.bodies] == [300, 60]
    assert allocation.state["calls"]["policy"]["attempted"] == 2
    assert allocation.state["calls"]["search"]["attempted"] == 2
    assert all(arm["status"] == "selected" for arm in result.arms.values())
    assert all("seed" in [r["chunk_id"] for r in arm["evidence"]]
               for arm in result.arms.values())


@pytest.mark.parametrize("entity", ["Imaginary", "123", "", "{s9.unknown}"])
def test_unusable_entity_is_legitimate_skip(run, entity):
    result, _, _, seen = run(models=Models(entity=entity))
    assert result.trace["status"] == "no_valid_bridge"
    assert not seen
    assert all(arm["evidence"] is not None for arm in result.arms.values())


@pytest.mark.parametrize("change", ["unknown", "foreach", "history", "extract2", "three",
                                    "duplicate", "list", "nodeps", "twice", "bare"])
def test_rejects_unsupported_shape_before_plan_repairs(run, change):
    p = plan()
    if change == "unknown":
        p["steps"][1]["query"] = "{s1.unknown} phone"
    elif change == "foreach":
        p["steps"][1]["foreach"] = True
    elif change == "history":
        p["steps"][1].update(as_of="2020-01-01", when="at")
    elif change == "extract2":
        p["steps"][1]["extract"] = p["steps"][0]["extract"]
    elif change == "three":
        p["steps"].append(p["steps"][1])
    elif change == "duplicate":
        p["steps"][1]["id"] = "s1"
    elif change == "list":
        p["steps"][0]["extract"]["kind"] = "list"
    elif change == "nodeps":
        p["steps"][1]["depends_on"] = []
    elif change == "twice":
        p["steps"][1]["query"] += " {s1.carrier}"
    else:
        p["steps"][1]["query"] = "{oops} phone"
    result, model, _, seen = run(models=Models(planning=p))
    assert result.trace["status"] == "no_valid_bridge"
    assert len(model.bodies) == 1 and not seen


def test_invalid_json_and_backend_failure_are_distinct(run):
    invalid, _, allocation, _ = run(models=Models(planning="not JSON"))
    failed, _, failed_budget, _ = run(models=Models(fail=1))
    assert invalid.trace["status"] == "no_valid_bridge"
    assert failed.trace["status"] == "operational_failure"
    assert failed.arms["bridge"]["evidence"] is None
    assert allocation.state["calls"]["policy"]["failed"] == 1
    assert failed_budget.state["calls"]["policy"]["failed"] == 1


def test_scope_validation_checks_both_before_dispatch(run):
    scopes = []
    def scope(original, query):
        scopes.append((original, query))
        return "Acme" not in query
    result, _, _, seen = run(validate_scope=scope)
    assert result.trace["status"] == "scope_mismatch"
    assert len(scopes) == 2 and all(original == QUESTION for original, _ in scopes)
    assert not seen


def test_oversize_original_is_never_truncated(run):
    result, _, _, seen = run(question=QUESTION + "x" * 500)
    assert result.trace["status"] == "query_too_long" and not seen


@pytest.mark.parametrize("call", [1, 2, 3, 4, 5, 6, 7])
def test_revocation_or_version_change_is_not_silently_repaired(run, call):
    reads = 0
    def auth(rows):
        nonlocal reads
        reads += 1
        return [dict(r, version_id="new-private-version") for r in rows] if reads == call else deepcopy(rows)
    result, model, _, _ = run(reauthorize=auth)
    assert any(a["status"] == "failed" for a in result.arms.values())
    if call <= 3:
        assert len(model.bodies) < call


def test_revoked_seed_omission_fails_before_any_model(run):
    result, model, _, seen = run(reauthorize=lambda rows: [])
    assert result.trace["status"] == "operational_failure"
    assert not seen and not model.bodies


def test_rotation_and_failed_attempt_do_not_retry_or_block_other_arm(run):
    seen = []
    def search(args):
        seen.append(args.query)
        if len(seen) == 1:
            raise RuntimeError("private network data")
        return [row("fresh", "Phone is 555.")]
    result, _, allocation, _ = run(order=("bridge", "control"), search=search)
    assert seen == [QUESTION + "\nAcme phone", QUESTION + "\nphone"]
    assert result.arms["bridge"]["evidence"] is None
    assert result.arms["control"]["status"] == "selected"
    assert allocation.state["calls"]["search"]["failed"] == 1
    assert allocation.state["calls"]["search"]["attempted"] == 2


def test_policy_budget_exhaustion_honest_null(run):
    result, _, allocation, seen = run(budget=budget(policy=1))
    assert result.trace["status"] == "operational_failure"
    assert allocation.state["calls"]["policy"]["attempted"] == 1 and not seen
    assert result.arms["control"]["evidence"] is None


def test_round_robin_does_not_compare_incompatible_scores(run):
    old = [row(), row("old2", "More old data", rerank_score=900)]
    fresh = [row("new1", "Fresh new data", fusion_score=0.001), row("new2", "More new data")]
    result, _, _, _ = run(candidates=old, search=lambda args: deepcopy(fresh))
    union = result.arms["bridge"]["candidates"]
    assert [(r["chunk_id"], r["parent_rank"]) for r in union] == [
        ("seed", 1), ("new1", 2), ("old2", 3), ("new2", 4)]


def test_required_source_cannot_be_hidden_ninth_or_truncated(run):
    result, _, _, _ = run(caps={"token_budget": 1})
    assert all(a["status"] == "failed" and a["evidence"] is None for a in result.arms.values())


def test_public_trace_excludes_raw_material_and_inputs_unchanged(run):
    seed, candidates = [row()], [row(), row("other", "Secret pool material")]
    before = deepcopy((seed, candidates))
    result, model, _, _ = run(seed=seed, candidates=candidates)
    trace = json.dumps(result.trace)
    assert all(s not in trace for s in [QUESTION, "Acme", "carrier", "Secret", "Orders"])
    assert result.trace["bridge_chunk_id"] == "seed"
    assert result.trace["bridge_source_sha256"] == "seed-sha"
    assert "Secret pool material" not in json.dumps(model.bodies)
    assert (seed, candidates) == before


@pytest.mark.parametrize("question,key", [("Compare both reports", "multi_source_intent"),
                                          ("分别以及查承运商", "needs_document_diversity")])
def test_intent_gate_can_fire_without_missing_facets(run, monkeypatch, question, key):
    monkeypatch.setattr("agent.evidence_supplement.evidence_coverage",
                        lambda facets, rows, question: {f: ["seed"] for f in facets})
    result, _, _, _ = run(question=question)
    assert result.trace["gates"][key]
    assert result.trace["status"] != "gate_false"


def test_focus_does_not_read_other_subjects_and_row_anchoring_rejects_wrong_value(run):
    seed = [row(text="Order A-1 carrier Acme.\nOrder B-2 carrier Other."),
            row("unrelated", "Order Z-9 carrier Unrelated.")]
    result, model, _, seen = run(seed=seed, candidates=seed, models=Models(entity="Other"))
    assert result.trace["status"] == "no_valid_bridge" and not seen
    assert "Unrelated" not in model.bodies[1]["messages"][1]["content"]


def test_rare_subject_focus_mismatch_rejects_before_extraction(run, monkeypatch):
    monkeypatch.setattr("agent.evidence_supplement.ungrounded_terms", lambda *args: ["private"])
    result, model, _, seen = run()
    assert result.trace["status"] == "no_valid_bridge" and len(model.bodies) == 1 and not seen
    assert "private" not in json.dumps(result.trace)


def test_plan_is_checked_after_existing_validator_mutates_it(run, monkeypatch):
    def mutate(question, accepted):
        accepted.steps[1].foreach = True
        return []
    monkeypatch.setattr("agent.evidence_supplement.validate_plan", mutate)
    result, model, _, seen = run()
    assert result.trace["status"] == "no_valid_bridge" and len(model.bodies) == 1 and not seen


def test_placeholder_only_allows_original_only_control(run):
    p = plan()
    p["steps"][1]["query"] = "{s1.carrier}"
    result, _, _, seen = run(models=Models(planning=p))
    assert result.trace["status"] == "completed"
    assert [a.query for a in seen] == [QUESTION, QUESTION + "\nAcme"]


def test_whitespace_normalized_literal_is_still_source_bound(run):
    seed = [row(text="Order A-1 carrier Ac me.")]
    result, _, _, _ = run(seed=seed, candidates=seed, models=Models(entity="Acme"))
    assert result.trace["status"] == "completed"


def test_source_date_and_original_query_preserved_in_both_queries(run):
    question = QUESTION + " according to BBC after 2024-01-01"
    originals = []
    def scope(original, query):
        originals.append(original)
        return query.startswith(question)
    result, _, _, seen = run(question=question, validate_scope=scope)
    assert result.trace["status"] == "completed"
    assert all(a.query.startswith(question) for a in seen) and originals == [question] * 4


def test_deadline_during_failed_first_search_blocks_second_dispatch(run):
    allocation = budget()
    seen = []
    def search(args):
        seen.append(args)
        allocation.state["deadline"] = allocation.clock() - 1
        raise RuntimeError("failed")
    result, _, _, _ = run(budget=allocation, search=search)
    assert len(seen) == 1
    assert all(a["evidence"] is None for a in result.arms.values())
    assert allocation.state["calls"]["search"]["attempted"] == 1


def test_search_budget_is_cumulative_and_not_reset(run):
    allocation = budget()
    with allocation.call("search"):
        pass
    result, _, _, seen = run(budget=allocation)
    assert len(seen) == 1 and result.arms["bridge"]["evidence"] is None
    assert allocation.state["calls"]["search"]["attempted"] == 2
    assert result.trace["calls"]["search"]["attempted"] == 1


def test_real_duplicate_lanes_are_preserved_and_conflicts_rejected(run):
    original = row()
    old = [original, dict(original, lane_type="source", lane_sha256="real-source")]
    result, _, _, _ = run(candidates=old)
    assert sum(r["chunk_id"] == "seed" for r in result.arms["bridge"]["candidates"]) == 2
    failed, _, _, _ = run(search=lambda args: [dict(original, text="Conflict private text")])
    assert all(a["status"] == "failed" for a in failed.arms.values())


def test_hard_excluded_required_source_fails_and_soft_new_rows_allowed(run):
    blocked = dict(row(), excluded_because="tenant_scope")
    failed, _, _, _ = run(candidates=[blocked])
    assert all(a["evidence"] is None for a in failed.arms.values())
    good, _, _, _ = run(search=lambda args: [row("fresh", "Acme phone 555",
                                                excluded_because="source_quota")])
    assert all("fresh" in [r["chunk_id"] for r in a["evidence"]] for a in good.arms.values())


def test_whole_seed_payload_never_truncates_inside_existing_call(run):
    seed = [row(text="carrier Acme A-1 " + "a" * 13000)]
    result, model, _, seen = run(seed=seed, candidates=seed)
    assert result.trace["status"] in {"operational_failure", "planning_payload_too_large"}
    assert not model.bodies and not seen


def test_extraction_failure_and_inactive_seed_are_operational_failures(run):
    failed, _, allocation, seen = run(models=Models(fail=2))
    assert failed.trace["status"] == "operational_failure" and not seen
    assert allocation.state["calls"]["policy"]["failed"] == 1
    inactive, model, _, _ = run(reauthorize=lambda rows: [dict(r, active=False) for r in rows])
    assert inactive.trace["status"] == "operational_failure" and not model.bodies


def test_scope_rechecked_before_second_search_after_first_arm(run):
    reads = 0
    def scope(original, query):
        nonlocal reads
        reads += 1
        return reads != 4
    result, _, allocation, seen = run(validate_scope=scope)
    assert len(seen) == 1 and reads == 4
    assert result.arms["bridge"]["status"] == "failed"
    assert result.arms["control"]["status"] == "selected"
    assert allocation.state["calls"]["search"]["attempted"] == 1


def test_invalid_seed_bounds_never_carried_as_valid_context(run):
    seed = [row(str(i)) for i in range(9)]
    result, model, _, seen = run(seed=seed, candidates=seed)
    assert result.trace["status"] == "operational_failure" and not model.bodies and not seen
    assert all(a["evidence"] is None for a in result.arms.values())


def test_long_whole_source_quote_has_no_additional_arbitrary_cap(run):
    seed = [row(text="Order A-1 carrier Acme " + "detail " * 170 + ".")]
    result, _, _, seen = run(seed=seed, candidates=seed)
    assert result.trace["status"] == "completed" and len(seen) == 2


def test_required_seed_missing_from_recorded_pool_is_explicit_arm_failure(run):
    result, _, _, _ = run(candidates=[row("old", "Old unrelated")])
    assert all(a["status"] == "failed" and a["evidence"] is None for a in result.arms.values())
    assert all(error["stage"] == "select" for error in result.trace["errors"])


def test_budget_caps_and_local_persistence_are_applied_without_reset(run):
    snapshots = []
    allocation = budget(policy=9)
    allocation.state["deadline"] = allocation.clock() + 1000
    allocation.state["limits"]["search"] = 9
    allocation.persist = snapshots.append
    result, _, _, _ = run(budget=allocation)
    assert result.trace["status"] == "completed"
    assert allocation.state["limits"]["policy"] == allocation.state["limits"]["search"] == 2
    assert allocation.state["deadline"] <= allocation.clock() + 180
    assert snapshots[-1]["calls"]["search"]["attempted"] == 2


def test_callbacks_receive_copies_and_cannot_mutate_input_snapshots(run):
    seed, old = [row()], [row()]
    snapshot = deepcopy((seed, old))
    def auth(rows):
        rows[0]["parent_rank"] = 999
        return rows
    result, _, _, _ = run(seed=seed, candidates=old, reauthorize=auth)
    assert result.trace["status"] == "completed" and (seed, old) == snapshot
    assert result.arms["bridge"]["candidates"][0]["original_parent_rank"] == 1


def test_schema_errors_do_not_reveal_model_reply(run):
    result, _, allocation, _ = run(models=Models(planning="PRIVATE_MODEL_RESPONSE"))
    assert result.trace["status"] == "no_valid_bridge"
    assert "PRIVATE_MODEL_RESPONSE" not in json.dumps(result.trace)
    assert allocation.state["calls"]["policy"]["attempted"] == 1


def test_context_churn_telemetry_measures_final_context_against_initial_seed(run):
    bridge = row()
    weak = row("weak", "Old unrelated context")
    old = [dict(bridge, parent_rank=7), row("old", "Old background"),
           dict(weak, parent_rank=99)]
    fresh = dict(row("fresh", "Acme phone is 555"), parent_rank=13)
    result, _, _, _ = run(seed=[bridge, weak], candidates=old,
                          search=lambda args: [fresh], caps={"limit": 2})
    assert result.trace["status"] == "completed"
    for arm in result.trace["arms"].values():
        assert arm["added_ids"] == ["fresh"]
        assert arm["removed_ids"] == ["weak"]
        assert arm["replacement_count"] == 1
        assert arm["algorithmic_swap_count"] == 0
        assert [(r["original_parent_rank"], r["parent_rank"], r["origin"])
                for r in arm["candidate_order"]] == [
                    (7, 1, "original"), (13, 2, "supplement"),
                    (1, 3, "original"), (99, 5, "original")]


def test_context_append_is_distinct_from_replacement(run):
    result, _, _, _ = run()
    for arm in result.trace["arms"].values():
        assert arm["added_ids"] == ["fresh"] and arm["removed_ids"] == []
        assert arm["replacement_count"] == 0


def test_invalid_original_rank_cannot_leak_into_public_provenance(run):
    bad = dict(row("fresh", "Acme phone 555"), parent_rank="PRIVATE_RANK_VALUE")
    result, _, _, _ = run(search=lambda args: [bad])
    assert all(a["status"] == "failed" for a in result.arms.values())
    assert "PRIVATE_RANK_VALUE" not in json.dumps(result.trace)


@pytest.mark.parametrize("seed", [[row(str(i)) for i in range(9)],
                                   [row(text="carrier " * 6000)]])
def test_seed_bounds_failure_precedes_oversized_query_skip(run, seed):
    result, model, _, seen = run(question="x" * 501, seed=seed, candidates=seed)
    assert result.trace["status"] == "operational_failure"
    assert not seen and not model.bodies
    assert all(a["status"] == "failed" and a["evidence"] is None for a in result.arms.values())


@pytest.mark.parametrize("rank", [0, -1, None, True, "PRIVATE_RANK_VALUE"])
def test_original_rank_provenance_requires_positive_number(run, rank):
    candidate = dict(row("fresh", "Acme phone 555"), parent_rank=rank)
    result, _, _, _ = run(search=lambda args: [candidate])
    assert all(a["status"] == "failed" for a in result.arms.values())
    assert "PRIVATE_RANK_VALUE" not in json.dumps(result.trace)


@pytest.mark.parametrize("seed,caps", [([row(), row("second")], {"limit": 1}),
                                      ([row()], {"token_budget": 1})])
def test_declared_caps_bound_gate_false_carryforward(run, monkeypatch, seed, caps):
    monkeypatch.setattr("agent.evidence_supplement.evidence_coverage",
                        lambda facets, rows, question: {f: ["seed"] for f in facets})
    result, model, _, seen = run(seed=seed, candidates=seed, caps=caps)
    assert result.trace["status"] == "operational_failure" and not model.bodies and not seen
    assert all(a["evidence"] is None for a in result.arms.values())


def test_lower_declared_count_cap_precedes_oversized_query_skip(run):
    seed = [row(str(i)) for i in range(3)]
    result, model, _, seen = run(question="x" * 501, seed=seed, candidates=seed,
                                 caps={"limit": 2})
    assert result.trace["status"] == "operational_failure" and not model.bodies and not seen
    assert all(a["evidence"] is None for a in result.arms.values())
