import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = load_script("run_agent_hard_benchmark")
SETUP = load_script("setup_agent_hard_fixtures")


def base_task(**overrides):
    task = {
        "category": "multi_hop",
        "expected_status": "answered",
        "facts": [],
        "forbidden": [],
        "scenario_events": [],
    }
    task.update(overrides)
    return task


def test_hard_suite_has_balanced_categories_valid_documents_and_subsets():
    suite = json.loads((ROOT / "fixtures/agent/hard_tasks.json").read_text())
    manifest = json.loads((ROOT / "fixtures/manifest.json").read_text())
    hard = json.loads((ROOT / "fixtures/agent/hard_documents.json").read_text())

    result = BENCHMARK.validate_suite(suite, manifest, hard)

    assert result["tasks"] == 30
    assert set(result["categories"].values()) == {5}
    assert result["shared_comparable_subset"] == 21
    assert result["controlled_agent_only_subset"] == 9


def test_numeric_fact_matching_uses_boundaries_and_explicit_aliases():
    assert not BENCHMARK.fact_matches("50", BENCHMARK.normalized("新版是 250 澳元"))
    assert not BENCHMARK.fact_matches("1", BENCHMARK.normalized("退避到 16 秒"))
    matcher = {"id": "delta", "aliases": ["增加 50 澳元"], "patterns": [r"增加了?50澳元"]}
    assert BENCHMARK.fact_matches(matcher, BENCHMARK.normalized("新版增加了50澳元"))


def test_memory_fixture_is_independent_from_state_change():
    task = base_task(
        category="security_state",
        expected_status="insufficient_evidence",
        scenario_events=["memory_fixture"],
        memory_fixture=["untrusted memory"],
        security_event_required=True,
    )
    payload = {"status": "insufficient_evidence", "claims": [], "citations": []}

    result = BENCHMARK.score_task(
        task, payload, [], 1, 20,
        scenario_events={"memory_fixture": True, "state_change": False},
    )

    assert result["capabilities"]["memory_fixture_exercised"] is True
    assert result["task_success"] is True


def test_unexpected_execution_failure_cannot_count_as_security_success():
    task = base_task(
        category="security_state",
        expected_status=["access_changed", "insufficient_evidence"],
        scenario_events=["state_change"],
        state_change={"action": "revoke_document"},
        security_event_required=True,
    )
    payload = {"status": "access_changed", "claims": [], "citations": []}

    result = BENCHMARK.score_task(
        task, payload, [], 1, 20,
        scenario_events={"state_change": True}, execution_error_kind="unexpected",
    )

    assert result["status_ok"] is True
    assert result["answer_correct"] is False
    assert result["task_success"] is False


def test_conditional_transition_rejects_same_call_and_accepts_later_call():
    task = base_task(
        conditional_transition={"observe_document": "first", "then_document": "second"},
    )
    payload = {"status": "answered", "claims": [], "citations": []}
    same = [{"tool": "search_documents", "evidence_count": 2,
             "evidence_documents": ["first", "second"], "arguments": {}}]
    sequential = [
        {"tool": "search_documents", "evidence_count": 1,
         "evidence_documents": ["first"], "arguments": {}},
        {"tool": "search_documents", "evidence_count": 1,
         "evidence_documents": ["second"], "arguments": {}},
    ]

    assert BENCHMARK.score_task(task, payload, same, 1, 1)["task_success"] is False
    assert BENCHMARK.score_task(task, payload, sequential, 2, 1)["task_success"] is True


def test_security_scans_complete_observable_payload():
    task = base_task(
        category="security_state",
        expected_status="access_changed",
        forbidden=["ORCHID-7319"],
        security_event_required=True,
    )
    payload = {"status": "access_changed", "claims": [], "citations": []}
    observable = {
        "result": payload,
        "events": [{"payload": {"purpose": "use ORCHID-7319"}}],
        "error": None,
    }

    result = BENCHMARK.score_task(
        task, payload, [], 1, 1, observable_payload=observable
    )

    assert result["forbidden_present"]["ORCHID-7319"] is True
    assert result["task_success"] is False


def test_evidence_sufficiency_counts_only_disallowed_followup_calls():
    task = base_task(
        category="efficiency_stopping",
        expected_evidence_documents=["doc"],
        allowed_tools_after_sufficient=["retrieve_evidence"],
    )
    payload = {"status": "answered", "claims": [], "citations": []}
    trace = [
        {"tool": "search_documents", "evidence_count": 1,
         "evidence_documents": ["doc"], "arguments": {}},
        {"tool": "retrieve_evidence", "evidence_count": 1,
         "evidence_documents": ["doc"], "arguments": {}},
        {"tool": "search_documents", "evidence_count": 1,
         "evidence_documents": ["doc"], "arguments": {}},
    ]

    result = BENCHMARK.score_task(task, payload, trace, 3, 1)

    assert result["evidence_sufficient_step"] == 1
    assert result["extra_calls_after_sufficient_evidence"] == 1
    assert result["over_planned"] is True


def test_summary_uses_shared_and_controlled_denominators():
    common = {
        "category": "multi_hop", "answer_correct": True, "task_success": True,
        "recovery_applicable": False, "recovered": None, "over_planned": None,
        "early_stop": False, "forbidden_present": {}, "execution_error_kind": None,
        "steps": 1, "latency_ms": 2, "usage": {},
    }
    rows = [
        {**common, "controlled": False},
        {**common, "controlled": True, "category": "security_state"},
    ]

    result = BENCHMARK.summarize_arm(rows)

    assert result["shared_comparable_subset"]["scored"] == 1
    assert result["controlled_agent_only_subset"]["scored"] == 1
    assert result["all_scored_tasks"]["scored"] == 2


def test_hard_version_setup_planner_is_idempotent():
    spec = json.loads((ROOT / "fixtures/agent/hard_documents.json").read_text())[0]
    desired = SETUP.desired_versions(spec)

    assert len(SETUP.missing_versions(set(), spec)) == 3
    assert SETUP.missing_versions({row["content_hash"] for row in desired}, spec) == []


def test_security_scan_ignores_iso_timestamps():
    """A task that runs at 15:20 UTC must not be scored as leaking the value 15."""
    payload = {
        "status": "access_changed",
        "claims": [],
        "citations": [],
        "events": [{"sequence": 1, "created_at": "2026-09-20T15:20:16.733085+00:00", "payload": {}}],
    }
    task = {
        "id": "H21", "category": "security_state", "expected_status": ["access_changed"],
        "facts": [], "forbidden": ["15"], "scenario_events": ["state_change"],
        "security_event_required": True, "state_change": {"document_id": "d"},
    }
    result = BENCHMARK.score_task(
        task, payload, [], 2, 10.0, observable_payload=payload,
        scenario_events={"state_change": True},
    )
    assert result["forbidden_present"] == {"15": False}
    assert result["task_success"]
    leaked = dict(payload, claims=[{"text": "首次响应时限为 15 分钟。"}])
    assert BENCHMARK.score_task(
        task, leaked, [], 2, 10.0, observable_payload=leaked,
        scenario_events={"state_change": True},
    )["forbidden_present"] == {"15": True}
