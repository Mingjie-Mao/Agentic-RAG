"""Real worker synchronization proves optional shadow never waits on the judge."""
from threading import Event

from app import slot_shadow
from app.config import settings
from app.execution_budget import ExecutionBudget, current_budget, use_budget


def test_dispatch_returns_while_observer_is_blocked_and_queue_is_bounded(monkeypatch):
    started, release, returned = Event(), Event(), Event()
    seen = []
    def observe(*args):
        seen.append((args, current_budget()))
        started.set()
        assert release.wait(5)
    monkeypatch.setattr(slot_shadow, "observe_completed", observe)
    dispatcher = slot_shadow.ShadowDispatcher()
    try:
        assert dispatcher.submit("agent", "first", "user", 1)["status"] == "queued"
        returned.set()
        assert started.wait(5) and returned.is_set()
        assert dispatcher.submit("agent", "first", "user", 1)["reason"] == "already_pending"
        assert dispatcher.submit("answer", "waiting", "user", 1)["status"] == "queued"
        assert dispatcher.submit("answer", "overflow", "user", 1)["reason"] == "queue_full"
    finally:
        release.set()
        dispatcher.queue.join()
    assert [r[0][1] for r in seen] == ["first", "waiting"]
    assert all(budget is None for _, budget in seen)
    assert not dispatcher.pending


def test_no_primary_budget_transfer_and_zero_sampling_never_dispatches(monkeypatch):
    dispatcher = slot_shadow.ShadowDispatcher()
    with use_budget(ExecutionBudget(settings())):
        assert dispatcher.submit("agent", "a", "user", 1)["reason"] == "primary_budget_active"
    assert dispatcher.submit("agent", "a", "user", 0)["reason"] == "not_sampled"
    assert dispatcher.worker is None


def test_observer_failure_does_not_kill_worker_or_hold_pending_key(monkeypatch):
    done = Event()
    def fail(*args):
        done.set()
        raise RuntimeError("diagnostic failure")
    monkeypatch.setattr(slot_shadow, "observe_completed", fail)
    dispatcher = slot_shadow.ShadowDispatcher()
    assert dispatcher.submit("agent", "a", "user", 1)["status"] == "queued"
    assert done.wait(5)
    dispatcher.queue.join()
    assert dispatcher.worker.is_alive() and not dispatcher.pending
    assert dispatcher.submit("agent", "a", "user", 1)["status"] == "queued"
    dispatcher.queue.join()


def test_unknown_condition_does_not_hide_pending_branches_as_inactive():
    from app.task_contract import build_contract
    from app.semantic_evidence import EvidenceJudgment, aggregate_slots
    contract = build_contract("先查RPO，如果它超过30分钟，再给出回滚超时。")
    items = slot_shadow.observation_items(contract, {"usage": {"task_contract": {
        "condition": {"result": "unknown", "active_slot_ids": ["operand"]}}}})
    report = aggregate_slots(items, [EvidenceJudgment(subgoal_id="operand", status="supported")])
    assert report["unknown"] == ["true_1"] and not report["inactive"] and not report["ready"]


def test_missing_version_chain_is_unknown_even_when_current_value_is_present():
    from app.task_contract import build_contract
    from app.semantic_evidence import aggregate_slots
    contract = build_contract("操作日志保留时间各版本有没有变化？")
    assert contract.intent == "history"
    items = slot_shadow.observation_items(contract, {})
    assert all(i["prerequisite_pending"] for i in items)
    assert aggregate_slots(items, [])["unknown"] == ["history"]
