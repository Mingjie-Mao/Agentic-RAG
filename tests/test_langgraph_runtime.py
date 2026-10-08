"""Fault injection around real LangGraph checkpoints and application boundaries."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from agent import controller, hybrid
from agent.controller import create_task, run_task
from agent.graph_runtime import graph_checkpointer, graph_identity
from agent.tools import ToolResult
from app.clients import AgentDecision, Claim, GeneratedAnswer
from app.config import settings
from app.execution_budget import BudgetExceeded, bounded_model
from app.models import AgentEvent, AgentTask, Document, ToolExecution, User
from test_agent import agent_db


class Tools:
    def __init__(self, user):
        self.user, self.calls = user, []

    def call(self, name, args):
        self.calls.append((name, args))
        return ToolResult("ok", {"matches": [{"chunk_id": "c2", "title": "恢复政策",
            "text": "RPO 为 15 分钟。", "document_id": "doc-a"}]}, ["c2"], {})


class Models:
    def __init__(self):
        self.policy_calls = self.generations = 0

    @bounded_model("policy")
    def decide_agent_action(self, *args):
        self.policy_calls += 1
        return AgentDecision(action="search_documents", arguments={"query": "当前RPO"}, purpose="查证")

    def generate(self, question, evidence, **options):
        self.generations += 1
        assert evidence[0]["chunk_id"] == "c2"
        return GeneratedAnswer(answerable=True, claims=[Claim(text="RPO 为 15 分钟。",
            evidence_ids=["E1"], quotes=["RPO 为 15 分钟。"])]), {}


@pytest.fixture(autouse=True)
def ordinary_graph(monkeypatch):
    for name in ("task_contract_enabled", "semantic_slot_control_enabled", "semantic_coverage_control",
                 "semantic_coverage_shadow", "semantic_slot_shadow_enabled"):
        monkeypatch.setattr(settings(), name, False)


def fail_finish(monkeypatch):
    original = controller._finish
    def fail(*args, **kwargs):
        raise RuntimeError("injected-before-generation")
    monkeypatch.setattr(controller, "_finish", fail)
    return original


@pytest.mark.parametrize("mode", ["workflow", "dynamic", "hybrid"])
def test_resume_continues_pending_node_without_repeating_completed_tools(monkeypatch, mode):
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 是多少？", mode, 4)
    tools, models = Tools(user), Models()
    original = fail_finish(monkeypatch)
    with pytest.raises(RuntimeError, match="injected-before-generation"):
        run_task(db, user, task, models=models, tools=tools)
    before = (len(tools.calls), task.step_no, models.policy_calls)
    deadline = task.input["_execution_budget"]["deadline"]
    with graph_checkpointer(db) as saver:
        saved = saver.get_tuple(graph_identity(task, user))
        assert saved and saved.checkpoint["channel_values"]["refs"] == ["c2"]
    monkeypatch.setattr(controller, "_finish", original)
    run_task(db, user, task, models=models, tools=tools)
    assert task.status == "completed" and task.result["status"] == "answered"
    assert (len(tools.calls), task.step_no, models.policy_calls) == before
    assert models.generations == 1
    assert task.input["_execution_budget"]["deadline"] == deadline


@pytest.mark.parametrize("change", ["acl", "version", "disabled"])
def test_resume_rechecks_permissions_and_active_version_before_generation(monkeypatch, change):
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 是多少？", "workflow", 4)
    tools, models = Tools(user), Models()
    original = fail_finish(monkeypatch)
    with pytest.raises(RuntimeError):
        run_task(db, user, task, models=models, tools=tools)
    doc = db.get(Document, "doc-a")
    if change == "acl":
        doc.owner_id, doc.read_groups = "someone-else", ["support"]
    elif change == "version":
        doc.active_version_id = "v1"
    else:
        user.active = False
    db.commit()
    monkeypatch.setattr(controller, "_finish", original)
    with pytest.raises(HTTPException):
        run_task(db, user, task, models=models, tools=tools)
    assert models.generations == 0 and not task.result


@pytest.mark.parametrize("change", ["input", "config", "retrieval_mode", "gate_file"])
def test_changed_execution_contract_refuses_resume(monkeypatch, change, tmp_path):
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 是多少？", "workflow", 4)
    tools, models = Tools(user), Models()
    if change == "gate_file":
        gate = tmp_path / "gate.json"
        gate.write_text('{"accepted": false}')
        monkeypatch.setattr(settings(), "semantic_slot_gate_path", str(gate))
    original = fail_finish(monkeypatch)
    with pytest.raises(RuntimeError):
        run_task(db, user, task, models=models, tools=tools)
    if change == "input":
        task.input = {**task.input, "document_id": "doc-a"}
        db.commit()
    elif change == "config":
        monkeypatch.setattr(settings(), "task_contract_enabled", True)
    elif change == "retrieval_mode":
        monkeypatch.setattr(settings(), "retrieval_mode", "bm25")
    else:
        gate.write_text('{"accepted": true}')
    monkeypatch.setattr(controller, "_finish", original)
    with pytest.raises(HTTPException) as raised:
        run_task(db, user, task, models=models, tools=tools)
    assert raised.value.status_code == 409 and models.generations == 0


def test_dynamic_tool_commit_before_node_checkpoint_replays_without_new_policy(monkeypatch):
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 是多少？", "dynamic", 2)
    tools, models = Tools(user), Models()
    original = controller._execute
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected-after-tool-commit")
    monkeypatch.setattr(controller, "_execute", crash)
    with pytest.raises(RuntimeError, match="injected-after-tool-commit"):
        run_task(db, user, task, models=models, tools=tools)
    assert len(tools.calls) == task.step_no == models.policy_calls == 1
    monkeypatch.setattr(controller, "_execute", original)
    run_task(db, user, task, models=models, tools=tools)
    assert task.status == "completed" and task.result["status"] == "answered"
    assert len(tools.calls) == 1 and task.step_no == models.policy_calls == 1
    assert len(db.scalars(select(ToolExecution).where(ToolExecution.task_id == task.id)).all()) == 1


def test_hybrid_recovers_pending_subgraph_node_with_bound_intent(monkeypatch):
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 和 RTO 是多少？", "hybrid", 4)
    tools, models = Tools(user), Models()
    monkeypatch.setattr(hybrid, "evidence_ready", lambda state: any(n == "open_document" for n, _ in tools.calls))
    intents = []
    def choose(*args):
        intents.append(True)
        return hybrid.ActionIntent(kind="inspect_document", target="恢复政策")
    monkeypatch.setattr(hybrid, "choose_intent", choose)
    original = controller._execute
    def crash(db, task, tools, name, arguments, **kwargs):
        if name == "open_document":
            raise RuntimeError("injected-before-bound-tool")
        return original(db, task, tools, name, arguments, **kwargs)
    monkeypatch.setattr(controller, "_execute", crash)
    with pytest.raises(RuntimeError, match="injected-before-bound-tool"):
        run_task(db, user, task, models=models, tools=tools)
    assert len(intents) == 1 and len(tools.calls) == 1
    monkeypatch.setattr(controller, "_execute", original)
    run_task(db, user, task, models=models, tools=tools)
    assert task.status == "completed"
    assert len(intents) == 1 and [n for n, _ in tools.calls][:2] == ["search_documents", "open_document"]


def test_official_sqlite_checkpoints_survive_a_new_engine_and_session(tmp_path, monkeypatch):
    from app.models import Base
    db, user = agent_db()
    path = tmp_path / "agent.sqlite"
    disk = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(disk)
    destination = disk.raw_connection()
    try:
        with db.get_bind().connect() as source:
            source.connection.driver_connection.backup(destination.driver_connection)
    finally:
        destination.close()
    db.close()
    db = Session(disk, expire_on_commit=False)
    user = db.get(User, "user-a")
    task = create_task(db, user, "当前 RPO 是多少？", "workflow", 4)
    task_id = task.id
    tools, models = Tools(user), Models()
    original = fail_finish(monkeypatch)
    with pytest.raises(RuntimeError):
        run_task(db, user, task, models=models, tools=tools)
    before = len(tools.calls)
    db.close()
    disk.dispose()
    replacement = create_engine(f"sqlite:///{path}")
    with Session(replacement, expire_on_commit=False) as reopened:
        task, user = reopened.get(AgentTask, task_id), reopened.get(User, "user-a")
        tools.user = user
        monkeypatch.setattr(controller, "_finish", original)
        run_task(reopened, user, task, models=models, tools=tools)
        assert task.status == "completed" and len(tools.calls) == before


def test_task_identity_cannot_be_overridden_by_client_input_or_another_owner():
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO", task_input={"_langgraph_config": {"thread_id": "victim"},
                                                     "_langgraph_state": {"refs": ["private"]}})
    assert "_langgraph_config" not in task.input and "_langgraph_state" not in task.input
    assert graph_identity(task, user)["configurable"]["thread_id"] == f"tenant-a:user-a:{task.id}"
    stranger = SimpleNamespace(id="other-user", tenant_id=user.tenant_id, active=True)
    with pytest.raises(HTTPException):
        graph_identity(task, stranger)
    other_tenant = SimpleNamespace(id=user.id, tenant_id="tenant-b", active=True)
    with pytest.raises(HTTPException):
        graph_identity(task, other_tenant)


def test_checkpoint_failure_after_completed_answer_preserves_terminal_result(monkeypatch):
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 是多少？", "workflow", 4)
    tools, models = Tools(user), Models()
    original = controller._finish
    def commit_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected-after-answer-commit")
    monkeypatch.setattr(controller, "_finish", commit_then_fail)
    run_task(db, user, task, models=models, tools=tools)
    assert task.status == "completed" and task.result["status"] == "answered"
    run_task(db, user, task, models=models, tools=tools)
    assert models.generations == 1


def test_resume_cannot_reset_model_call_budget(monkeypatch):
    db, user = agent_db()
    monkeypatch.setattr(settings(), "agent_policy_max_calls", 1)
    task = create_task(db, user, "当前 RPO 以及 RTO 分别是多少？", "dynamic", 4)
    tools, models = Tools(user), Models()
    original = controller._execute
    def crash(*args, **kwargs):
        raise RuntimeError("injected-before-tool")
    monkeypatch.setattr(controller, "_execute", crash)
    with pytest.raises(RuntimeError):
        run_task(db, user, task, models=models, tools=tools)
    saved = task.input["_execution_budget"]
    assert saved["calls"]["policy"]["attempted"] == 1
    monkeypatch.setattr(controller, "_execute", original)
    with pytest.raises(BudgetExceeded, match="policy_call_budget_exhausted"):
        run_task(db, user, task, models=models, tools=tools)
    assert models.policy_calls == 1 and len(tools.calls) == 1
    assert task.input["_execution_budget"]["deadline"] == saved["deadline"]


def test_trial_reset_removes_framework_checkpoints(monkeypatch):
    from app.trial import reset_trial
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 是多少？", "workflow", 4)
    original = fail_finish(monkeypatch)
    with pytest.raises(RuntimeError):
        run_task(db, user, task, models=Models(), tools=Tools(user))
    identity = graph_identity(task, user)
    with graph_checkpointer(db) as saver:
        assert saver.get_tuple(identity)
    assert reset_trial(db, user)["tasks"] == 1
    with graph_checkpointer(db) as saver:
        assert saver.get_tuple(identity) is None
    monkeypatch.setattr(controller, "_finish", original)


def test_revoked_evidence_in_pending_hybrid_subgraph_is_not_reused(monkeypatch):
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 是多少？", "hybrid", 4)
    tools, models = Tools(user), Models()
    monkeypatch.setattr(hybrid, "evidence_ready", lambda state: False)
    def unavailable(*args):
        raise RuntimeError("injected-at-subgraph-policy")
    monkeypatch.setattr(hybrid, "choose_intent", unavailable)
    with pytest.raises(RuntimeError, match="injected-at-subgraph-policy"):
        run_task(db, user, task, models=models, tools=tools)
    document = db.get(Document, "doc-a")
    document.owner_id, document.read_groups = "someone-else", ["support"]
    db.commit()
    def never_call(*args):
        pytest.fail("cached subgraph state reached policy after permission revocation")
    monkeypatch.setattr(hybrid, "choose_intent", never_call)
    with pytest.raises(HTTPException):
        run_task(db, user, task, models=models, tools=tools)
    assert not task.result and models.generations == 0 and len(tools.calls) == 1


def test_checkpoint_write_failure_blocks_dependent_nodes_and_can_be_resumed(monkeypatch):
    monkeypatch.setattr(settings(), "adaptive_routing_enabled", False)
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 是多少？", "workflow", 4)
    tools, models = Tools(user), Models()
    with graph_checkpointer(db) as saver:
        original = saver.put
        def unavailable(config, checkpoint, metadata, new_versions):
            if checkpoint["channel_values"].get("workflow_search"):
                raise RuntimeError("injected-checkpoint-write-failure")
            return original(config, checkpoint, metadata, new_versions)
        monkeypatch.setattr(saver, "put", unavailable)
        with pytest.raises(RuntimeError, match="injected-checkpoint-write-failure"):
            run_task(db, user, task, models=models, tools=tools)
        assert len(tools.calls) == 1 and models.generations == 0 and not task.result
        events = db.scalars(select(AgentEvent).where(AgentEvent.task_id == task.id,
                                                    AgentEvent.event_type == "subgoal_state")).all()
        assert all(event.payload["phase"] != "retrieval" for event in events)
        monkeypatch.setattr(saver, "put", original)
    run_task(db, user, task, models=models, tools=tools)
    assert task.status == "completed" and task.result["status"] == "answered"
    assert [name for name, _ in tools.calls] == ["search_documents", "retrieve_evidence"]
