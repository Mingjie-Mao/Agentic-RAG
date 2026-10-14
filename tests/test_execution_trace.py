"""Requested mode, selected mode and the graph nodes a task actually ran."""

import pytest

from agent import controller
from agent.controller import create_task, run_task, task_payload
from app.config import settings
from test_agent import agent_db
from test_langgraph_runtime import Models, Tools

CONDITIONAL = "先查生产数据库的RPO，如果它超过30分钟，再给出生产数据库的RTO。"


@pytest.fixture(autouse=True)
def ordinary_graph(monkeypatch):
    for name in ("task_contract_enabled", "semantic_slot_control_enabled", "semantic_coverage_control",
                 "semantic_coverage_shadow", "semantic_slot_shadow_enabled"):
        monkeypatch.setattr(settings(), name, False)


def test_workflow_records_requested_selected_mode_and_executed_path():
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 是多少？", "workflow", 4, {"requested_mode": "workflow"})
    run_task(db, user, task, models=Models(), tools=Tools(user))
    execution = task_payload(db, user, task)["execution"]
    assert execution["requested_mode"] == "workflow" and execution["selected_mode"] == "workflow"
    path = execution["executed_path"]
    assert path[:3] == ["prepare", "memory", "workflow_search"] and path[-1] == "finish"


def test_client_cannot_inject_an_executed_path():
    db, user = agent_db()
    task = create_task(db, user, "当前 RPO 是多少？", "workflow", 4, {"_executed_path": ["hybrid"]})
    assert "_executed_path" not in task.input
    assert task_payload(db, user, task)["execution"]["executed_path"] is None


@pytest.mark.parametrize("mode", ["dynamic", "hybrid"])
def test_contract_short_circuit_is_visible_in_executed_path(monkeypatch, mode):
    """A non-ordinary contract skips the agent subgraph; the mode label alone hides it."""
    import app.task_contract_runtime as runtime

    monkeypatch.setattr(settings(), "task_contract_enabled", True)
    monkeypatch.setattr(runtime, "prepare_contract", lambda *args, **kwargs: ([], {"stub": True}))
    seen = {}

    def finish(db, user, task, models, refs, memory_context, **kwargs):
        seen["contract"] = kwargs["contract"]
        task.status, task.result = "completed", {"status": "answered", "claims": [], "citations": []}
        db.commit()
    monkeypatch.setattr(controller, "_finish", finish)
    db, user = agent_db()
    tools, models = Tools(user), Models()
    task = create_task(db, user, CONDITIONAL, mode, 4)
    run_task(db, user, task, models=models, tools=tools)
    execution = task_payload(db, user, task)["execution"]
    assert seen["contract"].intent == "conditional"
    assert execution["selected_mode"] == mode
    assert execution["executed_path"] == ["prepare", "finish"]
    assert models.policy_calls == 0
