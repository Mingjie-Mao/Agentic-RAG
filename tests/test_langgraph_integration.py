"""Real PostgreSQL checkpoint recovery across worker sessions."""

import os

import pytest
from sqlalchemy import delete

from agent import controller
from agent.controller import create_task, run_task
from agent.graph_runtime import graph_checkpointer, graph_identity
from app.config import settings
from app.db import SessionLocal
from app.models import AgentEvent, AgentTask, ToolExecution, User

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.getenv("RAG_RUN_INTEGRATION") != "1", reason="requires local demo services")]


@pytest.mark.parametrize("mode", ["workflow", "hybrid"])
def test_postgres_pending_node_recovers_in_new_session_without_retrieval_replay(monkeypatch, mode):
    monkeypatch.setattr(settings(), "task_contract_enabled", False)
    monkeypatch.setattr(settings(), "semantic_slot_shadow_enabled", False)
    task_id, identity = None, None
    original = controller._finish
    def unavailable(*args, **kwargs):
        raise RuntimeError("injected-at-answer-node")
    try:
        with SessionLocal() as db:
            assert db.get_bind().dialect.name == "postgresql"
            user = db.get(User, "xq-engineer")
            assert user is not None
            task = create_task(db, user, "生产数据库的RPO是多少？", mode, 4, benchmark_execution=True)
            task_id, identity = task.id, graph_identity(task, user)
            monkeypatch.setattr(controller, "_finish", unavailable)
            with pytest.raises(RuntimeError, match="injected-at-answer-node"):
                run_task(db, user, task)
            before_step = task.step_no
            assert before_step > 0
            with graph_checkpointer(db) as saver:
                saved = saver.get_tuple(identity)
                assert saved and saved.checkpoint["channel_values"]["refs"]
        monkeypatch.setattr(controller, "_finish", original)
        with SessionLocal() as db:
            user, task = db.get(User, "xq-engineer"), db.get(AgentTask, task_id)
            run_task(db, user, task)
            assert task.status == "completed" and task.result["status"] == "answered"
            assert task.step_no == before_step
            assert task.result["citations"]
    finally:
        if task_id:
            with SessionLocal() as db:
                with graph_checkpointer(db) as saver:
                    saver.delete_thread(identity["configurable"]["thread_id"])
                db.execute(delete(AgentEvent).where(AgentEvent.task_id == task_id))
                db.execute(delete(ToolExecution).where(ToolExecution.task_id == task_id))
                db.execute(delete(AgentTask).where(AgentTask.id == task_id))
                db.commit()
