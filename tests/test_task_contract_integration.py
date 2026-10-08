"""Opt-in real PostgreSQL/OpenSearch/BGE/Qwen smoke; existing seeded corpus only."""
import os
import pytest
from sqlalchemy import delete, select

from agent.controller import create_task, run_task
from app.clients import Models
from app.config import settings
from app.db import SessionLocal
from app.models import AgentEvent, AgentTask, Answer, ToolExecution, User
from app.qa import answer_question

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.getenv("RAG_RUN_INTEGRATION") != "1", reason="requires isolated local demo services")]


@pytest.mark.parametrize("arm", ["rag", "workflow", "hybrid"])
def test_shared_contract_real_retrieval_and_comparison(monkeypatch, arm):
    monkeypatch.setattr(settings(), "task_contract_enabled", True)
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.username == "engineer@xingqiao.demo"))
        assert user is not None
        task_ids, answer_ids = [], []
        try:
            def run(question):
                if arm == "rag":
                    result = answer_question(db, user, question)
                    answer_ids.append(result["id"])
                    return result
                task = create_task(db, user, question, arm, 6)
                task_ids.append(task.id)
                run_task(db, user, task)
                assert task.status == "completed"
                return task.result
            false = run("先查生产数据库的RPO，如果它超过30分钟，再给出生产数据库的RTO。")
            assert false["status"] == "answered"
            assert false["usage"]["task_contract"]["condition"]["result"] == "false"
            assert all("RTO" not in c["text"] for c in false["claims"])
            true = run("先查生产数据库的RPO，如果它至少15分钟，再给出生产数据库的RTO。")
            assert true["status"] == "answered"
            assert len(true["claims"]) == 2
            comparison = run("生产数据库的RPO和生产数据库的RTO相差多少？")
            assert comparison["status"] == "answered"
            texts = " ".join(c["text"] for c in comparison["claims"])
            assert "15 分钟" in texts and "60 分钟" in texts and "45 分钟" in texts
            assert len(comparison["claims"][-1]["evidence_ids"]) == 2
            # One ordinary call ensures the real model's generation usage is counted
            # separately from policy/judge calls. Deterministic contracts need none.
            if arm == "rag":
                normal = run("生产数据库的RPO是多少？")
                budget = normal["usage"]["execution_budget"]["calls"]
                assert budget["generation"]["attempted"] >= 1
                assert budget["generation"]["prompt_tokens"] > 0
        finally:
            db.rollback()
            if task_ids:
                db.execute(delete(AgentEvent).where(AgentEvent.task_id.in_(task_ids)))
                db.execute(delete(ToolExecution).where(ToolExecution.task_id.in_(task_ids)))
                db.execute(delete(AgentTask).where(AgentTask.id.in_(task_ids)))
            if answer_ids:
                db.execute(delete(Answer).where(Answer.id.in_(answer_ids)))
            db.commit()


def test_real_policy_response_has_independent_attempt_and_token_accounting():
    from app.execution_budget import ExecutionBudget, use_budget
    budget = ExecutionBudget(settings())
    with use_budget(budget):
        Models().decide_agent_action("查询生产数据库RPO", [], 0)
    usage = budget.snapshot()["calls"]["policy"]
    assert usage["attempted"] == usage["succeeded"] == 1
    assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0
    assert "generation" not in budget.state["calls"]


def test_real_historical_version_chain_keeps_old_and_current_values(monkeypatch):
    monkeypatch.setattr(settings(), "task_contract_enabled", True)
    with SessionLocal() as db:
        user = db.get(User, "xq-admin")
        task = create_task(db, user, "各个历史版本的首次响应时限有没有变化？", "hybrid", 6,
                           {"document_id": "hard-support-sla-chain"})
        try:
            run_task(db, user, task)
            assert task.status == "completed" and task.result["status"] == "answered"
            assert len({c["version_id"] for c in task.result["citations"]}) == 3
            assert "发生变化" in task.result["claims"][-1]["text"]
        finally:
            db.rollback()
            db.execute(delete(AgentEvent).where(AgentEvent.task_id == task.id))
            db.execute(delete(ToolExecution).where(ToolExecution.task_id == task.id))
            db.execute(delete(AgentTask).where(AgentTask.id == task.id))
            db.commit()
