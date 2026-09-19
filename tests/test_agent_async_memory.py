from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import agent.controller as controller
from agent.controller import claim_agent_task, create_task, resume_task
from agent.memory_adapter import LongTermMemoryAdapter, MemoryUnavailable, _compact_memories
from app.main import AgentTaskBody
from app.config import settings
from app.models import Base, Tenant, User


def database():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    db = Session(engine, expire_on_commit=False)
    db.add(Tenant(id="tenant-a", name="A"))
    user = User(
        id="user-a",
        tenant_id="tenant-a",
        username="a@example.test",
        display_name="A",
        password_hash="unused",
        role="member",
        groups=[],
        active=True,
    )
    db.add(user)
    db.commit()
    return db, user, factory


def test_worker_claims_queued_task_with_a_lease(monkeypatch):
    db, user, factory = database()
    task = create_task(db, user, "核对恢复目标", "workflow", 4)
    monkeypatch.setattr(controller, "SessionLocal", factory)

    task_id, token = claim_agent_task()

    db.expire_all()
    claimed = db.get(type(task), task_id)
    assert task_id == task.id
    assert claimed.status == "running"
    assert claimed.attempts == 1
    assert claimed.lease_token == token
    assert claim_agent_task() is None


def test_manual_resume_requeues_and_resets_attempt_budget():
    db, user, _ = database()
    task = create_task(db, user, "恢复失败任务", "workflow", 4)
    task.status = "failed"
    task.attempts = 3
    task.error = "dependency failed"
    db.commit()

    resumed = resume_task(db, user, task.id)

    assert resumed.status == "queued"
    assert resumed.attempts == 0
    assert resumed.error is None


class Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "memories": [
                {
                    "id": "m1",
                    "content": "回答时优先给出实施步骤",
                    "status": "active",
                    "score": 0.91,
                }
            ],
            "candidates_considered": 3,
        }


class Client:
    def __init__(self):
        self.calls = []

    def post(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return Response()


def test_memory_adapter_binds_bearer_token_to_server_user_namespace(monkeypatch):
    _, user, _ = database()
    cfg = settings()
    monkeypatch.setattr(cfg, "memory_url", "http://memory.test")
    monkeypatch.setattr(cfg, "memory_tokens_json", '{"tenant-a:user-a":"secret-token"}')
    client = Client()
    adapter = LongTermMemoryAdapter(user, client=client)

    result = adapter.search("输出偏好")

    path, request = client.calls[0]
    assert path == "/v1/memories/search"
    assert request["headers"]["Authorization"] == "Bearer secret-token"
    assert request["json"]["user_id"] == "tenant-a:user-a"
    assert result["memories"][0]["content"] == "回答时优先给出实施步骤"


def test_memory_adapter_fails_closed_without_a_namespace_token(monkeypatch):
    _, user, _ = database()
    cfg = settings()
    monkeypatch.setattr(cfg, "memory_url", "http://memory.test")
    monkeypatch.setattr(cfg, "memory_tokens_json", "{}")
    adapter = LongTermMemoryAdapter(user, client=Client())

    assert adapter.enabled is False
    try:
        adapter.search("输出偏好")
    except MemoryUnavailable as exc:
        assert "受信命名空间" in str(exc)
    else:
        raise AssertionError("missing token must fail closed")


def test_long_term_memory_is_opt_in():
    assert AgentTaskBody(goal="核对灾备策略").use_memory is False


def test_memory_context_has_item_and_character_budgets():
    rows = [
        {"id": f"m{i}", "content": ("偏好 " * 200), "status": "active"}
        for i in range(5)
    ]

    compact = _compact_memories(rows)

    assert len(compact) == 3
    assert all(len(row["content"]) <= 240 for row in compact)
    assert sum(len(row["content"]) for row in compact) <= 600
