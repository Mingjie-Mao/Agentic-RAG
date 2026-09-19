from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agent.controller import create_task, run_task, task_payload
from agent.tools import KnowledgeTools, ToolResult
from app.clients import AgentDecision, Claim, GeneratedAnswer
from app.models import AgentEvent, Base, Chunk, Document, DocumentVersion, Tenant, User


def agent_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = Session(engine, expire_on_commit=False)
    db.add(Tenant(id="tenant-a", name="A"))
    user = User(
        id="user-a",
        tenant_id="tenant-a",
        username="a@example.test",
        display_name="A",
        password_hash="unused",
        role="member",
        groups=["engineering"],
        active=True,
    )
    db.add(user)
    document = Document(
        id="doc-a",
        tenant_id="tenant-a",
        owner_id="user-a",
        title="恢复政策",
        active_version_id="v2",
        read_groups=["engineering"],
        tenant_public=False,
        deleted=False,
        revision=2,
        metadata_json={},
    )
    db.add(document)
    db.add_all(
        [
            DocumentVersion(
                id="v1",
                document_id="doc-a",
                filename="old.md",
                media_type="text/markdown",
                content_hash="1" * 64,
                storage_key="old",
                status="ready",
                pipeline={},
                timings={},
                parsed_blocks=[],
            ),
            DocumentVersion(
                id="v2",
                document_id="doc-a",
                filename="new.md",
                media_type="text/markdown",
                content_hash="2" * 64,
                storage_key="new",
                status="ready",
                pipeline={},
                timings={},
                parsed_blocks=[],
            ),
        ]
    )
    db.add_all(
        [
            Chunk(id="c1", version_id="v1", ordinal=0, text="RPO 为 30 分钟。", locator={}),
            Chunk(id="c2", version_id="v2", ordinal=0, text="RPO 为 15 分钟。", locator={}),
        ]
    )
    db.commit()
    return db, user


def test_version_tools_are_acl_scoped_and_distinguish_current_from_history():
    db, user = agent_db()
    tools = KnowledgeTools(db, user, models=object(), search=object())
    versions = tools.call("get_document_version", {"document_id": "doc-a"})
    assert versions.status == "ok"
    assert versions.data["active_version_id"] == "v2"

    comparison = tools.call("compare_versions", {"document_id": "doc-a"})
    assert comparison.status == "ok"
    assert any("-RPO 为 30 分钟" in line for line in comparison.data["diff_lines"])
    assert any("+RPO 为 15 分钟" in line for line in comparison.data["diff_lines"])
    assert set(comparison.evidence_refs) == {"c1", "c2"}

    access = tools.call("verify_chunk_access", {"chunk_ids": ["c1", "c2"]})
    assert access.data["checks"][0]["reason"] == "superseded"
    assert access.data["checks"][1]["readable"] is True

    db.get(Document, "doc-a").read_groups = ["support"]
    db.get(Document, "doc-a").owner_id = "someone-else"
    db.commit()
    denied = tools.call("open_document", {"document_id": "doc-a"})
    assert denied.status == "error" and denied.error_code == "http_404"
    assert denied.data == {} and denied.evidence_refs == []


def test_workflow_agent_persists_trace_and_returns_verified_citations():
    db, user = agent_db()

    class FakeTools:
        def __init__(self):
            self.user = user

        def call(self, name, arguments):
            assert name in {"search_documents", "retrieve_evidence"}
            return ToolResult("ok", {}, ["c2"], {"checked_now": True})

    class FakeModels:
        def generate(self, question, evidence):
            assert question == "当前 RPO 是多少？"
            assert evidence[0]["chunk_id"] == "c2"
            return (
                GeneratedAnswer(
                    answerable=True,
                    claims=[
                        Claim(
                            text="当前 RPO 为 15 分钟。",
                            evidence_ids=["E1"],
                            quotes=["RPO 为 15 分钟。"],
                        )
                    ],
                ),
                {"answer_status": "answered"},
            )

    task = create_task(db, user, "当前 RPO 是多少？", "workflow", 4)
    run_task(db, user, task, models=FakeModels(), tools=FakeTools())
    payload = task_payload(db, user, task)
    assert payload["status"] == "completed"
    assert payload["result"]["status"] == "answered"
    assert payload["result"]["citations"][0]["chunk_id"] == "c2"
    assert [event["event_type"] for event in payload["events"]] == [
        "task_created",
        "task_started",
        "tool_completed",
        "tool_completed",
        "task_completed",
    ]

    db.get(Document, "doc-a").active_version_id = "v1"
    first_tool_event = next(
        event
        for event in db.query(AgentEvent).filter(AgentEvent.task_id == task.id).all()
        if event.event_type == "tool_completed"
    )
    first_tool_event.payload = {
        **first_tool_event.payload,
        "titles": ["敏感恢复政策"],
        "arguments": {"query": "敏感恢复政策"},
        "purpose": "读取敏感恢复政策",
    }
    # The historical tool event alone is enough to make the task dependent on c2.
    task.evidence_chunk_ids = []
    task.error = "敏感恢复政策处理失败"
    db.commit()
    stale = task_payload(db, user, task)
    assert stale["result"]["status"] == "access_changed"
    assert stale["error"] is None
    assert all(event["evidence_refs"] == [] for event in stale["events"])
    observable = str(stale)
    assert "敏感恢复政策" not in observable
    assert "arguments" not in observable
    assert "purpose" not in observable
    db.get(Document, "doc-a").active_version_id = "v2"
    db.commit()

    db.get(Document, "doc-a").deleted = True
    db.commit()
    hidden = task_payload(db, user, task)
    assert hidden["result"]["status"] == "access_changed"
    assert hidden["result"]["citations"] == []


def test_dynamic_agent_repeated_action_consumes_budget_and_stops():
    db, user = agent_db()

    class RepeatingModels:
        def decide_agent_action(self, *args):
            return AgentDecision(
                action="search_documents", arguments={"query": "重复查询"}, purpose="查找资料"
            )

    class EmptyTools:
        def __init__(self):
            self.user = user

        def call(self, name, arguments):
            return ToolResult("ok", {}, [], {"checked_now": True})

    task = create_task(db, user, "测试重复动作能否停止", "dynamic", 2)
    run_task(db, user, task, models=RepeatingModels(), tools=EmptyTools())
    payload = task_payload(db, user, task)
    assert payload["status"] == "completed" and payload["step_no"] == 2
    assert payload["result"]["status"] == "insufficient_evidence"
    assert any(event["event_type"] == "tool_rejected" for event in payload["events"])


def test_dynamic_compound_task_adds_coverage_retrieval_before_finishing():
    db, user = agent_db()

    class EarlyStopModels:
        def decide_agent_action(self, *args):
            return AgentDecision(action="final", arguments={}, purpose="已经完成")

        def generate(self, question, evidence):
            assert question == "当前 RPO 以及恢复目标分别是什么？"
            assert [row["chunk_id"] for row in evidence] == ["c2"]
            return (
                GeneratedAnswer(
                    answerable=True,
                    claims=[
                        Claim(
                            text="当前 RPO 为 15 分钟。",
                            evidence_ids=["E1"],
                            quotes=["RPO 为 15 分钟。"],
                        )
                    ],
                ),
                {"answer_status": "answered"},
            )

    class CoverageTools:
        def __init__(self):
            self.user = user
            self.calls = []

        def call(self, name, arguments):
            self.calls.append((name, arguments))
            return ToolResult("ok", {}, ["c2"], {"checked_now": True})

    tools = CoverageTools()
    task = create_task(
        db,
        user,
        "当前 RPO 以及恢复目标分别是什么？",
        "dynamic",
        2,
    )
    run_task(db, user, task, models=EarlyStopModels(), tools=tools)
    payload = task_payload(db, user, task)

    assert tools.calls == [
        (
            "search_documents",
            {"query": "当前 RPO 以及恢复目标分别是什么？", "top_k": 8},
        )
    ]
    assert payload["step_no"] == 1
    assert any(event["event_type"] == "coverage_retrieval" for event in payload["events"])
