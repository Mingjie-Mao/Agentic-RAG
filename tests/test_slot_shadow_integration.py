"""Real DB/retrieval checks: post-completion shadow cannot mutate task/answer state."""
import json
import os
from pathlib import Path
from threading import Event

import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.db import SessionLocal
from app.execution_budget import current_budget
from app.models import AgentTask, AgentEvent, Answer, ToolExecution
from app.semantic_evidence import SlotEvidenceEvaluator

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.getenv("RAG_RUN_INTEGRATION") != "1", reason="requires local demo services")]


@pytest.mark.parametrize("kind", ["agent", "answer"])
def test_post_completion_shadow_has_own_budget_and_no_primary_writes(monkeypatch, kind):
    from app import slot_shadow
    from app.qa import answer_question
    from app.models import User
    from agent.controller import create_task, run_task

    monkeypatch.setattr(settings(), "task_contract_enabled", True)
    monkeypatch.setattr(settings(), "semantic_slot_shadow_enabled", True)
    monkeypatch.setattr(settings(), "semantic_slot_shadow_sample_rate", 1)
    finished = Event()
    primary_returned = Event()
    original_observer = slot_shadow.observe_completed
    def observe(*args):
        try:
            return original_observer(*args)
        finally:
            finished.set()
    monkeypatch.setattr(slot_shadow, "observe_completed", observe)
    called = []
    def judge(payload):
        assert current_budget().state["limits"]["judge"] == 2
        called.append(payload)
        assert primary_returned.wait(10), "primary request waited on its shadow judge"
        first = payload["evidence"][0]
        return {"output": {"judgments": [{"subgoal_id": s["id"], "status": "supported",
            "support_refs": [first["id"]], "quote_spans": [{"ref": first["id"], "quote": first["text"]}],
            "missing_slots": [], "reason_code": "isolation_fixture_only", "confidence": 0}
            for s in payload["subgoals"]]}, "prompt_tokens": 12, "completion_tokens": 5}
    class FakeJudgeEvaluator(SlotEvidenceEvaluator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, judge=judge, relevance_scorer=lambda q, texts: [5]*len(texts))
    monkeypatch.setattr(slot_shadow, "SlotEvidenceEvaluator", FakeJudgeEvaluator)
    with SessionLocal() as db:
        user = db.get(User, "xq-engineer")
        question = "先查生产数据库的RPO，如果它超过30分钟，再给出生产数据库的RTO。"
        rid = None
        try:
            if kind == "agent":
                task = create_task(db, user, question, "hybrid", 6, benchmark_execution=True)
                rid = task.id
                run_task(db, user, task)
                primary = task.result
                assert task.status == "completed"
                assert not db.scalar(select(AgentEvent).where(AgentEvent.task_id == rid,
                                                             AgentEvent.event_type == "semantic_coverage_control"))
            else:
                result = answer_question(db, user, question)
                rid = result["id"]
                primary = db.get(Answer, rid).payload
            primary_returned.set()
            assert primary["usage"]["execution_budget"]["calls"].get("judge", {}).get("attempted", 0) == 0
            assert primary["status"] == "answered"
            assert finished.wait(30), "background observation did not finish"
            slot_shadow._dispatcher.queue.join()
            sidecar = Path(f".runtime/slot-shadow/{kind}/{rid}.json")
            shadow = json.loads(sidecar.read_text())
            assert shadow["status"] == "observed"
            assert shadow["execution_budget"]["calls"]["judge"]["attempted"] == 1
            assert called and current_budget() is None
            assert shadow["report"]["inactive"] == ["true_1"]
            cached = slot_shadow.observe_completed(kind, rid, user.id)
            assert cached == shadow and len(called) == 1
            db.expire_all()
            stored = db.get(AgentTask, rid).result if kind == "agent" else db.get(Answer, rid).payload
            assert stored == primary
        finally:
            db.rollback()
            if rid:
                if kind == "agent":
                    db.execute(delete(AgentEvent).where(AgentEvent.task_id == rid))
                    db.execute(delete(ToolExecution).where(ToolExecution.task_id == rid))
                    db.execute(delete(AgentTask).where(AgentTask.id == rid))
                else:
                    db.execute(delete(Answer).where(Answer.id == rid))
                db.commit()
                Path(f".runtime/slot-shadow/{kind}/{rid}.json").unlink(missing_ok=True)


def test_qa_shadow_uses_uncited_generation_context_and_legacy_context_is_unknown(monkeypatch):
    from types import SimpleNamespace
    from app import qa, slot_shadow
    from app.clients import GeneratedAnswer, Claim, Models
    from app.models import Chunk, Document, DocumentVersion, User
    from app.security import require_chunk
    monkeypatch.setattr(settings(), "task_contract_enabled", False)
    monkeypatch.setattr(settings(), "semantic_slot_shadow_enabled", False)
    contexts = []
    def judge(payload):
        contexts.append(payload["evidence"])
        return {"output": {"judgments": []}}
    class Evaluator(SlotEvidenceEvaluator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, judge=judge, relevance_scorer=lambda q, texts: [5]*len(texts))
    monkeypatch.setattr(slot_shadow, "SlotEvidenceEvaluator", Evaluator)
    with SessionLocal() as db:
        user = db.get(User, "xq-engineer")
        evidence = []
        for did in ("seed-db-backup", "seed-sdk-retry"):
            chunk = db.scalar(select(Chunk).join(DocumentVersion).join(Document).where(
                Document.id == did, Document.active_version_id == DocumentVersion.id).order_by(Chunk.ordinal))
            c, v, d = require_chunk(db, user, chunk.id, active_only=True)
            evidence.append(dict(id=f"E{len(evidence)+1}", chunk_id=c.id, version_id=v.id,
                                 document_id=d.id, text=c.text, title=d.title, locator=c.locator))
        candidates = [{"chunk_id": p["chunk_id"]} for p in evidence]
        monkeypatch.setattr(qa, "retrieve_authorized", lambda *a, **k: SimpleNamespace(
            evidence=evidence, candidates=candidates, readable_documents=2, searchable_documents=2,
            context_tokens=200, embed_ms=0, retrieval_ms=0))
        def generate(*args, **kwargs):
            return GeneratedAnswer(answerable=True, claims=[Claim(text="资料说明", evidence_ids=["E1"],
                    quotes=[evidence[0]["text"]])]), {}
        monkeypatch.setattr(Models, "generate", generate)
        rid = None
        try:
            answer = qa.answer_question(db, user, "说明数据库备份制度。")
            rid = answer["id"]
            assert len(answer["citations"]) == 1
            assert answer["trace"]["generation_context_chunk_ids"] == [p["chunk_id"] for p in evidence]
            before = db.get(Answer, rid).payload
            shadow = slot_shadow.observe_completed("answer", rid, user.id)
            assert shadow["status"] == "observed" and len(contexts[0]) == 2
            db.expire_all()
            assert db.get(Answer, rid).payload == before
            record = db.get(Answer, rid)
            # A legacy answer has no recoverable generation snapshot. Do not
            # present its citation subset as a full-context observation.
            record.payload = {**record.payload, "trace": {}}
            db.commit()
            legacy = slot_shadow.observe_completed("answer", rid, user.id)
            assert legacy["status"] == "unknown" and legacy["reason"] == "generation_context_unavailable"
            assert len(contexts) == 1
        finally:
            db.rollback()
            if rid:
                db.execute(delete(Answer).where(Answer.id == rid))
                db.commit()
                Path(f".runtime/slot-shadow/answer/{rid}.json").unlink(missing_ok=True)


def test_real_revocation_during_judge_cannot_be_written_as_supported(monkeypatch):
    from uuid import uuid4
    from app import slot_shadow
    from app.models import Chunk, Document, DocumentVersion, User
    did, vid, cid, rid = ["shadow-followup-" + uuid4().hex for _ in range(4)]
    text = "调度流程必须先记录故障，再通知值班人员。"
    def judge(payload):
        with SessionLocal() as changed:
            changed.get(Document, did).deleted = True
            changed.commit()
        return {"output": {"judgments": [{"subgoal_id": s["id"], "status": "supported",
            "support_refs": ["E1"], "quote_spans": [{"ref": "E1", "quote": text}],
            "missing_slots": [], "reason_code": "fixture", "confidence": 0}
            for s in payload["subgoals"]]}}
    class Evaluator(SlotEvidenceEvaluator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, judge=judge, relevance_scorer=lambda q, texts: [5]*len(texts))
    monkeypatch.setattr(slot_shadow, "SlotEvidenceEvaluator", Evaluator)
    with SessionLocal() as db:
        user = db.get(User, "xq-engineer")
        payload = {"status": "answered", "claims": [], "citations": [],
                   "trace": {"generation_context_chunk_ids": [cid]}, "usage": {}}
        try:
            db.add(Document(id=did, tenant_id=user.tenant_id, owner_id="xq-admin", title="临时撤权测试",
                            read_groups=["engineering"], tenant_public=False, active_version_id=vid))
            db.flush()
            db.add(DocumentVersion(id=vid, document_id=did, filename="fixture.md", media_type="text/markdown",
                                   content_hash="fixture", storage_key="unused", status="ready"))
            db.flush()
            db.add(Chunk(id=cid, version_id=vid, ordinal=0, text=text, locator={}))
            db.add(Answer(id=rid, user_id=user.id, tenant_id=user.tenant_id, question="调度流程是什么？",
                          payload=payload, evidence_chunk_ids=[cid]))
            db.commit()
            shadow = slot_shadow.observe_completed("answer", rid, user.id)
            assert shadow["status"] == "unknown" and shadow["reason"] == "HTTPException"
            assert shadow["execution_budget"]["calls"]["judge"]["attempted"] == 1
            assert "report" not in shadow
            db.expire_all()
            assert db.get(Answer, rid).payload == payload
        finally:
            db.rollback()
            db.execute(delete(Answer).where(Answer.id == rid))
            db.execute(delete(Chunk).where(Chunk.id == cid))
            db.execute(delete(DocumentVersion).where(DocumentVersion.id == vid))
            db.execute(delete(Document).where(Document.id == did))
            db.commit()
            Path(f".runtime/slot-shadow/answer/{rid}.json").unlink(missing_ok=True)
