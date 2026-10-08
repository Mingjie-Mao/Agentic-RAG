"""Source identity parity and honest final-answer failure diagnostics."""
import json

import pytest

from agent.controller import _evidence_from_refs, _finish, create_task
from app.clients import Claim, GeneratedAnswer, Models, evidence_spans
from app.config import settings
from app.models import Document
from app.qa import validate_claims
from tests.test_agent import agent_db


def test_rebuilt_current_evidence_matches_shared_generation_source_identity():
    db, user = agent_db()
    row = _evidence_from_refs(db, user, ["c2"])[0]
    # Shared retrieval sends the original document title, not a display badge.
    retrieved = {**row, "title": db.get(Document, "doc-a").title}
    assert evidence_spans([row]) == evidence_spans([retrieved])
    calls = []
    def backend(body):
        calls.append(body)
        return {"message": {"content": json.dumps({"status": "answered", "claims": [
            {"text": "RPO 为 15 分钟。", "source_ids": ["E1:S1"]}]})}}
    generated, _ = Models(backend).generate("RPO是多少？", [row])
    assert json.loads(calls[0]["messages"][1]["content"])["evidence"][0]["title"] == "恢复政策"
    claims, status = validate_claims(generated, [row])
    assert status == "answered" and claims[0]["evidence_ids"] == ["c2"]


def test_title_parity_preserves_historical_scope_and_rechecks_authorization():
    db, user = agent_db()
    rows = _evidence_from_refs(db, user, ["c1", "c2"], allow_historical=True, limit=None)
    by_version = {r["version_id"]: r for r in rows}
    assert "历史版本" in by_version["v1"]["title"]
    assert by_version["v1"]["is_active"] is False
    assert by_version["v2"]["title"] == "恢复政策"
    assert by_version["v2"]["is_active"] is True
    assert {r["version_id"] for r in _evidence_from_refs(db, user, ["c1", "c2"])} == {"v2"}
    doc = db.get(Document, "doc-a")
    doc.owner_id = "other"
    doc.read_groups = ["finance"]
    db.commit()
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _evidence_from_refs(db, user, ["c2"])


def test_table_citation_binds_header_and_rows_without_crossing_other_text():
    text = "## 配额\n| 套餐 | 每分钟调用上限 |\n| --- | --- |\n| 基础版 | 300 次 |\n| 专业版 | 1200 次 |\n\n补充说明。第二句。"
    row = {"id": "E1", "chunk_id": "table", "document_id": "doc", "title": "配额", "text": text}
    sources, context = evidence_spans([row])
    containing_value = [s for s in sources.values() if "1200" in s["quote"]]
    assert len(containing_value) == 1
    table = containing_value[0]["quote"]
    assert "每分钟调用上限" in table and "专业版" in table and "基础版" in table
    assert "补充说明" not in table
    source_id = next(s["id"] for s in context[0]["sources"] if "1200" in s["text"])
    def backend(_):
        return {"message": {"content": json.dumps({"status": "answered", "claims": [
            {"text": "专业版每分钟调用上限为1200次。", "source_ids": [source_id]}]})}}
    generated, _ = Models(backend).generate("专业版配额是多少？", [row])
    claims, status = validate_claims(generated, [row])
    assert status == "answered" and claims[0]["quotes"] == [table]
    assert claims[0]["evidence_ids"] == ["table"]


def test_citation_units_keep_distinct_tables_and_unmodified_prose():
    row = {"id": "E1", "title": "说明", "text": "前句。后句。\n| A | B |\n| x | y |\n\n隔离说明。\n| C | D |\n| u | v |"}
    sources, _ = evidence_spans([row])
    quotes = [s["quote"] for s in sources.values()]
    assert quotes[:2] == ["前句。", "后句。"]
    assert len([q for q in quotes if "|" in q]) == 2
    assert not any("| A |" in q and "| C |" in q for q in quotes)


@pytest.mark.parametrize("answerable,status,reason", [
    (False, "insufficient_evidence", "generator_abstained"),
    (True, "verification_failed", "claim_validation_failed"),
])
def test_model_abstention_is_distinguished_from_invalid_citations(monkeypatch, answerable, status, reason):
    cfg = settings()
    for key in ("task_contract_enabled", "passage_window_enabled", "focused_generation_enabled"):
        monkeypatch.setattr(cfg, key, False)
    db, user = agent_db()
    class Model:
        def generate(self, *args, **kwargs):
            claims = [Claim(text="RPO为999分钟。", evidence_ids=["E1"], quotes=["伪造的原文。"]) ] if answerable else []
            return GeneratedAnswer(answerable=answerable, claims=claims), {
                "answer_status": "answered" if answerable else "insufficient_evidence"}
    task = create_task(db, user, "RPO是多少？", "dynamic")
    _finish(db, user, task, Model(), ["c2"])
    assert task.result["status"] == status
    assert task.result["claims"] == [] and task.result["citations"] == []
    assert task.result["usage"]["answer_validation"]["reason"] == reason
    assert ("引用校验" in task.result["message"]) is answerable
