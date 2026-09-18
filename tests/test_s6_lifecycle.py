"""S6: version replacement, access change and deletion, checked through every read entry.

The property under test is one sentence: once the business database says a reader may
no longer see something, no entry point may return it — regardless of what the search
index still contains.
"""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.main import app
from app.models import Chunk, Document, DocumentVersion

pytestmark = pytest.mark.skipif(
    os.getenv("RAG_RUN_INTEGRATION") != "1", reason="requires isolated local demo services"
)
HEADERS = {"X-Requested-With": "AgenticRAG"}
SAMPLE = "# 鹊桥临时通道规定\n\n鹊桥临时通道的单日调用上限为 77 次。\n鹊桥临时通道仅在演练期间开放。\n".encode()
REVISED = "# 鹊桥临时通道规定\n\n鹊桥临时通道的单日调用上限为 88 次。\n鹊桥临时通道仅在演练期间开放。\n".encode()


def login(username):
    client = TestClient(app, headers=HEADERS)
    client.__enter__()
    assert (
        client.post(
            "/api/auth/login",
            json={"username": username, "password": settings().demo_password},
        ).status_code
        == 200
    )
    return client


@pytest.fixture
def owner():
    client = login("support@xingqiao.demo")
    yield client
    client.__exit__(None, None, None)


@pytest.fixture
def admin():
    client = login("admin@xingqiao.demo")
    yield client
    client.__exit__(None, None, None)


@pytest.fixture
def outsider():
    client = login("engineer@xingqiao.demo")
    yield client
    client.__exit__(None, None, None)


@pytest.fixture
def uploaded(owner):
    response = owner.post(
        "/api/documents",
        files={"file": ("s6-sample.md", SAMPLE, "text/markdown")},
        data={"title": "S6 临时规定", "groups": "[]", "tenant_public": "false"},
    )
    assert response.status_code == 202, response.text
    document_id = response.json()["id"]
    for _ in range(120):
        state = owner.get(f"/api/documents/{document_id}").json()
        if state["status"] == "ready":
            break
        time.sleep(1)
    assert state["status"] == "ready", state
    yield document_id
    owner.delete(f"/api/documents/{document_id}")


def chunk_of(document_id):
    with SessionLocal() as db:
        document = db.get(Document, document_id)
        chunk = db.scalar(select(Chunk).where(Chunk.version_id == document.active_version_id))
        return document.active_version_id, chunk.id


def every_read_entry(client, document_id, version_id, chunk_id, answer_id=None):
    """Every route through which stored content can surface.

    History and the retrieval trace are included because they are the easiest to
    overlook and the most revealing: history stores the claims and citations of a past
    answer, and the trace stores the titles of every candidate that was considered.
    """
    listed = client.get("/api/documents").json()
    entries = {
        "list": any(d["id"] == document_id for d in listed),
        "detail": client.get(f"/api/documents/{document_id}").status_code == 200,
        "processing": client.get(f"/api/documents/{document_id}/processing").status_code == 200,
        "evidence": client.get(f"/api/evidence/{chunk_id}").status_code == 200,
        "original": client.get(f"/api/versions/{version_id}/original").status_code == 200,
        "readable_probe": client.get(f"/api/access/chunks/{chunk_id}").json()["readable"],
    }
    if answer_id:
        history = client.get("/api/history").json()
        record = next((row for row in history if row["id"] == answer_id), None)
        entries["history_list"] = record is not None
        # A hidden answer must not leak through its citations or its trace candidates.
        entries["history_citations"] = bool(record and record.get("citations"))
        entries["history_trace"] = bool(record and (record.get("trace") or {}).get("candidates"))
        single = client.get(f"/api/history/{answer_id}")
        entries["history_detail"] = single.status_code == 200 and bool(single.json().get("citations"))
    return entries


def ask_about(client, question):
    response = client.post("/api/chat", json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()


def test_private_document_is_invisible_to_another_group_everywhere(uploaded, owner, outsider):
    version_id, chunk_id = chunk_of(uploaded)
    assert all(every_read_entry(owner, uploaded, version_id, chunk_id).values())
    assert not any(every_read_entry(outsider, uploaded, version_id, chunk_id).values())


def test_losing_access_hides_the_past_answer_its_citations_and_its_trace(uploaded, owner, admin):
    """An answer is only as private as the weakest route back to its evidence."""
    version_id, chunk_id = chunk_of(uploaded)
    admin.patch(
        f"/api/documents/{uploaded}/access",
        json={"groups": ["support", "engineering"], "tenant_public": False},
    )
    engineer = login("engineer@xingqiao.demo")
    try:
        answer = ask_about(engineer, "鹊桥临时通道的单日调用上限是多少？")
        assert any(c["document_id"] == uploaded for c in answer["citations"]), answer["status"]
        opened = every_read_entry(engineer, uploaded, version_id, chunk_id, answer["id"])
        assert opened["history_citations"] and opened["history_trace"]

        admin.patch(
            f"/api/documents/{uploaded}/access",
            json={"groups": ["support"], "tenant_public": False},
        )
        closed = every_read_entry(engineer, uploaded, version_id, chunk_id, answer["id"])
        # The record may remain as a tombstone — it is the reader's own past action —
        # but it must reveal nothing, including the question, which names the subject.
        assert not any(value for key, value in closed.items() if key != "history_list")
        tombstone = engineer.get(f"/api/history/{answer['id']}").json()
        assert tombstone["status"] == "access_changed"
        assert tombstone["claims"] == [] and tombstone["citations"] == []
        assert "鹊桥临时通道" not in json.dumps(tombstone, ensure_ascii=False)
        assert "trace" not in tombstone
    finally:
        engineer.__exit__(None, None, None)


def test_revoking_access_closes_every_entry_without_waiting_for_the_index(
    uploaded, owner, admin, outsider
):
    version_id, chunk_id = chunk_of(uploaded)
    assert (
        owner.patch(
            f"/api/documents/{uploaded}/access",
            json={"groups": ["support"], "tenant_public": False},
        ).status_code
        == 200
    )
    assert not any(every_read_entry(outsider, uploaded, version_id, chunk_id).values())

    # Cross-group sharing is an administrator action; a support member cannot do it.
    assert (
        admin.patch(
            f"/api/documents/{uploaded}/access",
            json={"groups": ["support", "engineering"], "tenant_public": False},
        ).status_code
        == 200
    )
    assert all(every_read_entry(outsider, uploaded, version_id, chunk_id).values())

    with SessionLocal() as db:
        indexed_before = db.scalar(select(Chunk).where(Chunk.version_id == version_id)) is not None
    assert indexed_before, "The chunk must still exist; this test is about authorisation, not cleanup"

    admin.patch(
        f"/api/documents/{uploaded}/access", json={"groups": ["support"], "tenant_public": False}
    )
    # No refresh, no sleep: the database decides, so the next request is already refused.
    assert not any(every_read_entry(outsider, uploaded, version_id, chunk_id).values())


def test_replacement_publishes_only_after_it_is_searchable(uploaded, owner):
    original_version, original_chunk = chunk_of(uploaded)
    response = owner.post(
        f"/api/documents/{uploaded}/versions",
        files={"file": ("s6-sample.md", REVISED, "text/markdown")},
    )
    assert response.status_code == 202, response.text
    new_version = response.json()["version_id"]
    assert response.json()["active_version_id"] == original_version, (
        "The previous version must keep serving until the replacement is searchable"
    )

    for _ in range(120):
        with SessionLocal() as db:
            document = db.get(Document, uploaded)
            promoted = document.active_version_id == new_version
        if promoted:
            break
        time.sleep(1)
    assert promoted, "Replacement never became active"

    with SessionLocal() as db:
        assert db.get(DocumentVersion, original_version).status == "ready"
    # The old chunk is no longer the active version, so it must stop being readable
    # while its own version record stays intact for auditing older answers.
    assert owner.get(f"/api/access/chunks/{original_chunk}").json() == {
        "chunk_id": original_chunk,
        "readable": False,
        "reason": "superseded",
        "document_id": uploaded,
        "active_version_id": new_version,
    }
    detail = owner.get(f"/api/documents/{uploaded}/processing").json()
    assert len(detail["versions"]) == 2
    assert [v["active"] for v in detail["versions"]].count(True) == 1


def test_deletion_closes_every_entry_and_purges_the_index(owner):
    response = owner.post(
        "/api/documents",
        files={"file": ("s6-doomed.md", SAMPLE, "text/markdown")},
        data={"title": "S6 待删除", "groups": "[]", "tenant_public": "false"},
    )
    document_id = response.json()["id"]
    for _ in range(120):
        state = owner.get(f"/api/documents/{document_id}").json()
        if state["status"] == "ready":
            break
        time.sleep(1)
    version_id, chunk_id = chunk_of(document_id)

    removed = owner.delete(f"/api/documents/{document_id}")
    assert removed.status_code == 200, removed.text
    assert removed.json()["deleted"] is True
    assert not any(every_read_entry(owner, document_id, version_id, chunk_id).values())
    assert owner.get(f"/api/access/chunks/{chunk_id}").json()["reason"] == "deleted"
    assert removed.json()["cleanup"]["errors"] == []
    assert removed.json()["cleanup"]["files"] >= 1


def test_only_a_writer_may_replace_change_access_or_delete(uploaded, outsider):
    assert (
        outsider.post(
            f"/api/documents/{uploaded}/versions",
            files={"file": ("x.md", REVISED, "text/markdown")},
        ).status_code
        == 404
    )
    assert (
        outsider.patch(
            f"/api/documents/{uploaded}/access", json={"groups": [], "tenant_public": False}
        ).status_code
        == 404
    )
    assert outsider.delete(f"/api/documents/{uploaded}").status_code == 404


def test_a_member_cannot_grant_beyond_their_own_groups(uploaded, owner):
    refused = owner.patch(
        f"/api/documents/{uploaded}/access",
        json={"groups": ["engineering", "support"], "tenant_public": False},
    )
    assert refused.status_code == 403
    assert (
        owner.patch(
            f"/api/documents/{uploaded}/access", json={"groups": [], "tenant_public": True}
        ).status_code
        == 403
    )
    assert (
        owner.patch(
            f"/api/documents/{uploaded}/access", json={"groups": ["marketing"], "tenant_public": False}
        ).status_code
        == 422
    )
