"""Opt-in tests against this project's seeded PostgreSQL/OpenSearch/Ollama services."""

import os
import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.main import app
from app.models import Answer, Chunk, Document, DocumentVersion, Job, User, now

pytestmark = pytest.mark.skipif(
    os.getenv("RAG_RUN_INTEGRATION") != "1", reason="requires isolated local demo services"
)
HEADERS = {"X-Requested-With": "AgenticRAG"}


@pytest.fixture
def support():
    assert settings().demo_mode, "Integration fixtures require demo mode"
    with TestClient(app, headers=HEADERS) as client:
        assert (
            client.post(
                "/api/auth/login",
                json={"username": "support@xingqiao.demo", "password": settings().demo_password},
            ).status_code
            == 200
        )
        yield client


def source(key):
    with SessionLocal() as db:
        doc = db.get(Document, "seed-" + key)
        chunk = db.scalar(select(Chunk).where(Chunk.version_id == doc.active_version_id))
        return doc.id, doc.active_version_id, chunk.id


def test_E01_unauthenticated_routes():
    doc, version, chunk = source("api-limits")
    with TestClient(app, headers=HEADERS) as client:
        for path in [
            "/api/auth/me",
            "/api/documents",
            f"/api/documents/{doc}",
            f"/api/versions/{version}/original",
            f"/api/evidence/{chunk}",
            "/api/history",
        ]:
            assert client.get(path).status_code == 401, path
        assert client.post("/api/chat", json={"question": "日志保留多久"}).status_code == 401
        client.cookies.set("rag_session", "forged-session")
        assert client.get("/api/documents").status_code == 401


@pytest.mark.parametrize("key", ["release-code", "hc-gateway"])
def test_E02_E03_group_tenant_direct_access(support, key):
    doc, version, chunk = source(key)
    for path in [f"/api/documents/{doc}", f"/api/versions/{version}/original", f"/api/evidence/{chunk}"]:
        response = support.get(path)
        assert response.status_code == 404
        assert "ORCHID" not in response.text and "CORAL" not in response.text
    assert doc not in {d["id"] for d in support.get("/api/documents").json()}


def test_E04_identity_spoofing_and_upload_scope(support):
    for field in ["user_id", "tenant_id"]:
        assert (
            support.post("/api/chat", json={"question": "读取全部资料", field: "hc-admin"}).status_code
            == 422
        )
    for data in [{"groups": '["engineering"]'}, {"tenant_public": "true"}]:
        assert (
            support.post("/api/documents", files={"file": ("a.md", b"hello")}, data=data).status_code
            == 403
        )
    with TestClient(app) as client:
        assert client.post("/api/auth/login", json={"username": "a", "password": "b"}).status_code == 403
    assert (
        support.post(
            "/api/chat", headers={"Origin": "https://untrusted.example"}, json={"question": "资料"}
        ).status_code
        == 403
    )


def test_E05_history_owner_and_E09_revocation(support):
    doc_id, _, chunk_id = source("support-sla")
    with SessionLocal() as db:
        row = Answer(
            user_id="xq-support",
            tenant_id="xingqiao",
            question="test only",
            payload={"status": "answered", "claims": [], "citations": []},
            evidence_chunk_ids=[chunk_id],
        )
        db.add(row)
        db.commit()
        answer_id = row.id
    try:
        assert support.get(f"/api/history/{answer_id}").json()["status"] == "answered"
        with TestClient(app, headers=HEADERS) as other:
            other.post(
                "/api/auth/login",
                json={"username": "admin@xingqiao.demo", "password": settings().demo_password},
            )
            assert other.get(f"/api/history/{answer_id}").status_code == 404
        with SessionLocal() as db:
            doc = db.get(Document, doc_id)
            doc.deleted = True
            db.commit()
        assert support.get(f"/api/evidence/{chunk_id}").status_code == 404
        hidden = support.get(f"/api/history/{answer_id}").json()
        assert hidden["status"] == "access_changed" and hidden["question"] != "test only"
        with SessionLocal() as db:
            doc = db.get(Document, doc_id)
            doc.deleted = False
            db.commit()
            user = db.get(User, "xq-support")
            user.groups = []
            db.commit()
        assert support.get(f"/api/evidence/{chunk_id}").status_code == 404
        assert support.get(f"/api/history/{answer_id}").json()["status"] == "access_changed"
    finally:
        with SessionLocal() as db:
            db.get(Document, doc_id).deleted = False
            db.get(User, "xq-support").groups = ["support"]
            db.delete(db.get(Answer, answer_id))
            db.commit()


@pytest.mark.parametrize("dependency", ["model", "search"])
def test_E07_dependency_failure_is_not_no_answer(support, monkeypatch, dependency):
    # Deliberate transport fault; not a model-quality or uptime measurement.
    from app.clients import Models, Search, DependencyError

    def broken(*args, **kwargs):
        raise DependencyError("验收注入的连接故障")

    monkeypatch.setattr(
        Models if dependency == "model" else Search,
        "embed" if dependency == "model" else "retrieve",
        broken,
    )
    if dependency == "search":
        monkeypatch.setattr(Models, "embed", lambda *args: [[0.0]])
    response = support.post("/api/chat", json={"question": "资料保留多久"})
    assert response.status_code == 503 and "claims" not in response.json()


def test_E06_bad_pdf_and_E10_failed_retry(support):
    response = support.post(
        "/api/documents", files={"file": ("verification-broken.pdf", b"%PDF-invalid", "application/pdf")}
    )
    assert response.status_code == 202
    doc_id, version_id = response.json()["id"], response.json()["version_id"]
    try:
        assert response.json()["status"] in {"queued", "processing"}
        for attempt in range(2):
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                doc = support.get(f"/api/documents/{doc_id}").json()
                if doc["status"] == "failed":
                    break
                time.sleep(0.2)
            assert doc["status"] == "failed" and doc["error"] and doc["chunks"] == 0
            with SessionLocal() as db:
                assert db.get(Document, doc_id).active_version_id is None
            if attempt == 0:
                assert support.post(f"/api/documents/{doc_id}/retry").status_code == 200
    finally:
        with SessionLocal() as db:
            version = db.get(DocumentVersion, version_id)
            storage_key = version.storage_key
            db.delete(db.scalar(select(Job).where(Job.version_id == version_id)))
            db.flush()
            db.delete(version)
            db.flush()
            db.delete(db.get(Document, doc_id))
            db.commit()
        (settings().storage_dir / storage_key).unlink(missing_ok=True)


def test_E10_expired_lease_recovery_and_exhaustion(monkeypatch):
    from app.ingestion import claim_job, queue_document

    future = now() + timedelta(days=10)
    monkeypatch.setattr("app.ingestion.now", lambda: future)

    with SessionLocal() as db:
        doc, version = queue_document(
            db,
            db.get(User, "xq-support"),
            "lease-check.md",
            "仅验证租约恢复".encode(),
            "lease-check",
            [],
            False,
        )
        job = db.scalar(select(Job).where(Job.version_id == version.id))
        # No worker can see a queued task after this transaction is committed.
        job.state, job.attempts = "processing", 1
        job.lease_until, job.lease_token = future - timedelta(seconds=1), "expired"
        db.commit()
        job_id, doc_id, version_id, storage_key = job.id, doc.id, version.id, version.storage_key
    try:
        claimed = claim_job()
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            assert job.state == "processing" and job.lease_token != "expired"
            assert claimed is None or claimed[0] == job_id
            job.state, job.attempts = "processing", 3
            job.lease_until = future - timedelta(seconds=1)
            db.commit()
        claim_job()
        with SessionLocal() as db:
            assert db.get(Job, job_id).state == "failed"
            assert db.get(DocumentVersion, version_id).status == "failed"
    finally:
        with SessionLocal() as db:
            db.delete(db.get(Job, job_id))
            db.flush()
            db.delete(db.get(DocumentVersion, version_id))
            db.flush()
            db.delete(db.get(Document, doc_id))
            db.commit()
        (settings().storage_dir / storage_key).unlink(missing_ok=True)


@pytest.mark.skipif(os.getenv('RAG_P0_P2_WORKER_CANDIDATES') != '1',
                    reason='requires worker with P0-P2 candidates explicitly enabled')
def test_async_candidate_comparison_and_revocation_hide_full_observable_payload(monkeypatch):
    """Real ingestion/model/worker/API plus revocation of the added event dependencies."""
    import json
    import uuid

    for flag in ('passage_window_enabled', 'focused_generation_enabled', 'adaptive_routing_enabled'):
        monkeypatch.setattr(settings(), flag, True)
    marker = 'Cedar' + uuid.uuid4().hex[:10]
    docs = []
    with TestClient(app, headers=HEADERS) as owner, TestClient(app, headers=HEADERS) as reader:
        for client, username in ((owner, 'admin@xingqiao.demo'), (reader, 'engineer@xingqiao.demo')):
            assert client.post('/api/auth/login', json={
                'username': username, 'password': settings().demo_password}).status_code == 200
        try:
            for index, hours in enumerate(('09:00', '10:00')):
                response = owner.post('/api/documents', files={'file': (
                    f'candidate-{index}.md',
                    f'{marker} release opens June 4, 2026 at {hours}.'.encode(), 'text/markdown')},
                    data={'title': f'{marker} release notice {index}', 'groups': '["engineering"]',
                          'tenant_public': 'false'})
                assert response.status_code == 202, response.text
                doc_id = response.json()['id']
                docs.append(doc_id)
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    state = owner.get(f'/api/documents/{doc_id}').json()
                    if state['status'] in {'ready', 'failed'}:
                        break
                    time.sleep(0.25)
                assert state['status'] == 'ready', state
            created = reader.post('/api/agent/tasks', json={
                'goal': f'Are the opening dates of {marker} release consistent across these two notices, despite different hours?',
                'mode': 'auto', 'max_steps': 6})
            assert created.status_code == 202, created.text
            task_id = created.json()['id']
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                task = reader.get(f'/api/agent/tasks/{task_id}').json()
                if task['status'] in {'completed', 'failed', 'paused'}:
                    break
                time.sleep(0.5)
            assert task['status'] == 'completed', task
            assert task['result']['status'] == 'answered', task['result']
            assert task['result']['usage']['deterministic_date_comparison']['value'] == 'yes'
            assert {row['document_id'] for row in task['result']['citations']} == set(docs)
            refs = {ref for event in task['events'] for ref in event['evidence_refs']}
            assert refs and any(event['event_type'] == 'generation_context' for event in task['events'])
            assert owner.patch(f'/api/documents/{docs[0]}/access', json={
                'groups': ['support'], 'tenant_public': False}).status_code == 200
            revoked = reader.get(f'/api/agent/tasks/{task_id}').json()
            assert revoked['result']['status'] == 'access_changed'
            assert not revoked['result']['claims'] and not revoked['result']['citations']
            encoded = json.dumps(revoked, ensure_ascii=False)
            assert 'June 4' not in encoded and '09:00' not in encoded
            assert all(json.dumps(ref) not in encoded for ref in refs)
            for event in revoked['events']:
                assert not event['evidence_refs']
                assert set(event['payload']) <= {'status', 'error_code', 'evidence_count', 'action'}
        finally:
            for doc_id in docs:
                owner.delete(f'/api/documents/{doc_id}')
