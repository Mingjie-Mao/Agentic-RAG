from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func, delete

from app.config import settings
from app.db import SessionLocal
from app.main import app
from app.models import Chunk, Document, DocumentVersion, Job, User, now

pytestmark = pytest.mark.skipif(
    os.getenv("RAG_RUN_INTEGRATION") != "1", reason="requires running demo services"
)
HEADERS = {"X-Requested-With": "AgenticRAG"}


def login(username="support@xingqiao.demo"):
    c = TestClient(app, headers=HEADERS)
    r = c.post("/api/auth/login", json={"username": username, "password": settings().demo_password})
    assert r.status_code == 200
    return c


def ready(client, doc_id):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        row = client.get(f"/api/documents/{doc_id}").json()
        if row["status"] == "ready":
            return row
        assert row["status"] != "failed", row
        time.sleep(0.25)
    raise AssertionError("Worker did not publish within 120 seconds")


def cleanup(doc_id):
    with SessionLocal() as db:
        versions = db.scalars(select(DocumentVersion).where(DocumentVersion.document_id == doc_id)).all()
        for version in versions:
            db.execute(delete(Job).where(Job.version_id == version.id))
            db.execute(delete(Chunk).where(Chunk.version_id == version.id))
            (settings().storage_dir / version.storage_key).unlink(missing_ok=True)
            db.delete(version)
        db.flush()
        db.delete(db.get(Document, doc_id))
        db.commit()


@pytest.mark.parametrize("filename", ["headings.docx", "formula-missing.xlsx", "text.pdf"])
def test_office_pdf_upload_preview_and_access(filename):
    client = login()
    response = client.post(
        "/api/documents", files={"file": (filename, Path("fixtures/s2", filename).read_bytes())}
    )
    assert response.status_code == 202, response.text
    doc_id = response.json()["id"]
    try:
        row = ready(client, doc_id)
        result = client.get(f"/api/documents/{doc_id}/processing").json()
        assert result["blocks"] and result["chunks"]
        original = client.get(result["original_url"])
        assert original.content == Path("fixtures/s2", filename).read_bytes()
        other = login("engineer@xingqiao.demo")
        assert other.get(f"/api/documents/{doc_id}/processing").status_code == 404
        assert other.get(result["original_url"]).status_code == 404
        with SessionLocal() as db:
            job = db.scalar(select(Job).where(Job.version_id == row["version_id"]))
            token = job.lease_token
        from app.ingestion import process_job

        process_job(job.id, token)
        with SessionLocal() as db:
            assert db.scalar(
                select(func.count()).select_from(Chunk).where(Chunk.version_id == row["version_id"])
            ) == len(result["chunks"])
    finally:
        cleanup(doc_id)
        client.close()


def test_concurrent_duplicate_upload_has_one_document_and_job():
    data = ("# 并发重复上传\n\n测试编号 " + str(uuid.uuid4())).encode()

    def upload(_):
        with login() as client:
            response = client.post("/api/documents", files={"file": ("dedup.md", data)})
            assert response.status_code == 202, response.text
            return response.json()

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(upload, range(4)))
    ids = {r["id"] for r in responses}
    assert len(ids) == 1 and sum(r["deduplicated"] for r in responses) == 3
    doc_id = next(iter(ids))
    try:
        with login() as client:
            ready(client, doc_id)
        with SessionLocal() as db:
            versions = list(
                db.scalars(select(DocumentVersion).where(DocumentVersion.document_id == doc_id))
            )
            assert len(versions) == 1
            assert (
                db.scalar(select(func.count()).select_from(Job).where(Job.version_id == versions[0].id))
                == 1
            )
    finally:
        cleanup(doc_id)


def test_transient_index_failure_can_retry_and_publish(monkeypatch):
    from app.ingestion import queue_document, process_job
    from app.clients import Search, DependencyError

    token = str(uuid.uuid4())
    with SessionLocal() as db:
        doc, version = queue_document(
            db,
            db.get(User, "xq-support"),
            "retry-valid.md",
            ("# 临时故障恢复\n\n" + token).encode(),
            "retry-valid",
            [],
            False,
        )
        job = db.scalar(select(Job).where(Job.version_id == version.id))
        job.state = "processing"
        job.lease_token = token
        job.lease_until = now() + timedelta(minutes=10)
        db.commit()
        job_id, doc_id, version_id = job.id, doc.id, version.id
    try:

        def fail(*args, **kwargs):
            raise DependencyError("验收注入：索引暂不可用")

        with monkeypatch.context() as fault:
            fault.setattr(Search, "index_chunks", fail)
            with pytest.raises(DependencyError):
                process_job(job_id, token)
        with SessionLocal() as db:
            assert db.get(Document, doc_id).active_version_id is None
            assert db.get(DocumentVersion, version_id).status == "failed"
        with login() as client:
            assert client.post(f"/api/documents/{doc_id}/retry").status_code == 200
            row = ready(client, doc_id)
            assert row["version_id"] == version_id
        with SessionLocal() as db:
            assert db.get(Job, job_id).state == "done"
    finally:
        cleanup(doc_id)
