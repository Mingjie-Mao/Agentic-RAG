import hashlib
import json
from pathlib import Path
import time
import uuid
from datetime import timedelta

from sqlalchemy import delete, or_, select, update, text

from app.clients import Models, Search, DependencyError
from app.config import settings
from app.db import SessionLocal
from app.models import Chunk, Document, DocumentVersion, Job, User, now, uid
from app.parsing import parse_document, split_passages
from app.security import can_write


def queue_document(
    db, user, filename, data, title, groups, tenant_public, metadata=None, document_id=None
):
    extension = Path(filename).suffix.lower()
    media_type = {
        ".md": "text/markdown",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }.get(extension)
    if not media_type:
        raise ValueError("支持 .md、文本型 .pdf、.docx 和 .xlsx")
    if not data or len(data) > settings().max_upload_bytes:
        raise ValueError("文件不能为空，且不能超过 10 MB")
    version_id = uid()
    pipeline = {
        "parser": settings().parser_version,
        "chunking": settings().chunk_strategy,
        "chunk_chars": settings().chunk_chars,
        "overlap": settings().chunk_overlap,
        "embedding_model": settings().embed_model,
        "embedding_dimension": settings().embed_dimension,
        "token_counter": "cl100k_base (estimate, not generation tokenizer)",
    }
    content_hash = hashlib.sha256(data).hexdigest()
    identity = [
        user.tenant_id,
        user.id,
        content_hash,
        media_type,
        sorted(set(groups)),
        bool(tenant_public),
        pipeline,
        document_id,
    ]
    ingest_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": int(ingest_key[:15], 16)})
    existing = db.scalar(select(Document).where(Document.ingest_key == ingest_key))
    if existing and not existing.deleted:
        latest = db.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == existing.id)
            .order_by(DocumentVersion.created_at.desc())
            .limit(1)
        )
        existing._deduplicated = True
        db.commit()
        return existing, latest
    if existing:
        existing.ingest_key = None
        db.flush()
    storage = settings().storage_dir.resolve()
    storage.mkdir(parents=True, exist_ok=True)
    storage_key = version_id + extension
    path = storage / storage_key
    document = Document(
        id=document_id or uid(),
        tenant_id=user.tenant_id,
        owner_id=user.id,
        title=title,
        ingest_key=ingest_key,
        read_groups=groups,
        tenant_public=tenant_public,
        metadata_json=metadata or {},
    )
    version = DocumentVersion(
        id=version_id,
        document_id=document.id,
        filename=Path(filename).name,
        media_type=media_type,
        content_hash=content_hash,
        storage_key=storage_key,
        pipeline=pipeline,
    )
    try:
        path.write_bytes(data)
        path.chmod(0o600)
        db.add(document)
        db.flush()
        db.add(version)
        db.flush()
        db.add(Job(version_id=version_id, actor_id=user.id))
        db.commit()
    except Exception:
        db.rollback()
        path.unlink(missing_ok=True)
        raise
    return document, version


def claim_job():
    with SessionLocal() as db, db.begin():
        exhausted = db.scalars(
            select(Job)
            .where(Job.state == "processing", Job.lease_until < now(), Job.attempts >= 3)
            .with_for_update(skip_locked=True)
        ).all()
        for stale in exhausted:
            stale.state, stale.lease_until, stale.lease_token = "failed", None, None
            version = db.get(DocumentVersion, stale.version_id)
            version.status, version.error = "failed", "处理多次中断，请检查服务后手动重试"
        job = db.scalar(
            select(Job)
            .where(
                or_(Job.state == "queued", (Job.state == "processing") & (Job.lease_until < now())),
                Job.attempts < 3,
            )
            .order_by(Job.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if job is None:
            return None
        job.state, job.lease_token = "processing", uid()
        job.lease_until = now() + timedelta(minutes=10)
        job.attempts += 1
        version = db.get(DocumentVersion, job.version_id)
        version.status, version.error = "processing", None
        return job.id, job.lease_token


def heartbeat(job_id, token):
    with SessionLocal() as db:
        result = db.execute(
            update(Job)
            .where(Job.id == job_id, Job.lease_token == token, Job.state == "processing")
            .values(lease_until=now() + timedelta(minutes=10))
        )
        db.commit()
        if result.rowcount != 1:
            raise RuntimeError("Task lease was replaced")


def process_job(job_id: str, token: str):
    started = time.monotonic()
    try:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if not job or job.state != "processing" or job.lease_token != token:
                return
            version = db.get(DocumentVersion, job.version_id)
            document = db.get(Document, version.document_id)
            actor = db.get(User, job.actor_id)
            if not actor or not can_write(actor, document):
                raise ValueError("上传者已无权发布此资料")
            version_id, document_id, tenant_id = version.id, document.id, document.tenant_id
            title = document.title
            revision = document.revision
            pipeline = version.pipeline
            if pipeline["embedding_model"] != settings().embed_model:
                raise ValueError("任务模型配置已改变，请重新上传")
            data = (settings().storage_dir.resolve() / version.storage_key).read_bytes()
            structured = pipeline["parser"] != "pypdf6+utf8-v1"
            passages = parse_document(data, version.media_type, structured=structured)
            from app.chunking import chunk_blocks

            pieces = (
                chunk_blocks(
                    passages, pipeline["chunk_chars"], pipeline["overlap"], pipeline["chunking"]
                )
                if structured
                else split_passages(passages, pipeline["chunk_chars"], pipeline["overlap"])
            )
        if len(pieces) > 500:
            raise ValueError("分块超过 S1 单份文件的 500 块上限")
        parse_ms = round((time.monotonic() - started) * 1000, 1)
        chunks = [
            Chunk(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{version_id}:{n}")),
                version_id=version_id,
                ordinal=n,
                text=p.text,
                locator=p.locator,
            )
            for n, p in enumerate(pieces)
        ]
        models, search = Models(), Search()
        search.ensure_index()
        vectors = []
        embedding_start = time.monotonic()
        for offset in range(0, len(chunks), 8):
            heartbeat(job_id, token)
            vectors.extend(models.embed([c.text for c in chunks[offset : offset + 8]]))
        embedding_ms = round((time.monotonic() - embedding_start) * 1000, 1)
        heartbeat(job_id, token)
        search.index_chunks(tenant_id, document_id, version_id, chunks, vectors, title=title)
        with SessionLocal() as db, db.begin():
            job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
            document = db.scalar(select(Document).where(Document.id == document_id).with_for_update())
            actor = db.get(User, job.actor_id)
            if job.lease_token != token or job.state != "processing":
                return
            if not actor or not can_write(actor, document) or document.revision != revision:
                raise ValueError("资料或权限已变化，本次发布已取消")
            db.execute(delete(Chunk).where(Chunk.version_id == version_id))
            db.add_all(chunks)
            version = db.get(DocumentVersion, version_id)
            version.status = "ready"
            version.parsed_blocks = [{"text": p.text, "locator": p.locator} for p in passages]
            version.timings = {
                "parse_ms": parse_ms,
                "embedding_ms": embedding_ms,
                "total_ms": round((time.monotonic() - started) * 1000, 1),
                "chunks": len(chunks),
                "pages": len(passages),
            }
            document.active_version_id = version_id
            job.state, job.lease_until = "done", None
    except Exception as exc:
        # Persist a safe user-facing error; no source text or credentials in logs.
        error = (
            str(exc) if isinstance(exc, (ValueError, DependencyError)) else "处理失败，请检查服务后重试"
        )
        with SessionLocal() as db, db.begin():
            job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
            if job and job.lease_token == token:
                job.state, job.lease_until = "failed", None
                version = db.get(DocumentVersion, job.version_id)
                version.status, version.error = "failed", error
        raise
