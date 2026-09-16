"""Document lifecycle: replace a version, change who may read, delete (S6).

Two rules shape all of it.

The business database decides visibility, not the search index. Every one of these
operations commits the authoritative state first; index and file cleanup happen
afterwards and may fail without affecting correctness, because reads are filtered by
the database and every chunk is re-checked before it reaches a model or a reader.

A version becomes active only after its chunks are indexed and confirmed searchable.
Until then the previous version keeps serving, so a failed or slow reprocess degrades
to "nothing changed" rather than to an empty document.
"""

from sqlalchemy import delete, select

from app.clients import DependencyError, Search
from app.config import settings
from app.ingestion import build_version
from app.models import Chunk, Document, DocumentVersion, Job
from app.security import can_read, require_document


def add_version(db, user, document_id, filename, data):
    """Queue a replacement version. The active version is untouched until it publishes."""
    document = require_document(db, user, document_id, write=True)
    version, job = build_version(db, document, user, filename, data)
    # Any publish that was already in flight is now stale: it would make an older
    # version active after a newer one was requested.
    document.revision += 1
    db.commit()
    return document, version, job


def set_access(db, user, document_id, groups, tenant_public):
    document = require_document(db, user, document_id, write=True)
    document.read_groups = list(groups)
    document.tenant_public = bool(tenant_public)
    # Bumps the generation so an in-flight publish cannot land under the old grant.
    document.revision += 1
    db.commit()
    return document


def soft_delete(db, user, document_id):
    """Commit the deletion, then report which versions still need purging."""
    document = require_document(db, user, document_id, write=True)
    rows = db.scalars(select(DocumentVersion).where(DocumentVersion.document_id == document.id)).all()
    versions = [row.id for row in rows]
    storage_keys = [row.storage_key for row in rows]
    document.deleted = True
    document.active_version_id = None
    # Freeing the ingest key lets the same file be uploaded again later as a new document.
    document.ingest_key = None
    document.revision += 1
    if versions:
        db.execute(delete(Job).where(Job.version_id.in_(versions)))
    db.commit()
    return document, versions, storage_keys


def purge(version_ids, storage_keys=()):
    """Best-effort cleanup after the deletion is already committed.

    A failure here leaves searchable-but-unreadable chunks, which the read path
    already refuses; `scripts/reconcile_index.py` sweeps them later.
    """
    removed = {"chunks": 0, "files": 0, "errors": []}
    if version_ids:
        try:
            result = Search().request(
                "POST",
                f"/{Search().index}/_delete_by_query?refresh=true&conflicts=proceed",
                json={"query": {"terms": {"version_id": list(version_ids)}}},
            )
            removed["chunks"] = result.get("deleted", 0)
        except DependencyError as exc:
            removed["errors"].append(str(exc))
    storage = settings().storage_dir.resolve()
    for key in storage_keys:
        target = storage / key
        try:
            if target.is_file():
                target.unlink()
                removed["files"] += 1
        except OSError as exc:
            removed["errors"].append(str(exc))
    return removed


def readable_chunk(db, user, chunk_id):
    """Answer one question for callers that derived content from this corpus:
    may this user read this chunk *right now*? No titles or text are returned, so a
    negative answer reveals nothing beyond the answer itself."""
    chunk = db.get(Chunk, chunk_id)
    if not chunk:
        return {"chunk_id": chunk_id, "readable": False, "reason": "not_found"}
    version = db.get(DocumentVersion, chunk.version_id)
    document = db.get(Document, version.document_id) if version else None
    if not document or document.deleted:
        return {"chunk_id": chunk_id, "readable": False, "reason": "deleted"}
    if document.tenant_id != user.tenant_id or not can_read(user, document):
        return {"chunk_id": chunk_id, "readable": False, "reason": "not_permitted"}
    if document.active_version_id != version.id:
        return {
            "chunk_id": chunk_id,
            "readable": False,
            "reason": "superseded",
            "document_id": document.id,
            "active_version_id": document.active_version_id,
        }
    return {
        "chunk_id": chunk_id,
        "readable": True,
        "document_id": document.id,
        "version_id": version.id,
        "revision": document.revision,
    }


def orphan_versions(db):
    """Version IDs whose chunks must not remain searchable."""
    deleted = set(db.scalars(select(Document.id).where(Document.deleted.is_(True))))
    rows = db.execute(
        select(DocumentVersion.id, DocumentVersion.document_id, Document.active_version_id).join(
            Document, Document.id == DocumentVersion.document_id
        )
    ).all()
    return [
        version_id
        for version_id, document_id, active in rows
        if document_id in deleted or version_id != active
    ]
