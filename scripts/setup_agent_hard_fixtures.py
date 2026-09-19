"""Idempotently create the benchmark-only three-version policy chain."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.ingestion import claim_job, process_job, queue_document
from app.lifecycle import add_version
from app.models import Document, DocumentVersion, User


ROOT = Path(__file__).resolve().parents[1]


def desired_versions(spec):
    return [
        {
            **row,
            "data": (ROOT / row["path"]).read_bytes(),
            "content_hash": hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest(),
        }
        for row in spec["versions"]
    ]


def missing_versions(existing_hashes, spec):
    return [row for row in desired_versions(spec) if row["content_hash"] not in existing_hashes]


def drain_jobs():
    while claimed := claim_job():
        process_job(*claimed)


def set_created_at(document_id, desired):
    dates = {row["content_hash"]: datetime.fromisoformat(row["effective_from"].replace("Z", "+00:00"))
             for row in desired}
    with SessionLocal() as db:
        versions = db.scalars(
            select(DocumentVersion).where(DocumentVersion.document_id == document_id)
        ).all()
        for version in versions:
            if version.content_hash in dates:
                version.created_at = dates[version.content_hash]
        db.commit()


def setup_document(spec):
    desired = desired_versions(spec)
    with SessionLocal() as db:
        user = db.get(User, spec["user"])
        if user is None:
            raise RuntimeError(f"missing seeded user {spec['user']}")
        document = db.get(Document, spec["document_id"])
        if document is None:
            first = desired[0]
            document, _ = queue_document(
                db, user, first["filename"], first["data"], spec["title"],
                spec["groups"], spec["tenant_public"],
                metadata={"benchmark_only": True}, document_id=spec["document_id"],
            )
        else:
            document.title = spec["title"]
            document.read_groups = spec["groups"]
            document.tenant_public = spec["tenant_public"]
            document.metadata_json = {"benchmark_only": True}
            db.commit()
    drain_jobs()
    for version_spec in desired[1:]:
        with SessionLocal() as db:
            user = db.get(User, spec["user"])
            hashes = set(db.scalars(
                select(DocumentVersion.content_hash)
                .where(DocumentVersion.document_id == spec["document_id"])
            ).all())
            if version_spec["content_hash"] not in hashes:
                add_version(
                    db, user, spec["document_id"], version_spec["filename"], version_spec["data"]
                )
        drain_jobs()
    set_created_at(spec["document_id"], desired)
    with SessionLocal() as db:
        document = db.get(Document, spec["document_id"], populate_existing=True)
        versions = db.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == spec["document_id"])
            .order_by(DocumentVersion.created_at)
        ).all()
        if [row.content_hash for row in versions] != [row["content_hash"] for row in desired]:
            raise RuntimeError(f"{spec['document_id']}: version chain differs from fixture")
        if any(row.status != "ready" for row in versions) or document.active_version_id != versions[-1].id:
            raise RuntimeError(f"{spec['document_id']}: version chain is not fully published")
        return {"document_id": document.id, "versions": len(versions), "active": versions[-1].filename}


def main():
    specs = json.loads((ROOT / "fixtures/agent/hard_documents.json").read_text())
    results = [setup_document(spec) for spec in specs]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
