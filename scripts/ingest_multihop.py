"""Ingest the MultiHop-RAG news corpus into an isolated tenant.

The 609 articles go through the same upload path as any other document: parse, chunk,
embed, index, publish. They are put in their own tenant so that nothing about the
existing Chinese corpus, its ACLs or its frozen evaluations changes — retrieval is
filtered by tenant, so the two corpora cannot see each other.

The article text is never committed; it is materialised from the cached dataset under
`.runtime/` at ingest time. Running this twice is a no-op: `queue_document` deduplicates
on content hash, ACL and pipeline settings.
"""

import argparse
import hashlib
import json
from pathlib import Path

from app.db import SessionLocal
from app.ingestion import claim_job, process_job, queue_document
from app.models import Document, Tenant, User
from app.security import hasher

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".runtime/multihop"
TENANT = {"id": "multihop", "name": "MultiHop-RAG External Corpus"}
USER = {
    "id": "mh-eval",
    "tenant_id": "multihop",
    "username": "eval@multihop.external",
    "display_name": "MultiHop 评测账号",
    "role": "admin",
    "groups": ["engineering", "support"],
    "active": True,
}


def document_id(title: str) -> str:
    return "mh-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:24]


def markdown(row: dict) -> bytes:
    header = " · ".join(
        part
        for part in (
            f"Source: {row.get('source')}" if row.get("source") else None,
            f"Author: {row.get('author')}" if row.get("author") else None,
            f"Published: {row.get('published_at')}" if row.get("published_at") else None,
            f"Category: {row.get('category')}" if row.get("category") else None,
        )
        if part
    )
    return f"# {row['title']}\n\n{header}\n\n{row['body'].strip()}\n".encode("utf-8")


def ensure_accounts(db):
    if not db.get(Tenant, TENANT["id"]):
        db.add(Tenant(**TENANT))
    if not db.get(User, USER["id"]):
        # No usable password: this account exists to own evaluation documents, and
        # nothing should be able to sign in as it.
        db.add(User(**USER, password_hash=hasher.hash(hashlib.sha256(b"unusable").hexdigest())))
    db.commit()


def drain():
    processed = 0
    while claimed := claim_job():
        process_job(*claimed)
        processed += 1
        if processed % 25 == 0:
            print(f"  processed {processed} jobs", flush=True)
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--drain-only", action="store_true")
    args = parser.parse_args()
    corpus = json.loads((CACHE / "corpus.json").read_text())[: args.limit]
    queued = 0
    with SessionLocal() as db:
        ensure_accounts(db)
        user = db.get(User, USER["id"])
        if not args.drain_only:
            for row in corpus:
                identifier = document_id(row["title"])
                if db.get(Document, identifier):
                    continue
                queue_document(
                    db,
                    user,
                    f"{identifier}.md",
                    markdown(row),
                    row["title"][:200],
                    [],
                    True,
                    metadata={
                        "source": row.get("source"),
                        "published_at": row.get("published_at"),
                        "category": row.get("category"),
                        "url": row.get("url"),
                        "dataset": "yixuantt/MultiHopRAG",
                    },
                    document_id=identifier,
                )
                queued += 1
    print(f"queued {queued} documents", flush=True)
    print(f"processed {drain()} jobs")
    with SessionLocal() as db:
        ready = db.query(Document).filter(Document.tenant_id == "multihop").count()
    print(f"documents in tenant multihop: {ready}")


if __name__ == "__main__":
    main()
