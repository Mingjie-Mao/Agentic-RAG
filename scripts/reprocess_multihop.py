"""Re-ingest the MultiHop-RAG tenant under a new pipeline, as new versions.

The pipeline comes from the environment (`RAG_CHUNK_CONTEXT`, `RAG_BOILERPLATE_FILTER`,
`RAG_CHUNK_CHARS`, ...) exactly as for any upload, and is recorded on every new version.
Chunks are never rewritten in place: each article gets a genuine replacement version,
indexed first and promoted only once searchable, with the old version kept — the same
path as `scripts/reprocess_seed.py`. Articles already on the requested pipeline are
skipped, so an interrupted run resumes where it stopped.

Only tenant `multihop` is touched; the Chinese corpus and its frozen results are not.
"""

import argparse
import json
from pathlib import Path
import time

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.ingestion import claim_job, current_pipeline, process_job
from app.lifecycle import add_version
from app.models import Document, DocumentVersion, User

TENANT = "multihop"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    target = current_pipeline()
    started = time.monotonic()
    with SessionLocal() as db:
        rows = db.execute(
            select(Document, DocumentVersion)
            .join(DocumentVersion, Document.active_version_id == DocumentVersion.id)
            .where(Document.tenant_id == TENANT, Document.deleted.is_(False))
            .order_by(Document.id)
        ).all()
        stale = [(doc, version) for doc, version in rows if version.pipeline != target][: args.limit]
        print(f"{len(rows)} articles, {len(stale)} not on the requested pipeline", flush=True)
        queued = []
        for doc, old in stale:
            data = (settings().storage_dir.resolve() / old.storage_key).read_bytes()
            _, version, _ = add_version(db, db.get(User, doc.owner_id), doc.id, old.filename, data)
            queued.append({"document_id": doc.id, "old": old.id, "new": version.id})

    failures, processed = [], 0
    while claim := claim_job():
        try:
            process_job(*claim)
        except Exception as exc:  # one failure must not abandon the rest
            failures.append(str(exc)[:200])
        processed += 1
        if processed % 50 == 0:
            print(f"  processed {processed}/{len(queued)}", flush=True)

    with SessionLocal() as db:
        promoted = sum(
            db.get(Document, item["document_id"]).active_version_id == item["new"] for item in queued
        )
        versions = db.scalars(
            select(DocumentVersion)
            .join(Document, Document.active_version_id == DocumentVersion.id)
            .where(Document.tenant_id == TENANT)
        ).all()
        chunks = sum((v.timings or {}).get("chunks", 0) for v in versions)
        removed = sum((v.timings or {}).get("furniture_blocks_removed", 0) for v in versions)
        on_target = sum(v.pipeline == target for v in versions)
    report = {
        "tenant": TENANT,
        "pipeline": target,
        "queued": len(queued),
        "promoted": promoted,
        "failures": failures,
        "articles_on_pipeline": on_target,
        "active_chunks": chunks,
        "furniture_blocks_removed": removed,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
