"""Reprocess seeded documents through the current parser, as new versions (S6).

The seed corpus was ingested in S1 with a parser that recorded no layout coordinates,
so its citations cannot be highlighted in the page. Rewriting those chunks in place
would make each chunk's recorded pipeline a lie, so instead every affected document
gets a genuine new version: indexed first, promoted only once it is searchable, with
the previous version left intact for auditing answers that cite it.
"""

import argparse
import json
from pathlib import Path
import time

from sqlalchemy import select

from app.db import SessionLocal
from app.ingestion import claim_job, current_pipeline, process_job
from app.lifecycle import add_version
from app.models import Document, DocumentVersion, User


def stale_documents(db, parser):
    rows = db.execute(
        select(Document, DocumentVersion)
        .join(DocumentVersion, Document.active_version_id == DocumentVersion.id)
        .where(Document.deleted.is_(False))
    ).all()
    return [(doc, version) for doc, version in rows if version.pipeline.get("parser") != parser]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="artifacts/s6-reprocess.json")
    args = parser.parse_args()

    target = current_pipeline()["parser"]
    with SessionLocal() as db:
        stale = stale_documents(db, target)[: args.limit]
        report = {
            "target_parser": target,
            "stale_documents": len(stale),
            "documents": [
                {"id": doc.id, "title": doc.title, "from_parser": version.pipeline.get("parser")}
                for doc, version in stale
            ],
        }
        if args.dry_run or not stale:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        queued = []
        for doc, old in stale:
            actor = db.get(User, doc.owner_id)
            data = (Path(".runtime/files") / old.storage_key).read_bytes()
            _, version, _ = add_version(db, actor, doc.id, old.filename, data)
            queued.append({"document_id": doc.id, "old": old.id, "new": version.id})
        print(f"已排队 {len(queued)} 份资料的新版本，开始处理", flush=True)

    started = time.monotonic()
    processed = 0
    while True:
        claim = claim_job()
        if not claim:
            break
        try:
            process_job(*claim)
        except Exception as exc:  # a single failure must not abandon the rest
            report.setdefault("failures", []).append(str(exc)[:200])
        processed += 1
        if processed % 5 == 0:
            print(f"已处理 {processed}/{len(queued)}", flush=True)

    with SessionLocal() as db:
        promoted = 0
        for item in queued:
            doc = db.get(Document, item["document_id"])
            item["promoted"] = doc.active_version_id == item["new"]
            promoted += item["promoted"]
        remaining = len(stale_documents(db, target))
    report |= {
        "queued": queued,
        "promoted": promoted,
        "still_stale": remaining,
        "seconds": round(time.monotonic() - started, 1),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"完成：{promoted}/{len(queued)} 份已切换到新版本，仍使用旧解析器的还有 {remaining} 份，"
        f"用时 {report['seconds']} 秒"
    )


if __name__ == "__main__":
    main()
