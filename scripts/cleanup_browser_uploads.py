"""Soft-delete only S1 browser-test residue proven by filename, owner and marker.

Dry-run by default. Original files and historical reports are preserved.
"""

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.config import settings
from app.lifecycle import soft_delete
from app.models import Document, DocumentVersion, User


def receipt_ids(directory):
    """Only completed upload receipts, never credentials or arbitrary ID lists."""
    ids = set()
    for path in Path(directory).glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("kind") != "browser-test-upload-receipts":
            continue
        uploads = payload.get("uploads")
        if not isinstance(uploads, list):
            continue
        for row in uploads:
            if isinstance(row, dict) and row.get("created_by_test") is True and isinstance(row.get("document_id"), str):
                ids.add(row["document_id"])
    return ids


def find_residue(db, owner, storage_dir, receipts=()):
    rows = db.scalars(select(Document).where(
        Document.owner_id == owner.id,
        Document.tenant_id == owner.tenant_id,
        Document.deleted.is_(False),
        Document.tenant_public.is_(False),
    )).all()
    matched = []
    for row in rows:
        if row.id.startswith("seed-") or row.read_groups:
            continue
        if row.id in receipts:
            matched.append(row)
            continue
        if row.title != "网页验收记录.md":
            continue
        version = db.get(DocumentVersion, row.active_version_id) if row.active_version_id else None
        if not version or not version.storage_key:
            continue
        path = (Path(storage_dir) / version.storage_key).resolve()
        if not path.is_relative_to(Path(storage_dir).resolve()):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "仅供网页上传验收的虚构文档" in text and "UI-S1-OK" in text:
            matched.append(row)
    return matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipts", type=Path, default=Path(__file__).resolve().parents[1] / ".runtime/browser-uploads")
    args = parser.parse_args()
    with SessionLocal() as db:
        owner = db.scalar(select(User).where(User.username == "support@xingqiao.demo"))
        if not owner:
            return
        matched = find_residue(db, owner, settings().storage_dir, receipt_ids(args.receipts))
        for row in matched:
            print(row.id, "soft-delete" if args.apply else "would soft-delete")
            if args.apply:
                soft_delete(db, owner, row.id)
        print(f"matched={len(matched)}; originals and evaluation artifacts preserved")


if __name__ == "__main__":
    main()
