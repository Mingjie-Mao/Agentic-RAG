"""Idempotently insert the fictional S0 accounts and queue source documents."""

import json
from pathlib import Path

from app.config import settings
from app.db import SessionLocal
from app.ingestion import queue_document
from app.models import Document, Tenant, User
from app.security import hasher

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = settings()
    if not cfg.demo_mode or not cfg.demo_password:
        raise SystemExit(
            "Set RAG_DEMO_MODE=true and a local RAG_DEMO_PASSWORD to seed synthetic accounts."
        )
    catalog = json.loads((ROOT / "fixtures/catalog.json").read_text())
    manifest = json.loads((ROOT / "fixtures/manifest.json").read_text())
    with SessionLocal() as db:
        for item in catalog["tenants"]:
            if not db.get(Tenant, item["id"]):
                db.add(Tenant(**item))
        db.commit()
        for item in catalog["users"]:
            if not db.get(User, item["id"]):
                db.add(User(**item, password_hash=hasher.hash(cfg.demo_password)))
        db.commit()
        count = 0
        for item in manifest:
            if db.get(Document, item["id"]):
                continue
            user = db.get(User, "xq-admin" if item["tenant_id"] == "xingqiao" else "hc-admin")
            path = ROOT / item["path"]
            queue_document(
                db,
                user,
                path.name,
                path.read_bytes(),
                item["title"],
                item["groups"],
                item["tenant_public"],
                metadata=item,
                document_id=item["id"],
            )
            count += 1
    print(f"Seed accounts ready; queued {count} new documents.")


if __name__ == "__main__":
    main()
