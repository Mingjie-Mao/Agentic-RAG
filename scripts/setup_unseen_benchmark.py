"""Create the unseen benchmark's tenant, accounts and documents, idempotently.

The corpus (fixtures/unseen/documents/) is new: no rule, prompt, threshold or evaluator
was developed on it. It lives in its own tenant so that neither the Hard 30 regression
set nor the frozen Chinese evaluations see it. Documents go through the normal upload,
parse, index and publish path; version chains are added as real versions and their
creation dates set to the effective dates the documents state.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys
import secrets

from app.db import SessionLocal
from app.ingestion import claim_job, process_job, queue_document
from app.lifecycle import add_version
from app.models import Document, DocumentVersion, Tenant, User
from app.security import hasher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = ROOT / "fixtures/unseen/documents.json"


def drain():
    while claimed := claim_job():
        process_job(*claimed)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, default=SPEC.parent)
    args = parser.parse_args()
    spec = json.loads((args.suite_dir / "documents.json").read_text())
    suite_file = args.suite_dir / "tasks.json"
    if suite_file.exists():
        suite = json.loads(suite_file.read_text())
        if suite.get("evaluation_protocol") == "unseen-v2-semantic-review":
            from scripts.unseen_v2_scoring import reviewed_suite
            reviewed_suite(suite, ROOT)
        if suite.get("evaluation_protocol") == "enterprise-benchmark-v1" and suite.get("split") != "dev":
            from scripts.benchmark_package_scoring import require_task_review
            require_task_review(suite, spec, ROOT)
    with SessionLocal() as db:
        for tenant in [spec["tenant"], *spec.get("tenants", [])]:
            if not db.get(Tenant, tenant["id"]):
                db.add(Tenant(**tenant))
        for user in spec["users"]:
            if not db.get(User, user["id"]):
                # No usable password: these accounts exist to own and read benchmark data.
                db.add(User(**user, active=True, password_hash=hasher.hash(secrets.token_urlsafe(48))))
        db.commit()
        owner = db.get(User, spec["owner"])
        for doc in spec["documents"]:
            owner = db.get(User, doc.get("owner", spec["owner"]))
            versions = doc["versions"]
            if db.get(Document, doc["document_id"]) is None:
                first = versions[0]
                queue_document(db, owner, Path(first["path"]).name, (ROOT / first["path"]).read_bytes(),
                               doc["title"][:200], doc["groups"], doc["tenant_public"],
                               document_id=doc["document_id"], metadata=doc.get("metadata"))
                drain()
            existing = {v.content_hash for v in db.query(DocumentVersion).filter_by(document_id=doc["document_id"])}
            for row in versions[1:]:
                data = (ROOT / row["path"]).read_bytes()
                if hashlib.sha256(data).hexdigest() not in existing:
                    add_version(db, owner, doc["document_id"], Path(row["path"]).name, data)
                    drain()
                    existing.add(hashlib.sha256(data).hexdigest())
            dates = {hashlib.sha256((ROOT / r["path"]).read_bytes()).hexdigest():
                     datetime.fromisoformat(r["effective_from"].replace("Z", "+00:00")) for r in versions}
            for version in db.query(DocumentVersion).filter_by(document_id=doc["document_id"]):
                if version.content_hash in dates:
                    version.created_at = dates[version.content_hash]
            db.commit()
        rows = db.query(Document).filter_by(tenant_id=spec["tenant"]["id"]).all()
        ready = sum(1 for d in rows if d.active_version_id)
        versions = db.query(DocumentVersion).join(Document).filter(Document.tenant_id == spec["tenant"]["id"]).count()
    print(f"tenant {spec['tenant']['id']}: {len(rows)} documents, {ready} published, {versions} versions")


if __name__ == "__main__":
    main()
