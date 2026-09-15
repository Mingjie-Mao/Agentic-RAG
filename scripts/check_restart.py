"""Snapshot/check real API history, seed originals and persisted database identities."""

import argparse
import hashlib
import json
from pathlib import Path

import httpx
from sqlalchemy import select

from app.config import settings
from app.clients import Search
from app.db import SessionLocal
from app.models import Answer, Chunk, Document, DocumentVersion

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["snapshot", "check"])
args = parser.parse_args()
cfg = settings()
assert cfg.demo_mode, "Only run against this project's fictional demo environment"
private = Path(".runtime/restart-snapshot.json")


def seed_state():
    result = []
    with SessionLocal() as db:
        for source in json.loads(Path("fixtures/manifest.json").read_text()):
            doc = db.get(Document, source["id"])
            version = db.get(DocumentVersion, doc.active_version_id)
            chunks = list(
                db.scalars(
                    select(Chunk.id).where(Chunk.version_id == version.id).order_by(Chunk.ordinal)
                )
            )
            actual = hashlib.sha256((cfg.storage_dir / version.storage_key).read_bytes()).hexdigest()
            assert actual == source["sha256"] == version.content_hash
            assert version.status == "ready" and chunks
            result.append(
                {"document_id": doc.id, "version_id": version.id, "sha256": actual, "chunks": chunks}
            )
    return result


with httpx.Client(
    base_url="http://127.0.0.1:8000",
    headers={"X-Requested-With": "EnterpriseRAG"},
    trust_env=False,
    timeout=15,
) as client:
    if args.action == "snapshot":
        client.post(
            "/api/auth/login", json={"username": "support@xingqiao.demo", "password": cfg.demo_password}
        ).raise_for_status()
        with SessionLocal() as db:
            row = db.scalar(
                select(Answer).where(Answer.user_id == "xq-support").order_by(Answer.created_at.desc())
            )
            assert row is not None
            answer_id = row.id
        response = client.get("/api/history/" + answer_id)
        response.raise_for_status()
        snapshot = {
            "cookies": dict(client.cookies),
            "answer_id": answer_id,
            "answer": response.json(),
            "seed": seed_state(),
        }
        private.write_text(json.dumps(snapshot, ensure_ascii=False))
        private.chmod(0o600)
        print(
            "Snapshot saved privately: 30 seed originals, chunk/version IDs, login session and history"
        )
    else:
        snapshot = json.loads(private.read_text())
        client.cookies.update(snapshot["cookies"])
        assert client.get("/api/auth/me").status_code == 200, "Session did not survive restart"
        actual = client.get("/api/history/" + snapshot["answer_id"])
        assert actual.status_code == 200 and actual.json() == snapshot["answer"]
        assert seed_state() == snapshot["seed"]
        source = next(d for d in snapshot["seed"] if d["document_id"] == "seed-upload-guide")
        original = client.get(f"/api/versions/{source['version_id']}/original")
        assert hashlib.sha256(original.content).hexdigest() == source["sha256"]
        indexed = Search().request("GET", f"/{cfg.search_index}/_doc/{source['chunks'][0]}")
        assert indexed["_source"]["version_id"] == source["version_id"]
        hits = Search().retrieve(indexed["_source"]["embedding"], "xingqiao", [source["version_id"]], 1)
        assert hits[0]["chunk_id"] == source["chunks"][0]
        result = {
            "scenario": "E08",
            "passed": True,
            "seed_documents": 30,
            "checks": [
                "seed file SHA-256",
                "version and chunk identities",
                "ready state",
                "login session",
                "answer payload",
                "authenticated PDF original",
                "persisted search vector and filtered retrieval",
            ],
            "answer_id": snapshot["answer_id"],
        }
        Path("artifacts/s1-restart.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        print("E08 restart verification passed")
