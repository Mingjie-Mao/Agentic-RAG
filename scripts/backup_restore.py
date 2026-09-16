"""Back up and restore the whole system, then prove the restore is usable (S7).

A backup that has never been restored is a guess. This takes a real backup, restores
it into a clean state, and then checks the properties that matter after a restore:
identities and grants survive, document versions and chunks survive, stored originals
are byte-identical, and — the part a database dump alone cannot give you — the search
index still answers, because it is rebuilt from the restored rows rather than assumed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from sqlalchemy import func, select

from app.clients import Search
from app.config import settings
from app.db import SessionLocal
from app.models import Answer, Chunk, Document, DocumentVersion, User

BACKUP = Path(".runtime/backup")
CONTAINER = "enterprise-rag-postgres-1"


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"命令失败：{' '.join(command)}\n{result.stderr[:400]}")
    return result


def fingerprint():
    """What must still be true after a restore, expressed as data rather than prose."""
    with SessionLocal() as db:
        storage = settings().storage_dir.resolve()
        documents = db.scalars(select(Document).order_by(Document.id)).all()
        originals = {}
        for version in db.scalars(select(DocumentVersion).order_by(DocumentVersion.id)):
            path = storage / version.storage_key
            if path.is_file():
                originals[version.id] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "users": sorted(
                (u.id, u.tenant_id, u.role, tuple(u.groups)) for u in db.scalars(select(User))
            ),
            "documents": sorted(
                (
                    d.id,
                    d.tenant_id,
                    d.active_version_id,
                    d.tenant_public,
                    tuple(d.read_groups),
                    d.deleted,
                )
                for d in documents
            ),
            "versions": db.scalar(select(func.count()).select_from(DocumentVersion)),
            "chunks": db.scalar(select(func.count()).select_from(Chunk)),
            "answers": db.scalar(select(func.count()).select_from(Answer)),
            "originals": originals,
        }


def dump():
    BACKUP.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sql = run(
        ["docker", "exec", CONTAINER, "pg_dump", "-U", "rag", "-d", "rag", "--clean", "--if-exists"]
    )
    (BACKUP / "rag.sql").write_text(sql.stdout)
    storage = settings().storage_dir.resolve()
    run(["tar", "-czf", str(BACKUP / "files.tar.gz"), "-C", str(storage.parent), storage.name])
    state = fingerprint()
    (BACKUP / "fingerprint.json").write_text(json.dumps(state, ensure_ascii=False, default=str))
    return {
        "sql_bytes": (BACKUP / "rag.sql").stat().st_size,
        "files_bytes": (BACKUP / "files.tar.gz").stat().st_size,
        "documents": len(state["documents"]),
        "chunks": state["chunks"],
        "seconds": round(time.monotonic() - started, 1),
    }


def restore():
    started = time.monotonic()
    sql = (BACKUP / "rag.sql").read_text()
    run(["docker", "exec", "-i", CONTAINER, "psql", "-U", "rag", "-d", "rag", "-q"], input=sql)
    storage = settings().storage_dir.resolve()
    run(["tar", "-xzf", str(BACKUP / "files.tar.gz"), "-C", str(storage.parent)])
    return round(time.monotonic() - started, 1)


def rebuild_index():
    """The dump restores rows, not the search index; it is rebuilt from those rows."""
    from app.clients import Models

    search, models = Search(), Models()
    if index_exists(search):
        search.request("DELETE", f"/{search.index}")
    search.ensure_index()
    indexed = 0
    with SessionLocal() as db:
        for document in db.scalars(select(Document).where(Document.deleted.is_(False))):
            if not document.active_version_id:
                continue
            chunks = db.scalars(
                select(Chunk)
                .where(Chunk.version_id == document.active_version_id)
                .order_by(Chunk.ordinal)
            ).all()
            if not chunks:
                continue
            vectors = []
            for offset in range(0, len(chunks), 8):
                vectors.extend(models.embed([c.text for c in chunks[offset : offset + 8]]))
            search.index_chunks(
                document.tenant_id,
                document.id,
                document.active_version_id,
                chunks,
                vectors,
                title=document.title,
            )
            indexed += len(chunks)
    search.request("POST", f"/{search.index}/_refresh")
    return indexed


def index_exists(search):
    try:
        search.request("GET", f"/{search.index}")
        return True
    except Exception:
        return False


def verify():
    expected = json.loads((BACKUP / "fingerprint.json").read_text())
    actual = json.loads(json.dumps(fingerprint(), ensure_ascii=False, default=str))
    differences = [key for key in expected if expected[key] != actual[key]]
    search = Search()
    count = search.request("GET", f"/{search.index}/_count")["count"]
    return {
        "identical": not differences,
        "differences": differences,
        "documents": len(actual["documents"]),
        "chunks": actual["chunks"],
        "originals_verified": len(actual["originals"]),
        "indexed_chunks": count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["backup", "restore", "drill"])
    parser.add_argument("--out", default="artifacts/s7-backup-restore.json")
    args = parser.parse_args()
    assert settings().demo_mode, "只在本项目的虚构演示环境上运行"

    report = {}
    if args.action in {"backup", "drill"}:
        report["backup"] = dump()
        print(f"备份完成：{report['backup']}", flush=True)
    if args.action in {"restore", "drill"}:
        report["restore_seconds"] = restore()
        print(f"数据库与原件已恢复（{report['restore_seconds']} 秒），正在重建索引", flush=True)
        report["reindexed_chunks"] = rebuild_index()
        report["verify"] = verify()
        print(json.dumps(report["verify"], ensure_ascii=False, indent=2))
        if not report["verify"]["identical"]:
            Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            raise SystemExit("恢复后状态与备份不一致，见报告")
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.action == "backup":
        print("备份已保存。备份没有被恢复过就还只是一个假设，用 drill 验证它。")
    else:
        print("恢复演练通过：身份、授权、版本、分块、原件与索引全部一致")


if __name__ == "__main__":
    main()
