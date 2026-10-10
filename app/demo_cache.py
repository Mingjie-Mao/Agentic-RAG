"""Exact verified-answer reuse via existing Answer records, only for trial Q&A.

No semantic similarity cache or second persistence framework. Changes to any
visible document, access identity, configuration or server epoch invalidate reuse.
Benchmark requests, follow-ups, failures and Agent tasks never use this cache.
"""

from copy import deepcopy
from datetime import timedelta
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

from sqlalchemy import select

from app.config import settings
from app.models import Answer, DocumentVersion, now
from app.security import readable_documents
from app.trial import is_trial_user

_EPOCH = uuid4().hex


@lru_cache
def code_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(_EPOCH.encode())
    for path in sorted((root / "app").glob("*.py")) + sorted((root / "agent").glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_key(db, user, question, history=None, *, benchmark_run_token=None):
    cfg = settings()
    if not cfg.demo_answer_cache_enabled or history or benchmark_run_token or not is_trial_user(user):
        return None
    documents = readable_documents(db, user)
    if not user.active or not documents:
        return None
    versions = list(db.scalars(select(DocumentVersion).where(
        DocumentVersion.document_id.in_([d.id for d in documents])
    ).execution_options(populate_existing=True)))
    identity = {"principal": [user.tenant_id, user.id, user.role, sorted(user.groups), user.active],
                "question": question, "code": code_identity(),
                "configuration": cfg.model_dump(mode="json", exclude={"database_url", "demo_password", "memory_tokens_json"}),
                "documents": sorted([[d.id, d.active_version_id, d.revision, d.owner_id,
                                      sorted(d.read_groups), d.tenant_public, d.title] for d in documents]),
                "versions": sorted([[v.id, v.content_hash, v.status, v.pipeline] for v in versions], key=lambda v: v[0])}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def store_key(db, user, question, key, payload):
    if (not key or payload.get("status") != "answered" or not payload.get("claims")
            or not payload.get("citations") or cache_key(db, user, question) != key):
        return
    row = db.get(Answer, payload["id"])
    if row is None or row.user_id != user.id or row.tenant_id != user.tenant_id:
        return
    payload["trace"] = {**payload.get("trace", {}), "demo_cache": {"hit": False, "key": key}}
    row.payload = {k: deepcopy(v) for k, v in payload.items() if k not in {"id", "question"}}
    db.commit()


def reuse_answer(db, user, question, key):
    from app.qa import visible_answer
    if not key:
        return None
    started = time.monotonic()
    cutoff = now() - timedelta(seconds=settings().demo_answer_cache_seconds)
    source = db.scalar(select(Answer).where(
        Answer.user_id == user.id, Answer.tenant_id == user.tenant_id, Answer.question == question,
        Answer.created_at >= cutoff,
        Answer.payload["trace"]["demo_cache"]["key"].as_string() == key,
        Answer.payload["trace"]["demo_cache"]["hit"].as_boolean().is_(False),
    ).order_by(Answer.created_at.desc()).limit(1))
    if source is None or source.payload.get("status") != "answered":
        return None
    payload = deepcopy(visible_answer(db, user, source))
    if payload.get("status") != "answered" or cache_key(db, user, question) != key:
        return None
    trace = payload.get("trace", {})
    trace.update(method="exact_verified_cache", generation_ms=0, embed_ms=0,
                 retrieval_ms=0, total_ms=round((time.monotonic() - started) * 1000, 1),
                 demo_cache={"hit": True, "source_answer_id": source.id,
                             "source_created_at": source.created_at.isoformat(),
                             "source_total_ms": source.payload.get("trace", {}).get("total_ms"),
                             "checked_current_access_and_versions": True})
    payload["trace"] = trace
    payload["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "model_calls": 0, "cache_hit": True}
    row = Answer(user_id=user.id, tenant_id=user.tenant_id, question=question,
                 evidence_chunk_ids=list(source.evidence_chunk_ids),
                 payload={k: v for k, v in payload.items() if k not in {"id", "question"}})
    db.add(row)
    db.commit()
    return {"id": row.id, "question": question, **row.payload}
