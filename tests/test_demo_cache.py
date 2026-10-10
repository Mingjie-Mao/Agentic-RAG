from copy import deepcopy
from datetime import timedelta

import pytest

from app.config import settings
from app.demo_cache import cache_key, reuse_answer, store_key
from app.models import Answer, Chunk, Document, DocumentVersion, User, now
from app.trial import reserve_trial_request, trial_status
from test_trial import enable_trial, trial_db


@pytest.fixture
def cached(monkeypatch):
    db, user = trial_db()
    enable_trial(monkeypatch, limit=10)
    monkeypatch.setattr(settings(), "demo_answer_cache_enabled", True)
    doc = Document(id="doc", tenant_id=user.tenant_id, owner_id="someone-else", title="Quota",
                   active_version_id="v1", read_groups=["support"], tenant_public=False)
    version = DocumentVersion(id="v1", document_id="doc", filename="quota.md", media_type="text/markdown",
                              content_hash="source-sha", storage_key="unused", status="ready")
    chunk = Chunk(id="c1", version_id="v1", ordinal=0, text="120 calls", locator={})
    payload = {"status": "answered", "message": "", "claims": [{"text": "120 calls", "quotes": ["120 calls"], "evidence_ids": ["c1"]}],
               "citations": [{"chunk_id": "c1"}], "trace": {"total_ms": 39000, "generation_ms": 36500},
               "usage": {"prompt_tokens": 1273, "completion_tokens": 51}}
    row = Answer(id="answer", user_id=user.id, tenant_id=user.tenant_id, question="quota?",
                 evidence_chunk_ids=["c1"], payload=payload)
    db.add_all([doc, version, chunk, row])
    db.commit()
    key = cache_key(db, user, "quota?")
    store_key(db, user, "quota?", key, {"id": row.id, **deepcopy(payload)})
    return db, user, doc, version, row, key


def test_hit_revalidates_sources_records_new_history_and_zero_model_cost(cached):
    db, user, _, _, original, key = cached
    before = deepcopy(original.payload)
    reserve_trial_request(db, user, "chat")
    result = reuse_answer(db, user, "quota?", key)
    assert result["id"] != original.id
    assert result["claims"] == before["claims"] and result["citations"] == before["citations"]
    assert result["usage"]["model_calls"] == 0
    assert result["trace"]["generation_ms"] == 0
    assert result["trace"]["demo_cache"]["source_total_ms"] == 39000
    assert trial_status(db, user)["used"] == 1
    assert original.payload == before


@pytest.mark.parametrize("change", ["revoke", "delete", "version", "revision", "group", "role", "config", "new_document", "source_hash", "disable_user"])
def test_cache_misses_on_any_corpus_permission_identity_or_configuration_change(cached, change):
    db, user, doc, version, _, key = cached
    if change == "revoke":
        doc.read_groups = []
    elif change == "delete":
        doc.deleted = True
    elif change == "version":
        doc.active_version_id = "v2"
    elif change == "revision":
        doc.revision += 1
    elif change == "group":
        user.groups = ["engineering"]
    elif change == "role":
        user.role = "admin"
    elif change == "config":
        settings().compact_prompt_json = not settings().compact_prompt_json
    elif change == "new_document":
        db.add(Document(id="new", tenant_id=user.tenant_id, owner_id=user.id, title="Conflicting quota", read_groups=[]))
    elif change == "source_hash":
        version.content_hash = "changed"
    else:
        user.active = False
    db.commit()
    try:
        assert reuse_answer(db, user, "quota?", key) is None
    finally:
        if change == "config":
            settings().compact_prompt_json = not settings().compact_prompt_json


def test_no_reuse_across_principals_followups_benchmarks_or_expiry(cached):
    db, user, _, _, row, key = cached
    other = User(id="other", tenant_id=user.tenant_id, username="other@test", display_name="Other",
                 password_hash="unused", role="member", groups=["support"], active=True)
    db.add(other)
    db.commit()
    assert reuse_answer(db, other, "quota?", key) is None
    assert cache_key(db, user, "quota?", ["previous question"]) is None
    assert cache_key(db, user, "quota?", benchmark_run_token="dev-run") is None
    assert reuse_answer(db, user, "different?", key) is None
    row.created_at = now() - timedelta(days=2)
    db.commit()
    assert reuse_answer(db, user, "quota?", key) is None


def test_negative_or_unverified_answers_are_not_registered(cached):
    db, user, _, _, row, key = cached
    payload = deepcopy(row.payload)
    payload["status"] = "verification_failed"
    payload["trace"].pop("demo_cache")
    row.payload = payload
    db.commit()
    store_key(db, user, "quota?", key, {"id": row.id, **deepcopy(payload)})
    assert reuse_answer(db, user, "quota?", key) is None


def test_normal_question_entry_uses_cache_without_calling_generator(cached, monkeypatch):
    from app.qa import answer_question
    db, user, _, _, _, _ = cached
    monkeypatch.setattr("app.qa._answer_question", lambda *a, **kw: pytest.fail("cache hit must not generate"))
    result = answer_question(db, user, "quota?")
    assert result["trace"]["demo_cache"]["hit"] is True
