import json

from app.models import Document, DocumentVersion, User
from scripts.cleanup_browser_uploads import find_residue, receipt_ids
from test_trial import trial_db


def add_doc(db, owner, tmp_path, identifier, text="", **kwargs):
    row = Document(id=identifier, tenant_id=owner.tenant_id, owner_id=owner.id,
                   title="网页验收记录.md", active_version_id=identifier + "-v", **kwargs)
    version = DocumentVersion(id=row.active_version_id, document_id=row.id,
        filename="网页验收记录.md", media_type="text/markdown", content_hash="test",
        storage_key=identifier + ".md", status="ready")
    db.add_all([row, version])
    db.commit()
    (tmp_path / version.storage_key).write_text(text)
    return row


def test_legacy_cleanup_requires_both_markers_and_keeps_real_documents(tmp_path):
    db, owner = trial_db()
    add_doc(db, owner, tmp_path, "proven", "仅供网页上传验收的虚构文档 UI-S1-OK")
    add_doc(db, owner, tmp_path, "one-marker", "UI-S1-OK")
    add_doc(db, owner, tmp_path, "real-doc", "正式资料，文件名相同也不能清理")
    assert [row.id for row in find_residue(db, owner, tmp_path)] == ["proven"]


def test_interrupted_upload_receipt_cannot_remove_seed_shared_or_other_owner(tmp_path):
    db, owner = trial_db()
    other = User(id="other", tenant_id=owner.tenant_id, username="other@example.test",
        display_name="Other", password_hash="unused", role="member", groups=[], active=True)
    db.add(other)
    db.commit()
    for identifier, user, kwargs in [
        ("own-private", owner, {}), ("seed-protected", owner, {}),
        ("shared", owner, {"read_groups": ["support"]}),
        ("public", owner, {"tenant_public": True}), ("others", other, {}),
    ]:
        add_doc(db, user, tmp_path, identifier, **kwargs)
    receipts={"own-private", "seed-protected", "shared", "public", "others"}
    assert [row.id for row in find_residue(db, owner, tmp_path, receipts)] == ["own-private"]


def test_missing_original_is_skipped_but_interrupted_receipt_can_recover(tmp_path):
    db, owner = trial_db()
    row = add_doc(db, owner, tmp_path, "missing")
    (tmp_path / "missing.md").unlink()
    assert find_residue(db, owner, tmp_path) == []
    assert find_residue(db, owner, tmp_path, {row.id}) == [row]
    assert not row.deleted  # Discovery itself must remain dry-run.


def test_receipts_require_test_schema_and_ignore_corrupted_files(tmp_path):
    (tmp_path / "broken.json").write_text("{")
    (tmp_path / "not-test.json").write_text(json.dumps({"uploads": [{"document_id": "real"}]}))
    (tmp_path / "interrupted.json").write_text(json.dumps({"kind": "browser-test-upload-receipts", "uploads": [{"document_id": "private-test", "created_by_test": True}]}))
    assert receipt_ids(tmp_path) == {"private-test"}


def test_recovered_cleanup_soft_deletes_but_preserves_original_and_version(tmp_path):
    from app.lifecycle import soft_delete
    from app.security import can_read

    db, owner = trial_db()
    row = add_doc(db, owner, tmp_path, "interrupted", "已上传，测试进程中断")
    version_id = row.active_version_id
    matched = find_residue(db, owner, tmp_path, {row.id})
    assert matched == [row]
    for document in matched:
        soft_delete(db, owner, document.id)
    assert row.deleted and row.active_version_id is None
    assert not can_read(owner, row)
    assert db.get(DocumentVersion, version_id) is not None
    assert (tmp_path / "interrupted.md").read_text() == "已上传，测试进程中断"
