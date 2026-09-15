import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.clients import Claim, GeneratedAnswer
from app.parsing import ParseError, parse_document, split_passages
from app.qa import validate_claims
from app.security import can_read, can_write


@pytest.mark.parametrize(
    "data,media",
    [
        (b"\xff", "text/markdown"),
        (b"\x00", "text/markdown"),
        (b"  ", "text/markdown"),
        (b"broken", "application/pdf"),
        (b"%PDF-1.7\nbroken", "application/pdf"),
    ],
)
def test_bad_files_fail_explicitly(data, media):
    with pytest.raises(ParseError):
        parse_document(data, media)


def test_chunk_locators_reconstruct_source_without_gaps():
    source = "第一行：数据保留 90 天。\n第二行：审计保留 180 天。\n" * 70
    passages = parse_document(source.encode(), "text/markdown")
    chunks = split_passages(passages)
    covered = set()
    for chunk in chunks:
        loc = chunk.locator
        assert source[loc["char_start"] : loc["char_end"]] == chunk.text
        assert loc["line_start"] == source[: loc["char_start"]].count("\n") + 1
        assert len(chunk.text) <= 700
        covered.update(range(loc["char_start"], loc["char_end"]))
    assert len(covered) == len(source)


def test_all_fixture_pdfs_have_real_page_locators():
    for row in json.loads(Path("fixtures/manifest.json").read_text()):
        if row["path"].endswith(".pdf"):
            pages = parse_document(Path(row["path"]).read_bytes(), "application/pdf")
            assert pages and all(p.locator["page"] >= 1 and p.text.strip() for p in pages)


@pytest.mark.parametrize(
    "evidence_ids,quotes,status",
    [
        (["E1"], ["业务日志保留 90 天"], "answered"),
        (["E9"], ["业务日志保留 90 天"], "verification_failed"),
        (["E1"], ["业务日志保留 900 天"], "verification_failed"),
        (["E1", "E1"], ["业务日志保留 90 天"], "verification_failed"),
    ],
)
def test_fabricated_citation_or_quote_is_rejected(evidence_ids, quotes, status):
    generated = GeneratedAnswer(
        answerable=True, claims=[Claim(text="保留 90 天。", evidence_ids=evidence_ids, quotes=quotes)]
    )
    checked, actual = validate_claims(
        generated, [{"id": "E1", "chunk_id": "real-chunk", "text": "业务日志保留 90 天。"}]
    )
    assert actual == status
    if checked:
        assert checked[0]["evidence_ids"] == ["real-chunk"]


def test_tenant_boundary_even_for_admin_and_owner():
    doc = SimpleNamespace(
        tenant_id="a", owner_id="u", tenant_public=True, read_groups=["engineering"], deleted=False
    )
    user = SimpleNamespace(tenant_id="b", id="u", role="admin", groups=["engineering"], active=True)
    assert not can_read(user, doc) and not can_write(user, doc)
    user.tenant_id = "a"
    assert can_read(user, doc) and can_write(user, doc)
    doc.deleted = True
    assert not can_read(user, doc) and not can_write(user, doc)


def test_model_protocol_pairs_ids_and_quotes_and_excludes_database_ids(monkeypatch):
    from app.clients import Models

    def reply(self, path, body):
        assert path == "/api/chat"
        assert body["format"]["$defs"]["ModelClaim"]["properties"]["source_ids"]["items"]["enum"] == [
            "E1:S1"
        ]
        assert "database-private-id" not in body["messages"][1]["content"]
        return {
            "message": {
                "content": json.dumps(
                    {
                        "status": "answered",
                        "claims": [
                            {
                                "text": "保留 90 天。",
                                "source_ids": ["E1:S1"],
                            }
                        ],
                    }
                )
            }
        }

    monkeypatch.setattr(Models, "_post", reply)
    generated, _ = Models().generate(
        "保留多久？",
        [
            {
                "id": "E1",
                "title": "保留政策",
                "text": "业务日志保留 90 天",
                "chunk_id": "database-private-id",
            }
        ],
    )
    assert generated.claims[0].evidence_ids == ["E1"]
    assert generated.claims[0].quotes == ["业务日志保留 90 天"]


def test_permission_revoked_during_generation_never_returns_answer(monkeypatch):
    from app import qa

    doc = SimpleNamespace(active_version_id="v", id="d", title="restricted", metadata_json={})
    chunk = SimpleNamespace(id="c", text="业务日志保留 90 天。", locator={})
    state = {"revoked": False}

    class FakeModels:
        def embed(self, _):
            return [[0.0]]

        def generate(self, *args):
            state["revoked"] = True
            return GeneratedAnswer(answerable=False, claims=[]), {}

    class FakeSearch:
        def retrieve(self, *args):
            return [{"chunk_id": "c", "cosine_similarity": 0.8}]

    def read(*args, **kwargs):
        if state["revoked"]:
            raise HTTPException(404)
        return chunk, SimpleNamespace(id="v"), doc

    monkeypatch.setattr(qa, "Models", FakeModels)
    monkeypatch.setattr(qa, "Search", FakeSearch)
    monkeypatch.setattr(qa, "readable_documents", lambda *args: [doc])
    monkeypatch.setattr(qa, "require_chunk", read)
    with pytest.raises(HTTPException) as exc:
        qa.answer_question(None, SimpleNamespace(tenant_id="a"), "日志保留多久？")
    assert exc.value.status_code == 404
