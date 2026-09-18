from app.clients import evidence_spans, schedule_conflict
from app.models import Base, Tenant, User
from app.retrieval import _boilerplate_only, foreign_tenant_mentioned, retrieve_authorized
from app.task_analysis import acceptance_items, conflict_intent, needs_document_diversity
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from types import SimpleNamespace


def test_compound_question_exposes_acceptance_items_and_requests_diversity():
    question = "生产数据库的 RPO、RTO 以及代码审查人数分别是多少？"
    items = acceptance_items(question)
    assert len(items) == 3
    assert any("RPO" in item for item in items)
    assert any("RTO" in item for item in items)
    assert needs_document_diversity(question) is True


def test_schedule_conflict_rule_requires_intent_different_docs_and_different_ranges():
    evidence = [
        {
            "id": "E1",
            "document_id": "a",
            "title": "安排 A",
            "text": "工作日 09:00 至 18:00。",
        },
        {
            "id": "E2",
            "document_id": "b",
            "title": "安排 B",
            "text": "每日 08:00 至 20:00。",
        },
    ]
    sources, _ = evidence_spans(evidence)
    cited = ["E1:S1", "E2:S1"]
    assert conflict_intent("到底哪些时段支持？") is True
    assert schedule_conflict("到底哪些时段支持？", sources, cited) == (
        "E1:S1",
        "E2:S1",
    )
    assert schedule_conflict("请列出两个团队的时段", sources, cited) is None


def test_schedule_rule_does_not_compare_two_spans_from_one_document():
    evidence = [
        {
            "id": "E1",
            "document_id": "a",
            "title": "等级表",
            "text": "P1 09:00 至 18:00。\nP2 08:00 至 20:00。",
        }
    ]
    sources, _ = evidence_spans(evidence)
    assert schedule_conflict("到底按哪个时段？", sources, ["E1:S1", "E1:S2"]) is None


def test_compound_retrieval_limits_one_document_from_crowding_out_others():
    documents = {
        "a": SimpleNamespace(id="a", active_version_id="va", title="A", metadata_json={}),
        "b": SimpleNamespace(id="b", active_version_id="vb", title="B", metadata_json={}),
    }
    chunks = {
        f"a{i}": (
            SimpleNamespace(id=f"a{i}", text=f"A {i}", locator={}),
            SimpleNamespace(id="va"),
            documents["a"],
        )
        for i in range(4)
    }
    chunks["b1"] = (
        SimpleNamespace(id="b1", text="B 1", locator={}),
        SimpleNamespace(id="vb"),
        documents["b"],
    )

    class Models:
        def embed(self, _):
            return [[0.0]]

    class Search:
        def retrieve_hybrid(self, *args):
            assert args[-1] == 12
            return [
                {"chunk_id": key, "score": 0.1, "bm25_rank": rank, "dense_rank": rank}
                for rank, key in enumerate(["a0", "a1", "a2", "a3", "b1"], 1)
            ]

    result = retrieve_authorized(
        None,
        SimpleNamespace(tenant_id="tenant"),
        "A 以及 B 分别是什么？",
        cfg=SimpleNamespace(
            top_k=4,
            retrieval_mode="hybrid",
            min_similarity=0.0,
            context_token_budget=5000,
        ),
        models=Models(),
        search=Search(),
        readable_documents_fn=lambda *_: list(documents.values()),
        require_chunk_fn=lambda _db, _user, chunk_id, **_kwargs: chunks[chunk_id],
    )
    assert [row["document_id"] for row in result.evidence] == ["a", "a", "b"]
    assert any(row["excluded_because"] == "document_quota" for row in result.candidates)


def test_fixture_provenance_chunk_does_not_consume_evidence_budget():
    assert _boilerplate_only(
        "# 文档标题\n本项目自建的虚构资料，仅用于开发与演示，不代表任何真实公司的制度。"
    )
    assert not _boilerplate_only(
        "本项目自建的虚构资料，仅用于开发与演示，不代表任何真实公司的制度。\n"
        "业务日志保留 90 天，审计日志保留 180 天。"
    )


def test_explicit_foreign_tenant_is_rejected_before_retrieval():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all([Tenant(id="a", name="星桥软件"), Tenant(id="b", name="海川工作室")])
        user = User(
            id="u",
            tenant_id="a",
            username="u@example.test",
            display_name="U",
            password_hash="unused",
            role="member",
            groups=[],
            active=True,
        )
        db.add(user)
        db.commit()
        assert foreign_tenant_mentioned(db, user, "海川工作室的网关标记是什么？") is True
        assert foreign_tenant_mentioned(db, user, "星桥软件的网关标记是什么？") is False
