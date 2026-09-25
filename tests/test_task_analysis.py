from app.clients import evidence_spans, schedule_conflict
from app.models import Base, Tenant, User
from app.retrieval import _boilerplate_only, foreign_tenant_mentioned, retrieve_authorized
from app.task_analysis import (
    acceptance_items,
    conflict_intent,
    judgment_intent,
    multi_source_intent,
    needs_document_diversity,
)
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
        blocked = retrieve_authorized(
            db, user, "海川工作室的网关标记是什么？", cfg=SimpleNamespace(top_k=4),
            models=None, search=None, readable_documents_fn=lambda *_: [],
        )
        assert blocked.blocked_reason == "tenant_scope"
        assert blocked.evidence == []


def test_judgment_questions_are_recognized_in_both_languages():
    # The auxiliary verb can open the question or the last clause of a long one.
    assert judgment_intent("Do the TechCrunch and Hacker News articles both report a rise?")
    assert judgment_intent(
        "After the October 7 report on Flexport, and the October 30 article, "
        "was there a change in the perception of its leadership?"
    )
    assert judgment_intent("试点安排 A 和 B 的支持时段是否一致？")
    # A long fronted adjunct pushes the inverted auxiliary into the middle.
    assert judgment_intent(
        "After the report from Fortune on October 4, which discussed the claims, "
        "did The Verge's report on October 12 maintain consistency with it?"
    )
    # A question that asks for a name or a number is not a verdict question.
    assert not judgment_intent("Who is the individual facing a criminal trial on fraud charges?")
    assert not judgment_intent("Which article reported the acquisition first?")
    # A wh-question can still end in an auxiliary clause; it is asking for a name.
    assert not judgment_intent(
        "What institution, frequently mentioned in articles from one paper, "
        "is the focal point of investors' hopes regarding a halt to rate rises?"
    )
    assert not judgment_intent("当前 P1 工单首次响应时限是多少？")
    assert not judgment_intent("Python SDK 默认最多重试几次？")
    assert not judgment_intent("Did Fortune report a larger or smaller decrease?")


def test_multi_source_intent_does_not_change_the_chinese_retrieval_budget():
    assert multi_source_intent("Do both articles report an increase in revenue, respectively?")
    assert multi_source_intent("Compare the Verge report and the TechCrunch story on the trial.")
    assert multi_source_intent("The BBC article and the Times of India report disagree.")
    # The internal corpus and its frozen evaluations are Chinese; none of these may
    # start asking for a different number of chunks as a side effect.
    for question in (
        "生产数据库的 RPO、RTO 以及代码审查人数分别是多少？",
        "试点安排 A 和 B 的支持时段是否一致？",
        "当前 P1 工单首次响应时限是多少？",
    ):
        assert not multi_source_intent(question)


def test_multi_source_question_gets_the_full_budget_and_a_document_quota():
    documents = {
        key: SimpleNamespace(id=key, active_version_id=f"v{key}", title=key.upper(), metadata_json={})
        for key in ("a", "b", "c", "d")
    }
    chunks = {
        f"{key}{index}": (
            SimpleNamespace(id=f"{key}{index}", text=f"{key} {index}", locator={}),
            SimpleNamespace(id=f"v{key}"),
            documents[key],
        )
        for key in documents
        for index in range(4)
    }
    order = ["a0", "a1", "a2", "a3", "b0", "b1", "c0", "c1", "d0", "d1"]

    class Models:
        def embed(self, _):
            return [[0.0]]

    class Search:
        def retrieve_hybrid(self, *args):
            assert args[-1] == 24  # eight admitted slots need a deeper candidate pool
            return [
                {"chunk_id": key, "score": 0.1, "bm25_rank": rank, "dense_rank": rank}
                for rank, key in enumerate(order, 1)
            ]

    result = retrieve_authorized(
        None,
        SimpleNamespace(tenant_id="tenant"),
        "Do the A report and the B article both mention the outage, respectively?",
        cfg=SimpleNamespace(
            top_k=4, retrieval_mode="hybrid", min_similarity=0.0, context_token_budget=5000
        ),
        models=Models(),
        search=Search(),
        top_k=8,
        readable_documents_fn=lambda *_: list(documents.values()),
        require_chunk_fn=lambda _db, _user, chunk_id, **_kwargs: chunks[chunk_id],
    )
    # Four documents instead of one document using the whole budget.
    assert [row["document_id"] for row in result.evidence] == ["a", "a", "b", "b", "c", "c", "d", "d"]
