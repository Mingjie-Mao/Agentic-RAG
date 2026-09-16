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

        def retrieve_hybrid(self, *args, **kwargs):
            return [{"chunk_id": "c", "score": 0.03, "bm25_rank": 1, "dense_rank": 1}]

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


def conflict_harness(monkeypatch, evidence, cited, verdict):
    """Drive Models.generate with a scripted answer and conflict verdict."""
    from app.clients import Models

    calls = []

    def reply(self, path, body):
        calls.append(body)
        if len(calls) == 1:
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "status": "conflict",
                            "claims": [{"text": "两条规定。", "source_ids": cited}],
                        }
                    )
                }
            }
        return {"message": {"content": json.dumps(verdict)}}

    monkeypatch.setattr(Models, "_post", reply)
    _, usage = Models().generate("支持时段是什么？", evidence)
    return usage, calls


def span(slot, document, text):
    return {"id": slot, "document_id": document, "title": "规定", "text": text}


def test_conflict_check_is_skipped_when_all_spans_share_one_document(monkeypatch):
    evidence = [span("E1", "doc-a", "首次响应 30 分钟。恢复时限 4 小时。")]
    usage, calls = conflict_harness(
        monkeypatch,
        evidence,
        ["E1:S1", "E1:S2"],
        {"conflict": True, "left_id": "E1:S1", "right_id": "E1:S2", "reason": "x"},
    )
    assert len(calls) == 1, "One document cannot contradict itself; no second call"
    assert usage["answer_status"] == "answered"
    assert usage["conflict_check"] is None


def test_conflict_verdict_is_rejected_when_both_spans_share_one_document(monkeypatch):
    evidence = [span("E1", "doc-a", "首次响应 30 分钟。"), span("E2", "doc-b", "升级时限 2 小时。")]
    usage, _ = conflict_harness(
        monkeypatch,
        evidence,
        ["E1:S1", "E2:S1"],
        {"conflict": True, "left_id": "E1:S1", "right_id": "E1:S1", "reason": "x"},
    )
    assert usage["answer_status"] == "answered"
    assert usage["conflict_check"]["accepted"] is False


def test_conflict_is_accepted_only_across_documents(monkeypatch):
    evidence = [
        span("E1", "doc-a", "每日 08:00 至 20:00。"),
        span("E2", "doc-b", "工作日 09:00 至 18:00。"),
    ]
    usage, _ = conflict_harness(
        monkeypatch,
        evidence,
        ["E1:S1", "E2:S1"],
        {"conflict": True, "left_id": "E1:S1", "right_id": "E2:S1", "reason": "时段不一致"},
    )
    assert usage["answer_status"] == "conflict"
    assert usage["conflict_check"]["accepted"] is True


def test_rrf_fuses_ranks_not_scores(monkeypatch):
    """BM25 and cosine scores are not comparable, so fusion must use positions."""
    from app.clients import Search

    lexical = [
        {"chunk_id": "a", "score": 91.4, "bm25_score": 91.4},
        {"chunk_id": "b", "score": 12.0, "bm25_score": 12.0},
    ]
    dense = [
        {"chunk_id": "b", "score": 0.81, "cosine_similarity": 0.62},
        {"chunk_id": "c", "score": 0.80, "cosine_similarity": 0.60},
    ]
    monkeypatch.setattr(Search, "retrieve_bm25", lambda self, *a: lexical)
    monkeypatch.setattr(Search, "retrieve", lambda self, *a: dense)
    fused = Search("i").retrieve_hybrid("q", [0.0], "t", ["v"], 3, constant=60)

    # b is second in one run and first in the other, so it must beat a single first place.
    assert [hit["chunk_id"] for hit in fused] == ["b", "a", "c"]
    assert fused[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0]["bm25_score"] == 12.0 and fused[0]["cosine_similarity"] == 0.62
    assert fused[0]["bm25_rank"] == 2 and fused[0]["dense_rank"] == 1
    # A huge BM25 score must not dominate; only its rank counts.
    assert fused[1]["chunk_id"] == "a" and fused[1]["score"] == pytest.approx(1 / 61)


def test_rrf_respects_requested_depth_and_is_deterministic(monkeypatch):
    from app.clients import Search

    run = [{"chunk_id": f"c{i}", "score": 1.0 / (i + 1)} for i in range(10)]
    monkeypatch.setattr(Search, "retrieve_bm25", lambda self, *a: run)
    monkeypatch.setattr(Search, "retrieve", lambda self, *a: list(reversed(run)))
    first = Search("i").retrieve_hybrid("q", [0.0], "t", ["v"], 4)
    second = Search("i").retrieve_hybrid("q", [0.0], "t", ["v"], 4)
    assert len(first) == 4 and first == second


def answer_with_documents(monkeypatch, documents):
    """Run the QA path without a model; only the pre-retrieval states are exercised."""
    from app import qa

    class Unreachable:
        def __getattr__(self, name):
            raise AssertionError("No model or search call may happen without searchable documents")

    monkeypatch.setattr(qa, "Models", Unreachable)
    monkeypatch.setattr(qa, "Search", Unreachable)
    monkeypatch.setattr(qa, "readable_documents", lambda *args: documents)
    monkeypatch.setattr(qa, "Answer", lambda **kwargs: SimpleNamespace(id="a", **kwargs))
    db = SimpleNamespace(add=lambda row: None, commit=lambda: None)
    return qa.answer_question(db, SimpleNamespace(id="u", tenant_id="a"), "保留多久？")


def test_empty_knowledge_base_is_not_reported_as_missing_evidence(monkeypatch):
    result = answer_with_documents(monkeypatch, [])
    assert result["status"] == "no_readable_documents"
    assert "没有可访问的资料" in result["message"]
    assert result["trace"]["scope"] == {"readable_documents": 0, "searchable_documents": 0}


def test_unpublished_documents_report_processing_not_missing_evidence(monkeypatch):
    pending = SimpleNamespace(active_version_id=None, id="d", title="t", metadata_json={})
    result = answer_with_documents(monkeypatch, [pending])
    assert result["status"] == "documents_processing"
    assert "还在处理" in result["message"]
    assert result["trace"]["scope"] == {"readable_documents": 1, "searchable_documents": 0}


def test_dependency_outage_is_retried_but_other_failures_are_not(monkeypatch):
    """Losing hours of completed work to a transient outage is not acceptable;
    losing a run to a real bug must still happen."""
    import sys

    sys.path.insert(0, "scripts")
    import run_s3_baseline as runner
    from app.clients import DependencyError

    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    state = {"calls": 0}

    class FlakySearch:
        def request(self, *args, **kwargs):
            state["calls"] += 1
            if state["calls"] < 3:
                raise DependencyError("检索服务暂时不可用")
            return {"status": "green"}

    class Models:
        def embed(self, _):
            return [[0.0]]

    assert runner.await_dependencies(FlakySearch(), Models()) is True
    assert state["calls"] == 3

    class DeadSearch:
        def request(self, *args, **kwargs):
            raise DependencyError("检索服务暂时不可用")

    assert runner.await_dependencies(DeadSearch(), Models(), attempts=2) is False

    class BrokenSearch:
        def request(self, *args, **kwargs):
            raise ValueError("a real bug, not an outage")

    with pytest.raises(ValueError):
        runner.await_dependencies(BrokenSearch(), Models())


def rewrite_with(monkeypatch, reply, mode, history, question="那请求超时呢？"):
    from app import rewrite

    captured = {}

    class FakeModels:
        def _post(self, path, body):
            captured["path"] = path
            captured["body"] = body
            return reply

    query, trace = rewrite.rewrite_query(question, history, mode=mode, models=FakeModels())
    return query, trace, captured


def test_rewrite_off_and_ruleless_history_change_nothing(monkeypatch):
    from app.rewrite import rewrite_query

    assert rewrite_query("那请求超时呢？", ["星桥批量导入的每批记录上限是多少？"], mode="off")[0] == (
        "那请求超时呢？"
    )
    # Rule mode without history has nothing to prepend and must stay a no-op.
    assert rewrite_query("那请求超时呢？", [], mode="rule")[0] == "那请求超时呢？"


def test_rewrite_only_sends_earlier_question_text_never_evidence(monkeypatch):
    reply = {"message": {"content": json.dumps({"query": "星桥批量导入的请求超时是多少？"})}}
    history = ["星桥批量导入的每批记录上限是多少？"]
    query, trace, captured = rewrite_with(monkeypatch, reply, "llm_history", history)
    payload = json.loads(captured["body"]["messages"][1]["content"])
    assert set(payload) == {"history_questions", "question"}
    assert payload["history_questions"] == history
    assert query == "星桥批量导入的请求超时是多少？" and trace["rewritten"] is True


def test_rewrite_history_is_capped_and_ordered(monkeypatch):
    from app.rewrite import HISTORY_TURNS

    reply = {"message": {"content": json.dumps({"query": "q"})}}
    history = [f"问题{i}" for i in range(10)]
    _, trace, captured = rewrite_with(monkeypatch, reply, "llm_history", history)
    payload = json.loads(captured["body"]["messages"][1]["content"])
    assert payload["history_questions"] == history[-HISTORY_TURNS:]
    assert trace["history_turns"] == HISTORY_TURNS


def test_rewrite_without_history_mode_sees_no_history(monkeypatch):
    reply = {"message": {"content": json.dumps({"query": "q"})}}
    _, _, captured = rewrite_with(monkeypatch, reply, "llm", ["星桥批量导入的每批记录上限是多少？"])
    assert json.loads(captured["body"]["messages"][1]["content"])["history_questions"] == []


def test_failed_rewrite_falls_back_to_the_original_question(monkeypatch):
    from app import rewrite
    from app.clients import DependencyError

    class Dead:
        def _post(self, path, body):
            raise DependencyError("本地模型暂时不可用")

    query, trace = rewrite.rewrite_query(
        "那请求超时呢？", ["上一个问题"], mode="llm_history", models=Dead()
    )
    assert query == "那请求超时呢？" and trace["rewritten"] is False and "error" in trace

    class Garbage:
        def _post(self, path, body):
            return {"message": {"content": "not json"}}

    query, trace = rewrite.rewrite_query("那请求超时呢？", ["上一个问题"], mode="llm", models=Garbage())
    assert query == "那请求超时呢？" and trace["rewritten"] is False


def test_rule_rewrite_is_a_noop_for_single_turn_questions():
    """Single-turn baselines must stay valid after rewriting became the default."""
    from app.config import settings
    from app.rewrite import rewrite_query

    assert settings().rewrite_mode == "rule"
    for history in ([], None, ["", "   "]):
        query, trace = rewrite_query("星桥标准套餐每分钟可以调用多少次 API？", history)
        assert query == "星桥标准套餐每分钟可以调用多少次 API？"
        assert trace["rewritten"] is False


def test_replayed_history_cannot_widen_the_searchable_scope(monkeypatch):
    """History steers retrieval only; the permitted version set is computed from the
    user, so replaying someone else's question must not reach their documents."""
    from app import qa

    doc = SimpleNamespace(active_version_id="v", id="d", title="t", metadata_json={})
    seen = {}

    class FakeModels:
        def embed(self, texts):
            seen["query"] = texts[0]
            return [[0.0]]

        def generate(self, *args):
            return GeneratedAnswer(answerable=False, claims=[]), {}

    class FakeSearch:
        def retrieve_hybrid(self, query, vector, tenant_id, versions, top_k):
            seen["versions"] = list(versions)
            return []

    monkeypatch.setattr(qa, "Models", FakeModels)
    monkeypatch.setattr(qa, "Search", FakeSearch)
    monkeypatch.setattr(qa, "readable_documents", lambda *args: [doc])
    monkeypatch.setattr(qa, "Answer", lambda **kwargs: SimpleNamespace(id="a", **kwargs))
    db = SimpleNamespace(add=lambda row: None, commit=lambda: None)
    result = qa.answer_question(
        db, SimpleNamespace(id="u", tenant_id="a"), "那上限呢？", ["海川工作室的机密配额是多少？"]
    )
    # The replayed line reaches the query text, but not the authorised version set.
    assert "海川" in seen["query"] and seen["versions"] == ["v"]
    assert result["trace"]["retrieval_query"] != result["trace"]["original_question"]
    assert result["status"] == "insufficient_evidence"


def test_request_log_carries_numbers_but_never_text():
    """Logs outlive access grants, so they must not become a second copy of the corpus.

    Questions are excluded along with evidence: a question routinely contains the very
    value the asker was not entitled to learn.
    """
    from app.observability import answer_record

    secret = "生产数据库恢复点目标 RPO 为 15 分钟"
    question = "海川工作室的机密配额是多少？"
    payload = {
        "id": "answer-1",
        "question": question,
        "status": "answered",
        "claims": [{"text": secret, "evidence_ids": ["c1"], "quotes": [secret]}],
        "citations": [{"chunk_id": "c1", "document_id": "d1", "title": "备份规则", "text": secret}],
        "message": "",
        "trace": {
            "method": "hybrid",
            "prompt_version": "grounded-v5-conflict-gated",
            "original_question": question,
            "retrieval_query": question,
            "query_rewritten": False,
            "embed_ms": 12.0,
            "retrieval_ms": 30.0,
            "generation_ms": 900.0,
            "total_ms": 942.0,
            "context_tokens_estimate": 512,
            "candidates": [{"chunk_id": "c1", "title": "备份规则", "score": 0.9}],
        },
        "usage": {"prompt_tokens": 800, "completion_tokens": 40, "api_cost": 0, "currency": "AUD"},
    }
    record = json.dumps(answer_record(payload), ensure_ascii=False)

    for forbidden in [secret, question, "备份规则", "RPO", "海川"]:
        assert forbidden not in record, f"日志中出现了不应记录的内容：{forbidden}"
    # What operating the system actually needs is all present.
    for required in ["hybrid", "generation_ms", "prompt_tokens", "api_cost", "c1"]:
        assert required in record
    assert json.loads(record)["citations"] == 1


def test_outage_reports_the_stage_so_the_reader_is_not_told_a_guess():
    """ "The model never saw this" and "the model may have answered" differ for the
    reader: one makes retrying free, the other does not. Only the server knows which."""
    from app.clients import DependencyError, Search

    with pytest.raises(DependencyError) as retrieval:
        Search("i").request("GET", "/nowhere", json={})
    assert retrieval.value.stage == "retrieval"

    def dead(self, path, body):
        raise DependencyError("本地模型暂时不可用", stage="generation")

    with pytest.raises(DependencyError) as generation:
        dead(None, "/api/chat", {})
    assert generation.value.stage == "generation"

    # An unannotated failure must not claim the model was never reached.
    assert DependencyError("x").stage == "unknown"


def test_shadow_scores_are_recorded_and_never_withhold_an_answer():
    """Calibration put three scorers inside overlapping confidence intervals on 34
    cases, so none of them has earned the right to refuse. They record only."""
    from app.qa import shadow_scores

    evidence = [{"text": "星桥批量导入的每批记录上限为 240 条。"}]
    claims = [{"text": "每批记录上限为 240 条。"}]
    on_topic = shadow_scores("星桥批量导入每批记录上限是多少？", evidence, claims)
    off_topic = shadow_scores(
        "星桥恢复抽查的检查间隔是多少？",
        [{"text": "班车发车间隔为 20 分钟。"}],
        [{"text": "班车发车间隔为 20 分钟。"}],
    )
    assert on_topic["relevance"] > off_topic["relevance"]
    assert on_topic["mode"] == off_topic["mode"] == "shadow"
    assert "no answer is withheld" in on_topic["action"]
    assert len(on_topic["claim_support"]) == len(claims)


def test_shadow_scoring_cannot_change_the_answer(monkeypatch):
    """The property that matters is not the score but that nothing depends on it."""
    from app import qa

    doc = SimpleNamespace(active_version_id="v", id="d", title="备份规则", metadata_json={})
    chunk = SimpleNamespace(id="c", text="生产数据库恢复点目标 RPO 为 15 分钟。", locator={})
    version = SimpleNamespace(id="v")

    class FakeModels:
        def embed(self, _):
            return [[0.0]]

        def generate(self, *args):
            return GeneratedAnswer(
                answerable=True,
                claims=[Claim(text="RPO 为 15 分钟。", evidence_ids=["E1"], quotes=[chunk.text])],
            ), {}

    class FakeSearch:
        def retrieve_hybrid(self, *args, **kwargs):
            return [{"chunk_id": "c", "score": 0.03}]

    monkeypatch.setattr(qa, "Models", FakeModels)
    monkeypatch.setattr(qa, "Search", FakeSearch)
    monkeypatch.setattr(qa, "readable_documents", lambda *a: [doc])
    monkeypatch.setattr(qa, "require_chunk", lambda *a, **k: (chunk, version, doc))
    monkeypatch.setattr(qa, "Answer", lambda **kw: SimpleNamespace(id="a", **kw))
    db = SimpleNamespace(add=lambda row: None, commit=lambda: None)

    # An off-topic question scores low, and the answer is returned regardless.
    result = qa.answer_question(db, SimpleNamespace(id="u", tenant_id="t"), "火星补贴是多少？")
    assert result["status"] == "answered"
    assert result["claims"], "影子评分不得拦下任何回答"
    assert result["trace"]["shadow_scores"]["relevance"] < 0.3
