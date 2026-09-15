import math
from types import SimpleNamespace

import pytest

from app.evaluation import answer_scores, allowed_sources, fact_present, retrieval_scores


@pytest.mark.parametrize(
    "fact,text,expected",
    [
        ("15", "150", False),
        ("15", "115", False),
        ("15", "15 分钟", True),
        ("35 秒", "需要３５秒", True),
        ("12%", "比例 12%", True),
        ("240 条", "240条记录", True),
        ("240 条", "1240条", False),
        ("RPO", "rpo", True),
        ("90", "九十天", False),
    ],
)
def test_literal_scoring_has_numeric_boundaries(fact, text, expected):
    assert fact_present(fact, text) is expected


def test_ranking_deduplicates_sources_and_accounts_for_multiple_gold():
    result = retrieval_scores(["a", "b"], ["x", "x", "a", "b"])
    assert result["recall_at_5"] == 1
    assert result["mrr"] == 0.5
    assert result["ndcg_at_10"] == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))


def test_missing_and_empty_gold_have_distinct_denominators():
    assert retrieval_scores(["a"], []) == dict(recall_at_5=0, recall_at_10=0, mrr=0, ndcg_at_10=0)
    assert all(v is None for v in retrieval_scores([], ["a"]).values())


def test_eval_uses_same_tenant_and_group_rules_as_product():
    user = SimpleNamespace(id="reader", tenant_id="a", role="member", groups=["support"], active=True)
    sources = [
        dict(id="public", tenant_id="a", groups=[], tenant_public=True),
        dict(id="private", tenant_id="a", groups=["engineering"], tenant_public=False),
        dict(id="foreign", tenant_id="b", groups=["support"], tenant_public=True),
    ]
    assert [s["id"] for s in allowed_sources(sources, user)] == ["public"]


def test_citation_must_match_stored_source_identity_and_text():
    c = dict(chunk_id="chunk", document_id="doc", version_id="v", text="原文", locator={"page": 1})
    answer = dict(status="answered", claims=[{"text": "15 分钟"}], citations=[c])
    question = dict(facts=["15"], source_ids=["doc"], expected="answered")
    chunks = {"chunk": dict(c)}
    assert answer_scores(question, answer, chunks)["citation_identity_valid"] is True
    c["text"] = "替换后的伪造原文"
    assert answer_scores(question, answer, chunks)["citation_identity_valid"] is False
