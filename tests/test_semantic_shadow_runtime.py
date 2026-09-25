"""The calibrated scorer observes cited claims without controlling the answer."""

from app import qa, semantic_shadow


def test_semantic_shadow_scores_only_cited_evidence(monkeypatch):
    seen = []

    def fake_rerank(query, candidates, *, window):
        seen.append((query, [row["text"] for row in candidates], window))
        return [{**row, "rerank_score": 3.0} for row in candidates]

    monkeypatch.setattr(semantic_shadow, "rerank", fake_rerank)
    evidence = [
        {"chunk_id": "a", "text": "event_id 为 EVT-7。"},
        {"chunk_id": "b", "text": "另一租户的内容不应参与。"},
    ]
    claims = [{"text": "event_id 是 EVT-7。", "evidence_ids": ["a"]}]
    result = semantic_shadow.score_claims(evidence, claims)
    assert result["claim_support_scores"] == [3.0]
    assert seen == [(claims[0]["text"], [evidence[0]["text"]], 1)]
    assert result["mode"] == "shadow"


def test_semantic_shadow_failure_is_fail_open(monkeypatch):
    def fail(*_args):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(semantic_shadow, "score_claims", fail)
    result = qa.shadow_scores(
        "event_id 是什么？",
        [{"chunk_id": "a", "text": "event_id 为 EVT-7。"}],
        [{"text": "event_id 为 EVT-7。", "evidence_ids": ["a"]}],
        semantic=True,
    )
    assert result["semantic"] == {"mode": "shadow", "status": "unavailable"}
    assert result["action"] == "recorded only; no answer is withheld on these scores"
