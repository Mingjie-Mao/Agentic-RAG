from collections import Counter
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app import retrieval
from app.clients import Search
from app.config import Settings


QUERY = "What do TechCrunch and Fortune report about the finding?"


def corpus():
    documents = [
        SimpleNamespace(id=key, active_version_id=f"v{key}", title=key,
                        metadata_json={"source": source})
        for key, source in (("a", "TechCrunch"), ("b", "Fortune"), ("c", "Wired"),
                            ("d", "TechCrunch"), ("e", "Fortune"))
    ]
    chunks = {
        f"{document.id}{i}": (
            SimpleNamespace(id=f"{document.id}{i}", text="A useful fact.", locator={}),
            SimpleNamespace(id=document.active_version_id), document,
        )
        for document in documents for i in range(30)
    }
    return documents, chunks


class ScopedSearch:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve_hybrid(self, text, vector, tenant, versions, size, **kwargs):
        assert tenant == "tenant"
        assert set(versions) <= {"va", "vb", "vc", "vd", "ve"}
        self.calls.append((text, tuple(versions), size, kwargs))
        return [{"chunk_id": key, "score": 1 / (index + 1)}
                for index, (key, (_, version, _)) in enumerate(self.chunks.items())
                if version.id in versions][:size]


def run(*, depth=0, query=QUERY, top_k=8, search=None, require=None,
        mode="hybrid", **options):
    documents, chunks = corpus()
    search = search or ScopedSearch(chunks)
    cfg = SimpleNamespace(top_k=4, retrieval_mode=mode, min_similarity=0,
                          context_token_budget=5000, retrieval_candidate_depth=depth,
                          **options)
    result = retrieval.retrieve_authorized(
        None, SimpleNamespace(tenant_id="tenant"), query, cfg=cfg,
        models=SimpleNamespace(embed=lambda texts: [[0.0] for _ in texts]),
        search=search, top_k=top_k, readable_documents_fn=lambda *_: documents,
        require_chunk_fn=require or (lambda _db, _user, key, **_kw: chunks[key]),
    )
    return result, search


@pytest.mark.parametrize("value", [-1, 101])
def test_settings_reject_out_of_range_depth(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_candidate_depth=value)


def test_settings_default_is_opt_in():
    assert Settings(_env_file=None).retrieval_candidate_depth == 0


def test_zero_preserves_previous_request_sizes_and_evidence_order():
    result, search = run()
    assert [(scope, size, kwargs) for _, scope, size, kwargs in search.calls] == [
        (("va", "vd"), 9, {}), (("vb", "ve"), 9, {}),
        (("va", "vb", "vc", "vd", "ve"), 24, {}),
    ]
    assert [item["chunk_id"] for item in result.evidence] == ["a0", "b0", "a1", "b1"]
    assert all(item["rerank_score"] is None for item in result.candidates)
    assert [(c["chunk_id"], c["retrieval_rank"]) for c in result.candidates[:4]] == [
        ("a0", 1), ("b0", 1), ("a1", 2), ("b1", 2),
    ]


@pytest.mark.parametrize("depth", [50, 100])
@pytest.mark.parametrize("rerank_depth", [None, 24, 80])
def test_candidate_depth_widens_scoped_lanes_without_rerank_undoing_it(
    depth, rerank_depth, monkeypatch,
):
    options = {}
    if rerank_depth:
        options = {"passage_rerank": True, "passage_rerank_depth": rerank_depth}
        monkeypatch.setattr(retrieval, "_reranked", lambda _db, _user, _q, hits, _req: hits)
    result, search = run(depth=depth, **options)
    assert [scope for _, scope, _, _ in search.calls] == [
        ("va", "vd"), ("vb", "ve"), ("va", "vb", "vc", "vd", "ve"),
    ]
    assert all(size == max(depth, rerank_depth or 0) for _, _, size, _ in search.calls)
    assert all(kwargs == ({"depth": 100} if depth == 100 else {})
               for _, _, _, kwargs in search.calls)
    assert len(result.evidence) <= 8


def test_fifty_keeps_existing_five_argument_search_interface():
    class ExistingSearch:
        def retrieve_hybrid(self, _text, _vector, _tenant, _versions, size):
            assert size == 50
            return []

    run(depth=50, search=ExistingSearch())


@pytest.mark.parametrize("mode", ["bm25", "dense"])
@pytest.mark.parametrize("depth", [0, 50, 100])
def test_single_routes_apply_depth_without_changing_scope_or_evidence(mode, depth):
    class SingleRouteSearch:
        def __init__(self):
            self.calls = []

        def retrieve_bm25(self, _text, tenant, versions, size):
            self.calls.append((tenant, tuple(versions), size))
            return [{"chunk_id": "a0", "score": 1.0, "cosine_similarity": 1.0}]

        def retrieve(self, _vector, tenant, versions, size):
            return self.retrieve_bm25(None, tenant, versions, size)

    search = SingleRouteSearch()
    result, _ = run(depth=depth, mode=mode, search=search, query="Explain the finding", top_k=2)
    assert search.calls == [("tenant", ("va", "vb", "vc", "vd", "ve"), max(2, depth))]
    assert [item["chunk_id"] for item in result.evidence] == ["a0"]


def test_hundred_widens_real_hybrid_routes_not_only_merged_output():
    class HybridSearch(Search):
        def __init__(self):
            self.route_sizes = []

        def retrieve_bm25(self, _text, _tenant, versions, size):
            self.route_sizes.append((tuple(versions), size))
            return [{"chunk_id": f"a{i}", "score": 1} for i in range(min(size, 30))]

        def retrieve(self, _vector, _tenant, versions, size):
            self.route_sizes.append((tuple(versions), size))
            return [{"chunk_id": f"a{i}", "score": 1} for i in range(min(size, 30))]

    search = HybridSearch()
    run(depth=100, query="Explain the finding", search=search)
    assert search.route_sizes == [
        (("va", "vb", "vc", "vd", "ve"), 100),
        (("va", "vb", "vc", "vd", "ve"), 100),
    ]


def test_wider_pool_keeps_document_source_and_final_top_k_quotas():
    result, _ = run(depth=100, top_k=100)
    assert len(result.evidence) == 8
    assert max(Counter(e["document_id"] for e in result.evidence).values()) == 2
    admitted_lanes = Counter(c["lane"] for c in result.candidates if c["admitted"])
    assert admitted_lanes["source:techcrunch"] == admitted_lanes["source:fortune"] == 3
    reasons = {c["excluded_because"] for c in result.candidates}
    assert {"document_quota", "source_quota", "top_k_full"} <= reasons


def test_context_budget_is_independent_of_candidate_depth():
    documents, chunks = corpus()
    cfg = SimpleNamespace(top_k=8, retrieval_mode="hybrid", min_similarity=0,
                          context_token_budget=220, retrieval_candidate_depth=100)
    result = retrieval.retrieve_authorized(
        None, SimpleNamespace(tenant_id="tenant"), QUERY, cfg=cfg,
        models=SimpleNamespace(embed=lambda _texts: [[0.0]]), search=ScopedSearch(chunks),
        readable_documents_fn=lambda *_: documents,
        require_chunk_fn=lambda _db, _user, key, **_kw: chunks[key],
    )
    assert len(result.evidence) == 2
    assert result.context_tokens <= 220
    assert any(c["excluded_because"] == "context_budget_exhausted" for c in result.candidates)


def test_final_evidence_is_reauthorized_even_with_expanded_candidate_pool():
    _, chunks = corpus()
    checks = Counter()

    def require(_db, _user, key, *, active_only):
        assert active_only is True
        checks[key] += 1
        if key == "a0" and checks[key] == 2:
            raise PermissionError("authorization revoked after admission")
        return chunks[key]

    with pytest.raises(PermissionError, match="revoked"):
        run(depth=100, query="Explain the finding", top_k=1, require=require)


def test_reranking_preserves_lane_retrieval_rank_and_scores(monkeypatch):
    def rerank(_db, _user, _query, hits, _require):
        return [{**hit, "rerank_score": float(index)}
                for index, hit in reversed(list(enumerate(hits)))]

    monkeypatch.setattr(retrieval, "_reranked", rerank)
    result, _ = run(depth=50, passage_rerank=True, passage_rerank_depth=24)
    first = result.candidates[0]
    assert first["lane"] == "source:techcrunch"
    assert (first["rank"], first["retrieval_rank"], first["rerank_score"]) == (1, 50, 49.0)
    assert "text" not in first


def test_lane_ranks_do_not_mutate_shared_raw_search_hits():
    _, chunks = corpus()
    raw = [{"chunk_id": key, "score": 1.0} for key in chunks]

    class SharedSearch:
        def retrieve_hybrid(self, _text, _vector, _tenant, versions, size):
            return [hit for hit in raw if chunks[hit["chunk_id"]][1].id in versions][:size]

    result, _ = run(depth=50, top_k=1, search=SharedSearch())
    b0 = [c for c in result.candidates if c["chunk_id"] == "b0"]
    assert [(c["lane"], c["retrieval_rank"]) for c in b0] == [
        ("source:fortune", 1), ("global", 31),
    ]
    assert all("retrieval_rank" not in hit for hit in raw)


def test_expanded_unranked_passage_has_null_retrieval_rank(monkeypatch):
    def expand(_db, _user, hits, _count, _require):
        return hits + [{"chunk_id": "a29", "score": 0.0}]

    def rerank(_db, _user, _query, hits, _require):
        return [{**hit, "rerank_score": 4.0} for hit in reversed(hits)]

    monkeypatch.setattr(retrieval, "_expanded", expand)
    monkeypatch.setattr(retrieval, "_reranked", rerank)
    result, _ = run(query="Explain the finding", top_k=2,
                    passage_rerank=True, passage_rerank_depth=2, passage_expand_documents=1)
    assert result.candidates[0]["chunk_id"] == "a29"
    assert result.candidates[0]["retrieval_rank"] is None
    assert result.candidates[0]["rerank_score"] == 4.0
    assert result.candidates[1]["retrieval_rank"] == 2


@pytest.mark.parametrize("option", ["source_focus_queries", "article_first_lanes",
                                   "source_facet_queries", "source_query_plan"])
def test_lane_query_variants_use_same_candidate_depth(option, monkeypatch):
    monkeypatch.setattr(retrieval, "retrieval_facets", lambda _query: ["first", "second"])
    if option == "source_query_plan":
        monkeypatch.setattr("app.query_planner.plan_search_queries",
                            lambda *_: (["first", "second"],
                                        {"prompt_tokens": 1, "completion_tokens": 1}))
    query = ("What does TechCrunch report about the first development, while Fortune "
             "reports about the second development and their shared finding?")
    _, search = run(depth=100, query=query, **{option: True})
    assert len(search.calls) >= 3
    assert all(size == 100 and kwargs == {"depth": 100}
               for _, _, size, kwargs in search.calls)
