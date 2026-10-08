from types import SimpleNamespace

from app.boilerplate import is_web_boilerplate
from app.ingestion import chunk_header, indexed_text
from app.retrieval import retrieve_authorized, route_sources


def article(key, source, published="2023-10-07T10:00:00+00:00"):
    return SimpleNamespace(
        id=key,
        active_version_id=f"v{key}",
        title=key.upper(),
        metadata_json={"source": source, "published_at": published},
    )


def test_named_publications_become_groups_and_a_date_narrows_one():
    documents = [
        article("a", "TechCrunch", "2023-10-07T08:00:00+00:00"),
        article("b", "TechCrunch", "2023-10-30T08:00:00+00:00"),
        article("c", "The Independent - Life and Style"),
        article("d", "The Age"),
    ]
    groups = route_sources(
        "After the TechCrunch report on October 7, 2023, did The Independent agree?", documents
    )
    assert [(g["mention"], g["dated"], g["key"]) for g in groups] == [
        ("techcrunch", True, ["a"]),
        ("the independent", False, ["c"]),
    ]


def test_ordinary_words_are_not_mistaken_for_a_publication():
    documents = [article("d", "The Age"), article("e", "CBSSports.com")]
    assert route_sources("Is this the age of AI?", documents) == []
    assert [g["mention"] for g in route_sources("Did CBSSports.com say so?", documents)] == [
        "cbssports.com"
    ]


def test_documents_without_a_source_are_never_routed():
    plain = SimpleNamespace(id="x", active_version_id="vx", title="X", metadata_json={})
    assert route_sources("TechCrunch 和海川的规定分别是什么？", [plain]) == []


def test_routed_lane_searches_only_the_named_publication_inside_the_readable_scope():
    documents = {"a": article("a", "TechCrunch"), "b": article("b", "Fortune"), "z": article("z", "Wired")}
    chunks = {
        f"{key}{i}": (
            SimpleNamespace(id=f"{key}{i}", text=f"{key} {i}", locator={}),
            SimpleNamespace(id=f"v{key}"),
            documents[key],
        )
        for key in documents
        for i in range(3)
    }
    scopes = []

    class Search:
        def retrieve_hybrid(self, _query, _vector, _tenant, versions, _size):
            scopes.append(sorted(versions))
            keys = [f"{d}{i}" for d in "zab" for i in range(3) if f"v{d}" in versions]
            return [{"chunk_id": key, "score": 0.1} for key in keys]

    result = retrieve_authorized(
        None,
        SimpleNamespace(tenant_id="t"),
        "Do the TechCrunch and Fortune articles both report it?",
        cfg=SimpleNamespace(top_k=4, retrieval_mode="hybrid", min_similarity=0.0, context_token_budget=5000),
        models=SimpleNamespace(embed=lambda _texts: [[0.0]]),
        search=Search(),
        top_k=8,
        readable_documents_fn=lambda *_: list(documents.values()),
        require_chunk_fn=lambda _db, _user, key, **_kw: chunks[key],
    )
    assert scopes[:2] == [["va"], ["vb"]]  # each lane stays inside its publication
    assert scopes[2] == ["va", "vb", "vz"]  # the global fill uses the full readable scope
    admitted = [row["document_id"] for row in result.evidence]
    assert admitted.count("a") == 2 and admitted.count("b") == 2
    assert [g["mention"] for g in result.source_groups] == ["techcrunch", "fortune"]


def test_focused_source_query_uses_its_own_clause_and_remains_acl_scoped():
    documents = {key: article(key, source) for key, source in
                 [('a', 'Sporting News'), ('b', 'CBSSports.com')]}
    chunks = {key: (SimpleNamespace(id=key, text='Example fact.', locator={}),
                    SimpleNamespace(id='v' + key), document)
              for key, document in documents.items()}
    calls = []
    class Search:
        def retrieve_hybrid(self, text, _vector, _tenant, versions, _size):
            calls.append((text, tuple(versions)))
            return [{'chunk_id': key, 'score': 0.1} for key in documents
                    if 'v' + key in versions]
    query = ('Did the Sporting News article anticipate Jordan Love at home, while '
             'the CBSSports.com article report Derrick Henry rushing yards?')
    result = retrieve_authorized(
        None, SimpleNamespace(tenant_id='t'), query,
        cfg=SimpleNamespace(top_k=4, retrieval_mode='hybrid', min_similarity=0.0,
                            context_token_budget=5000, source_focus_queries=True),
        models=SimpleNamespace(embed=lambda _texts: [[0.0]]), search=Search(),
        top_k=6, readable_documents_fn=lambda *_: list(documents.values()),
        require_chunk_fn=lambda _db, _user, key, **_kw: chunks[key],
    )
    assert set(row['document_id'] for row in result.evidence) == {'a', 'b'}
    assert calls[0][1] == ('va',) and 'CBSSports.com' not in calls[0][0]
    assert calls[1][1] == ('vb',) and 'Sporting News' not in calls[1][0]
    assert calls[2] == (query, ('va', 'vb'))


def test_facet_queries_interleave_article_candidates_within_acl_scope():
    documents = {key: article(key, source) for key, source in
                 [('a', 'TechCrunch'), ('b', 'TechCrunch'), ('z', 'Wired')]}
    chunks = {key: (SimpleNamespace(id=key, text='Evidence about ' + key, locator={}),
                    SimpleNamespace(id='v' + key), document)
              for key, document in documents.items()}
    calls = []
    class Search:
        def retrieve_hybrid(self, text, _vector, _tenant, versions, _size):
            calls.append((text, tuple(versions)))
            if 'trial' in text and 'fraud' not in text:
                return [{'chunk_id': 'a', 'score': 1.0}]
            if 'fraud' in text and 'trial' not in text:
                return [{'chunk_id': 'b', 'score': 1.0}]
            return [{'chunk_id': 'z', 'score': 1.0}]
    query = ('Between the TechCrunch report on the trial and the subsequent '
             'TechCrunch report on fraud allegations, was the portrayal different?')
    result = retrieve_authorized(
        None, SimpleNamespace(tenant_id='t'), query,
        cfg=SimpleNamespace(top_k=4, retrieval_mode='hybrid', min_similarity=0.0,
                            context_token_budget=5000, source_facet_queries=True),
        models=SimpleNamespace(embed=lambda _texts: [[0.0]]), search=Search(),
        top_k=6, readable_documents_fn=lambda *_: list(documents.values()),
        require_chunk_fn=lambda _db, _user, key, **_kw: chunks[key],
    )
    assert {row['document_id'] for row in result.evidence} >= {'a', 'b'}
    assert calls[0][1] == calls[1][1] == ('va', 'vb')
    assert calls[2] == (query, ('va', 'vb', 'vz'))


def test_document_facets_reserve_distinct_authorized_articles():
    documents = {key: article(key, 'TechCrunch') for key in ('a', 'b')}
    chunks = {key: (SimpleNamespace(id=key, text='Relevant passage', locator={}),
                    SimpleNamespace(id='v' + key), documents[key]) for key in documents}
    class Search:
        def retrieve_hybrid(self, _text, _vector, _tenant, versions, _size):
            assert set(versions) <= {'va', 'vb'}
            return [{'chunk_id': 'a', 'score': 1.0}, {'chunk_id': 'b', 'score': 0.9}]
    query = ('Between the TechCrunch report on the trial and the subsequent '
             'TechCrunch report on fraud allegations, was the portrayal different?')
    result = retrieve_authorized(
        None, SimpleNamespace(tenant_id='t'), query,
        cfg=SimpleNamespace(top_k=4, retrieval_mode='hybrid', min_similarity=0.0,
                            context_token_budget=5000, source_facet_document_queries=True),
        models=SimpleNamespace(embed=lambda _texts: [[0.0]]), search=Search(),
        top_k=6, readable_documents_fn=lambda *_: list(documents.values()),
        require_chunk_fn=lambda _db, _user, key, **_kw: chunks[key],
    )
    assert [row['document_id'] for row in result.evidence[:2]] == ['a', 'b']


def test_article_first_lane_keeps_document_diversity_and_acl():
    documents = {key: article(key, 'TechCrunch') for key in ('a', 'b', 'c')}
    chunks = {f'{key}{i}': (SimpleNamespace(id=f'{key}{i}', text='Relevant passage', locator={}),
                              SimpleNamespace(id='v' + key), document)
              for key, document in documents.items() for i in range(3)}
    scopes = []
    class Search:
        def retrieve_hybrid(self, _text, _vector, _tenant, versions, size):
            scopes.append((tuple(versions), size))
            return [{'chunk_id': key, 'score': 1.0} for key in
                    ('a0', 'a1', 'a2', 'b0', 'c0') if 'v' + key[0] in versions]
    result = retrieve_authorized(
        None, SimpleNamespace(tenant_id='t'), 'What do the TechCrunch reports say about this case?',
        cfg=SimpleNamespace(top_k=4, retrieval_mode='hybrid', min_similarity=0.0,
                            context_token_budget=5000, article_first_lanes=True),
        models=SimpleNamespace(embed=lambda _texts: [[0.0]]), search=Search(),
        top_k=6, readable_documents_fn=lambda *_: list(documents.values()),
        require_chunk_fn=lambda _db, _user, key, **_kw: chunks[key],
    )
    assert [row['document_id'] for row in result.evidence[:3]] == ['a', 'b', 'c']
    assert scopes[0] == (('va', 'vb', 'vc'), 48)


def test_page_furniture_is_dropped_but_content_about_sign_ups_is_kept():
    assert is_web_boilerplate("Advertisement")
    assert is_web_boilerplate("CLICK HERE TO SIGN UP FOR OUR HEALTH NEWSLETTER")
    assert is_web_boilerplate(
        "Stay ahead of the trend with our free weekly Lifestyle Edit newsletter. "
        "Please enter a valid email address. Read our privacy notice."
    )
    assert not is_web_boilerplate("Viewers in Canada can sign up for DAZN, which carries NFL Game Pass.")
    assert not is_web_boilerplate(
        "The AFL Draft will be shown on Fox Footy and Kayo. CLICK HERE for a seven-day free trial."
    )
    assert not is_web_boilerplate("# Newsletter strategy at TechCrunch")


def test_document_header_is_indexed_but_only_when_the_pipeline_recorded_it():
    metadata = {"source": "Fortune", "published_at": "2023-11-27T08:45:59+00:00"}
    header = chunk_header("SBF trial", metadata, {"chunk_context": "document_header"})
    assert header == "SBF trial · Fortune · 2023-11-27"
    assert indexed_text(header, "body") == "SBF trial · Fortune · 2023-11-27\nbody"
    assert chunk_header("SBF trial", metadata, {}) == ""
    assert indexed_text("", "body") == "body"


def test_conflict_check_is_skipped_only_for_listed_tenants(monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings(), "conflict_check_disabled_tenants", ["multihop"])
    assert config.conflict_check_enabled("xingqiao")
    assert not config.conflict_check_enabled("multihop")


def test_agent_rewritten_query_keeps_publications_named_in_the_user_goal():
    from app.retrieval import routing_goal

    documents = {"a": article("a", "TechCrunch"), "b": article("b", "Fortune"), "z": article("z", "Wired")}
    chunks = {
        f"{key}{i}": (SimpleNamespace(id=f"{key}{i}", text=f"{key} {i}", locator={}),
                      SimpleNamespace(id=f"v{key}"), documents[key])
        for key in documents for i in range(3)
    }

    class Search:
        def retrieve_hybrid(self, _query, _vector, _tenant, versions, _size):
            return [{"chunk_id": f"{d}{i}", "score": 0.1} for d in "zab" for i in range(3) if f"v{d}" in versions]

    def run(query):
        return retrieve_authorized(
            None, SimpleNamespace(tenant_id="t"), query,
            cfg=SimpleNamespace(top_k=4, retrieval_mode="hybrid", min_similarity=0.0, context_token_budget=5000),
            models=SimpleNamespace(embed=lambda _texts: [[0.0]]), search=Search(), top_k=8,
            readable_documents_fn=lambda *_: list(documents.values()),
            require_chunk_fn=lambda _db, _user, key, **_kw: chunks[key])

    rewritten = "privacy control reported in both articles"
    assert run(rewritten).source_groups == []
    with routing_goal("Do the TechCrunch and Fortune articles both report it?"):
        routed = run(rewritten)
    assert [g["mention"] for g in routed.source_groups] == ["techcrunch", "fortune"]
    assert {"a", "b"} <= {row["document_id"] for row in routed.evidence}
    assert run(rewritten).source_groups == []  # the goal does not leak past its task


def test_reported_before_dates_bound_each_time_point_and_exact_dates_are_unchanged():
    documents = [
        article("a", "Sporting News", "2023-10-02T08:00:00+00:00"),
        article("b", "Sporting News", "2023-10-04T22:00:00+00:00"),
        article("c", "Sporting News", "2023-10-24T08:00:00+00:00"),
        article("d", "Sporting News", "2023-11-06T08:00:00+00:00"),
    ]
    question = "Has the approach, as reported by Sporting News before October 4, 2023, and before November 1, 2023, stayed the same?"
    groups = route_sources(question, documents, strict_dates=True)
    assert [g["key"] for g in groups] == [["a", "b"], ["a", "b", "c"]]  # the date's own report counts
    assert all(g["bounded"] for g in groups)
    after = route_sources("Did Sporting News report after October 24, 2023 that it changed?", documents, strict_dates=True)
    assert [g["key"] for g in after] == [["c", "d"]]
    # An event date or a non-strict question keeps the previous behaviour.
    assert not route_sources("Did Sporting News say the game before October 4, 2023 mattered?", documents)[0].get("bounded")
    exact = route_sources("Sporting News articles published on October 24, 2023", documents, strict_dates=True)
    assert [(g["key"], g["bounded"]) for g in exact] == [(["c"], False)]


def test_global_fill_skips_a_bounded_publication_outside_its_windows():
    documents = {"a": article("a", "Sporting News", "2023-10-02T08:00:00+00:00"),
                 "d": article("d", "Sporting News", "2023-11-06T08:00:00+00:00"),
                 "w": article("w", "Wired", "2023-11-06T08:00:00+00:00")}
    chunks = {f"{key}{i}": (SimpleNamespace(id=f"{key}{i}", text=f"{key} {i}", locator={}),
                            SimpleNamespace(id=f"v{key}"), documents[key]) for key in documents for i in range(3)}

    class Search:
        def retrieve_hybrid(self, _query, _vector, _tenant, versions, _size):
            return [{"chunk_id": f"{d}{i}", "score": 0.1} for d in "dwa" for i in range(3) if f"v{d}" in versions]

    result = retrieve_authorized(
        None, SimpleNamespace(tenant_id="t"), "What did Sporting News report before October 4, 2023 about odds?",
        cfg=SimpleNamespace(top_k=8, retrieval_mode="hybrid", min_similarity=0.0, context_token_budget=5000,
                            focused_generation_enabled=True),
        models=SimpleNamespace(embed=lambda _texts: [[0.0]]), search=Search(), top_k=8,
        readable_documents_fn=lambda *_: list(documents.values()),
        require_chunk_fn=lambda _db, _user, key, **_kw: chunks[key])
    admitted = {row["document_id"] for row in result.evidence}
    assert "d" not in admitted and "a" in admitted and "w" in admitted  # other publications still fill
    assert any(c["excluded_because"] == "outside_requested_publication_dates" for c in result.candidates)


def test_missing_constraints_finds_dropped_names_dates_and_identifiers():
    from app.retrieval import missing_constraints

    goal = "Did TechCrunch report on October 31, 2023 that Google paid millions, per incident 417?"
    assert missing_constraints(goal, "Google paid millions TechCrunch October 31, 2023 incident 417") == []
    assert missing_constraints(goal, "Google payment default search") == ["417", "TechCrunch", "date:2023-10-31"]
    assert missing_constraints("事故 417 的导入任务有什么特征？", "导入任务 特征") == ["417"]


def test_goal_lane_is_added_only_for_model_written_queries_that_dropped_constraints():
    from app.retrieval import routing_goal

    documents = {"a": article("a", "TechCrunch"), "z": article("z", "Wired")}
    chunks = {f"{key}{i}": (SimpleNamespace(id=f"{key}{i}", text=f"{key} {i}", locator={}),
                            SimpleNamespace(id=f"v{key}"), documents[key]) for key in documents for i in range(3)}
    queries = []

    class Search:
        def retrieve_hybrid(self, query, _vector, _tenant, versions, _size):
            queries.append(query)
            first = "a" if "Kane" in query else "z"
            order = [first] + [d for d in "za" if d != first]
            return [{"chunk_id": f"{d}{i}", "score": 0.1} for d in order for i in range(3) if f"v{d}" in versions]

    def run(query):
        return retrieve_authorized(
            None, SimpleNamespace(tenant_id="t"), query,
            cfg=SimpleNamespace(top_k=4, retrieval_mode="hybrid", min_similarity=0.0, context_token_budget=5000),
            models=SimpleNamespace(embed=lambda _texts: [[0.0]]), search=Search(), top_k=4,
            readable_documents_fn=lambda *_: list(documents.values()),
            require_chunk_fn=lambda _db, _user, key, **_kw: chunks[key])

    goal = "Why did Patrick Kane move to Detroit?"
    with routing_goal(goal):  # workflow: no fusion
        plain = run("player move reasons")
    assert plain.missing_constraints == [] and goal not in queries
    with routing_goal(goal, fuse=True):
        fused = run("player move reasons")
    assert fused.missing_constraints == ["Detroit", "Kane", "Patrick"] and goal in queries
    assert "a" in {row["document_id"] for row in fused.evidence}
    assert any(c["lane"] == "goal" and c["admitted"] for c in fused.candidates)
