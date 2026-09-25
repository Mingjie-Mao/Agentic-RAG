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
