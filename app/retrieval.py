"""Permission-scoped retrieval shared by single-turn RAG and Agent tools."""

from contextlib import contextmanager
from contextvars import ContextVar
import re
import time
from dataclasses import dataclass

from sqlalchemy import select

from app.chunking import token_count
from app.models import Tenant
from app.security import readable_documents, require_chunk
from app.task_analysis import multi_source_intent, needs_document_diversity
from app.retrieval_queries import retrieval_facets


# Publication names of the documents this caller could read in the latest retrieval,
# with those document IDs as a guard. Validation can then notice a claim naming a
# readable publication that was never retrieved, without widening any ACL.
_READABLE_SOURCES: ContextVar[tuple[frozenset, frozenset] | None] = ContextVar("readable_sources", default=None)


# The user's own question during an Agent task. An Agent may rewrite its search
# query and drop a publication the user named; routing still honours the question.
_TASK_GOAL: ContextVar[str | None] = ContextVar("task_goal", default=None)


_FUSE_GOAL: ContextVar = ContextVar("fuse_goal", default=False)  # bool or a zero-argument callable


@contextmanager
def routing_goal(goal, *, fuse=False):
    """`fuse` is for model-written queries (Dynamic/Hybrid), never workflow subgoals."""
    token, fuse_token = _TASK_GOAL.set(goal), _FUSE_GOAL.set(fuse)
    try:
        yield
    finally:
        _TASK_GOAL.reset(token)
        _FUSE_GOAL.reset(fuse_token)


_QUESTION_WORDS = {
    "does", "did", "do", "is", "are", "was", "were", "has", "have", "had", "who", "what", "which",
    "when", "where", "why", "how", "considering", "based", "between", "after", "before", "according",
    "the", "a", "an", "in", "on", "of", "and", "or", "if", "while", "both", "as",
}


def missing_constraints(goal: str, query: str) -> list[str]:
    """Constraints the user wrote that a rewritten query dropped: dates, quoted spans,
    proper names and alphanumeric identifiers. Lexical only; it decides when the
    original question is searched too, never what the answer is."""
    wanted = set()
    days, _ = _dates_in(goal)
    have_days, _ = _dates_in(query)
    wanted.update(f"date:{day}" for day in days - have_days)
    text = _DATE.sub(" ", goal)
    for quoted in re.findall(r"[\"“‘']([^\"”’']{3,80})[\"”’']", text):
        if quoted.casefold() not in query.casefold():
            wanted.add(quoted)
    for token in re.findall(r"\b[A-Z][A-Za-z0-9&.\-]{2,}\b|\b[A-Za-z]*\d[A-Za-z0-9\-]*\b", text):
        if token.casefold() in _QUESTION_WORDS:
            continue
        if not re.search(r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])", query, re.I):
            wanted.add(token)
    return sorted(wanted)


def readable_source_names(evidence) -> set[str]:
    """Readable publication names, only if the snapshot covers every evidence document."""
    snapshot = _READABLE_SOURCES.get()
    if not snapshot or not evidence:
        return set()
    ids, names = snapshot
    if not {row.get("document_id") for row in evidence} <= ids:
        return set()
    return set(names)


def _boilerplate_only(text: str) -> bool:
    remainder = text.replace("本项目自建的虚构资料，仅用于开发与演示，不代表任何真实公司的制度。", "")
    remainder = remainder.replace("#", "").strip()
    return "本项目自建的虚构资料" in text and len(remainder) < 18


_MONTHS = (
    "january february march april may june july august september october november december"
).split()
_DATE = re.compile(
    r"\b(?P<month>" + "|".join(_MONTHS) + r")\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})\b"
    r"|\b(?P<day2>\d{1,2})\s+(?P<month2>" + "|".join(_MONTHS) + r"),?\s+(?P<year2>\d{4})\b"
    r"|\b(?P<iso>\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_CLOCK = re.compile(r"\b(\d{2}:\d{2}:\d{2})\b")


def _source_aliases(name: str) -> set[str]:
    """A publication is named in a question by its masthead, not its feed name:
    "The Independent - Life and Style" is asked about as "The Independent"."""
    aliases = {name.strip()}
    for separator in (" - ", " | "):
        if separator in name:
            aliases.add(name.split(separator)[0].strip())
    return {alias for alias in aliases if len(alias) >= 3}


def _written_as_name(text: str) -> bool:
    """"The Age" is a newspaper; "the age of AI" is not. A mention counts only when it is
    written like a name: every word capitalised, or an all-caps acronym ("CNBC")."""
    words = re.findall(r"[^\W\d_]+", text)
    return bool(words) and (text.isupper() or all(word[0].isupper() for word in words))


def _dates_in(text: str) -> tuple[set[str], set[str]]:
    days = set()
    for match in _DATE.finditer(text):
        if match["iso"]:
            days.add(match["iso"])
            continue
        month = (match["month"] or match["month2"]).lower()
        day, year = match["day"] or match["day2"], match["year"] or match["year2"]
        days.add(f"{year}-{_MONTHS.index(month) + 1:02d}-{int(day):02d}")
    return days, set(_CLOCK.findall(text))


def _publication_dates(text):
    days, clocks = set(), set()
    for match in _DATE.finditer(text):
        prefix = text[max(0, match.start() - 70):match.start()]
        if re.search(r'(?:published|posted|dated)(?:\s+on)?\s*$'
                     r'|(?:articles?|reports?|stories|pieces?|coverage)(?:\s+(?:published|dated))?'
                     r'\s+(?:on|from|dated)\s*$|(?:发表于|发布于|刊登于)\s*$', prefix, re.I):
            found, _ = _dates_in(match.group())
            days.update(found)
    for match in _CLOCK.finditer(text):
        prefix = text[max(0, match.start() - 40):match.start()]
        if re.search(r'(?:published|posted|article|report)(?:\s+at)?\s*$', prefix, re.I):
            clocks.add(match.group())
    return days, clocks


_BOUND = re.compile(r"\b(before|prior to|after|since)\s+$", re.I)


def _publication_bounds(text):
    """"Reported before/after DATE" bounds a publication's date; the date's own report counts."""
    bounds = []
    for match in _DATE.finditer(text):
        found = _BOUND.search(text[max(0, match.start() - 30):match.start()])
        if found:
            day = next(iter(_dates_in(match.group())[0]))
            bounds.append(("<=" if found.group(1).lower() in {"before", "prior to"} else ">=", day))
    return bounds


def source_mentions(question: str, authorized_documents) -> list[dict]:
    """Exact legacy alias occurrences, without document/publication deduplication.

    Only caller-authorized metadata is consulted. The original offsets and the
    complete authorized candidate documents are retained for each occurrence.
    """
    by_alias: dict[str, list] = {}
    spelled: dict[str, str] = {}
    for document in authorized_documents:
        source = (document.metadata_json or {}).get("source")
        if source:
            for alias in _source_aliases(source):
                by_alias.setdefault(alias.lower(), []).append(document)
                spelled[alias.lower()] = alias
    if not by_alias:
        return []
    found = []
    for alias in by_alias:
        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
        found.extend(
            (m.start(), m.end(), alias)
            for m in re.finditer(pattern, question, re.IGNORECASE)
            if m.group(0) == spelled[alias] or _written_as_name(m.group(0))
        )
    mentions, cursor = [], -1
    for start, end, alias in sorted(found, key=lambda row: (row[0], -(row[1] - row[0]))):
        if start >= cursor:
            mentions.append({"start": start, "end": end, "mention": alias,
                             "documents": by_alias[alias]})
            cursor = end
    return mentions


def route_sources(query: str, documents, *, strict_dates=False) -> list[dict]:
    """Which named publications a question asks about, one group per mention.

    Only document metadata that the caller can already read is consulted. A date or a
    clock time written after a mention (and before the next one) narrows that group to
    the articles published then, when there are any; otherwise the group keeps every
    article of that publication. Documents without a `source` produce no groups, so the
    Chinese corpus is untouched.
    """
    mentions = source_mentions(query, documents)
    groups = []
    for index, occurrence in enumerate(mentions):
        start, end, alias = occurrence["start"], occurrence["end"], occurrence["mention"]
        stop = mentions[index + 1]["start"] if index + 1 < len(mentions) else len(query)
        days, clocks = (_publication_dates(query[end:stop]) if strict_dates
                        else _dates_in(query[end:stop]))
        members = occurrence["documents"]

        def published(document):
            return str((document.metadata_json or {}).get("published_at") or "")

        dated = [
            document
            for document in members
            if published(document)
            and (published(document)[:10] in days or any(clock in published(document) for clock in clocks))
        ]
        selections = [dated] if dated else []
        bounded_by_date = False
        if not dated and strict_dates:
            # Each "before/after DATE" is its own time point inside one publication.
            for op, day in _publication_bounds(query[end:stop]):
                bounded = [d for d in members if published(d) and (
                    published(d)[:10] <= day if op == "<=" else published(d)[:10] >= day)]
                if bounded:
                    selections.append(bounded)
                    bounded_by_date = True
        clause = query[start:stop].strip(" ,;'\"")
        for chosen in selections or [members]:
            key = sorted(document.id for document in chosen)
            if any(group["key"] == key for group in groups):
                continue
            groups.append(
                {
                    "mention": alias,
                    "clause": clause if len(clause.split()) >= 4 else "",
                    "dated": bool(selections),
                    "bounded": bounded_by_date,
                    "key": key,
                    "versions": [d.active_version_id for d in chosen if d.active_version_id],
                }
            )
    return [group for group in groups if group["versions"]]


def _expanded(db, user, hits, documents, require_chunk_fn):
    """Add every active passage of the lane's first `documents` articles to the lane.

    A lane ranks passages across every article of a publication, so the right article
    often arrives with the wrong paragraph. The article-level signal is kept by taking
    the top articles from the lane's own order; the paragraph is then left to the
    cross-encoder. Every added passage is still checked by `require_chunk_fn`."""
    from sqlalchemy import select

    from app.models import Chunk

    top_versions = []
    for hit in hits:
        _chunk, version, _document = require_chunk_fn(db, user, hit["chunk_id"], active_only=True)
        if version.id not in top_versions:
            top_versions.append(version.id)
        if len(top_versions) >= documents:
            break
    seen = {hit["chunk_id"] for hit in hits}
    extra = db.scalars(
        select(Chunk.id).where(Chunk.version_id.in_(top_versions)).order_by(Chunk.version_id, Chunk.ordinal)
    ).all()
    return hits + [{"chunk_id": key, "score": 0.0} for key in extra if key not in seen]


def _reranked(db, user, query, hits, require_chunk_fn):
    """Order one lane by cross-encoder relevance of the passage, with its document's
    title, source and date in front of it: which article a sentence is from is part of
    whether it answers a question that names the article."""
    from app.rerank import rerank

    def passage(hit):
        chunk, _version, document = require_chunk_fn(db, user, hit["chunk_id"], active_only=True)
        metadata = document.metadata_json or {}
        header = " · ".join(
            part
            for part in (document.title, metadata.get("source"), str(metadata.get("published_at") or "")[:10])
            if part
        )
        return f"{header}\n{chunk.text}"

    return rerank(query, hits, text_of=passage, window=len(hits))


def foreign_tenant_mentioned(db, user, query: str) -> bool:
    """Reject an explicit tenant switch before unrelated local evidence reaches a model."""
    if db is None or not hasattr(db, "execute") or not getattr(user, "tenant_id", None):
        return False
    tenants = db.execute(select(Tenant.id, Tenant.name)).all()
    return any(
        tenant_id != user.tenant_id and len(name.strip()) >= 2 and name.strip() in query
        for tenant_id, name in tenants
    )


@dataclass
class RetrievalResult:
    evidence: list[dict]
    candidates: list[dict]
    readable_documents: int
    searchable_documents: int
    context_tokens: int
    embed_ms: float
    retrieval_ms: float
    source_groups: list | None = None
    blocked_reason: str | None = None
    planning_ms: float = 0.0
    planning_tokens: int = 0
    planned_queries_count: int = 0
    planning_error: str | None = None
    missing_constraints: list | None = None


def retrieve_authorized(
    db,
    user,
    query,
    *,
    cfg,
    models,
    search,
    top_k=None,
    readable_documents_fn=readable_documents,
    require_chunk_fn=require_chunk,
):
    """Retrieve evidence without generating an answer or writing history.

    Identity and ACL scope come from the server. Every returned chunk is checked
    against the business database before it is exposed to either a model or caller.
    """
    documents = readable_documents_fn(db, user)
    _READABLE_SOURCES.set((
        frozenset(document.id for document in documents),
        frozenset(source for document in documents if (source := (document.metadata_json or {}).get("source"))),
    ))
    versions = [document.active_version_id for document in documents if document.active_version_id]
    evidence, candidates = [], []
    embed_ms = retrieval_ms = 0.0
    context_tokens = 0
    limit = min(max(top_k or cfg.top_k, 1), 8)
    if foreign_tenant_mentioned(db, user, query):
        return RetrievalResult(
            evidence, candidates, len(documents), len(versions), 0, embed_ms, retrieval_ms,
            blocked_reason="tenant_scope",
        )
    if not versions:
        return RetrievalResult(
            evidence, candidates, len(documents), 0, 0, embed_ms, retrieval_ms
        )

    started = time.monotonic()
    vector = models.embed([query])[0] if cfg.retrieval_mode != "bm25" else None
    embed_ms = (time.monotonic() - started) * 1000
    started = time.monotonic()
    groups = route_sources(query, documents, strict_dates=getattr(cfg, 'focused_generation_enabled', False)) if getattr(cfg, "source_routing", True) else []
    goal = _TASK_GOAL.get()
    if goal and goal != query and getattr(cfg, "source_routing", True):
        keys = {tuple(group["key"]) for group in groups}
        for group in route_sources(goal, documents, strict_dates=getattr(cfg, 'focused_generation_enabled', False)):
            if tuple(group["key"]) not in keys:
                groups.append(group)
                keys.add(tuple(group["key"]))
    fuse = _FUSE_GOAL.get()
    fuse = fuse() if callable(fuse) else fuse  # adaptive tasks switch once they escalate
    missing = (missing_constraints(goal, query)
               if goal and goal != query and fuse else [])
    diversify = bool(groups) or needs_document_diversity(query) or multi_source_intent(query)
    search_limit = min(limit * 3, 24) if diversify else limit
    planned_queries, plan_vectors = [], []
    planning_ms = planning_tokens = 0
    planning_error = None
    if groups and len(query.split()) >= 14 and getattr(cfg, 'source_query_plan', False):
        from app.query_planner import plan_search_queries

        planning_started = time.monotonic()
        try:
            planned_queries, usage = plan_search_queries(models, query, cfg)
            plan_vectors = (models.embed(planned_queries) if vector is not None
                            else [None] * len(planned_queries))
            planning_tokens = usage['prompt_tokens'] + usage['completion_tokens']
        except (ValueError, KeyError, TypeError):
            planning_error = 'invalid_search_plan'
        planning_ms = round((time.monotonic() - planning_started) * 1000, 1)

    candidate_depth = getattr(cfg, "retrieval_candidate_depth", 0)

    def run(scope, size, text=query, embedding=vector):
        size = max(size, candidate_depth)
        if cfg.retrieval_mode == "hybrid":
            # Hybrid otherwise searches 50 hits per route. An explicit larger
            # candidate pool must widen those routes too, not only the fused slice.
            if candidate_depth > 50:
                return search.retrieve_hybrid(
                    text, embedding, user.tenant_id, scope, size, depth=candidate_depth,
                )
            return search.retrieve_hybrid(text, embedding, user.tenant_id, scope, size)
        if embedding is not None:
            return search.retrieve(embedding, user.tenant_id, scope, size)
        return search.retrieve_bm25(text, user.tenant_id, scope, size)

    def lane_hits(group, size):
        if planned_queries:
            searches = [run(group['versions'], max(size, 12), text, embedded)
                        for text, embedded in zip(planned_queries, plan_vectors, strict=True)]
            merged, seen_chunks, seen_documents = [], set(), set()
            for hits in searches:
                for hit in hits:
                    _chunk, _version, document = require_chunk_fn(
                        db, user, hit['chunk_id'], active_only=True)
                    if document.id not in seen_documents:
                        merged.append(hit)
                        seen_chunks.add(hit['chunk_id'])
                        seen_documents.add(document.id)
                        break
            for position in range(max(map(len, searches), default=0)):
                for hits in searches:
                    if position < len(hits) and hits[position]['chunk_id'] not in seen_chunks:
                        merged.append(hits[position])
                        seen_chunks.add(hits[position]['chunk_id'])
            return merged
        if getattr(cfg, 'article_first_lanes', False):
            hits = run(group['versions'], max(size, 48))
            first, rest, seen_documents = [], [], set()
            for hit in hits:
                _chunk, _version, document = require_chunk_fn(
                    db, user, hit['chunk_id'], active_only=True)
                if document.id in seen_documents:
                    rest.append(hit)
                else:
                    first.append(hit)
                    seen_documents.add(document.id)
            return first + rest
        document_facets = getattr(cfg, 'source_facet_document_queries', False)
        if getattr(cfg, 'source_facet_queries', False) or document_facets:
            facets = retrieval_facets(query)
            if facets:
                searches = []
                for facet in facets:
                    embedded = models.embed([facet])[0] if vector is not None else None
                    searches.append(run(group['versions'], size, facet, embedded))
                merged, seen = [], set()
                if document_facets:
                    seen_documents = set()
                    for hits in searches:
                        for hit in hits:
                            _chunk, _version, document = require_chunk_fn(
                                db, user, hit['chunk_id'], active_only=True)
                            if document.id not in seen_documents:
                                merged.append(hit)
                                seen.add(hit['chunk_id'])
                                seen_documents.add(document.id)
                                break
                for index in range(max(map(len, searches), default=0)):
                    for hits in searches:
                        if index < len(hits) and hits[index]['chunk_id'] not in seen:
                            merged.append(hits[index])
                            seen.add(hits[index]['chunk_id'])
                return merged
        # Inside one publication the question's own words about that publication
        # pick the paragraph; the whole question keeps the shared predicate.
        if not group["clause"] or not (getattr(cfg, "source_clause_queries", False)
                                      or getattr(cfg, "source_focus_queries", False)):
            return run(group["versions"], size)
        text = (group['clause'] if getattr(cfg, "source_focus_queries", False)
                else f"{group['clause']} {query}")
        embedding = models.embed([text])[0] if vector is not None else None
        return run(group["versions"], size, text, embedding)

    # A publication the question names gets its own share of the budget, searched
    # inside that publication's articles only; the rest is filled from the ordinary
    # ranking. Every group scope is a subset of the readable versions above.
    per_document = getattr(cfg, "document_quota", 2)
    quota = max(2, (limit - 2) // len(groups)) if groups else 0
    rerank_on = getattr(cfg, "passage_rerank", False)
    depth = getattr(cfg, "passage_rerank_depth", 24) if rerank_on else None
    lanes = [
        (f"source:{group['mention']}", lane_hits(group, depth or quota * 3)) for group in groups
    ]
    if missing:
        # The model's query dropped something the user asked about: search the
        # user's own question as well and interleave it with the routed lanes.
        goal_vector = models.embed([goal])[0] if vector is not None else None
        lanes.append(("goal", run(versions, max(depth or 0, search_limit), goal, goal_vector)))
    lanes.append(("global", run(versions, max(depth or 0, search_limit))))
    # Record each lane's retrieved order after its query merging, before expansion
    # or reranking. Expanded passages have no retrieval rank. Copies keep a hit
    # shared by multiple lanes from acquiring another lane's rank.
    lanes = [(lane, [{**hit, "retrieval_rank": rank} for rank, hit in enumerate(hits, 1)])
             for lane, hits in lanes]
    if rerank_on:
        expand = getattr(cfg, "passage_expand_documents", 0)
        if expand:
            lanes = [(lane, _expanded(db, user, hits, expand, require_chunk_fn)) for lane, hits in lanes]
        lanes = [(lane, _reranked(db, user, query, hits, require_chunk_fn)) for lane, hits in lanes]
    routed = []
    for position in range(max((len(hits) for _, hits in lanes[:-1]), default=0)):
        routed.extend((lane, hits[position]) for lane, hits in lanes[:-1] if position < len(hits))
    ordered = routed + [("global", hit) for hit in lanes[-1][1]]
    retrieval_ms = (time.monotonic() - started) * 1000

    # "Reported by X before/after DATE": X's articles outside every requested window
    # cannot fill in through the global lane. Exact-date groups keep their old behaviour.
    windows = {}
    for group in groups:
        if group.get("bounded"):
            windows.setdefault(group["mention"], set()).update(group["key"])

    def outside_requested_dates(document):
        source = (document.metadata_json or {}).get("source") or ""
        aliases = {alias.lower() for alias in _source_aliases(source)} if source else set()
        return any(mention in aliases and document.id not in keys for mention, keys in windows.items())

    admitted_by_document, admitted_by_lane, admitted_chunks = {}, {}, set()
    for rank, (lane, hit) in enumerate(ordered, 1):
        if hit["chunk_id"] in admitted_chunks:
            continue
        chunk, version, document = require_chunk_fn(db, user, hit["chunk_id"], active_only=True)
        candidate = {
            "chunk_id": chunk.id,
            "rank": rank,
            "retrieval_rank": hit.get("retrieval_rank"),
            "rerank_score": hit.get("rerank_score"),
            "title": document.title,
            "document_id": document.id,
            "version_id": version.id,
            "locator_label": chunk.locator.get("label", ""),
            "score": round(hit.get("cosine_similarity", hit.get("score", 0)), 4),
            "bm25_rank": hit.get("bm25_rank"),
            "dense_rank": hit.get("dense_rank"),
            "fusion_score": round(hit["score"], 6) if cfg.retrieval_mode == "hybrid" else None,
            "admitted": False,
            "excluded_because": None,
            "lane": lane,
        }
        candidates.append(candidate)
        if cfg.retrieval_mode == "dense" and hit["cosine_similarity"] < cfg.min_similarity:
            candidate["excluded_because"] = "below_min_similarity"
            continue
        if _boilerplate_only(chunk.text):
            candidate["excluded_because"] = "boilerplate_only"
            continue
        if lane == "global" and outside_requested_dates(document):
            candidate["excluded_because"] = "outside_requested_publication_dates"
            continue
        if len(evidence) >= limit:
            candidate["excluded_because"] = "top_k_full"
            continue
        if lane == "goal" and admitted_by_lane.get(lane, 0) >= max(2, limit // 2):
            candidate["excluded_because"] = "goal_quota"
            continue
        if lane not in {"global", "goal"} and admitted_by_lane.get(lane, 0) >= quota:
            candidate["excluded_because"] = "source_quota"
            continue
        if diversify and admitted_by_document.get(document.id, 0) >= per_document:
            candidate["excluded_because"] = "document_quota"
            continue
        cost = token_count(chunk.text) + token_count(document.title) + 100
        if context_tokens + cost > cfg.context_token_budget:
            candidate["excluded_because"] = "context_budget_exhausted"
            continue
        context_tokens += cost
        candidate["admitted"] = True
        candidate["evidence_id"] = f"E{len(evidence) + 1}"
        evidence.append(
            {
                "id": candidate["evidence_id"],
                "chunk_id": chunk.id,
                "document_id": document.id,
                "version_id": version.id,
                "title": document.title,
                "text": chunk.text,
                "locator": chunk.locator,
                "metadata": document.metadata_json,
            }
        )
        admitted_by_document[document.id] = admitted_by_document.get(document.id, 0) + 1
        admitted_by_lane[lane] = admitted_by_lane.get(lane, 0) + 1
        admitted_chunks.add(chunk.id)

    for item in evidence:
        require_chunk_fn(db, user, item["chunk_id"], active_only=True)
    return RetrievalResult(
        evidence,
        candidates,
        len(documents),
        len(versions),
        context_tokens,
        embed_ms,
        retrieval_ms,
        [{k: group[k] for k in ("mention", "dated")} | {"documents": len(group["key"])} for group in groups],
        planning_ms=planning_ms, planning_tokens=planning_tokens,
        planned_queries_count=len(planned_queries), planning_error=planning_error,
        missing_constraints=missing,
    )
