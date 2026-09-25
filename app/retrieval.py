"""Permission-scoped retrieval shared by single-turn RAG and Agent tools."""

import re
import time
from dataclasses import dataclass

from sqlalchemy import select

from app.chunking import token_count
from app.models import Tenant
from app.security import readable_documents, require_chunk
from app.task_analysis import multi_source_intent, needs_document_diversity


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


def route_sources(query: str, documents) -> list[dict]:
    """Which named publications a question asks about, one group per mention.

    Only document metadata that the caller can already read is consulted. A date or a
    clock time written after a mention (and before the next one) narrows that group to
    the articles published then, when there are any; otherwise the group keeps every
    article of that publication. Documents without a `source` produce no groups, so the
    Chinese corpus is untouched.
    """
    by_alias: dict[str, list] = {}
    spelled: dict[str, str] = {}
    for document in documents:
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
            for m in re.finditer(pattern, query, re.IGNORECASE)
            if m.group(0) == spelled[alias] or _written_as_name(m.group(0))
        )
    mentions, cursor = [], -1
    for start, end, alias in sorted(found, key=lambda row: (row[0], -(row[1] - row[0]))):
        if start >= cursor:
            mentions.append((start, end, alias))
            cursor = end
    groups = []
    for index, (start, end, alias) in enumerate(mentions):
        stop = mentions[index + 1][0] if index + 1 < len(mentions) else len(query)
        days, clocks = _dates_in(query[end:stop])
        members = by_alias[alias]
        dated = [
            document
            for document in members
            if (published := str((document.metadata_json or {}).get("published_at") or ""))
            and (published[:10] in days or any(clock in published for clock in clocks))
        ]
        chosen = dated or members
        key = sorted(document.id for document in chosen)
        if any(group["key"] == key for group in groups):
            continue
        clause = query[start:stop].strip(" ,;'\"")
        groups.append(
            {
                "mention": alias,
                "clause": clause if len(clause.split()) >= 4 else "",
                "dated": bool(dated),
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
    groups = route_sources(query, documents) if getattr(cfg, "source_routing", True) else []
    diversify = bool(groups) or needs_document_diversity(query) or multi_source_intent(query)
    search_limit = min(limit * 3, 24) if diversify else limit

    def run(scope, size, text=query, embedding=vector):
        if cfg.retrieval_mode == "hybrid":
            return search.retrieve_hybrid(text, embedding, user.tenant_id, scope, size)
        if embedding is not None:
            return search.retrieve(embedding, user.tenant_id, scope, size)
        return search.retrieve_bm25(text, user.tenant_id, scope, size)

    def lane_hits(group, size):
        # Inside one publication the question's own words about that publication
        # pick the paragraph; the whole question keeps the shared predicate.
        if not group["clause"] or not getattr(cfg, "source_clause_queries", False):
            return run(group["versions"], size)
        text = f"{group['clause']} {query}"
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
    lanes.append(("global", run(versions, max(depth or 0, search_limit))))
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

    admitted_by_document, admitted_by_lane, admitted_chunks = {}, {}, set()
    for rank, (lane, hit) in enumerate(ordered, 1):
        if hit["chunk_id"] in admitted_chunks:
            continue
        chunk, version, document = require_chunk_fn(db, user, hit["chunk_id"], active_only=True)
        candidate = {
            "chunk_id": chunk.id,
            "rank": rank,
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
        if len(evidence) >= limit:
            candidate["excluded_because"] = "top_k_full"
            continue
        if lane != "global" and admitted_by_lane.get(lane, 0) >= quota:
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
    )
