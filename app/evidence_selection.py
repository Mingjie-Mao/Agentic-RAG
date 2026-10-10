"""Sentence-level evidence selection (the CRAG / FILCO idea), off the production path.

A generator given the gold passages and nothing else answers the external yes/no set
far better than one given the same gold passages plus the retrieval around them, and
far better than one given retrieval alone. Whole chunks carry both problems: the right
sentence is often not in them, and when it is, it arrives with a paragraph of material
that pulls the answer elsewhere.

This keeps retrieval as it is and changes what reaches the model: the chunks of each
lane are split into sentences, every sentence is scored against the question (or
against the part of the question about that lane's publication), and only the best few
per publication are kept. Every selected sentence is a verbatim substring of an
authorized chunk, so evidence items keep their chunk id and citation validation works
unchanged. Nothing here grants access: it only narrows chunks retrieval already
returned for this user.
"""

import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import isfinite

from agent.planner import _EN_STOP, _grams, evidence_coverage, keywords, subgoals
from app.chunking import token_count
from app.verdict import question_slots

# A boundary is end punctuation followed by whitespace (so "$14.99" stays whole), not
# after an initial or a title ("U.S. District", "Mr. Smith"), or any Chinese full stop.
_BOUNDARY = re.compile(
    r"(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bvs\.)"
    r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)]))\s+"
    r"|(?<=[。！？])|\n+"
)


def split_sentences(text: str) -> list[str]:
    """Sentences as verbatim substrings of `text`, fragments under four words dropped."""
    sentences = []
    for piece in _BOUNDARY.split(text):
        piece = (piece or "").strip()
        if len(piece.split()) >= 4 or (piece and not piece.isascii() and len(piece) >= 8):
            sentences.append(piece)
    return sentences


def header_of(document) -> str:
    metadata = document.metadata_json or {}
    published = str(metadata.get("published_at") or "")[:10]
    return " · ".join(part for part in (document.title, metadata.get("source"), published) if part)


def select_sentences(
    question: str,
    lanes: list[dict],
    *,
    score,
    per_lane: int = 2,
    global_extra: int = 2,
    max_total: int = 8,
    neighbors: int = 0,
):
    """Pick the evidence sentences for `question`.

    `lanes` is a list of {"name", "query", "chunks": [{chunk_id, document_id, version_id,
    title, text, locator, metadata, header}]}, named lanes first and the global lane last.
    `score(query, texts) -> list[float]` is the relevance scorer (a cross-encoder in
    practice). Returns evidence items in the shape retrieval produces, one per chunk that
    contributed a sentence, with the chosen sentences joined in their original order.
    """
    chosen: dict[str, dict] = {}
    order: list[str] = []
    seen_sentences: set[str] = set()

    lane_of: dict[str, str] = {}

    def take(lane, count):
        candidates = []
        for chunk in lane["chunks"]:
            sentences = split_sentences(chunk["text"])
            for index, sentence in enumerate(sentences):
                candidates.append((chunk, sentences, index, sentence))
        if not candidates or count <= 0:
            return
        scores = score(lane["query"], [f"{c[0]['header']}\n{c[3]}" for c in candidates])
        ranked = sorted(zip(scores, range(len(candidates)), candidates), key=lambda row: (-row[0], row[1]))
        taken = 0
        for _score, _position, (chunk, sentences, index, sentence) in ranked:
            if taken >= count or sum(len(v["indices"]) for v in chosen.values()) >= max_total:
                break
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            entry = chosen.setdefault(chunk["chunk_id"], {"chunk": chunk, "sentences": sentences, "indices": set()})
            if chunk["chunk_id"] not in order:
                order.append(chunk["chunk_id"])
                lane_of[chunk["chunk_id"]] = lane["name"]
            for offset in range(-neighbors, neighbors + 1):
                if 0 <= index + offset < len(sentences):
                    entry["indices"].add(index + offset)
            taken += 1

    named, rest = lanes[:-1], lanes[-1:]
    for lane in named:
        take(lane, per_lane)
    for lane in rest:
        take(lane, global_extra if named else max_total)

    evidence = []
    for number, chunk_id in enumerate(order, 1):
        entry = chosen[chunk_id]
        chunk = entry["chunk"]
        text = " ".join(entry["sentences"][i] for i in sorted(entry["indices"]))
        evidence.append(
            {
                "id": f"E{number}",
                "chunk_id": chunk_id,
                "document_id": chunk["document_id"],
                "version_id": chunk["version_id"],
                "title": chunk["title"],
                "text": text,
                "locator": chunk["locator"],
                "metadata": chunk["metadata"],
                "selected_sentences": len(entry["indices"]),
                "lane": lane_of[chunk_id],
            }
        )
    return evidence


def cross_encoder_scorer(model_path=None):
    """Score with the project's cross-encoder (bge-reranker-v2-m3), batched, or with a
    fine-tuned copy of it saved at `model_path`."""
    from app.rerank import _model
    from app.config import settings

    loaded = None
    if model_path:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        loaded = (
            AutoTokenizer.from_pretrained(model_path),
            AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval(),
            device,
            torch,
        )

    def score(query: str, texts: list[str]) -> list[float]:
        tokenizer, model, device, torch = loaded or _model()
        cfg = settings()
        values = []
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            encoded = tokenizer(
                [query] * len(batch), batch, padding=True, truncation=True,
                max_length=cfg.rerank_max_tokens, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                values.extend(model(**encoded).logits.view(-1).float().tolist())
        return values

    return score


def build_lanes(db, user, question, *, cfg, models, search, top_k=8, chunks_per_lane=6, clause_queries=False):
    """Candidate chunks per retrieval lane, re-authorized, with the query each lane is
    scored against: the whole question, or the part of it about that publication."""
    from app.retrieval import retrieve_authorized, route_sources
    from app.security import readable_documents, require_chunk

    found = retrieve_authorized(
        db, user, question, cfg=cfg.model_copy(update={"passage_rerank": True}),
        models=models, search=search, top_k=top_k,
    )
    clauses = {}
    if clause_queries:
        for group in route_sources(question, readable_documents(db, user)):
            clauses[f"source:{group['mention']}"] = group.get("clause") or ""
    lanes: dict[str, list] = {}
    for candidate in found.candidates:
        lane = candidate.get("lane", "global")
        if len(lanes.setdefault(lane, [])) >= chunks_per_lane:
            continue
        chunk, version, document = require_chunk(db, user, candidate["chunk_id"], active_only=True)
        lanes[lane].append(
            {
                "chunk_id": chunk.id,
                "document_id": document.id,
                "version_id": version.id,
                "title": document.title,
                "text": chunk.text,
                "locator": chunk.locator,
                "metadata": document.metadata_json,
                "header": header_of(document),
            }
        )
    named = [name for name in lanes if name != "global"]
    ordered = [
        {"name": name, "query": (f"{clauses[name]} {question}" if clauses.get(name) else question), "chunks": lanes[name]}
        for name in named
    ]
    ordered.append({"name": "global", "query": question, "chunks": lanes.get("global", [])})
    return ordered, found.evidence


# These policies consume authorized rows only. They do not retrieve, authorize,
# call a model, or prove facts. The experiment caller must freeze the actual
# answer_quality_enabled setting used by evidence_coverage across all profiles.
_HARD_EXCLUSIONS = frozenset({
    "boilerplate_only", "below_min_similarity",
    "outside_requested_publication_dates", "tenant_scope",
})
_SOFT_EXCLUSIONS = frozenset({
    "document_quota", "source_quota", "top_k_full", "context_budget_exhausted",
})


@dataclass
class SelectionResult:
    evidence: list[dict]
    context_tokens: int
    trace: dict


def selection_facets(question: str) -> list[str]:
    """Reuse the longer existing decomposition; subgoals win equal lengths."""
    goals, slots = subgoals(question), question_slots(question)
    return list((slots if len(slots) > len(goals) else goals)[:6])


@dataclass(frozen=True)
class _Caps:
    limit: int
    tokens: int
    document: int | None
    source: int | None


def _positive(value, name, maximum=None, optional=False):
    if value is None and optional:
        return
    if (type(value) is not int or value < 1
            or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid {name}")


def _caps(limit, token_budget, document_quota, source_quota):
    _positive(limit, "limit", 8)
    _positive(token_budget, "token_budget", 5000)
    _positive(document_quota, "document_quota", optional=True)
    _positive(source_quota, "source_quota", optional=True)
    return _Caps(limit, token_budget, document_quota, source_quota)


def _rank(row):
    rank = row.get("parent_rank", row.get("rank"))
    if isinstance(rank, bool) or not isinstance(rank, (int, float)) or not isfinite(rank):
        raise ValueError("Invalid candidate rank")
    return rank


def _exclusions(row):
    value = row.get("excluded_because")
    if value is None or value == "":
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple)) and all(isinstance(reason, str) for reason in value):
        return set(value)
    raise ValueError("Invalid candidate exclusions")


def _digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


class _Pool:
    """Precomputed lexical features and real eligible lane occurrences."""

    def __init__(self, question, candidates, caps):
        self.caps = caps
        self.rows, self.options, self.blocked = {}, {}, {}
        identities = {}
        for source_row in candidates:
            row = deepcopy(source_row)
            keys = ("chunk_id", "document_id", "version_id", "source_sha256", "text", "title")
            if any(not isinstance(row.get(key), str) for key in keys):
                raise ValueError("Invalid candidate identity")
            if any(not row[key] for key in keys[:4]):
                raise ValueError("Invalid candidate identity")
            cid = row["chunk_id"]
            identity = tuple(row[key] for key in keys[1:])
            if cid in identities and identities[cid] != identity:
                raise ValueError("Conflicting candidate identity")
            identities[cid] = identity
            _rank(row)
            if row.get("lane_type") not in {"global", "source"}:
                raise ValueError("Invalid candidate lane")
            if not isinstance(row.get("lane_sha256"), str) or not row["lane_sha256"]:
                raise ValueError("Invalid candidate lane")
            reasons = _exclusions(row)
            hard = reasons & _HARD_EXCLUSIONS
            unknown = reasons - _HARD_EXCLUSIONS - _SOFT_EXCLUSIONS
            if hard or unknown:
                self.blocked.setdefault(cid, set()).update(hard or {"unrecognized_exclusion"})
                continue
            self.options.setdefault(cid, []).append(row)
        for cid, options in self.options.items():
            options.sort(key=_rank)
            # Repeated occurrences in the same real lane have identical cap
            # effects. Stable sorting retains the earliest rank and input tie.
            lanes = {}
            for row in options:
                lanes.setdefault((row["lane_type"], row["lane_sha256"]), row)
            self.options[cid] = list(lanes.values())
            self.rows[cid] = self.options[cid][0]
        self.facets = selection_facets(question)
        coverage = evidence_coverage(self.facets, list(self.rows.values()), question)
        wanted = _grams(keywords(question)) - _EN_STOP
        self.facets_by_id = {
            cid: {index for index, facet in enumerate(self.facets) if cid in coverage[facet]}
            for cid in self.rows
        }
        self.content = {cid: _grams(row["text"]) for cid, row in self.rows.items()}
        self.terms = {
            cid: wanted & _grams(f"{row['title']} {row['text']}") for cid, row in self.rows.items()
        }
        self.cost = {
            cid: token_count(row["text"]) + token_count(row["title"]) + 100
            for cid, row in self.rows.items()
        }
        self.lexical_signal = any(self.terms.values()) or any(self.facets_by_id.values())
        self.pairs = {}
        chunk_ids = sorted(self.rows)
        for index, left in enumerate(chunk_ids):
            for right in chunk_ids[index + 1:]:
                a, b = self.content[left], self.content[right]
                self.pairs[left, right] = Fraction(len(a & b), len(a | b)) if a | b else Fraction(0)

    def redundancy(self, left, right):
        return self.pairs[tuple(sorted((left, right)))]

    def assign(self, selected):
        """Find the earliest feasible *recorded* occurrence for every chunk.

        At most eight rows enter the small lane assignment search. Backtracking
        prevents a flexible required chunk from stealing a scarce source lane.
        """
        caps = self.caps
        if len(selected) > caps.limit or sum(self.cost[cid] for cid in selected) > caps.tokens:
            return None, "context_budget_exhausted" if len(selected) <= caps.limit else "top_k_full"
        documents = Counter(self.rows[cid]["document_id"] for cid in selected)
        if caps.document is not None and max(documents.values(), default=0) > caps.document:
            return None, "document_quota"
        lane_order = sorted(selected, key=lambda cid: (_rank(self.rows[cid]), cid))
        assigned, counts = {}, Counter()
        # A failed suffix depends only on the source usage affecting that suffix,
        # not on which earlier chunks consumed those slots. Memoize failures only
        # so successful first-fit occurrence order stays unchanged.
        relevant_lanes = [set() for _ in range(len(lane_order) + 1)]
        for index in range(len(lane_order) - 1, -1, -1):
            relevant_lanes[index] = relevant_lanes[index + 1] | {
                row["lane_sha256"] for row in self.options[lane_order[index]]
                if row["lane_type"] == "source" and caps.source is not None
            }
        relevant_lanes = [sorted(lanes) for lanes in relevant_lanes]
        failed = set()

        def visit(index):
            if index == len(lane_order):
                return True
            state = (index, tuple((lane, counts.get(lane, 0)) for lane in relevant_lanes[index]
                                  if counts.get(lane, 0)))
            if state in failed:
                return False
            cid = lane_order[index]
            for row in self.options[cid]:
                lane = row["lane_sha256"]
                scoped = row["lane_type"] == "source" and caps.source is not None
                if scoped and counts[lane] >= caps.source:
                    continue
                assigned[cid] = row
                if scoped:
                    counts[lane] += 1
                if visit(index + 1):
                    return True
                if scoped:
                    counts[lane] -= 1
                del assigned[cid]
            failed.add(state)
            return False

        return ([assigned[cid] for cid in selected], None) if visit(0) else (None, "source_quota")

    def objective(self, selected, assigned):
        facets = set().union(*(self.facets_by_id[cid] for cid in selected))
        terms = set().union(*(self.terms[cid] for cid in selected))
        redundancy = sum((self.redundancy(a, b) for i, a in enumerate(selected)
                          for b in selected[i + 1:]), Fraction(0))
        return len(facets), len(terms), -redundancy, -sum(_rank(row) for row in assigned)

    def trace(self, method, selected, assigned, required):
        caps = self.caps
        objective = self.objective(selected, assigned)
        rejected = []
        for cid in sorted(set(self.rows) | set(self.blocked)):
            if cid in selected:
                continue
            if cid not in self.rows:
                reasons = sorted(self.blocked[cid])
            else:
                _, reason = self.assign([*selected, cid])
                reasons = [reason or "lower_lexical_objective"]
            rejected.append({"chunk_id": cid, "reasons": reasons})
        return {
            "method": method, "coverage_type": "lexical_candidate_only",
            "facet_sha256": [_digest(facet) for facet in self.facets],
            "facet_count": len(self.facets), "candidate_count": len(self.rows),
            "selected_ids": list(selected), "required_ids": sorted(required),
            "context_tokens": sum(self.cost[cid] for cid in selected),
            "objective": _objective_trace(objective), "rejected": rejected,
            "caps": {"limit": caps.limit, "token_budget": caps.tokens,
                     "document_quota": caps.document, "source_quota": caps.source},
            "cap_flags": {
                "limit_full": len(selected) == caps.limit,
                "token_budget_full": sum(self.cost[cid] for cid in selected) == caps.tokens,
                "document_quota_active": caps.document is not None,
                "source_quota_active": caps.source is not None,
            },
        }


def _objective_trace(objective):
    return [float(value) if isinstance(value, Fraction) else value for value in objective]


def _required(pool, required_ids):
    required_ids = list(required_ids)
    if any(not isinstance(cid, str) or cid not in pool.rows for cid in required_ids):
        raise ValueError("Missing or excluded required evidence")
    required = set(required_ids)
    selected = sorted(required, key=lambda cid: (_rank(pool.rows[cid]), cid))
    assigned, _ = pool.assign(selected)
    if assigned is None:
        raise ValueError("Infeasible required evidence")
    return required, selected, assigned


def _result(pool, selected, assigned, trace):
    assigned = sorted(assigned, key=lambda row: (_rank(row), row["chunk_id"]))
    trace["selected_ids"] = [row["chunk_id"] for row in assigned]
    evidence = [dict(deepcopy(row), id=f"E{index}") for index, row in enumerate(assigned, 1)]
    return SelectionResult(evidence, sum(pool.cost[cid] for cid in selected), trace)


def select_complementary(question, candidates, *, limit=8, token_budget=5000,
                         document_quota=2, source_quota=None, required_ids=()):
    """Greedily cover facets and query terms within registered business caps.

    All signals are lexical proxies. Whole chunks are skipped when oversize;
    caller authorization and frozen coverage configuration remain prerequisites.
    """
    pool = _Pool(question, candidates, _caps(limit, token_budget, document_quota, source_quota))
    required, selected, assigned = _required(pool, required_ids)
    while len(selected) < limit:
        covered = set().union(*(pool.facets_by_id[cid] for cid in selected))
        found = set().union(*(pool.terms[cid] for cid in selected))
        choices = []
        for cid in sorted(pool.rows):
            if cid in selected:
                continue
            allocation, _ = pool.assign([*selected, cid])
            if allocation is None:
                continue
            redundancy = (max((pool.redundancy(cid, old) for old in selected), default=Fraction(0))
                          if pool.lexical_signal else Fraction(0))
            priority = (-len(pool.facets_by_id[cid] - covered), -len(pool.terms[cid] - found),
                        redundancy, _rank(allocation[-1]), cid)
            choices.append((priority, cid, allocation))
        if not choices:
            break
        _, cid, assigned = min(choices, key=lambda item: item[0])
        selected.append(cid)
    trace = pool.trace("complementary_lexical_v1", selected, assigned, required)
    return _result(pool, selected, assigned, trace)


def rank_relevance_candidates(candidates):
    """Comparable original-query scores; preserve every real lane occurrence."""
    rows = deepcopy(candidates)
    priorities = {}
    for row in rows:
        score = row.get('rerank_score')
        if type(score) not in (int, float) or not isfinite(score):
            raise ValueError('Missing or invalid comparable rerank score')
        key = (-score, _rank(row), row['chunk_id'])
        cid = row['chunk_id']
        priorities[cid] = min(priorities.get(cid, key), key)
    ordered = sorted(priorities, key=priorities.get)
    ranks = {cid: i for i, cid in enumerate(ordered, 1)}
    original = {cid: priorities[cid][1] for cid in ordered}
    rows.sort(key=lambda row: (ranks[row['chunk_id']], _rank(row)))
    for row in rows:
        row['parent_rank'] = ranks[row['chunk_id']]
    return rows, ordered, priorities, original


def select_relevant_evidence(candidates, *, limit=8, token_budget=5000,
                             document_quota=2, source_quota=None, required_ids=(),
                             source_groups=None):
    """Cached scores from the SAME original question, then feasible source floors.

    Caller binds authorization and score provenance. No query, references, novelty
    objective or added model call enters this strategy. Real occurrence assignment
    and business caps reuse the existing pool. Source floors never relax a cap.
    """
    rows, ordered, priorities, original = rank_relevance_candidates(candidates)
    pool = _Pool('', rows, _caps(limit, token_budget, document_quota, source_quota))
    required, selected, assigned = _required(pool, required_ids)
    floor = []
    for group, documents in (source_groups or {}).items():
        if (not isinstance(group, str) or not isinstance(documents, (list, tuple))
                or any(not isinstance(d, str) for d in documents)):
            raise ValueError('Invalid source group')
        choices = [cid for cid in ordered if cid in pool.rows
                   and pool.rows[cid]['document_id'] in documents]
        existing = next((cid for cid in choices if cid in selected), None)
        chosen, reason = existing, None
        if chosen is None:
            for cid in choices:
                allocation, reason = pool.assign([*selected, cid])
                if allocation is not None:
                    chosen = cid
                    selected.append(cid)
                    assigned = allocation
                    break
        floor.append({'group_sha256': group, 'status': 'reserved' if chosen else
                      'infeasible' if choices else 'no_eligible_candidate',
                      'chunk_id': chosen, 'reason': None if chosen else reason})
    for cid in ordered:
        if cid not in pool.rows or cid in selected:
            continue
        allocation, _ = pool.assign([*selected, cid])
        if allocation is not None:
            selected.append(cid)
            assigned = allocation
    trace = pool.trace('cached_original_question_relevance_v1', selected, assigned, required)
    trace.update(coverage_type='relevance_with_feasible_source_floor', source_floor=floor,
                 relevance_order=[{'chunk_id': cid, 'rerank_score': -priorities[cid][0],
                                   'original_parent_rank': original[cid]} for cid in ordered])
    # Pool objective is diagnostic only; selection above uses score order alone.
    return _result(pool, selected, assigned, trace)


def replace_weak_evidence(question, candidates, seed, *, limit=8, token_budget=5000,
                          document_quota=2, source_quota=None, max_replacements=8,
                          required_ids=()):
    """Bounded best one-for-one swaps (or appends), with strict improvement.

    The supplied seed must come from the same authorized candidate pool under
    these caps; no truncation or silent relaxation repairs an invalid seed.
    """
    if type(max_replacements) is not int or not 0 <= max_replacements <= 8:
        raise ValueError("Invalid max_replacements")
    pool = _Pool(question, candidates, _caps(limit, token_budget, document_quota, source_quota))
    required, _, _ = _required(pool, required_ids)
    selected = [row.get("chunk_id") for row in seed]
    if (any(not isinstance(cid, str) or cid not in pool.rows for cid in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Invalid seed evidence")
    if not required <= set(selected):
        raise ValueError("Seed missing required evidence")
    # Identity validation also rejects a seed that altered its authorized text.
    for row in seed:
        actual = pool.rows[row["chunk_id"]]
        if any(row.get(key) != actual[key] for key in
               ("document_id", "version_id", "source_sha256", "text", "title")):
            raise ValueError("Conflicting seed identity")
    assigned, _ = pool.assign(selected)
    if assigned is None:
        raise ValueError("Infeasible seed evidence")
    before = pool.objective(selected, assigned)
    objective, changes = before, []
    for _ in range(max_replacements):
        best = None
        for added in sorted(set(pool.rows) - set(selected)):
            removals = [cid for cid in selected if cid not in required]
            if len(selected) < limit:
                removals.append(None)
            for removed in removals:
                trial = [added if cid == removed else cid for cid in selected]
                if removed is None:
                    trial.append(added)
                allocation, _ = pool.assign(trial)
                if allocation is None:
                    continue
                score = pool.objective(trial, allocation)
                if score <= objective:
                    continue
                tie = (added, removed or "", tuple(trial))
                if best is None or score > best[0] or (score == best[0] and tie < best[1]):
                    best = score, tie, trial, allocation, added, removed
        if best is None:
            break
        objective, _, selected, assigned, added, removed = best
        changes.append({"added_ids": [added], "removed_ids": [removed] if removed else [],
                        "objective": _objective_trace(objective),
                        "context_tokens": sum(pool.cost[cid] for cid in selected)})
    trace = pool.trace("weak_evidence_replacement_lexical_v1", selected, assigned, required)
    trace.update(accepted_replacements=len(changes), max_replacements=max_replacements,
                 changes=changes, objective_before=_objective_trace(before),
                 objective_after=_objective_trace(objective))
    return _result(pool, selected, assigned, trace)


def select_slot_evidence(contract, candidates, *, limit=8, token_budget=5000,
                         document_quota=2, source_quota=None, required_ids=(), strategy='literal',
                         focus_queries=None):
    """Reserve source/date or coherent literal witnesses, then fill by input rank.

    Caller owns comparable original-query ranking or rank-only round-robin union.
    Unknown semantic requirements get only a labelled scoped lexical floor. Real
    lane alternatives and all hard caps/exclusions use the existing pool allocator.
    """
    from app.supplement_contract import assess_slot_presence
    if strategy not in {'source', 'literal'}:
        raise ValueError('Unsupported slot strategy')
    pool = _Pool('', candidates, _caps(limit, token_budget, document_quota, source_quota))
    required, selected, assigned = _required(pool, required_ids)
    presence = assess_slot_presence(contract, list(pool.rows.values()))
    focus_queries = dict(focus_queries or {})
    if (not set(focus_queries) <= {s.slot_id for s in contract.slots}
            or any(not isinstance(q, str) or not 2 <= len(q) <= 500 for q in focus_queries.values())):
        raise ValueError('Invalid focused slot query')
    floor = []
    for slot in presence['slots']:
        if slot['state'] == 'inactive':
            continue
        choices = slot['witness_ids'] if strategy == 'literal' and slot['witness_ids'] else slot['lexical_ids']
        basis = 'coherent_literal' if strategy == 'literal' and slot['witness_ids'] else 'scoped_lexical_only'
        if basis == 'scoped_lexical_only' and slot['slot_id'] in focus_queries:
            terms = _grams(focus_queries[slot['slot_id']]) - _EN_STOP
            choices = sorted(choices, key=lambda cid: (-len(terms & _grams(
                pool.rows[cid]['title'] + ' ' + pool.rows[cid]['text'])), _rank(pool.rows[cid]), cid))
        # A source-floor matrix does not create a floor for unscoped ordinary asks.
        original = next(s for s in contract.slots if s.slot_id == slot['slot_id'])
        if strategy == 'source' and not original.source_request_ids:
            continue
        existing = next((cid for cid in choices if cid in selected), None)
        chosen, reason = existing, None
        if chosen is None:
            for cid in choices:
                allocation, reason = pool.assign([*selected, cid])
                if allocation is not None:
                    chosen = cid
                    selected.append(cid)
                    assigned = allocation
                    break
        floor.append({'slot_id': slot['slot_id'], 'basis': basis,
            'status': 'reserved' if chosen else 'infeasible' if choices else 'no_eligible_candidate',
            'chunk_id': chosen, 'reason': None if chosen else reason})
    for cid in sorted(pool.rows, key=lambda c: (_rank(pool.rows[c]), c)):
        if cid in selected:
            continue
        allocation, _ = pool.assign([*selected, cid])
        if allocation is not None:
            selected.append(cid)
            assigned = allocation
    trace = pool.trace('source_slot_rank_fill_v1', selected, assigned, required)
    trace.update(slot_floor=floor, coverage_type='structural_literal_or_labelled_lexical_floor',
                 focus_query_sha256={key: sha256(q.encode()).hexdigest() for key, q in focus_queries.items()})
    return _result(pool, selected, assigned, trace)
