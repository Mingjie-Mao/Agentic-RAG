"""Query-derived aspect ranking of authorized whole chunks, experiment only.

Eight aspects match the existing eight-chunk ceiling and permit one reservation
per explicit topic. Larger decompositions fail explicitly instead of dropping
late requirements. Scores order candidates; they never certify semantic support.
"""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
import re
import time

from app.evidence_selection import _Pool, _caps, _rank, _required, _result
from app.retrieval import _DATE
from app.supplement_contract import assess_slot_presence
from app.task_contract import _UNIT, build_contract


@dataclass(frozen=True)
class FocusAspect:
    aspect_id: str
    slot_id: str
    query: str
    # None is the caller's authorization namespace; () is an empty source scope.
    document_versions: tuple[tuple[str, str], ...] | None
    source_request_ids: tuple[str, ...] = ()


def _content(query, requests):
    """Remove explicit publication scaffolding, leaving event/content wording."""
    text = query
    for request in requests:
        text = text.replace(request.original_span, '', 1)
    # Publication dates have a syntactic reporting prefix; an event date without
    # that prefix is content and remains verbatim (including its comma).
    expression = (r'\b(?:(?:articles?|reports?|reported|published|posted|dated)\s+'
                  r'(?:(?:published|posted|dated)\s+)?(?:on|from|before|after|since|prior to)\s+'
                  r'|and\s+then\s+on\s+)' + _DATE.pattern)
    text = re.sub(expression, '', text, flags=re.I)
    text = re.sub(r'^\s*(?:According\s+to|As\s+reported\s+by|Based\s+on)\s*[,;:]?\s*', '', text, flags=re.I)
    text = re.sub(r'\s+(?:according\s+to|as\s+reported\s+by|based\s+on)\s*[.?!]?\s*$', '', text, flags=re.I)
    text = re.sub(r'^\s*[,;:]?\s*(?:reported\s+)?that\s+', '', text, flags=re.I)
    text = re.sub(r'^\s*(?:articles?|reports?)\s+(?:about|on)\s+', '', text, flags=re.I)
    return text.strip(' ,;:.?')


def _topics(text):
    # Commas in dates are not topic separators. Coordinated dated reports have
    # already been separated by the source contract, never by this routine.
    separator = r',(?!\s*\d{4}\b)\s*|\s+and\s+'
    negation = re.search(r"\b(?:not|no|never|neither|without|avoid(?:s|ed|ing)?|cannot|\w+n['’]t)\b",
                         text, re.I)
    # A preceding negation can govern coordinated verbs or topics. Without a
    # syntax proof that it stops at a separator, retain the whole original clause
    # instead of producing a positive sibling. A final negated topic remains
    # intact and does not prevent the earlier explicit topics from being kept.
    if negation and re.search(separator, text[negation.end():], re.I):
        return [text]
    parts = [p.strip(' ,;:.?') for p in re.split(separator, text, flags=re.I)]
    parts = [re.sub(r'^and\s+', '', p, flags=re.I) for p in parts if p]
    if len(parts) < 2:
        return [text]
    # Carry only a written possessive entity, never an inferred person's name.
    named = re.search(r"\b([A-Z][\w’-]*(?:\s+[A-Z][\w’-]*)*)['’]s\b", text)
    # When no explicit possessive is written, retain the first short clause as
    # context. This copies the written subject without guessing a named entity.
    anchor = named[1] if named else ' '.join(parts[0].split()[:8])
    return [p if not anchor or re.search(r'\b'+re.escape(anchor)+r'\b', p)
            else f'{anchor} {p}' for p in parts]


def question_aspects(contract) -> list[FocusAspect]:
    """Ordered query-only topics with exact source/document/version bindings."""
    requests = {r.request_id: r for r in contract.source_requests}
    result = []
    for slot in contract.slots:
        source_ids = tuple(slot.source_request_ids)
        scope = None
        known = [requests[key] for key in source_ids if key in requests]
        if source_ids or requests:
            # Absent/unknown scopes never widen to the global namespace.
            scope = ()
            if (source_ids and len(known) == len(source_ids)
                    and all(r.resolution == 'supported' for r in known)):
                common = set(known[0].document_versions)
                for request in known[1:]:
                    common.intersection_update(request.document_versions)
                scope = tuple(pair for pair in known[0].document_versions if pair in common)
        original_query = slot.query
        # A sole publisher can be written after the content it governs. The
        # source parser's forward clause then starts at the publisher and misses
        # that prefix; using the original question retains it under the same
        # exact binding. Coordinated date groups and multiple publishers retain
        # their separate contract clauses.
        if len(requests) == 1 and len(contract.slots) == 1 and known:
            original_query = contract.question
        query = _content(original_query, known) if known else original_query.strip()
        # A shared requested output unit at the end of a Chinese comparison can
        # live only in the last parsed slot. Carry its original bytes to numeric
        # siblings instead of losing the question's requested unit.
        unit = re.search(rf'(?:多少|几)\s*({_UNIT})\s*[？?。.]?\s*$', contract.question, re.I)
        if (unit and slot.slot_spec and slot.slot_spec.value_type == 'number'
                and unit[1] not in query):
            query = f'{query} {unit[1]}'
        # Recognized task slots already encode domain subjects/units. Generic
        # ordinary slots keep their whole query if a safe list is not explicit.
        queries = ([query] if slot.slot_spec and slot.slot_spec.subject and slot.slot_spec.attribute
                   else _topics(query))
        for query in queries:
            if not query or len(query) > 500:
                raise ValueError('Invalid aspect query length')
            result.append(FocusAspect(f'aspect_{len(result)+1}', slot.slot_id, query, scope, source_ids))
    if len(result) > 8:
        raise ValueError('Too many question aspects (maximum 8)')
    return result


def _ranked(query, originals, ranker):
    supplied = deepcopy(originals)
    started = time.monotonic()
    output = ranker(query, supplied)
    elapsed = round((time.monotonic() - started) * 1000, 3)
    if supplied != originals or not isinstance(output, list) or len(output) != len(originals):
        raise ValueError('Invalid ranker output integrity')
    expected = {r['chunk_id']: r for r in originals}
    seen, scored, previous = set(), [], float('inf')
    annotations = {'rerank_score', 'fused_rank'}
    for row in output:
        if not isinstance(row, dict):
            raise ValueError('Invalid ranker row')
        cid, score = row.get('chunk_id'), row.get('rerank_score')
        if cid not in expected or cid in seen:
            raise ValueError('Invalid ranker identity/order')
        original = expected[cid]
        if ({k: v for k, v in row.items() if k not in annotations}
                != {k: v for k, v in original.items() if k not in annotations}):
            raise ValueError('Mutated ranker evidence')
        if type(score) not in (int, float) or not isfinite(score) or score > previous:
            raise ValueError('Invalid ranker score/order')
        previous = score
        seen.add(cid)
        scored.append((cid, score))
    # Model order within score ties is not used as a source of nondeterminism.
    scored.sort(key=lambda pair: (-pair[1], _rank(expected[pair[0]]), pair[0]))
    return scored, elapsed


def _reserve_condition_witness(contract, pool, presence, selected, assigned):
    """Deactivate a branch only when its deciding literal fits final context.

    The existing contract resolves one coherent operand value. Equivalent
    literal witnesses can live in several chunks; try each recorded witness
    under the existing allocator, and verify the selected rows still resolve
    the same inactive branches. Infeasibility keeps every aspect active.
    """
    proposed = {slot['slot_id'] for slot in presence['slots'] if slot['state'] == 'inactive'}
    if not proposed:
        return set(), selected, assigned, []
    operand_ids = {condition.operand_slot_id for condition in build_contract(contract.question).conditions}
    slots = {slot.slot_id: slot for slot in contract.slots}
    witnesses = {cid for state in presence['slots']
                 if slots[state['slot_id']].slot_spec
                 and slots[state['slot_id']].slot_spec.slot_id in operand_ids
                 for cid in state['witness_ids']}
    for cid in sorted(witnesses, key=lambda c: (_rank(pool.rows[c]), c)):
        prospective = selected if cid in selected else [*selected, cid]
        allocation, _ = pool.assign(prospective)
        if allocation is None:
            continue
        confirmed = {slot['slot_id'] for slot in assess_slot_presence(contract, allocation)['slots']
                     if slot['state'] == 'inactive'}
        if proposed <= confirmed:
            # Binding can omit a contradictory document version when all its
            # rows are present, yet expose a conflicting value in a selected
            # subset. Do not remove branch aspects if any eligible row can
            # change this decision. Contradictory rows remain in the rank/fill
            # pool; they are never suppressed to preserve an assumed branch.
            stable = True
            for other in pool.rows:
                if other in prospective:
                    continue
                combined = [*allocation, pool.rows[other]]
                later = {slot['slot_id'] for slot in assess_slot_presence(contract, combined)['slots']
                         if slot['state'] == 'inactive'}
                if not proposed <= later:
                    stable = False
                    break
            if stable:
                return proposed, prospective, allocation, [cid]
    return set(), selected, assigned, []


def select_aspect_evidence(contract, candidates, *, ranker, limit=8, token_budget=5000,
                           document_quota=None, source_quota=None, required_ids=()):
    """Reserve feasible whole chunks by aspect, then interleave per-query orders.

    The injected runner ranker should call app.rerank with the full scoped window.
    Scores from different queries are never compared. Call/pair/time telemetry is
    private and must be projected by the experiment runner before publication.
    """
    aspects = question_aspects(contract)
    caps = _caps(limit, token_budget, document_quota, source_quota)
    expected = {(d, v): sha for d, v, sha in contract.authorized_bindings}

    def compatible(aspect, row):
        pair = (row.get('document_id'), row.get('version_id'))
        return aspect.document_versions is None or pair in aspect.document_versions

    valid = []
    for row in candidates:
        pair = (row.get('document_id'), row.get('version_id'))
        if (row.get('authorized') is False or row.get('active') is False
                or (expected and (pair not in expected or
                    (expected[pair] and expected[pair] != row.get('source_sha256'))))):
            continue
        if any(compatible(a, row) for a in aspects):
            valid.append(row)
    pool = _Pool(contract.question, valid, caps)
    presence = assess_slot_presence(contract, list(pool.rows.values()))
    required, selected, assigned = _required(pool, required_ids)
    inactive, selected, assigned, condition_witnesses = _reserve_condition_witness(
        contract, pool, presence, selected, assigned)
    aspects = [aspect for aspect in aspects if aspect.slot_id not in inactive]
    orders, ranking, calls, pairs, wall = [], [], 0, 0, 0.0
    for aspect in aspects:
        rows = [pool.rows[cid] for cid in sorted(pool.rows, key=lambda c: (_rank(pool.rows[c]), c))
                if compatible(aspect, pool.rows[cid])]
        scores, elapsed = _ranked(aspect.query, rows, ranker) if rows else ([], 0.0)
        calls += bool(rows)
        pairs += len(rows)
        wall += elapsed
        orders.append([cid for cid, _ in scores])
        ranking.append({'aspect_id': aspect.aspect_id, 'slot_id': aspect.slot_id,
                        'query': aspect.query, 'document_versions': aspect.document_versions,
                        'scores': [{'chunk_id': cid, 'rerank_score': score} for cid, score in scores],
                        'pairs': len(rows), 'wall_ms': elapsed})
    floor = []
    for aspect, order in zip(aspects, orders, strict=True):
        chosen = None
        reason = None
        if chosen is None:
            for cid in order:
                if cid in selected:
                    chosen = cid
                    break
                allocation, reason = pool.assign([*selected, cid])
                if allocation is not None:
                    chosen = cid
                    selected.append(cid)
                    assigned = allocation
                    break
        floor.append({'aspect_id': aspect.aspect_id, 'slot_id': aspect.slot_id,
                      'status': 'reserved' if chosen else 'infeasible' if order else 'no_eligible_candidate',
                      'basis': 'scoped_relevance_only', 'chunk_id': chosen,
                      'reason': None if chosen else reason})
    # Round-robin retains query-local relevance order without treating distinct
    # query score scales as comparable. Infeasible rows remain whole and skipped.
    for position in range(max(map(len, orders), default=0)):
        for order in orders:
            if position >= len(order) or order[position] in selected:
                continue
            cid = order[position]
            allocation, _ = pool.assign([*selected, cid])
            if allocation is not None:
                selected.append(cid)
                assigned = allocation
    if inactive:
        final_inactive = {slot['slot_id'] for slot in assess_slot_presence(contract, assigned)['slots']
                          if slot['state'] == 'inactive'}
        if not inactive <= final_inactive:
            raise ValueError('Condition changed during evidence selection')
    trace = pool.trace('query_aspect_relevance_v1', selected, assigned, required)
    for rejected in trace['rejected']:
        if rejected['reasons'] == ['lower_lexical_objective']:
            rejected['reasons'] = ['lower_interleaved_relevance_order']
    trace.update(coverage_type='relevance_ranking_only', aspect_floor=floor, aspect_ranking=ranking,
                 ranker_calls=calls, ranker_pairs=pairs, ranker_wall_ms=round(wall, 3),
                 scope_rejected_count=len(candidates)-len(valid), inactive_slot_ids=sorted(inactive),
                 condition_witness_ids=condition_witnesses)
    return _result(pool, selected, assigned, trace)
