"""Pure, query-only supplement proposals over caller-authorized documents.

Raw spans and bindings stay local. ``public_summary`` is the only telemetry
projection. A nested syntax proposal is never proof of an entity dependency.
This module performs no retrieval, model calls, execution, or persistence.
"""

from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import re
from typing import Literal

from app.retrieval import _DATE, _dates_in, source_mentions
from app.task_contract import SlotSpec, bind_values, build_contract, evaluate_condition
from app.verdict import question_slots


Mode = Literal["independent", "entity_bridge", "unknown"]
Resolution = Literal["supported", "missing", "unknown"]
Reason = Literal[
    "source_resolved", "no_matching_publication_date", "missing_publication_metadata",
    "unsupported_publication_date", "ambiguous_date_attachment", "missing_document_identity",
    "explicit_source_requests", "explicit_independent_asks", "explicit_comparison", "recognized_subject_attribute",
    "nested_relationship_proposal", "ambiguous_anaphora", "unsupported_history",
    "unsupported_condition", "unresolved_query_structure",
]
_REASONS = set(Reason.__args__)
_MODES = set(Mode.__args__)
_RESOLUTIONS = set(Resolution.__args__)


def _fingerprint(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _public_enum(value, allowed, fallback):
    return value if value in allowed else fallback


def _public_id(value):
    return value if re.fullmatch(r"(?:source|occurrence|slot)_\d+", value) else "unknown"


@dataclass(frozen=True)
class SourceRequest:
    request_id: str
    occurrence_id: str
    start: int
    end: int
    original_span: str
    mention: str
    clause_start: int
    clause_end: int
    query_clause: str
    # The paired bindings preserve source -> document -> active version identity.
    document_versions: tuple[tuple[str, str], ...]
    publication_constraints: tuple[tuple[str, str], ...]
    resolution: Resolution
    reason: Reason

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(document_id for document_id, _ in self.document_versions)

    @property
    def version_ids(self) -> tuple[str, ...]:
        return tuple(version_id for _, version_id in self.document_versions)


@dataclass(frozen=True)
class SupplementSlot:
    slot_id: str
    query: str
    source_request_ids: tuple[str, ...] = ()
    slot_spec: SlotSpec | None = None


@dataclass(frozen=True)
class SupplementContract:
    question: str
    mode: Mode
    reason: Reason
    source_requests: tuple[SourceRequest, ...]
    slots: tuple[SupplementSlot, ...]
    authorized_bindings: tuple[tuple[str, str, str], ...] = ()

    def public_summary(self) -> dict:
        """Allowlist projection; neither raw corpus IDs nor model strings escape."""
        return {
            "mode": _public_enum(self.mode, _MODES, "unknown"),
            "reason": _public_enum(self.reason, _REASONS, "unresolved_query_structure"),
            "question_fingerprint": _fingerprint(self.question),
            "source_request_count": len(self.source_requests),
            "slot_count": len(self.slots),
            "source_requests": [{
                "request_id": _public_id(request.request_id),
                "occurrence_id": _public_id(request.occurrence_id),
                "resolution": _public_enum(request.resolution, _RESOLUTIONS, "unknown"),
                "reason": _public_enum(request.reason, _REASONS, "unresolved_query_structure"),
                "clause_fingerprint": _fingerprint(request.query_clause),
                "document_count": len(request.document_versions),
                "version_count": len(request.version_ids),
                "constraint_count": len(request.publication_constraints),
            } for request in self.source_requests],
            "slots": [{
                "slot_id": _public_id(slot.slot_id),
                "source_request_ids": [_public_id(key) for key in slot.source_request_ids],
                "query_fingerprint": _fingerprint(slot.query),
            } for slot in self.slots],
        }


_PUBLICATION_PREFIX = re.compile(
    r"\b(?:published|posted|reported|dated|articles?|reports?|sources?|stories|pieces?|coverage)"
    r"\s*(?:(on|from|dated|before|after|prior to|since)\s+)?$"
    r"|(?:发表于|发布于|刊登于)\s*$", re.I,
)
_PUBLICATION_EXPRESSION = re.compile(
    r"\b(?:published|posted|reported|dated|articles?|reports?|sources?)"
    r"\s+(?:on|from|dated|before|after|prior to|since|as of|at|during)\s+"
    r"|\b(?:published|posted|dated)\s+(?=\d|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"|(?:发表于|发布于|刊登于)", re.I,
)
_COORDINATED_REPORT = re.compile(r",?\s+and\s+then\s+on\s+", re.I)


def _coordinated_report_clauses(question, occurrence, stop):
    """Inherit one publisher only for an explicit dated reporting continuation."""
    text = question[occurrence["end"]:stop]
    lead = re.match(r"\s+reports?\s+on\s+|\s+reported\s+on\s+", text, re.I)
    joins = list(_COORDINATED_REPORT.finditer(text))
    if not lead or not joins:
        return None
    first_date = _DATE.match(text, lead.end())
    if not first_date:
        return (), "unsupported_publication_date", ()
    sources = {str((d.metadata_json or {}).get("source") or "").strip().casefold()
               for d in occurrence["documents"]}
    if len(sources) != 1:
        return (), "ambiguous_date_attachment", ()
    boundaries = [occurrence["start"]]
    complement_starts = [first_date.end()]
    for join in joins:
        day = _DATE.match(text, join.end())
        if not day:
            return (), "unsupported_publication_date", ()
        report = re.match(r"\s*,?\s+reported\b", text[day.end():], re.I)
        if not report:
            return (), "ambiguous_date_attachment", ()
        boundaries.append(occurrence["end"] + join.start())
        complement_starts.append(day.end() + report.end())
    boundaries.append(stop)
    # Only recognize "that" before an explicitly named subject. A demonstrative
    # such as "that company" still reaches the anaphora guard.
    complements = []
    for start in complement_starts:
        complement = re.match(r"\s*,?\s+(that)\b", text[start:], re.I)
        if complement and re.match(r"\s+[A-Z][\w'’-]*\b", text[start + complement.end():]):
            complements.append((occurrence["end"] + start + complement.start(1),
                                occurrence["end"] + start + complement.end(1)))
    return tuple(zip(boundaries, boundaries[1:])), None, tuple(complements)


def _constraints(question, occurrence, stop, *, inherited_start=None):
    """Return explicit date requirements; unsupported attachment is never widened."""
    text = question[occurrence["end"] if inherited_start is None else inherited_start:stop]
    found = []
    all_dates = list(_DATE.finditer(text))
    for match in all_dates:
        prefix = text[:match.start()]
        marker = _PUBLICATION_PREFIX.search(prefix)
        if inherited_start is not None and re.fullmatch(r"\s*,?\s*and\s+then\s+on\s+", prefix, re.I):
            marker = re.search(r"(on)\s+$", prefix, re.I)
        # "Reported by Publisher before DATE" has the publication verb before
        # the mention; require it immediately rather than treating any event bound
        # in the rest of the clause as a publication bound.
        if not marker and re.fullmatch(r"\s+(before|after|prior to|since)\s+", prefix, re.I):
            lead = question[max(0, occurrence["start"] - 80):occurrence["start"]]
            if re.search(r"\b(?:reported|published)\s+by\s+$", lead, re.I):
                marker = re.search(r"(before|after|prior to|since)\s+$", prefix, re.I)
        if not marker:
            continue
        day = next(iter(_dates_in(match.group())[0]))
        try:
            date.fromisoformat(day)
        except ValueError:
            return (), "unsupported_publication_date"
        op = (marker.group(1) or "on").lower()
        found.append((("lt" if op in {"before", "prior to"} else
                       "gt" if op == "after" else "ge" if op == "since" else "eq", day),))
    if found:
        if re.search(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:AM|PM)\b', text, re.I):
            return (), 'unsupported_publication_date'
        if len(found) != len(all_dates):
            # Mixing an unbound date into a publication expression cannot silently
            # discard an alternative, range endpoint, or possible event attachment.
            return (), "ambiguous_date_attachment"
        return tuple(found), None
    if _PUBLICATION_EXPRESSION.search(text):
        return (), "unsupported_publication_date"
    # A publication date before its publisher has no supported attachment grammar.
    lead = question[:occurrence["start"]]
    preceding_dates = list(_DATE.finditer(lead))
    if preceding_dates:
        last = preceding_dates[-1]
        if (_PUBLICATION_PREFIX.search(lead[:last.start()])
                and re.fullmatch(r"\s*(?:by|from)?\s*", lead[last.end():], re.I)):
            return (), "ambiguous_date_attachment"
    return ((),), None


def _resolve(members, constraints, error):
    if error:
        return (), "unknown", error
    selected = list(members)
    if constraints:
        published = []
        for document in selected:
            raw = str((document.metadata_json or {}).get("published_at") or "")
            try:
                day = datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
            except ValueError:
                return (), "unknown", "missing_publication_metadata"
            published.append((document, day))
        op, requested = constraints[0]
        selected = [document for document, day in published if
                    (day == requested if op == "eq" else day < requested if op == "lt"
                     else day > requested if op == "gt" else day >= requested)]
        if not selected:
            return (), "missing", "no_matching_publication_date"
    if any(not isinstance(getattr(d, 'id', None), str) or not d.id
           or not isinstance(getattr(d, 'active_version_id', None), str) or not d.active_version_id for d in selected):
        return (), "unknown", "missing_document_identity"
    bindings = tuple((d.id, d.active_version_id) for d in selected)
    return bindings, "supported", "source_resolved"


def _query_route(question, requests, publication_complements=()):
    # Mask publisher names: arbitrary metadata strings are data, never instructions
    # or grammatical evidence for anaphora, history or relationship classification.
    text = question
    for request in reversed(requests):
        text = text[:request.start] + " " * (request.end - request.start) + text[request.end:]
    for start, end in publication_complements:
        text = text[:start] + " " * (end - start) + text[end:]
    parsed = build_contract(text)
    if re.search(r"\b(?:it|its|he|his|she|her|they|their|them|this|that)\b|[他她它](?:的|们)?", text, re.I):
        return "unknown", "ambiguous_anaphora"
    if parsed.intent == "history" or re.search(r"\b(?:previously|historical|history|earliest)\b|历史|旧版", text, re.I):
        return "unknown", "unsupported_history"
    if parsed.intent == "conditional":
        return "unknown", "unsupported_condition"
    if len(requests) > 1:
        return "independent", "explicit_source_requests"
    if parsed.intent == "comparison" or re.search(r"\b(?:compare|difference)\b|比较|对比|相差|分别", text, re.I):
        return "independent", "explicit_comparison"
    if len(question_slots(question)) > 1:
        return "independent", "explicit_independent_asks"
    if re.search(r"[^的和与及，。；;?？]+的[^的和与及，。；;?？]+的", text) or (
        len(re.findall(r"['’]s\b|\bof\b", text, re.I)) >= 2
        and not re.search(r"\b(?:and|or)\b|[,;]", text, re.I)
    ):
        return "entity_bridge", "nested_relationship_proposal"
    if requests:
        return "independent", "explicit_source_requests"
    if parsed.slots and all(slot.subject and slot.attribute for slot in parsed.slots):
        return "independent", "recognized_subject_attribute"
    return "unknown", "unresolved_query_structure"


def build_supplement_contract(question: str, authorized_documents) -> SupplementContract:
    """Propose source scopes and query structure without evidence/answer access."""
    authorized_documents = list(authorized_documents)
    occurrences = source_mentions(question, authorized_documents)
    requests = []
    publication_complements = ()
    ambiguous_occurrences = set()
    for index in range(len(occurrences) - 1):
        current, following = occurrences[index:index + 2]
        gap = question[current["end"]:following["start"]]
        next_stop = occurrences[index + 2]["start"] if index + 2 < len(occurrences) else len(question)
        following_requirements, following_error = _constraints(question, following, next_stop)
        if (re.fullmatch(r"\s*(?:and|or|和|与|及|,)\s*", gap, re.I)
                and (any(following_requirements) or following_error)):
            ambiguous_occurrences.update((index, index + 1))
    for index, occurrence in enumerate(occurrences):
        stop = occurrences[index + 1]["start"] if index + 1 < len(occurrences) else len(question)
        coordinated = _coordinated_report_clauses(question, occurrence, stop) if len(occurrences) == 1 else None
        clauses, attachment_error = ((occurrence["start"], stop),), None
        if coordinated:
            clauses, attachment_error, publication_complements = coordinated
            clauses = clauses or ((occurrence["start"], stop),)
        for clause_index, (clause_start, clause_end) in enumerate(clauses):
            requirements, error = _constraints(question, occurrence, clause_end,
                                              inherited_start=clause_start if clause_index else None)
            if attachment_error or index in ambiguous_occurrences:
                error = attachment_error or "ambiguous_date_attachment"
            for constraints in requirements or ((),):
                bindings, resolution, reason = _resolve(occurrence["documents"], constraints, error)
                requests.append(SourceRequest(
                    request_id=f"source_{len(requests) + 1}", occurrence_id=f"occurrence_{index + 1}",
                    start=occurrence["start"], end=occurrence["end"],
                    original_span=question[occurrence["start"]:occurrence["end"]], mention=occurrence["mention"],
                    clause_start=clause_start, clause_end=clause_end,
                    query_clause=question[clause_start:clause_end], document_versions=bindings,
                    publication_constraints=constraints, resolution=resolution, reason=reason,
                ))
    parsed = build_contract(question)
    mode, reason = _query_route(question, requests, publication_complements)
    if requests:
        source_slots = []
        for i, request in enumerate(requests, 1):
            specs = build_contract(request.query_clause).slots
            spec = specs[0] if len(specs) == 1 else None
            source_slots.append(SupplementSlot(f"slot_{i}", request.query_clause, (request.request_id,), spec))
        slots = tuple(source_slots)
    else:
        slots = tuple(SupplementSlot(f"slot_{i}", spec.query, slot_spec=spec)
                      for i, spec in enumerate(parsed.slots, 1))
        # The existing contract parser is intentionally domain-bounded. Preserve
        # an explicit generic English comparison pair without inventing SlotSpec
        # subjects/attributes that the parser did not resolve.
        pair = re.fullmatch(r"\s*Compare\s+(.+?)\s+and\s+(.+?)['’]s\s+(.+?)[.?]?\s*", question, re.I)
        if pair and len(slots) == 1:
            slots = tuple(SupplementSlot(f"slot_{i}", f"{subject}'s {pair[3]}")
                          for i, subject in enumerate((pair[1], pair[2]), 1))
    bindings = tuple((d.id, d.active_version_id, getattr(d, 'source_sha256', ''))
                     for d in authorized_documents if isinstance(getattr(d, 'id', None), str) and d.id
                     and isinstance(getattr(d, 'active_version_id', None), str) and d.active_version_id)
    return SupplementContract(question, mode, reason, tuple(requests), slots, bindings)


def assess_slot_presence(contract: SupplementContract, rows: list[dict]) -> dict:
    """Bounded structural/literal presence, never semantic truth or answer accuracy.

    The caller reauthorizes whole rows. Frozen bindings, when supplied, additionally
    detect identity drift. Missing means missing from this inspected pool only.
    Public output hashes witness IDs; private IDs support cap-respecting selection.
    """
    from agent.planner import _grams
    expected = {(d, v): s for d, v, s in contract.authorized_bindings}
    valid, invalid = [], 0
    for row in rows:
        keys = ('chunk_id', 'document_id', 'version_id', 'source_sha256', 'text')
        pair = (row.get('document_id'), row.get('version_id'))
        if (any(not isinstance(row.get(k), str) or not row[k] for k in keys)
                or row.get('authorized') is False or row.get('active') is False
                or (expected and (pair not in expected or (expected[pair] and expected[pair] != row['source_sha256'])))):
            invalid += 1
        else:
            valid.append(dict(row, id=row.get('id') or row['chunk_id']))
    requests = {r.request_id: r for r in contract.source_requests}
    parsed = build_contract(contract.question)
    condition = evaluate_condition(parsed, [dict(r, title='') for r in valid]) if parsed.conditions else None
    results = []
    for slot in contract.slots:
        source_specs = [requests[key] for key in slot.source_request_ids]
        scoped = valid
        source_state = 'supported'
        date_state = 'supported'
        for request in source_specs:
            if request.resolution != 'supported':
                source_state = request.resolution
                if request.publication_constraints or request.reason in {
                        'unsupported_publication_date', 'ambiguous_date_attachment', 'missing_publication_metadata'}:
                    date_state = request.resolution
                scoped = []
            else:
                allowed = set(request.document_versions)
                scoped = [r for r in scoped if (r['document_id'], r['version_id']) in allowed]
                if not scoped:
                    source_state = 'missing'
                    if request.publication_constraints:
                        date_state = 'missing'
        spec = slot.slot_spec
        if spec and not spec.subject and spec.attribute:
            # A simple named Latin entity followed by an explicit possessive is
            # structurally resolvable; general prose/relationship clauses are not.
            named = re.fullmatch(r"([A-Za-z][A-Za-z0-9 _.-]{1,63}?)\s*的[^的]+", spec.query)
            if named and not re.search(r'\b(?:and|or|if|what|which|how|why)\b', named[1], re.I):
                spec = spec.model_copy(update={'subject': named[1].strip()})
        subject = spec.subject if spec else ''
        attribute = spec.attribute if spec else ''
        subject_rows = [r for r in scoped if subject and subject.casefold() in r['text'].casefold()]
        attribute_rows = [r for r in subject_rows if attribute and attribute.casefold() in r['text'].casefold()]
        # Lexical IDs guide ranking only. No overlap threshold can certify support.
        terms = _grams(slot.query)
        lexical = sorted(scoped, key=lambda r: (-len(terms & _grams(r.get('title', '') + r['text'])),
                                                r.get('parent_rank', 10**9), r['chunk_id']))
        values = []
        if spec and subject and attribute and spec.value_type != 'text':
            # A title mentioning A and B cannot lend A's subject to B's predicate.
            # Complete table headers/rows and explicit section headings still bind.
            values = bind_values(spec, [dict(r, title='') for r in scoped])
        distinct = {(v.value, v.unit) for v in values}
        witnesses = list(dict.fromkeys(v.chunk_id for v in values)) if len(distinct) == 1 else []
        inactive = bool(condition and condition.result != 'unknown' and spec and
                        spec.slot_id not in condition.active_slot_ids)
        ambiguous = bool(attribute_rows and not witnesses and any(
            re.search(r'不是|并非|不为|not|\d', r['text'], re.I) for r in attribute_rows))
        if inactive:
            state, reason = 'inactive', 'condition_inactive'
        elif invalid:
            state, reason = 'unknown', 'invalid_binding_in_pool'
        elif condition and condition.result == 'unknown':
            state, reason = 'unknown', 'condition_unknown'
        elif contract.reason == 'ambiguous_anaphora':
            state, reason = 'unknown', 'ambiguous_query_binding'
        elif contract.mode == 'entity_bridge':
            state, reason = 'unknown', 'entity_dependency_unresolved'
        elif contract.reason == 'unsupported_history':
            state, reason = 'unknown', 'history_requires_version_chain'
        elif source_state != 'supported':
            state, reason = source_state, 'source_scope_unresolved' if source_state == 'unknown' else 'source_absent_in_pool'
        elif not spec or not subject or not attribute or spec.value_type == 'text':
            state, reason = 'unknown', 'unsupported_subject_attribute'
        elif witnesses:
            state, reason = 'supported', 'coherent_literal_witness'
        elif len(distinct) > 1 or ambiguous:
            state, reason = 'unknown', 'ambiguous_literal_binding'
        else:
            state, reason = 'missing', 'literal_absent_in_pool'
        results.append({'slot_id': slot.slot_id, 'state': state, 'reason': reason,
            'source': source_state, 'date': date_state,
            'subject': 'supported' if witnesses or subject_rows else 'missing' if subject else 'unknown',
            'attribute': 'supported' if witnesses or attribute_rows else 'missing' if attribute else 'unknown',
            'witness_ids': witnesses if state == 'supported' else [],
            'scoped_ids': list(dict.fromkeys(r['chunk_id'] for r in scoped)),
            'lexical_ids': list(dict.fromkeys(r['chunk_id'] for r in lexical))})
    return {'slots': results, 'invalid_row_count': invalid,
            'gate': any(r['state'] in {'missing', 'unknown'} for r in results),
            'basis': 'source_bound_structural_literal_presence_not_entailment'}


def public_presence(presence):
    return {'gate': presence['gate'], 'invalid_row_count': presence['invalid_row_count'], 'basis': presence['basis'],
        'slots': [{k: r[k] for k in ('slot_id', 'state', 'reason', 'source', 'date', 'subject', 'attribute')} | {
            'witness_sha256': [_fingerprint(cid) for cid in r['witness_ids']],
            'scoped_count': len(r['scoped_ids']), 'lexical_count': len(r['lexical_ids'])}
            for r in presence['slots']]}


def diagnose_slot_loss(candidate_presence, selected_presence, *, feasible_slots=()):
    """Avoidability needs an actual feasible alternative, not a lexical assumption."""
    if [s['slot_id'] for s in candidate_presence['slots']] != [s['slot_id'] for s in selected_presence['slots']]:
        raise ValueError('Different slot contracts')
    result = []
    for old, selected in zip(candidate_presence['slots'], selected_presence['slots'], strict=True):
        if selected['state'] in {'supported', 'inactive'}:
            status = 'retained'
        elif old['state'] == 'supported':
            status = 'selection_loss'
        elif old['state'] == 'missing':
            status = 'candidate_absent'
        else:
            status = 'unresolved'
        result.append({'slot_id': old['slot_id'], 'status': status,
            'candidate_state': old['state'], 'selected_state': selected['state'],
            'avoidable': True if status == 'selection_loss' and old['slot_id'] in feasible_slots else None,
            'basis': 'inspected_pool_only'})
    return result
