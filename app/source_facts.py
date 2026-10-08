"""Source-bound fact extraction and narrow, deterministic conclusions.

Copied fields prove provenance, not arbitrary entailment. Unknown attributes and
free-text relations stay unresolved. No knowledge, benchmark IDs or gold is used.
"""
from datetime import date
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.answer_contract import comparison_dimension
from app.source_fact_guards import BINDING_VERSION, date_role, inclusive_end, literal_field


class ExtractedFact(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_id: str
    subject: str = Field(max_length=140)
    speaker: str = Field(max_length=140)
    modality: Literal['asserted', 'reported', 'alleged', 'uncertain']
    attribute: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=160)
    kind: Literal['opening_date', 'effective_date', 'publication_date', 'event_date', 'number', 'text']
    time_source_id: str
    time_value: str = Field(max_length=80)
    time_end: str = Field(max_length=80)
    time_role: Literal['effective', 'event', 'publication', 'none']


class FactAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answerable: bool
    facts: list[ExtractedFact] = Field(max_length=6)


class PartialFact(ExtractedFact):
    """Only provenance and a literal attribute/value pair are mandatory."""
    subject: str = Field(default='', max_length=140)
    speaker: str = Field(default='', max_length=140)
    modality: Literal['asserted', 'reported', 'alleged', 'uncertain'] = 'asserted'
    time_source_id: str = ''
    time_value: str = Field(default='', max_length=80)
    time_end: str = Field(default='', max_length=80)
    time_role: Literal['effective', 'event', 'publication', 'none'] = 'none'


class PartialFactAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answerable: bool
    facts: list[PartialFact] = Field(max_length=6)


_CUES = {
    'opening_date': r'\b(?:open(?:ed|ing|s)?|launch(?:ed|es)?|release(?:d)?)\b|开业|开幕|上线',
    'effective_date': r'\b(?:effective|takes? effect|in force)\b|生效|实施',
    'publication_date': r'\b(?:published|publication|posted)\b|发布|刊登',
    'event_date': r'\b(?:happened|occurred|incident|event|held|happen|occur)\b|发生|事件|举行',
}
_DATE = re.compile(r'(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)|(?<!\d)(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日')
_MONTHS = {name.lower(): i for i, name in enumerate(
    ['January','February','March','April','May','June','July','August','September','October','November','December'], 1)}
_MONTHS.update({name[:3]: number for name, number in list(_MONTHS.items())})
_MONTH_DATE = re.compile(r'\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b|\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b')
_REPORTING = re.compile(r'\b(?:according to|argued|said|claimed|alleged|testified|may|might|could|believes?)\b|表示|认为|指控|据称|可能|声称|主张', re.I)


def norm(text):
    return re.sub(r'\s+', ' ', text.casefold()).strip(' .。:：,，')


def dates(text):
    found = []
    for match in _DATE.finditer(text):
        values = match.groups()[:3] if match.group(1) else match.groups()[3:]
        try:
            found.append((match.group(), date(*(int(value) for value in values))))
        except ValueError:
            pass
    for match in _MONTH_DATE.finditer(text):
        month, day, year = match.groups()[:3] if match.group(1) else (match.group(5), match.group(4), match.group(6))
        if month.casefold() in _MONTHS:
            try:
                found.append((match.group(), date(int(year), _MONTHS[month.casefold()], int(day))))
            except ValueError:
                pass
    return found


def _date_value(raw):
    values = dates(raw)
    return values[0][1] if len(values) == 1 and norm(values[0][0]) == norm(raw) else None


def target_kind(question):
    target = comparison_dimension(question)
    for kind, cue in _CUES.items():
        if re.search(cue, target, re.I):
            return kind
    return None


def _bound_value(attribute, value, quote):
    # Same clause, with no intervening second date/number: avoid binding a policy
    # start or publication date to an adjacent, different requested attribute.
    left = quote.casefold().find(attribute.casefold())
    right = quote.casefold().find(value.casefold())
    if min(left, right) < 0:
        return False
    between = quote[min(left, right):max(left, right)]
    gap_start = left + len(attribute) if left < right else right + len(value)
    gap = quote[gap_start:max(left, right)]
    return (len(between) <= 100 and not re.search(r'[;；。!?！？\n]|\.(?:\s|$)', between)
            and not any(norm(raw) != norm(value) for raw, _ in dates(between))
            and not re.search(r'(?<![A-Za-z0-9_])\d+(?:\.\d+)?%?', gap))


def validate_facts(wire, sources, evidence, *, partial=False):
    by_id = {row['id']: row for row in evidence}
    accepted, rejected = [], []
    for index, fact in enumerate(wire.facts, 1):
        issues = []
        source = sources.get(fact.source_id)
        if not source:
            rejected.append({'fact_index': index, 'issues': ['unknown_source']})
            continue
        quote = source['quote']
        row = by_id[source['id']]
        if any(field and not literal_field(field, quote)
               for field in [fact.subject, fact.speaker, fact.attribute, fact.value]):
            issues.append('field_not_in_source')
        if not literal_field(fact.value, quote, complete_number=fact.kind == 'number'):
            issues.append('incomplete_or_partial_numeric_value')
        if not _bound_value(fact.attribute, fact.value, quote):
            issues.append('attribute_value_not_locally_bound')
        if fact.kind in _CUES and not re.search(_CUES[fact.kind], fact.attribute, re.I):
            issues.append('wrong_attribute_kind')
        if fact.kind.endswith('_date') and _date_value(fact.value) is None:
            issues.append('invalid_or_incomplete_date')
        if fact.kind in _CUES and date_role(fact.value, quote) != fact.kind.removesuffix('_date'):
            issues.append('date_attribute_not_locally_bound')
        if _REPORTING.search(quote) and fact.modality == 'asserted':
            issues.append('reported_or_uncertain_span_marked_asserted')
        if fact.speaker and not re.search(
            rf'(?:according to\s+(?:the\s+)?{re.escape(fact.speaker)}|{re.escape(fact.speaker)}.{{0,45}}'
            r'(?:argued|said|claimed|alleged|testified|believes?|表示|认为|指控|声称|主张))', quote, re.I
        ):
            issues.append('speaker_not_bound_to_reporting_cue')
        # Full server span is rendered verbatim, so omitted speaker/modality cannot
        # silently turn a reported allegation into an asserted paraphrase.
        attribution = 'source_span_retained' if _REPORTING.search(quote) else 'no_reporting_cue_detected'
        support = [source]
        time_date = None
        end_date = None
        time_quote = None
        if fact.time_role != 'none':
            time_source = sources.get(fact.time_source_id)
            if not time_source or time_source['id'] != source['id']:
                issues.append('time_not_from_same_evidence')
            else:
                time_quote = time_source['quote']
                time_date = _date_value(fact.time_value)
                cue_kind = {'effective':'effective_date','publication':'publication_date','event':'event_date'}[fact.time_role]
                if (not fact.time_value or fact.time_value not in time_quote or time_date is None
                        or not re.search(_CUES[cue_kind], time_quote, re.I)):
                    issues.append('invalid_time_role_or_value')
                if date_role(fact.time_value, time_quote) != fact.time_role:
                    issues.append('effective_date_not_locally_bound' if fact.time_role == 'effective'
                                  else 'date_role_not_locally_bound')
                else:
                    support.append(time_source)
                    if fact.time_end:
                        end_date = _date_value(fact.time_end)
                        if (not end_date or fact.time_end not in time_quote
                                or end_date < time_date
                                or not inclusive_end(fact.time_end, time_quote)):
                            issues.append('invalid_effective_end')
        elif fact.time_value or fact.time_source_id or fact.time_end:
            issues.append('time_role_missing')
        text = f"{source['title']} 原文：{quote}"
        if len(text) > 1000:
            issues.append('source_span_too_long')
        warnings = []
        if partial:
            # Salvage only optional metadata. Literal value, attribute association,
            # date role and source identity remain mandatory. Render full source.
            if any(field and not literal_field(field, quote) for field in [fact.subject, fact.speaker]):
                fact = fact.model_copy(update={'subject': '', 'speaker': ''})
                warnings.append('optional_subject_or_speaker_unknown')
                if literal_field(fact.attribute, quote) and literal_field(fact.value, quote):
                    issues = [issue for issue in issues if issue != 'field_not_in_source']
            if 'speaker_not_bound_to_reporting_cue' in issues:
                fact = fact.model_copy(update={'speaker': ''})
                warnings.append('speaker_role_unknown_source_retained')
                issues.remove('speaker_not_bound_to_reporting_cue')
            if 'reported_or_uncertain_span_marked_asserted' in issues:
                fact = fact.model_copy(update={'modality': 'uncertain' if re.search(
                    r'\b(?:may|might|could)\b|可能', quote, re.I) else 'reported'})
                warnings.append('modality_preserved_from_source')
                issues.remove('reported_or_uncertain_span_marked_asserted')
            time_issues = [issue for issue in issues if issue in {
                'time_not_from_same_evidence', 'invalid_time_role_or_value',
                'effective_date_not_locally_bound', 'date_role_not_locally_bound',
                'invalid_effective_end', 'time_role_missing'}]
            if time_issues:
                warnings.extend(time_issues)
                issues = [issue for issue in issues if issue not in time_issues]
                fact = fact.model_copy(update={'time_source_id': '', 'time_value': '',
                                               'time_end': '', 'time_role': 'none'})
                time_date = end_date = time_quote = None
                support = [source]
        if issues:
            rejected.append({'fact_index': index, 'issues': issues})
            continue
        accepted.append({
            'id': f'F{len(accepted)+1}', 'source_id': fact.source_id,
            'document_id': row.get('document_id', source['document_id']),
            'version_id': row.get('version_id'), 'chunk_id': row['chunk_id'],
            'source': (row.get('metadata') or {}).get('source') or row['title'],
            'subject': fact.subject, 'speaker': fact.speaker, 'modality': fact.modality,
            'attribute': fact.attribute, 'value': fact.value, 'kind': fact.kind,
            'time': {'raw':fact.time_value, 'role':fact.time_role,
                     'iso':time_date.isoformat() if time_date else None,
                     'end_iso':end_date.isoformat() if end_date else None},
            'quote': quote, 'time_quote': time_quote, 'attribution_check': attribution,
            'provenance': 'copied_fields_and_verbatim_span; not_general_entailment',
            'binding_version': BINDING_VERSION,
            'field_warnings': warnings,
            'claim': {'text':text, 'evidence_ids':[part['id'] for part in support],
                      'quotes':[part['quote'] for part in support]},
        })
    return accepted, rejected


def conclusion(question, facts):
    """Only date comparisons and explicit effective-date lookups are computed."""
    unknown = {'value':'unclear', 'fact_ids':[], 'reason':'unsupported_relation',
               'method':'source_bound_facts_v1'}
    kind = target_kind(question)
    same = re.search(r'\b(?:consistent|same|agree|agreement|identical|align)\b|一致|相同', question, re.I)
    order = re.search(r'\b(?:earlier|later|before|after)\b|更早|更晚', question, re.I)
    if kind and same:
        chosen = [fact for fact in facts if fact['kind'] == kind and fact['modality'] == 'asserted'
                  and _date_value(fact['value'])]
        groups = {}
        for fact in chosen:
            groups.setdefault(fact['document_id'], []).append(fact)
        if len(groups) != 2:
            return {**unknown, 'reason':'requires_two_unambiguous_documents'}
        # No selecting whichever two rows happen to agree; disagreements within a
        # source or differing named subjects make the relation unresolved.
        if any(len({_date_value(fact['value']) for fact in rows}) != 1 for rows in groups.values()):
            return {**unknown, 'reason':'ambiguous_values_within_source'}
        chosen = [rows[0] for rows in groups.values()]
        if not chosen[0]['subject'] or norm(chosen[0]['subject']) != norm(chosen[1]['subject']):
            return {**unknown, 'reason':'subject_not_aligned'}
        equal = _date_value(chosen[0]['value']) == _date_value(chosen[1]['value'])
        # Negation elsewhere in a compound question is not negation of this
        # comparison. Negated comparisons stay unresolved until bound explicitly.
        if re.search(r'\bnot\b.{0,12}(?:consistent|same)|不一致|不相同', question, re.I):
            return {**unknown, 'reason':'negated_comparison_requires_scope_binding'}
        return {'value':'yes' if equal else 'no', 'fact_ids':[row['id'] for row in chosen],
                'reason':'same_subject_and_requested_date_attribute', 'method':'source_bound_facts_v1',
                'comparison_dimension':comparison_dimension(question)}
    if kind and order:
        # Named source order and relation direction are not inferred from dict order.
        return {**unknown, 'reason':'source_order_requires_explicit_binding'}
    requested = dates(question)
    if requested and re.search(r'\bon\b|\bas of\b|在|截至|适用', question, re.I):
        selections = []
        for raw, day in requested:
            eligible = [fact for fact in facts if fact['time']['role'] == 'effective'
                        and fact['time']['iso'] and date.fromisoformat(fact['time']['iso']) <= day]
            subjects = {norm(fact['subject']) for fact in eligible}
            attributes = {norm(fact['attribute']) for fact in eligible}
            if not eligible or len(subjects) != 1 or len(attributes) != 1 or '' in subjects:
                return {**unknown, 'reason':'missing_or_ambiguous_effective_policy'}
            latest = max(fact['time']['iso'] for fact in eligible)
            rows = [fact for fact in eligible if fact['time']['iso'] == latest]
            if len({norm(fact['value']) for fact in rows}) != 1:
                return {**unknown, 'reason':'conflicting_effective_values'}
            selections.append({'requested':raw, 'iso':day.isoformat(), 'fact_id':rows[0]['id'],
                               'value':rows[0]['value']})
        by_id = {fact['id']: fact for fact in facts}
        for selected in selections:
            end = by_id[selected['fact_id']]['time']['end_iso']
            if not end or selected['iso'] > end:
                return {**unknown, 'reason':'effective_end_or_supersession_required'}
        return {'value':'not_applicable', 'fact_ids':[row['fact_id'] for row in selections],
                'selections':selections, 'reason':'explicit_effective_interval',
                'method':'source_bound_facts_v1'}
    return unknown


def public_facts(facts, claims):
    """Expose only fields bound to a quote in the final validated claims."""
    result, seen = [], set()
    for fact in facts:
        if not any(fact['chunk_id'] in claim['evidence_ids'] and fact['quote'] in claim['quotes']
                   for claim in claims):
            continue
        key = (fact['chunk_id'], fact['quote'], fact['attribute'], fact['value'], str(fact['time']))
        if key in seen:
            continue
        seen.add(key)
        result.append({**{key:value for key,value in fact.items() if key != 'claim'},
                       'id':f'F{len(result)+1}'})
    return result


def fact_verdict(question, facts, claims):
    relation = conclusion(question, facts)
    by_id = {fact['id']:fact for fact in facts}
    chunks = {by_id[key]['chunk_id'] for key in relation['fact_ids'] if key in by_id}
    indices = [index for index, claim in enumerate(claims, 1) if chunks & set(claim['evidence_ids'])]
    value = relation['value'] if relation['value'] in {'yes','no'} and indices else 'unclear'
    return {**relation, 'value':value, 'claim_index':indices[0] if value != 'unclear' else None,
            'claim_indices':indices if value != 'unclear' else [],
            'evidence_ids':sorted(chunks) if value != 'unclear' else [],
            'scope':'原文绑定字段的受限程序关系计算；未知关系保留 unclear，不调用额外判断模型'}
