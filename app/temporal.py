"""Explicit source-effective intervals. Creation/publication dates are never defaults."""
import re
from datetime import date

from app.source_facts import dates
from app.source_fact_guards import date_role

_MONTH = (r'(?:January|February|March|April|May|June|July|August|September|October|November|December)')
_SHARED_YEAR = re.compile(
    rf'(?P<cue>\beffective\s+)(?P<start>{_MONTH}\s+\d{{1,2}})'
    rf'(?P<link>\s+(?:through|until)\s+)(?P<end>{_MONTH}\s+\d{{1,2}},?\s+(?P<year>\d{{4}}))\b',
    re.I,
)


def effective_interval(text):
    # The year is explicit at the end of one effective range. Carry it only to
    # that range's start, never from a publication date or another version.
    normalized, carried = _SHARED_YEAR.subn(
        lambda match: (match['cue'] + match['start'] + ', ' + match['year']
                       + match['link'] + match['end']), text)
    text = normalized
    starts = []
    # Expiry must have its own immediately preceding marker.
    ends = []
    offsets = {}
    for raw, day in dates(text):
        index = text.find(raw, offsets.get(raw, 0))
        offsets[raw] = index + len(raw)
        prefix = text[max(0, index - 25):index]
        if re.search(r'取代|替代|supersedes?|replaces?', prefix, re.I):
            continue
        if re.search(r'(?:through|until|expires? on|到|至|截至)\s*$', prefix, re.I):
            tail = text[index + len(raw):index + len(raw) + 30]
            inclusive = bool(re.search(r'through|到|至|截至', prefix, re.I) or re.search(
                r'inclusive|including that day|含当日', tail, re.I))
            ends.append((day, inclusive))
        elif date_role(raw, text, position=index) == 'effective':
            starts.append((raw, day))
    unique = {day for _, day in starts}
    if len(unique) != 1 or len(set(ends)) > 1:
        return {'status': 'unknown', 'reason': 'missing_or_ambiguous_effective_interval'}
    start = next(iter(unique))
    end, inclusive = ends[0] if ends else (None, False)
    if end and (end < start or (end == start and not inclusive)):
        return {'status': 'unknown', 'reason': 'invalid_effective_interval'}
    open_ended = bool(re.search(r'until (?:superseded|replaced)|持续有效|直至被替代', text, re.I))
    if not end and not open_ended:
        return {'status': 'unknown', 'reason': 'effective_end_or_supersession_required',
                'valid_from': start.isoformat(), 'valid_to': None,
                'provenance': 'version_source_text', 'publication_or_creation_time_used': False}
    return {'status': 'explicit', 'valid_from': start.isoformat(),
            'valid_to': end.isoformat() if end else None, 'end_inclusive': inclusive,
            'open_ended': open_ended,
            'provenance': ('shared_year_within_effective_range' if carried else 'version_source_text'),
            'publication_or_creation_time_used': False}


def select_effective_versions(question, versions):
    requested = dates(question)
    # A month-only request must have one version throughout that month. Testing
    # both boundaries avoids silently assuming that the first day is intended.
    import calendar
    if not requested:
        for match in re.finditer(r'(\d{4})\s*年\s*(\d{1,2})\s*月(?!\s*\d)', question):
            year, month = map(int, match.groups())
            if 1 <= month <= 12:
                requested += [(match.group(), date(year, month, 1)),
                              (match.group(), date(year, month, calendar.monthrange(year, month)[1]))]
    if not requested:
        return {'status': 'not_requested', 'selections': []}
    selections = []
    for raw, day in requested:
        eligible = []
        for version in versions:
            interval = version.get('effective_interval') or {}
            if interval.get('status') != 'explicit':
                continue
            start = date.fromisoformat(interval['valid_from'])
            end = date.fromisoformat(interval['valid_to']) if interval.get('valid_to') else None
            if start <= day and (end is None or day < end or
                                 (interval.get('end_inclusive') and day == end)):
                eligible.append(version['version_id'])
        if len(eligible) != 1:
            return {'status': 'unresolved', 'reason': 'missing_or_overlapping_effective_versions',
                    'selections': []}
        selections.append({'requested': raw, 'date': day.isoformat(), 'version_id': eligible[0]})
    return {'status': 'selected', 'selections': selections}


def complete_version_intervals(versions, texts):
    """Close intervals only from a successor's explicit source supersession.

    Upload order never establishes policy validity. Unlinked or overlapping dates
    remain unresolved. The active version can be open-ended within this authorized
    manifest; this does not assert what will be valid in the future.
    """
    result = [dict(v, effective_interval=dict(v.get('effective_interval') or {})) for v in versions]
    by_start = {}
    for v in result:
        start = v['effective_interval'].get('valid_from')
        if start:
            by_start.setdefault(start, []).append(v)
    for successor in result:
        interval = successor['effective_interval']
        start = interval.get('valid_from')
        if not start:
            continue
        for sentence in re.split(r'[。\n]', texts.get(successor['version_id'], '')):
            marker = re.search(r'取代|替代|supersedes?|replaces?', sentence, re.I)
            if not marker:
                continue
            predecessors = dates(sentence[marker.end():])
            if len(predecessors) != 1:
                continue
            old_start = predecessors[0][1].isoformat()
            candidates = by_start.get(old_start, [])
            if len(candidates) != 1 or old_start >= start:
                continue
            old = candidates[0]['effective_interval']
            if old.get('valid_to') and old['valid_to'] != start:
                continue
            old.update(status='explicit', valid_to=start, end_inclusive=False,
                       open_ended=False, provenance='explicit_source_supersession')
    for v in result:
        interval = v['effective_interval']
        if (v.get('is_active') and interval.get('valid_from') and interval.get('status') != 'explicit' and
                interval.get('reason') == 'effective_end_or_supersession_required'):
            interval.update(status='explicit', open_ended=True, end_inclusive=False,
                            provenance='source_effective_start_and_authorized_active_version')
    return result
