"""Narrow explicit date equality, preserving subjects, roles and original citations."""
import re

from app.answer_contract import question_contract
from app.source_facts import FactAnswer, conclusion, dates, target_kind, validate_facts, _CUES


def date_comparison(question, sources, evidence):
    contract = question_contract(question)
    kind = target_kind(question)
    if not contract['comparison_target_extracted'] or not kind:
        return None
    if (re.search(r'\b(?:three|four|all)\s+(?:articles?|reports?|sources?|publications?|notices?|documents?)\b'
                  r'|(?:三|四|所有)(?:份|篇)?(?:报道|文章|资料|来源|文档|文件)', question, re.I)
            or re.search(r'\b[A-Z][\w.-]*,\s*[A-Z][\w.-]*\s+and\s+[A-Z][\w.-]*', question)):
        # Two available sources cannot settle an explicitly larger comparison.
        return None
    extracted = []
    for source_id, source in sources.items():
        quote = source['quote']
        values = dates(quote)
        if len(values) != 1:
            continue
        date_start = quote.casefold().find(values[0][0].casefold())
        cues = [cue for cue in re.finditer(_CUES[kind], quote, re.I)
                if cue.end() <= date_start and re.fullmatch(
                    r'\s*(?:(?:on|at|from|is|was|于|在|定于|为|是)\s*)?[,：:]?\s*',
                    quote[cue.end():date_start], re.I)]
        # "release" in a subject is a noun, not the date-role verb in
        # "The release opens June 4". Require a locally connected date cue.
        if len(cues) != 1:
            continue
        cue = cues[0]
        prefix = quote[:cue.start()].strip()
        if re.search(r'\b(?:according to|said|argued|claimed|may|might|could)\b|声称|认为|可能', quote, re.I):
            continue
        # Only an explicit uncomplicated subject before the date-role verb.
        prefix = prefix.replace(values[0][0], '').strip(' ,，于在:：')
        if not prefix or len(prefix) > 80 or re.search(r'[.!?。！？;；:]|\b(?:but|and|reported)\b', prefix, re.I):
            continue
        extracted.append({'source_id': source_id, 'subject': prefix, 'speaker': '',
                          'modality': 'asserted', 'attribute': cue.group(),
                          'value': values[0][0], 'kind': kind, 'time_source_id': '',
                          'time_value': '', 'time_end': '', 'time_role': 'none'})
    facts, _ = validate_facts(FactAnswer(answerable=True, facts=extracted[:6]), sources, evidence)
    relation = conclusion(question, facts)
    if relation['value'] not in {'yes', 'no'}:
        return None
    selected = [fact for fact in facts if fact['id'] in relation['fact_ids']]
    return {'value': relation['value'], 'source_ids': [fact['source_id'] for fact in selected],
            'text': '；'.join(f"{fact['subject']}：{fact['value']}" for fact in selected)
                    + ('；所问日期一致。' if relation['value'] == 'yes' else '；所问日期不一致。'),
            'method': 'explicit_same_subject_date_equality_v1'}
