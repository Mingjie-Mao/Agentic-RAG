"""Conservative source-negation and claim/verdict checks; no semantic model call."""
import re


_WORD = re.compile(r"[A-Za-z]+")
_STOP = set('a an the her his its their of for to from in on at by with was were is are be been has have had did does do'.split())
_NEGATION = re.compile(r"\b(?:not|never|no|without|cannot|can't|didn't|doesn't|don't)\b", re.I)
_CN_NEGATION = re.compile(r"(?:并非|没有|未|不)([\u4e00-\u9fff]{2,10})")


def _normalized_words(text):
    result = []
    for token in _WORD.findall(text.casefold()):
        if token in _STOP:
            continue
        # A small, auditable inflection normalization; do not infer synonyms.
        if token.endswith('ied') and len(token) > 5:
            token = token[:-3] + 'y'
        elif token.endswith('ing') and len(token) > 5:
            token = token[:-3]
        elif token.endswith('ed') and len(token) > 4:
            token = token[:-2]
        elif token.endswith('s') and len(token) > 4:
            token = token[:-1]
        result.append(token)
    return result


def _contains_sequence(haystack, needle):
    return any(haystack[index:index + len(needle)] == needle
               for index in range(len(haystack) - len(needle) + 1))


def source_polarity_issue(claim, quotes):
    """Flag only an explicit negative predicate restated positively in a claim."""
    if _NEGATION.search(claim) or re.search(r'并非|没有|未|不', claim):
        return False
    claim_words = _normalized_words(claim)
    for quote in quotes:
        for sentence in re.split(r'[.;；。!?！？\n]', quote):
            for match in _NEGATION.finditer(sentence):
                after = _normalized_words(sentence[match.end():])
                if len(after) >= 2 and len(claim_words) >= 2:
                    prefix = _normalized_words(sentence[:match.start()])
                    subject = prefix[-2:] if len(prefix) >= 2 and prefix[-1] in {
                        'policy', 'report', 'document', 'rule', 'company', 'team', 'source'
                    } else prefix[-1:]
                    if (subject and _contains_sequence(claim_words, subject)
                            and _contains_sequence(claim_words, after[:min(3, len(after))])):
                        return True
            for match in _CN_NEGATION.finditer(sentence):
                phrase = match.group(1)
                before = re.findall(r'[\u4e00-\u9fff]', sentence[:match.start()])
                if before and ''.join(before[-3:]) in claim and phrase[:min(4, len(phrase))] in claim:
                    return True
    return False


def explicit_consistency(question, claim):
    """Read a single explicit consistency relation, never a new source fact."""
    from app.answer_contract import comparison_dimension

    if not re.search(r'\b(?:consistent|consistency|agree|agreement)\b|一致|相同', question, re.I):
        return None
    target = comparison_dimension(question)
    extracted = target != question.strip().rstrip('?？')[:600]
    relations = []
    for clause in re.split(r'[.!?。！？;；\n]|\b(?:while|whereas|but)\b|然而|但是|但', claim, flags=re.I):
        if extracted and target.casefold() not in clause.casefold():
            continue
        if re.search(r'\b(?:not|never)\s+inconsistent\b|并非不一致|不是不一致|\bneither\b',
                     clause, re.I):
            return None
        negative = bool(re.search(
            r'\b(?:not|never)(?:\s+\w+){0,2}\s+consistent\b|\binconsistent\b'
            r'|\bnot(?:\s+\w+){0,2}\s+in agreement\b|\bdisagree\b|不一致|并不一致|未完全一致',
            clause, re.I))
        positive = bool(re.search(r'\bconsistent\b|\bin agreement\b|\bagree\b|一致|相同', clause, re.I))
        if negative:
            relations.append('no')
        elif positive and not re.search(r'\b(?:not|never|no|neither|nor)\b|未|不', clause, re.I):
            relations.append('yes')
    if len(relations) != 1:
        return None
    return relations[0]


def reconcile_verdict(question, claims, verdict):
    """Correct a classifier only when its cited claim says the opposite relation."""
    if not verdict or verdict.get('value') not in {'yes', 'no'}:
        return verdict
    index = verdict.get('claim_index')
    if not isinstance(index, int) or index < 1 or index > len(claims):
        return verdict
    relation = explicit_consistency(question, claims[index - 1]['text'])
    if relation is None or relation == verdict['value']:
        return verdict
    return {**verdict, 'value': relation, 'method': 'explicit_cited_claim_relation_v1',
            'relation_consistency_corrected': True}
