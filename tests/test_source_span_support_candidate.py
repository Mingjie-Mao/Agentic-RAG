"""Limits of the existing source renderer, before the single Dev component trial."""
import pytest

from app.answer_quality import bind_source_text
from app.clients import Claim, GeneratedAnswer
from app.qa import validate_claims


def evidence(text, eid='E1'):
    return {'id': eid, 'chunk_id': 'chunk-' + eid, 'title': 'Synthetic source', 'text': text}


def answer(text, quotes, ids=None):
    return GeneratedAnswer(answerable=True, claims=[
        Claim(text=text, evidence_ids=ids or ['E1'] * len(quotes), quotes=quotes)])


def test_renderer_exposes_actual_magnitude_instead_of_false_paraphrase():
    quote = 'The service paid $4.2 billion to its partners.'
    draft = answer('The service paid millions instead of billions.', [quote])
    before = draft.model_dump()
    rendered, trace = bind_source_text(draft, [evidence(quote)])
    assert rendered.claims[0].text == quote
    assert 'millions instead' not in rendered.claims[0].text
    assert validate_claims(rendered, [evidence(quote)])[1] == 'answered'
    assert trace['guarantee'].startswith('verbatim_provenance_only')
    assert draft.model_dump() == before


def test_rendering_does_not_turn_conditional_source_into_opposite_branch():
    quote = 'If the error rate exceeds 7%, the service must stop.'
    draft = answer('The service must not stop because its error rate is 5%.', [quote])
    rendered, trace = bind_source_text(draft, [evidence(quote)])
    assert rendered.claims[0].text.startswith('If ')
    assert '5%' not in rendered.claims[0].text
    # Literal transport passes. No inference or support judgment has been made.
    assert validate_claims(rendered, [evidence(quote)])[1] == 'answered'
    assert 'relevance_and_completeness_require_evaluation' in trace['guarantee']


def test_renderer_cannot_invent_omitted_required_fact_or_its_citation():
    rows = [evidence('Each batch contains 800 rows.'),
            evidence('A new backlog alarm notifies the on-call team.', 'E2')]
    rendered, _ = bind_source_text(answer('Batch size is 800 rows.', [rows[0]['text']]), rows)
    assert len(rendered.claims) == 1
    assert rendered.claims[0].evidence_ids == ['E1']
    assert 'alarm' not in rendered.claims[0].text


@pytest.mark.parametrize('ids,quotes', [
    (['E9'], ['Each batch contains 800 rows.']),
    (['E1'], ['Each batch contains 900 rows.']),
    (['E1', 'E1'], ['Each batch contains 800 rows.']),
])
def test_invalid_binding_is_not_repaired_by_fabricated_source(ids, quotes):
    draft = answer('A generated answer.', quotes, ids)
    before = draft.model_dump()
    rendered, trace = bind_source_text(draft, [evidence('Each batch contains 800 rows.')])
    assert rendered is draft and rendered.model_dump() == before
    assert trace['status'] == 'unrenderable_source_span'


def test_same_quote_in_distinct_sources_keeps_both_identities():
    quote = 'Each batch contains 800 rows.'
    rows = [evidence(quote), evidence(quote, 'E2')]
    rendered, _ = bind_source_text(answer('Both rules say 800 rows.', [quote, quote], ['E1', 'E2']), rows)
    assert [c.evidence_ids for c in rendered.claims] == [['E1'], ['E2']]
    claims, status = validate_claims(rendered, rows)
    assert status == 'answered'
    assert [c['evidence_ids'] for c in claims] == [['chunk-E1'], ['chunk-E2']]


def test_excess_source_claims_fail_instead_of_silently_dropping_facts():
    rows = [evidence(f'The synthetic setting number {i} is enabled.', f'E{i}') for i in range(1, 10)]
    draft = GeneratedAnswer(answerable=True, claims=[
        Claim(text='A generated summary.', evidence_ids=[r['id'] for r in group],
              quotes=[r['text'] for r in group]) for group in (rows[:5], rows[5:])])
    before = draft.model_dump()
    rendered, trace = bind_source_text(draft, rows)
    assert trace['status'] == 'source_claim_limit'
    assert rendered is draft and rendered.model_dump() == before


def test_long_original_span_is_not_truncated_into_new_unsupported_claim():
    quote = 'A' * 1001
    draft = answer('A short generated summary.', [quote])
    rendered, trace = bind_source_text(draft, [evidence(quote)])
    assert trace['status'] == 'unrenderable_source_span'
    assert rendered is draft and rendered.claims[0].quotes == [quote]


def test_default_free_form_validation_is_unchanged():
    quote = 'Each batch contains 800 rows.'
    draft = answer('Each batch contains 900 rows.', [quote])
    assert validate_claims(draft, [evidence(quote)]) == ([], 'verification_failed')
    assert draft.claims[0].text == 'Each batch contains 900 rows.'
