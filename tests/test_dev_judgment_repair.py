import json

import pytest

from app.answer_quality import recover_answer
from app.clients import Claim, GeneratedAnswer, Models
from app.config import Settings
from app.qa import answer_verdict, validate_claims
from app.verdict import EXPLICIT_NEGATION_PREFIX, judgment_expression, negate_verdict


@pytest.mark.parametrize('value,expected', [('yes', 'no'), ('no', 'yes'), ('unclear', 'unclear')])
def test_outer_not_preserves_unknown_and_bindings(value, expected):
    result = {'value': value, 'evidence_ids': ['allowed'], 'claim_indices': [1]}
    negated = negate_verdict(result)
    assert negated['value'] == expected
    assert negated['evidence_ids'] == ['allowed']
    question = 'Did Delta not approve the change?'
    assert judgment_expression(EXPLICIT_NEGATION_PREFIX + question) == (question, True)
    assert judgment_expression(question) == (question, False)


def test_structured_judge_reads_only_each_claims_bound_quote(monkeypatch):
    cfg = Settings(verdict_protocol='structured')
    monkeypatch.setattr('app.qa.settings', lambda: cfg)
    monkeypatch.setattr('app.clients.settings', lambda: cfg)
    calls = []

    def backend(body):
        calls.append(body)
        return {'message': {'content': json.dumps({'operator': 'atomic', 'question_complete': True,
            'comparison_dimension': '', 'propositions': [
                {'slot_index': 1, 'truth': 'contradicted', 'claim_indices': [1]}]})}}

    evidence = [{'chunk_id': 'c1', 'text': 'Delta did not approve the change.',
                 'title': 'Approval record', 'metadata': {'source': 'Records'}},
                {'chunk_id': 'c2', 'text': 'Unrelated private draft.', 'title': 'Other'}]
    claims = [{'text': 'Delta approved the change.', 'evidence_ids': ['c1'],
               'quotes': ['Delta did not approve the change.']}]
    result, _ = answer_verdict(Models(chat_backend=backend), 'Did Delta approve the change?',
                              claims, 'answered', evidence=evidence)
    assert result['value'] == 'no'
    inputs = json.loads(calls[0]['messages'][1]['content'])
    assert inputs['claims'][0]['bound_sources'][0]['quote'] == claims[0]['quotes'][0]
    assert 'Unrelated private draft.' not in calls[0]['messages'][1]['content']
    result, _ = answer_verdict(Models(chat_backend=backend),
        EXPLICIT_NEGATION_PREFIX + 'Did Delta approve the change?', claims, 'answered', evidence=evidence)
    assert result['value'] == 'yes' and result['base_value'] == 'no'
    cfg.verdict_protocol = 'legacy'
    Models(chat_backend=lambda body: calls.append(body) or {
        'message': {'content': '{"verdict":"yes","claim_index":1}'}}).decide_verdict(
            'Did Delta approve the change?', claims)
    assert 'bound_sources' not in calls[-1]['messages'][1]['content']


def test_existing_repair_graph_binds_both_numbers_without_relaxing_validation(monkeypatch):
    cfg = Settings()
    monkeypatch.setattr('app.answer_quality.settings', lambda: cfg)
    evidence = [{'id': 'E1', 'chunk_id': 'metric', 'document_id': 'd1', 'title': 'Rate',
                 'text': 'The rate is 4.5%.'},
                {'id': 'E2', 'chunk_id': 'rule', 'document_id': 'd2', 'title': 'Threshold',
                 'text': 'Escalate when the rate exceeds 3.0%.'}]
    bad = GeneratedAnswer(answerable=True, claims=[Claim(text='The rate is 4.5%, exceeding 3.0%.',
        evidence_ids=['E1'], quotes=['The rate is 4.5%.'])])
    fixed = GeneratedAnswer(answerable=True, claims=[Claim(text='The rate is 4.5%, exceeding 3.0%.',
        evidence_ids=['E1', 'E2'], quotes=['The rate is 4.5%.', 'Escalate when the rate exceeds 3.0%.'])])
    calls = []

    class Model:
        def generate(self, _question, _evidence, **kwargs):
            calls.append(kwargs)
            return (bad if len(calls) == 1 else fixed), {}

    answer, usage = recover_answer(Model(), 'Is escalation needed?', evidence)
    assert len(calls) == 2 and usage['answer_recovery']['repair_attempts'] == 1
    assert '观测值和规则阈值' in calls[1]['repair_context']['failed_claims'][0]['reason']
    assert validate_claims(bad, evidence)[1] == 'verification_failed'
    assert validate_claims(answer, evidence)[1] == 'answered'


def test_structured_unknown_stays_unknown_under_outer_not(monkeypatch):
    cfg = Settings(verdict_protocol='structured')
    monkeypatch.setattr('app.qa.settings', lambda: cfg)
    monkeypatch.setattr('app.clients.settings', lambda: cfg)
    models = Models(chat_backend=lambda _body: {'message': {'content': json.dumps({
        'operator': 'atomic', 'question_complete': True, 'comparison_dimension': '',
        'propositions': [{'slot_index': 1, 'truth': 'unknown', 'claim_indices': []}]})}})
    result, _ = answer_verdict(models, EXPLICIT_NEGATION_PREFIX + 'Did Delta approve?',
        [{'text': 'Delta has not announced a decision.', 'evidence_ids': ['c']}], 'answered')
    assert result['value'] == 'unclear' and result['evidence_ids'] == []
