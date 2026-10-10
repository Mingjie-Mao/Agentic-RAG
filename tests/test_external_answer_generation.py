from app.clients import Claim, GeneratedAnswer
from app.config import Settings
from scripts.run_external_answers_dev import generate_answer

import json


class Models:
    def __init__(self):
        self.events = []

    def generate(self, question, evidence, **kwargs):
        assert 'required_facts' not in kwargs
        assert kwargs['max_output_tokens'] == 700
        return GeneratedAnswer(answerable=True, claims=[Claim(text='Limit is nine units.',
            evidence_ids=['c'], quotes=['Limit is nine units.'])]), {'answer_status': 'answered'}


def test_generation_preserves_source_binding_and_rejects_fabricated_quote():
    evidence = [dict(chunk_id='c', document_id='d', version_id='v', source_sha256='s',
                     title='Limits', text='Limit is nine units.', locator={})]
    payload, usage = generate_answer('What is the limit?', evidence, models=Models(),
                                    cfg=Settings(), check_conflict=False)
    assert payload['status'] == 'answered'
    assert payload['citations'][0]['text'] == evidence[0]['text']
    assert payload['claims'][0]['evidence_ids'] == ['c']
    evidence[0]['text'] = 'Different fact.'
    payload, usage = generate_answer('What is the limit?', evidence, models=Models(),
                                    cfg=Settings(), check_conflict=False)
    assert payload['status'] == 'verification_failed'
    assert payload['claims'] == payload['citations'] == []


def test_real_transport_counts_generation_conflict_and_judgment(monkeypatch):
    from app import clients, qa
    cfg = Settings(focused_generation_enabled=False, answer_contract_enabled=False,
                   answer_quality_enabled=False, verdict_protocol='legacy')
    monkeypatch.setattr(clients, 'settings', lambda: cfg)
    monkeypatch.setattr(qa, 'settings', lambda: cfg)
    evidence = [dict(chunk_id=cid, document_id=cid, version_id='v', source_sha256='s',
        title='Limits', text='Limit is nine units.', locator={}) for cid in ('a', 'b')]
    responses = [
        {'status': 'answered', 'claims': [{'text': 'Both reports state a limit of nine units.',
            'source_ids': ['a:S1', 'b:S1']}]},
        {'conflict': False, 'left_id': 'a:S1', 'right_id': 'b:S1', 'reason': 'same limit'},
        {'verdict': 'yes', 'claim_index': 1}]
    bodies = []
    def backend(body):
        bodies.append(body)
        return {'message': {'content': json.dumps(responses[len(bodies)-1])},
                'prompt_eval_count': 10, 'eval_count': 5}
    payload, usage = generate_answer('Do both reports state a limit of nine units?', evidence,
        models=clients.Models(chat_backend=backend), cfg=cfg, check_conflict=True)
    assert payload['status'] == 'answered'
    assert len(payload['citations']) == 2
    assert [b['options']['num_predict'] for b in bodies] == [700, 200, 60]
    assert usage['business_budget']['calls']['generation']['attempted'] == 3
    assert usage['business_budget']['calls']['generation']['failed'] == 0
