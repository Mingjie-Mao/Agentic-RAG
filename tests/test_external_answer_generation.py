from app.clients import Claim, GeneratedAnswer
from app.config import Settings
from scripts.run_external_answers_dev import generate_answer

import json
import pytest
import time


class Models:
    def __init__(self):
        self.events = []

    def generate(self, question, evidence, **kwargs):
        assert 'required_facts' not in kwargs
        assert kwargs['max_output_tokens'] == 700
        assert evidence[0]['id'] == 'E1'
        return GeneratedAnswer(answerable=True, claims=[Claim(text='Limit is nine units.',
            evidence_ids=['E1'], quotes=['Limit is nine units.'])]), {'answer_status': 'answered'}


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


@pytest.mark.parametrize('protocol', ['direct', 'literal_fallback'])
def test_real_transport_counts_generation_conflict_and_judgment(monkeypatch, protocol):
    from app import clients, qa
    cfg = Settings(focused_generation_enabled=False, answer_contract_enabled=False,
                   answer_quality_enabled=False, verdict_protocol='legacy')
    monkeypatch.setattr(clients, 'settings', lambda: cfg)
    monkeypatch.setattr(qa, 'settings', lambda: cfg)
    evidence = [dict(chunk_id=cid, document_id=cid, version_id='v', source_sha256='s',
        title='Limits', text='Limit is nine units.', locator={}) for cid in ('a', 'b')]
    responses = [
        {'status': 'answered', 'claims': [{'text': 'Both reports state a limit of nine units.',
            'source_ids': ['E1:S1', 'E2:S1']}]},
        {'conflict': False, 'left_id': 'E1:S1', 'right_id': 'E2:S1', 'reason': 'same limit'},
        {'verdict': 'yes', 'claim_index': 1}]
    bodies = []
    def backend(body):
        bodies.append(body)
        return {'message': {'content': json.dumps(responses[len(bodies)-1])},
                'prompt_eval_count': 10, 'eval_count': 5}
    payload, usage = generate_answer('Do both reports state a limit of nine units?', evidence,
        models=clients.Models(chat_backend=backend), cfg=cfg, check_conflict=True, protocol=protocol)
    assert payload['status'] == 'answered'
    assert len(payload['citations']) == 2
    assert [b['options']['num_predict'] for b in bodies] == [700, 200, 60]
    assert usage['business_budget']['calls']['generation']['attempted'] == 3
    assert usage['business_budget']['calls']['generation']['failed'] == 0


def test_protocol_validation_precedes_any_call():
    import pytest
    for protocol in ('unknown', None, [], {}):
        with pytest.raises(ValueError, match='protocol'):
            generate_answer('question', [], models=Models(), cfg=Settings(),
                            check_conflict=False, protocol=protocol)


def test_shared_contract_routes_legacy_and_uses_original_question(monkeypatch):
    from app import task_contract
    calls = []
    def contract(models, question, evidence, **kwargs):
        calls.append((question, evidence, kwargs))
        return models.generate(question, evidence, **kwargs['options'])
    monkeypatch.setattr(task_contract, 'contract_generate', contract)
    evidence = [dict(chunk_id='c', document_id='d', version_id='v', source_sha256='s',
                     title='Limits', text='Limit is nine units.', locator={})]
    payload, _ = generate_answer('What is the limit?', evidence, models=Models(),
        cfg=Settings(), check_conflict=False, protocol='shared_contract')
    assert calls[0][0] == 'What is the limit?'
    assert calls[0][1][0]['id'] == 'E1'
    assert calls[0][2] == {'ordinary_policy': 'legacy',
                         'options': {'max_output_tokens': 700, 'check_conflict': False}}
    assert payload['claims'][0]['evidence_ids'] == ['c']


def test_literal_fallback_dispatches_same_policy_and_preserves_generation_limits(monkeypatch):
    from app import task_contract
    calls = []
    def contract(models, question, evidence, **kwargs):
        calls.append((question, evidence, kwargs))
        return models.generate(question, evidence, **kwargs['options'])
    monkeypatch.setattr(task_contract, 'contract_generate', contract)
    evidence = [dict(chunk_id='c', document_id='d', version_id='v', source_sha256='s',
                     title='Limits', text='Limit is nine units.', locator={})]
    started = time.time()
    payload, usage = generate_answer('What is the limit?', evidence, models=Models(),
        cfg=Settings(_env_file=None), check_conflict=False, protocol='literal_fallback')
    assert calls == [('What is the limit?', [dict(evidence[0], id='E1')], {
        'ordinary_policy': 'literal_fallback',
        'options': {'max_output_tokens': 700, 'check_conflict': False}})]
    assert payload['claims'][0]['evidence_ids'] == ['c']
    limits = usage['business_budget']['limits']
    assert limits['generation'] == 3
    assert all(limits[key] == 0 for key in ('policy', 'search', 'embedding', 'judge', 'judge_tokens'))
    assert usage['business_budget']['deadline'] <= started + 180.1


@pytest.mark.parametrize('protocol', ['shared_contract', 'literal_fallback'])
def test_comparison_budget_blocks_embedding_before_dispatch_for_both_protocols(monkeypatch, protocol):
    from app import task_contract
    from app.execution_budget import BudgetExceeded, current_budget
    entered, observed = [], []
    def contract(*args, **kwargs):
        budget = current_budget()
        observed.append(budget)
        with budget.call('embedding'):
            entered.append('transport')
        raise AssertionError('embedding must be blocked before dispatch')
    monkeypatch.setattr(task_contract, 'contract_generate', contract)
    with pytest.raises(BudgetExceeded, match='embedding_call_budget_exhausted'):
        generate_answer('Original question', [], models=Models(), cfg=Settings(_env_file=None),
            check_conflict=False, protocol=protocol, enforce_zero_embedding=True)
    assert entered == []
    assert observed[0].snapshot()['calls']['embedding']['attempted'] == 0
    assert observed[0].snapshot()['model_call_records'] == []


@pytest.mark.parametrize('model_type', ['ordinary', 'metered'])
@pytest.mark.parametrize('protocol', ['shared_contract', 'literal_fallback'])
def test_comparison_cap_blocks_real_embedding_entry_before_physical_transport(monkeypatch, protocol, model_type):
    from app import clients, task_contract
    from app.execution_budget import BudgetExceeded
    from scripts.run_slot_generation_dev import GenerationModels
    cfg = Settings(_env_file=None)
    monkeypatch.setattr(clients, 'settings', lambda: cfg)
    transports = []
    def post(*args, **kwargs):
        transports.append((args, kwargs))
        from types import SimpleNamespace
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'embeddings': []})
    monkeypatch.setattr(clients.httpx, 'post', post)
    models = clients.Models() if model_type == 'ordinary' else GenerationModels()
    def contract(actual, *args, **kwargs):
        assert actual.answer_budget.snapshot()['limits']['embedding'] == 0
        actual.embed([])
        raise RuntimeError('embedding transport was reached')
    monkeypatch.setattr(task_contract, 'contract_generate', contract)
    with pytest.raises(BudgetExceeded, match='embedding_call_budget_exhausted'):
        generate_answer('Original question', [], models=models, cfg=cfg,
            check_conflict=False, protocol=protocol, enforce_zero_embedding=True)
    assert transports == []
    snapshot = models.answer_budget.snapshot()
    assert snapshot['calls']['embedding']['attempted'] == 0
    assert snapshot['model_call_records'] == []
    if model_type == 'metered':
        assert models.events == []


@pytest.mark.parametrize('model_type', ['ordinary', 'metered'])
def test_v1_helper_without_explicit_embedding_cap_keeps_real_embedding_transport(monkeypatch, model_type):
    from app import clients, task_contract
    from scripts.run_slot_generation_dev import GenerationModels
    from types import SimpleNamespace
    cfg = Settings(_env_file=None)
    monkeypatch.setattr(clients, 'settings', lambda: cfg)
    transports = []
    def post(*args, **kwargs):
        transports.append((args, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'embeddings': []})
    monkeypatch.setattr(clients.httpx, 'post', post)
    models = clients.Models() if model_type == 'ordinary' else GenerationModels()
    def contract(actual, *args, **kwargs):
        assert 'embedding' not in actual.answer_budget.snapshot()['limits']
        assert actual.embed([]) == []
        return GeneratedAnswer(answerable=False, claims=[]), {}
    monkeypatch.setattr(task_contract, 'contract_generate', contract)
    payload, usage = generate_answer('Original question', [], models=models, cfg=cfg,
        check_conflict=False, protocol='shared_contract')
    assert payload['status'] == 'insufficient_evidence' and len(transports) == 1
    assert transports[0][0][0].endswith('/api/embed')
    assert 'embedding' not in usage['business_budget']['calls']
    assert usage['business_budget']['model_call_records'] == []


@pytest.mark.parametrize('limits', [None, {}, {'embedding': 1}])
def test_real_embedding_without_explicit_zero_retains_legacy_transport(monkeypatch, limits):
    from app import clients
    from app.execution_budget import ExecutionBudget, use_budget
    from contextlib import nullcontext
    from types import SimpleNamespace
    cfg = Settings(_env_file=None)
    monkeypatch.setattr(clients, 'settings', lambda: cfg)
    transports = []
    monkeypatch.setattr(clients.httpx, 'post', lambda *a, **kw:
        transports.append((a, kw)) or SimpleNamespace(raise_for_status=lambda: None,
                                                     json=lambda: {'embeddings': []}))
    context = nullcontext() if limits is None else use_budget(ExecutionBudget(cfg, saved={'limits': limits}))
    with context:
        assert clients.Models().embed([]) == []
    assert len(transports) == 1


@pytest.mark.parametrize('flag', [None, 0, 1, 'true', [], {}])
def test_comparison_budget_flag_requires_boolean_before_dispatch(monkeypatch, flag):
    from app import task_contract
    monkeypatch.setattr(task_contract, 'contract_generate', lambda *a, **kw:
                        pytest.fail('invalid budget flag dispatched'))
    with pytest.raises(ValueError, match='enforce_zero_embedding'):
        generate_answer('Question', [], models=Models(), cfg=Settings(_env_file=None),
            check_conflict=False, protocol='shared_contract', enforce_zero_embedding=flag)


@pytest.mark.parametrize('protocol', ['direct', 'shared_contract', 'literal_slots'])
def test_old_protocol_budget_shape_is_unchanged_when_comparison_cap_is_not_enabled(protocol):
    evidence = [dict(chunk_id='c', document_id='d', version_id='v', source_sha256='s',
                     title='Limits', text='Limit is nine units.', locator={})]
    expected = {'generation': 3, 'policy': 0, 'search': 0, 'judge': 0, 'judge_tokens': 0}
    for options in ({}, {'enforce_zero_embedding': False}):
        _, usage = generate_answer('What is the limit?', evidence, models=Models(),
            cfg=Settings(_env_file=None), check_conflict=False, protocol=protocol, **options)
        assert usage['business_budget']['limits'] == expected


@pytest.mark.parametrize('protocol', ['literal_slots', 'literal_fallback'])
def test_literal_slots_is_zero_call_with_source_binding(protocol):
    class NoCall:
        def generate(self, *args, **kwargs):
            raise AssertionError('literal answer should not dispatch')
    evidence = [dict(chunk_id='c', document_id='d', version_id='v', source_sha256='s',
                     title='日志服务', text='日志服务的超时阈值为9秒。', locator={})]
    payload, usage = generate_answer('日志服务的超时阈值是多少秒？', evidence,
        models=NoCall(), cfg=Settings(), check_conflict=False, protocol=protocol)
    assert payload['status'] == 'answered'
    assert payload['claims'][0]['evidence_ids'] == ['c']
    assert usage['generation']['task_contract']['status'] == 'ordinary_literal_complete'
    assert usage['business_budget']['model_call_records'] == []


def test_contract_history_incomplete_is_not_an_execution_failure():
    payload, usage = generate_answer('最早版本的操作日志保留多久？', [], models=Models(),
        cfg=Settings(), check_conflict=False, protocol='shared_contract')
    assert payload['status'] == 'insufficient_evidence'
    assert usage['generation']['task_contract']['status'] == 'history_incomplete'


def test_false_condition_does_not_generate_inactive_branch():
    class NoCall:
        def generate(self, *args, **kwargs):
            raise AssertionError('false condition should not dispatch')
    evidence = [dict(chunk_id='c', document_id='d', version_id='v', source_sha256='s',
        title='恢复目标', text='RPO为20分钟。回滚超时为90秒。', locator={})]
    for protocol in ('shared_contract', 'literal_slots', 'literal_fallback'):
        payload, usage = generate_answer('先查RPO，如果它超过30分钟，再给出回滚超时。', evidence,
            models=NoCall(), cfg=Settings(), check_conflict=False, protocol=protocol)
        assert payload['status'] == 'answered'
        assert '90' not in json.dumps(payload['claims'])
        assert usage['generation']['task_contract']['status'] == 'condition_gated'
