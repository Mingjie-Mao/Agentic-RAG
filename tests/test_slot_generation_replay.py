"""Synthetic replay inputs only; no services, real questions or benchmark artifacts."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from scripts.research_review import digest
from scripts.run_retrieval_quality_dev import sha


@pytest.fixture
def replay(tmp_path, monkeypatch):
    runner = importlib.import_module('scripts.run_slot_generation_dev')
    tasks = [dict(id=f't{i:02}', goal=f'Original question {i}', user='lt-eng',
                  category='synthetic', required_facts=[{'id': 'f', 'text': 'SECRET GOLD'}],
                  gold_documents=['d'], gold_evidence=[], forbidden=[], expected_behavior='SECRET PATH')
             for i in range(47)]
    suite = {'split': 'dev', 'tasks': tasks}
    selection = {'split': 'dev', 'count': 47, 'items': [
        dict(id=t['id'], query_sha256=sha(t['goal'].encode()), annotation_sha256=digest(t))
        for t in tasks]}
    row = dict(chunk_id='c', document_id='d', version_id='v', source_sha256='s',
               rank=1, retrieval_rank=1, bm25_rank=1, dense_rank=None, parent_rank=1,
               fusion_score=1, rerank_score=1, excluded_because=None, admitted=True,
               lane_type='global', lane_sha256=digest('global'))
    corpus = {'documents': []}
    index = {'index': 'source', 'chunks': []}
    parent = {'identity': {'corpus': corpus, 'corpus_sha256': digest(corpus),
                          'source_index': index, 'source_index_sha256': digest(index)}}
    chosen = {'version': 'guarded-selection-dev-v2', 'split': 'dev', 'subset': False,
        'valid': True, 'status': 'complete', 'error': None, 'drift': {'inputs': False},
        'selected_task_ids': [t['id'] for t in tasks], 'chosen_profile': 'baseline',
        'execution_errors': {'baseline': 0}, 'identity': {
            'parent_sha256': digest(parent), 'corpus_sha256': digest(corpus),
            'index_sha256': digest(index), 'code': {'previous_stage': 'different'},
            'registration': {'version': 'guarded-selection-input-v2',
                'profiles': ['baseline', 'rank_floor', 'guarded'],
                'caps': {'limit': 8, 'token_budget': 5000}}},
        'rows': {'baseline': [dict(task_id=t['id'], error=None, telemetry={'context': [row], 'context_tokens': 104})
                              for t in tasks]}}
    chosen['rows'] = {profile: deepcopy(chosen['rows']['baseline'])
                      for profile in ('baseline', 'rank_floor', 'guarded')}
    chosen['execution_errors'] = {profile: 0 for profile in chosen['rows']}
    context = tmp_path/'contexts'
    context.mkdir()
    for t in tasks:
        (context/(t['id']+'.json')).write_text(json.dumps(dict(task_id=t['id'], question=t['goal'],
            contexts={'baseline': [row]})))
    args = SimpleNamespace(suite=str(tmp_path/'suite.json'), selection_result=str(tmp_path/'chosen.json'),
        parent=str(tmp_path/'parent.json'), context_root=str(context),
        review_registration=str(tmp_path/'review.json'), source_audit=None,
        registration=str(tmp_path/'registration.json'), private_root=str(tmp_path/'private'),
        output=str(tmp_path/'output.json'))
    for name, value in [('suite', suite), ('selection_result', chosen), ('parent', parent),
                        ('review_registration', {'pending_task_ids': [t['id'] for t in tasks[:7]],
                                                  'reviewer_identity': {'name': 'Synthetic judge'}})]:
        Path(getattr(args, name)).write_text(json.dumps(value))
    selection_path = tmp_path/'selection.json'
    selection_path.write_text(json.dumps(selection))
    cfg = Settings(_env_file=None, chat_model='synthetic-generator', answer_quality_enabled=False,
                   source_facts_enabled=False)
    identity = {'policy': {'name': cfg.chat_model, 'digest': 'a'*64,
        'runtime': {'version': '1.0'}, 'endpoint_sha256': 'b'*64, 'status': 'active'},
        'libraries': {'langgraph': '1.2.12'}}
    monkeypatch.setattr(runner, 'SELECTION', selection_path)
    monkeypatch.setattr(runner, 'private_root', lambda p: Path(p))
    monkeypatch.setattr(runner, 'settings', lambda: cfg)
    monkeypatch.setattr(runner, 'code_fingerprints', lambda: {'current_stage.py': 'new-code'})
    identity_calls = []
    def model_identity(config):
        identity_calls.append(config)
        return deepcopy(identity)
    monkeypatch.setattr(runner.supplement, 'active_model_identity', model_identity)
    statements = []
    class DB:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def execute(self, sql):
            statements.append(str(sql))
    import app.db
    import app.clients
    monkeypatch.setattr(app.db, 'SessionLocal', DB)
    monkeypatch.setattr(app.clients, 'Search', lambda **kw: SimpleNamespace(index=kw['index']))
    monkeypatch.setattr(runner, 'active_scope', lambda db: ({'lt-eng': SimpleNamespace(tenant_id='tenant')}, [], corpus))
    monkeypatch.setattr(runner, 'index_ledger', lambda source, records: index)
    rebindings = []
    evidence = dict(row, title='SOURCE TITLE', text='SOURCE TEXT', metadata={}, locator={})
    def rebind(db, user, rows, frozen):
        rebindings.append(deepcopy(rows))
        return [dict(r, title=evidence['title'], text=evidence['text'], metadata={}, locator={}) for r in rows]
    monkeypatch.setattr(runner.replay, 'rebind_rows', rebind)
    calls, scored, embedding_caps = [], [], []
    def generate(question, supplied, *, models, cfg, check_conflict, protocol,
                 enforce_zero_embedding=False):
        embedding_caps.append(enforce_zero_embedding)
        calls.append((question, deepcopy(supplied), protocol))
        assert 'SECRET' not in json.dumps([question, supplied])
        return {'status': 'answered', 'claims': [], 'citations': [], 'verdict': None}, {
            'generation': {}, 'verdict': {}, 'business_budget': {'calls': {}, 'model_call_records': []}}
    monkeypatch.setattr(runner, 'generate_answer', generate)
    monkeypatch.setattr(runner, 'GenerationModels', lambda: SimpleNamespace(events=[]))
    def score(task, payload, trace, steps, latency, **kw):
        expected = 94 if getattr(args, 'comparison', 'frozen') == 'fallback' else 141
        assert len(calls) == expected, 'Gold-derived scoring ran before runtime completed'
        scored.append(task['id'])
        return {'answer_payload': payload, 'strict_task_success': None, 'usage': {},
                'early_stop': False, 'extra_calls_after_sufficient_evidence': 0, **kw}
    monkeypatch.setattr(runner, 'provisional_score', score)
    return SimpleNamespace(runner=runner, args=args, tasks=tasks, chosen=chosen, row=row,
        calls=calls, scored=scored, embedding_caps=embedding_caps, cfg=cfg,
        identity=identity, identity_calls=identity_calls,
        rebindings=rebindings, statements=statements, context=context, tmp=tmp_path)


def register(f):
    value = f.runner.registration_for(f.args)
    Path(f.args.registration).write_text(json.dumps(value))
    return value


def test_complete_same_context_three_arms_rotated_and_gold_after_runtime(replay):
    f = replay
    reg = register(f)
    assert reg['protocols'] == ['direct', 'shared_contract', 'literal_slots']
    assert reg['shared_context']['profile'] == 'baseline'
    assert all(c.agent_policy_model == c.chat_model and c.agent_policy_url == '' for c in f.identity_calls)
    result = f.runner.execute(f.args, {'run_id': 'complete'})
    assert result['valid'] and result['status'] == 'complete'
    assert len(f.calls) == len(f.scored) == 141
    assert f.embedding_caps == [False] * 141
    assert [x[2] for x in f.calls[:9]] == ['direct', 'shared_contract', 'literal_slots',
        'shared_contract', 'literal_slots', 'direct', 'literal_slots', 'direct', 'shared_contract']
    assert all(x[1] == f.calls[0][1] for x in f.calls)
    assert len(f.rebindings) == 47*6
    assert f.statements == ['SET TRANSACTION READ ONLY']
    raw = json.loads((f.tmp/'private/complete/raw.json').read_bytes())
    assert all(len(rows) == 47 for rows in raw['results'].values())
    assert all(r['steps'] is None and not r['trace_complete'] for rows in raw['results'].values() for r in rows)
    assert all(r['usage'] == r['physical_generation'] and r['early_stop'] is None
               and r['extra_calls_after_sufficient_evidence'] is None
               for rows in raw['results'].values() for r in rows)
    public = Path(f.args.output).read_text()
    for text in ('Original question', 'SOURCE TEXT', 'SECRET GOLD', 'SECRET PATH', 'SOURCE TITLE'):
        assert text not in public
    assert result['strict_task_success'] is result['semantic_recall'] is None


def test_fallback_comparison_freezes_two_arms_rotates_and_scores_only_after_94_calls(replay):
    f = replay
    f.args.comparison = 'fallback'
    reg = register(f)
    assert reg['version'] == 'slot-generation-input-v2'
    assert reg['comparison'] == 'fallback'
    assert reg['protocols'] == ['shared_contract', 'literal_fallback']
    assert reg['adoption_rule']['candidate'] == 'literal_fallback'
    assert reg['adoption_rule']['baseline'] == 'shared_contract'
    assert reg['shared_context']['binding'] == 'same_ordered_full_source_identities_all_declared_arms'
    assert reg['limits'] == f.runner.LIMITS
    assert reg['review']['pending_task_ids'] == [t['id'] for t in f.tasks[:7]]
    result = f.runner.execute(f.args, {'run_id': 'fallback-complete'})
    assert result['valid'] and result['status'] == 'complete'
    assert result['version'] == 'slot-generation-dev-v2' and result['comparison'] == 'fallback'
    assert list(result['rows']) == ['shared_contract', 'literal_fallback']
    assert len(f.calls) == len(f.scored) == 94
    assert f.embedding_caps == [True] * 94
    assert [x[2] for x in f.calls[:8]] == ['shared_contract', 'literal_fallback',
        'literal_fallback', 'shared_contract', 'shared_contract', 'literal_fallback',
        'literal_fallback', 'shared_contract']
    assert all(x[1] == f.calls[0][1] for x in f.calls)
    assert len(f.rebindings) == 188
    raw = json.loads((f.tmp/'private/fallback-complete/raw.json').read_bytes())
    reviewed_suite = json.loads((f.tmp/'private/fallback-complete/suite.json').read_bytes())
    assert reviewed_suite['arms'] == reg['protocols']
    assert list(raw['results']) == reg['protocols']
    assert all(len(rows) == 47 for rows in raw['results'].values())
    assert all(r['strict_task_success'] is None for rows in raw['results'].values() for r in rows)
    public = Path(f.args.output).read_text()
    assert all(value not in public for value in ('Original question', 'SOURCE TITLE', 'SOURCE TEXT',
                                                'SECRET GOLD', 'SECRET PATH'))
    assert result['strict_task_success'] is result['semantic_recall'] is None


@pytest.mark.parametrize('comparison', [None, [], {}, ['fallback'], True, 2, 'unknown',
                                      'shared_contract,literal_fallback', 'fallback,fallback'])
def test_comparison_rejects_arbitrary_arms_and_nonstring_modes(replay, comparison):
    f = replay
    f.args.comparison = comparison
    with pytest.raises(ValueError, match='comparison'):
        register(f)
    assert not f.calls


@pytest.mark.parametrize('change', ['comparison', 'protocols', 'duplicate', 'nonstring',
                                    'candidate', 'version'])
def test_fallback_registration_mismatch_rejects_before_dispatch_with_full_denominator(replay, change):
    f = replay
    f.args.comparison = 'fallback'
    reg = register(f)
    if change == 'comparison':
        reg['comparison'] = 'frozen'
    elif change == 'protocols':
        reg['protocols'] = ['direct', 'literal_fallback']
    elif change == 'duplicate':
        reg['protocols'] = ['shared_contract', 'shared_contract']
    elif change == 'nonstring':
        reg['protocols'] = ['shared_contract', None]
    elif change == 'candidate':
        reg['adoption_rule']['candidate'] = 'literal_slots'
    else:
        reg['version'] = 'slot-generation-input-v1'
    Path(f.args.registration).write_text(json.dumps(reg))
    result = f.runner.execute(f.args, {'run_id': 'fallback-mismatch'})
    assert not result['valid'] and not f.calls and not f.scored
    assert list(result['rows']) == ['shared_contract', 'literal_fallback']
    assert all(len(rows) == 47 and all(r['status'] == 'not_observed' and
        r['physical_generation'] is None and r['latency_ms'] is None for r in rows)
        for rows in result['rows'].values())


@pytest.mark.parametrize('registered,runtime', [('frozen', 'fallback'), ('fallback', 'frozen')])
def test_cli_mode_cannot_reuse_other_comparison_registration(replay, registered, runtime):
    f = replay
    f.args.comparison = registered
    register(f)
    f.args.comparison = runtime
    result = f.runner.execute(f.args, {'run_id': 'other-mode'})
    assert not result['valid'] and not f.calls and not f.scored
    assert list(result['rows']) == list(f.runner.protocols_for(f.args))
    assert all(len(rows) == 47 and all(r['physical_generation'] is None for r in rows)
               for rows in result['rows'].values())


def test_frozen_default_registration_keeps_v1_shape(replay):
    f = replay
    default = register(f)
    f.args.comparison = 'frozen'
    assert register(f) == default
    assert default['version'] == 'slot-generation-input-v1'
    assert 'comparison' not in default
    assert default['protocols'] == ['direct', 'shared_contract', 'literal_slots']


@pytest.mark.parametrize('change', ['chosen', 'parent', 'errors', 'drift', 'rows'])
def test_selection_binding_rejects_invalid_input_before_calls(replay, change):
    f = replay
    chosen = deepcopy(f.chosen)
    if change == 'chosen':
        chosen['chosen_profile'] = 'unregistered'
    elif change == 'parent':
        chosen['identity']['parent_sha256'] = 'wrong'
    elif change == 'errors':
        chosen['execution_errors']['baseline'] = 1
    elif change == 'drift':
        chosen['drift']['inputs'] = True
    else:
        chosen['rows']['baseline'].reverse()
    Path(f.args.selection_result).write_text(json.dumps(chosen))
    with pytest.raises(ValueError):
        register(f)
    assert not f.calls


@pytest.mark.parametrize('change', ['private', 'input', 'model', 'settings'])
@pytest.mark.parametrize('comparison', ['frozen', 'fallback'])
def test_registered_identities_changed_before_runtime_retain_null_positions(replay, change, comparison):
    f = replay
    f.args.comparison = comparison
    register(f)
    if change == 'private':
        (f.context/'t00.json').write_text('{}')
    elif change == 'input':
        Path(f.args.parent).write_text('{}')
    elif change == 'model':
        f.identity['policy']['digest'] = 'c'*64
    else:
        f.cfg.chat_model = 'changed'
    result = f.runner.execute(f.args, {'run_id': 'changed'})
    assert not result['valid'] and not f.calls
    assert all(len(rows) == 47 for rows in result['rows'].values())
    assert all(r['status'] == 'not_observed' and r['latency_ms'] is None
               and r['physical_generation'] is None for rows in result['rows'].values() for r in rows)


@pytest.mark.parametrize('comparison', ['frozen', 'fallback'])
def test_unauthorized_post_generation_binding_prevents_valid_run(replay, monkeypatch, comparison):
    f = replay
    f.args.comparison = comparison
    register(f)
    original = f.runner.replay.rebind_rows
    count = 0
    def bind(*a):
        nonlocal count
        count += 1
        if count == 2:
            raise PermissionError('PRIVATE REASON')
        return original(*a)
    monkeypatch.setattr(f.runner.replay, 'rebind_rows', bind)
    result = f.runner.execute(f.args, {'run_id': 'acl'})
    assert not result['valid']
    assert result['rows'][f.runner.protocols_for(f.args)[0]][0]['error'] == 'PermissionError'
    assert 'PRIVATE REASON' not in Path(f.args.output).read_text()


@pytest.mark.parametrize('comparison', ['frozen', 'fallback'])
def test_interrupt_retains_attempted_cost_and_all_null_unobserved_positions(replay, monkeypatch, comparison):
    from app.execution_budget import ExecutionBudget, use_budget
    f = replay
    f.args.comparison = comparison
    register(f)
    def interrupt(question, evidence, *, models, cfg, **kw):
        budget = ExecutionBudget(cfg)
        models.answer_budget = budget
        with use_budget(budget), budget.call('generation'):
            models.events.append(dict(role='generation', operation='generation', status='failed',
                prompt_tokens=None, completion_tokens=None, server_ms=None, wall_ms=2))
            raise KeyboardInterrupt()
    monkeypatch.setattr(f.runner, 'generate_answer', interrupt)
    result = f.runner.execute(f.args, {'run_id': 'interrupted'})
    assert not result['valid'] and result['status'] == 'interrupted'
    first_protocol, second_protocol = f.runner.protocols_for(f.args)[:2]
    first = result['rows'][first_protocol][0]
    assert first['physical_generation']['attempted'] == first['physical_generation']['failed'] == 1
    assert first['physical_generation']['prompt_tokens'] is None
    assert first['business_budget']['model_call_records'][0]['role'] == 'generation'
    assert all(len(rs) == 47 for rs in result['rows'].values())
    assert result['rows'][second_protocol][0]['physical_generation'] is None
    assert not f.scored
    assert (f.tmp/'private/interrupted/raw.json').exists()


def test_fallback_failed_transport_preserves_costs_full_denominator_and_null_provider_tokens(replay, monkeypatch):
    from app.execution_budget import ExecutionBudget, use_budget
    f = replay
    f.args.comparison = 'fallback'
    register(f)
    original = f.runner.generate_answer
    def generate(*a, models, cfg, **kw):
        result = original(*a, models=models, cfg=cfg, **kw)
        if len(f.calls) == 1:
            budget = ExecutionBudget(cfg)
            models.answer_budget = budget
            with use_budget(budget), budget.call('generation'):
                models.events.append(dict(role='generation', operation='generation', status='failed',
                    prompt_tokens=None, completion_tokens=None, server_ms=None, wall_ms=2))
                raise RuntimeError('PRIVATE ERROR')
        return result
    monkeypatch.setattr(f.runner, 'generate_answer', generate)
    result = f.runner.execute(f.args, {'run_id': 'fallback-failure'})
    assert not result['valid'] and result['status'] == 'failed'
    assert len(f.calls) == len(f.scored) == 94
    assert all(len(rows) == 47 for rows in result['rows'].values())
    first = result['rows']['shared_contract'][0]
    assert first['error'] == 'RuntimeError' and first['status'] == 'execution_failed'
    assert first['physical_generation']['attempted'] == first['physical_generation']['failed'] == 1
    assert first['business_budget']['calls']['generation']['failed'] == 1
    assert first['transport_budget_matched'] is True
    assert first['physical_generation']['prompt_tokens'] is None
    assert first['business_budget']['model_call_records'][0]['completion_tokens'] is None
    assert 'PRIVATE ERROR' not in Path(f.args.output).read_text()
    assert not (f.tmp/'private/fallback-failure/review-packet.json').exists()


@pytest.mark.parametrize('option', [None, 'frozen', 'fallback'])
def test_cli_declares_only_supported_comparison_modes(replay, monkeypatch, option):
    f = replay
    import sys
    argv = ['run_slot_generation_dev']
    for key in ('suite', 'selection_result', 'parent', 'context_root', 'review_registration',
                'registration', 'private_root', 'output'):
        argv.extend(['--'+key.replace('_', '-'), getattr(f.args, key)])
    if option is not None:
        argv.extend(['--comparison', option])
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(f.runner, 'reserve_output', lambda _: {'run_id': 'cli'})
    captured = []
    monkeypatch.setattr(f.runner, 'execute', lambda args, reservation:
                        captured.append(args.comparison) or {'valid': True})
    assert f.runner.main() == 0
    assert captured == [option or 'frozen']
    monkeypatch.setattr(sys, 'argv', argv[:1] + ['--comparison', 'fallback,fallback'] + argv[1:])
    with pytest.raises(SystemExit) as exc:
        f.runner.main()
    assert exc.value.code == 2 and captured == [option or 'frozen']


def test_transport_success_is_distinct_from_schema_failure_and_unknown_tokens(replay):
    f = replay
    models = SimpleNamespace(events=[dict(role='generation', operation='verdict', status='ok',
        prompt_tokens=None, completion_tokens=None, server_ms=None, wall_ms=2)])
    usage = {'business_budget': {'calls': {'generation': {'attempted': 1, 'succeeded': 1, 'failed': 0,
            'prompt_tokens': 0, 'completion_tokens': 0}}, 'model_call_records': [
                {'role': 'generation', 'status': 'ok', 'prompt_tokens': 0, 'completion_tokens': 0}]}}
    value = f.runner.cost_projection(models, usage)
    assert value['physical_generation']['succeeded'] == 1
    assert value['business_budget']['model_call_records'][0]['operation'] == 'verdict'
    assert value['business_budget']['model_call_records'][0]['prompt_tokens'] is None
    assert value['business_budget']['calls']['generation']['prompt_tokens'] is None


def test_public_contract_projection_contains_codes_not_full_trace(replay):
    usage = {'generation': {'task_contract': {'status': 'ordinary_slot_incomplete',
        'contract': {'question': 'PRIVATE QUESTION'}, 'bound_values': [{'quote': 'PRIVATE QUOTE'}],
        'slot_states': {'slot_1': 'unknown'}, 'slot_reasons': {'slot_1': ['missing_predicate']}}}}
    result = replay.runner.contract_projection(usage)
    assert result == {'status': 'ordinary_slot_incomplete', 'slot_states': {'slot_1': 'unknown'},
                      'slot_reasons': {'slot_1': ['missing_predicate']}}


def test_replay_entry_point_exists():
    assert importlib.util.find_spec('scripts.run_slot_generation_dev') is not None


@pytest.mark.parametrize('change', ['private', 'input', 'model', 'settings', 'code', 'corpus', 'index'])
@pytest.mark.parametrize('comparison', ['frozen', 'fallback'])
def test_after_runtime_identity_drift_invalidates_even_completed_answers(replay, monkeypatch, change, comparison):
    f = replay
    f.args.comparison = comparison
    total = 47 * len(f.runner.protocols_for(f.args))
    register(f)
    original = f.runner.generate_answer
    def generate(*a, **kw):
        result = original(*a, **kw)
        if len(f.calls) == total:
            if change == 'private':
                (f.context/'t00.json').write_text('{}')
            elif change == 'input':
                Path(f.args.review_registration).write_text('{}')
            elif change == 'model':
                f.identity['policy']['digest'] = 'e'*64
            elif change == 'settings':
                f.cfg.chat_model = 'changed-after-dispatch'
            elif change == 'code':
                monkeypatch.setattr(f.runner, 'code_fingerprints', lambda: {'new.py': 'changed'})
            elif change == 'corpus':
                monkeypatch.setattr(f.runner, 'active_scope', lambda _: ({}, [], {'documents': ['changed']}))
            else:
                monkeypatch.setattr(f.runner, 'index_ledger', lambda *_: {'index': 'changed'})
        return result
    monkeypatch.setattr(f.runner, 'generate_answer', generate)
    result = f.runner.execute(f.args, {'run_id': 'drift'})
    assert result['status'] == 'invalid_drift' and not result['valid']
    assert len(f.calls) == total
    assert any(v is True for v in result['drift'].values())


def test_fixed_public_context_order_is_rejected_before_model_dispatch(replay):
    f = replay
    original = json.loads((f.context/'t00.json').read_bytes())
    original['contexts']['baseline'][0]['version_id'] = 'changed-version'
    (f.context/'t00.json').write_text(json.dumps(original))
    register(f)
    result = f.runner.execute(f.args, {'run_id': 'bindings'})
    assert not result['valid'] and not f.calls
    assert result['error']['type'] == 'ValueError'


def test_exact_dev_denominator_rejects_subset_and_selection_annotation_change(replay):
    f = replay
    suite = json.loads(Path(f.args.suite).read_bytes())
    suite['tasks'] = suite['tasks'][:-1]
    Path(f.args.suite).write_text(json.dumps(suite))
    with pytest.raises(ValueError, match='47-task'):
        register(f)
    assert not f.calls


def test_budget_transport_mismatch_cannot_be_valid(replay, monkeypatch):
    f = replay
    register(f)
    original = f.runner.generate_answer
    def generate(*a, **kw):
        payload, usage = original(*a, **kw)
        usage['business_budget']['calls'] = {'generation': {'attempted': 1, 'succeeded': 1, 'failed': 0}}
        return payload, usage
    monkeypatch.setattr(f.runner, 'generate_answer', generate)
    result = f.runner.execute(f.args, {'run_id': 'mismatch'})
    assert not result['valid']
    assert result['rows']['direct'][0]['transport_budget_matched'] is False


def test_real_synthetic_transport_projection_labels_generation_conflict_and_verdict(monkeypatch):
    from app import clients, qa
    from scripts import run_slot_generation_dev as runner
    cfg = Settings(_env_file=None, focused_generation_enabled=False, answer_contract_enabled=False,
                   answer_quality_enabled=False, verdict_protocol='legacy')
    monkeypatch.setattr(clients, 'settings', lambda: cfg)
    monkeypatch.setattr(qa, 'settings', lambda: cfg)
    responses = [
        {'status': 'answered', 'claims': [{'text': 'Both reports state a limit of nine units.',
            'source_ids': ['E1:S1', 'E2:S1']}]},
        {'conflict': False, 'left_id': 'E1:S1', 'right_id': 'E2:S1', 'reason': 'same limit'},
        {'verdict': 'yes', 'claim_index': 1}]
    bodies = []
    def backend(self, path, body, base=None):
        assert path == '/api/chat'
        bodies.append(body)
        return {'message': {'content': json.dumps(responses[len(bodies)-1])}}
    monkeypatch.setattr(clients.Models, '_post', backend)
    models = runner.GenerationModels()
    evidence = [dict(chunk_id=cid, document_id=cid, version_id='v', source_sha256='s',
        title='Limits', text='Limit is nine units.', locator={}) for cid in ('a', 'b')]
    payload, usage = runner.generate_answer('Do both reports state a limit of nine units?', evidence,
        models=models, cfg=cfg, check_conflict=True)
    cost = runner.cost_projection(models, usage)
    assert payload['status'] == 'answered'
    assert cost['transport_budget_matched']
    assert [r['operation'] for r in cost['business_budget']['model_call_records']] == [
        'generation', 'conflict', 'verdict']
    assert all(r['role'] == 'generation' for r in models.events)
    assert cost['physical_generation']['prompt_tokens'] is None
    assert [r['status'] for r in cost['physical_call_records']] == ['ok', 'ok', 'ok']


def test_synthetic_schema_failure_preserves_successful_physical_transport(monkeypatch):
    from app import clients
    from app.clients import DependencyError
    from scripts import run_slot_generation_dev as runner
    monkeypatch.setattr(clients.Models, '_post', lambda *a, **kw: {'message': {'content': '{malformed'}})
    models = runner.GenerationModels()
    with pytest.raises(DependencyError):
        runner.generate_answer('Question?', [], models=models,
                               cfg=Settings(_env_file=None), check_conflict=False)
    cost = runner.cost_projection(models, {'business_budget': models.answer_budget.snapshot()})
    assert cost['physical_generation']['attempted'] == cost['physical_generation']['succeeded'] == 1
    assert cost['physical_generation']['failed'] == 0
    assert cost['business_budget']['calls']['generation']['succeeded'] == 1
    assert cost['physical_call_records'][0]['status'] == 'ok'


def test_public_result_schema_failure_codes_are_separate_from_transport(replay):
    row = replay.runner.missing('t', 'r')
    row.update(status='answered', not_observed=False, execution_error_kind=None,
        generation_usage={'verdict': {'verdict_status': 'unavailable',
            'verdict_error_kind': 'ValidationError', 'reason': 'PRIVATE REASON'}, 'generation': {}})
    value = replay.runner.public_row(row)
    assert value['result_pipeline']['verdict_status'] == 'unavailable'
    assert 'PRIVATE REASON' not in json.dumps(value)


@pytest.mark.parametrize('change', ['version', 'registration_version', 'profiles',
                                    'missing_caps', 'limit', 'token_budget', 'boolean_limit'])
def test_stage_one_provenance_and_caps_are_required_before_dispatch(replay, change):
    f = replay
    chosen = deepcopy(f.chosen)
    declaration = chosen['identity']['registration']
    if change == 'version':
        chosen['version'] = 'fabricated-selection'
    elif change == 'registration_version':
        declaration['version'] = 'guarded-selection-input-v1'
    elif change == 'profiles':
        declaration['profiles'] = ['baseline']
    elif change == 'missing_caps':
        declaration.pop('caps')
    elif change == 'limit':
        declaration['caps']['limit'] = 9
    elif change == 'token_budget':
        declaration['caps']['token_budget'] = 5001
    else:
        declaration['caps']['limit'] = True
    Path(f.args.selection_result).write_text(json.dumps(chosen))
    with pytest.raises(ValueError):
        register(f)
    result = f.runner.execute(f.args, {'run_id': 'invalid-caps'})
    assert not result['valid'] and not f.calls
    assert all(len(rows) == 47 and all(r['physical_generation'] is None for r in rows)
               for rows in result['rows'].values())


@pytest.mark.parametrize('change', ['duplicate', 'nine', 'negative_tokens', 'string_tokens',
                                    'bool_tokens', 'too_many_tokens', 'missing_tokens'])
def test_selected_public_context_caps_and_token_telemetry_fail_before_dispatch(replay, change):
    f = replay
    chosen = deepcopy(f.chosen)
    telemetry = chosen['rows']['baseline'][0]['telemetry']
    if change == 'duplicate':
        telemetry['context'].append(deepcopy(f.row))
    elif change == 'nine':
        telemetry['context'] = [dict(f.row, chunk_id=f'c{i}', rank=i+1) for i in range(9)]
    elif change == 'negative_tokens':
        telemetry['context_tokens'] = -1
    elif change == 'string_tokens':
        telemetry['context_tokens'] = '104'
    elif change == 'bool_tokens':
        telemetry['context_tokens'] = True
    elif change == 'too_many_tokens':
        telemetry['context_tokens'] = 5001
    else:
        telemetry.pop('context_tokens')
    Path(f.args.selection_result).write_text(json.dumps(chosen))
    with pytest.raises(ValueError):
        register(f)
    result = f.runner.execute(f.args, {'run_id': 'invalid-context'})
    assert not f.calls and not result['valid']
    assert all(len(rows) == 47 and all(r['physical_generation'] is None for r in rows)
               for rows in result['rows'].values())


def test_fabricated_but_in_range_token_count_is_checked_against_rebound_bytes(replay):
    f = replay
    chosen = deepcopy(f.chosen)
    chosen['rows']['baseline'][0]['telemetry']['context_tokens'] = 103
    Path(f.args.selection_result).write_text(json.dumps(chosen))
    register(f)
    result = f.runner.execute(f.args, {'run_id': 'fabricated-count'})
    assert not result['valid'] and not f.calls
    assert all(len(rows) == 47 and all(r['physical_generation'] is None for r in rows)
               for rows in result['rows'].values())


def bounded_context(f, monkeypatch, *, count, actual_tokens, registered_tokens):
    from scripts import run_evidence_selection_dev as evidence_replay
    chosen = deepcopy(f.chosen)
    public = [dict(f.row, chunk_id=f'c{i}', rank=i+1) for i in range(count)]
    full = [dict(row, title='', text=(' unit'*(actual_tokens-100*count) if i == 0 else ''),
                 metadata={}, locator={}) for i, row in enumerate(public)]
    assert evidence_replay.evidence_tokens(full) == actual_tokens
    for task, selected in zip(f.tasks, chosen['rows']['baseline'], strict=True):
        selected['telemetry'] = {'context': public, 'context_tokens': registered_tokens}
        path = f.context/(task['id']+'.json')
        record = json.loads(path.read_bytes())
        record['contexts']['baseline'] = full
        path.write_text(json.dumps(record))
    Path(f.args.selection_result).write_text(json.dumps(chosen))
    monkeypatch.setattr(f.runner.replay, 'rebind_rows', lambda db, user, rows, corpus: deepcopy(rows))


@pytest.mark.parametrize('comparison', ['frozen', 'fallback'])
def test_exact_eight_chunks_five_thousand_actual_tokens_is_valid(replay, monkeypatch, comparison):
    f = replay
    f.args.comparison = comparison
    bounded_context(f, monkeypatch, count=8, actual_tokens=5000, registered_tokens=5000)
    reg = register(f)
    assert reg['shared_context']['caps'] == {'limit': 8, 'token_budget': 5000}
    result = f.runner.execute(f.args, {'run_id': 'boundary'})
    assert result['valid'] and len(f.calls) == 47 * len(f.runner.protocols_for(f.args))


@pytest.mark.parametrize('comparison', ['frozen', 'fallback'])
def test_actual_five_thousand_and_one_tokens_blocks_all_dispatch_and_keeps_nulls(replay, monkeypatch, comparison):
    f = replay
    f.args.comparison = comparison
    bounded_context(f, monkeypatch, count=8, actual_tokens=5001, registered_tokens=5000)
    register(f)
    result = f.runner.execute(f.args, {'run_id': 'actual-over-budget'})
    assert not result['valid'] and not f.calls
    assert all(len(rows) == 47 and all(r['physical_generation'] is None for r in rows)
               for rows in result['rows'].values())


def test_adoption_rule_is_frozen_and_requires_primary_no_regression_before_efficiency(replay):
    f = replay
    reg = register(f)
    rule = reg['adoption_rule']
    assert rule['candidate'] == 'literal_slots' and rule['baseline'] == 'shared_contract'
    assert rule['required'] == ['full_valid_reviewed_original_primary', 'equal_denominator',
        'no_previously_correct_task_loss', 'no_required_fact_support_loss']
    assert rule['improvement'] == ['strict_task_success_gain',
        'equal_correctness_and_fewer_actual_generation_calls']
    assert rule['direct_to_shared_contract'] == 'evaluation_alignment_only'
    assert rule['headline'] == 'pending_frozen_reviewed_core'
    assert not f.calls


@pytest.mark.parametrize('change', ['missing_rows', 'missing_error_count', 'incomplete_other_profile'])
def test_complete_stage_one_requires_all_declared_profile_rows_and_error_counts(replay, change):
    f = replay
    chosen = deepcopy(f.chosen)
    if change == 'missing_rows':
        chosen['rows'].pop('guarded')
    elif change == 'missing_error_count':
        chosen['execution_errors'].pop('guarded')
    else:
        chosen['rows']['guarded'].pop()
    Path(f.args.selection_result).write_text(json.dumps(chosen))
    with pytest.raises(ValueError):
        register(f)
    assert not f.calls
