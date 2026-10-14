"""Synthetic replay fixtures: never load the frozen Dev tasks or source prose."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_guarded_selection_dev as runner
from scripts.research_review import digest


def candidate(cid='a', rank=1):
    return dict(chunk_id=cid, document_id='doc', version_id='v1', source_sha256='hash',
                title='Synthetic title', text='Synthetic source ' + cid, rank=rank,
                parent_rank=rank, rerank_score=999, lane_type='global',
                lane_sha256=digest('global'), admitted=True, excluded_because=None,
                metadata={'nested': [1]}, locator={'page': 1})


def original():
    rows = [candidate('a', 2), candidate('b', 1)]
    return {'pools': {'aspect': rows}, 'ranking_traces': [{
        'ranker_calls': 1, 'ranker_pairs': 2, 'aspect_ranking': [{
            'query': 'synthetic query', 'pairs': 2,
            'scores': [{'chunk_id': 'b', 'rerank_score': 2},
                       {'chunk_id': 'a', 'rerank_score': 1}]}]}]}


def test_observed_digest_reuses_original_cache_identity():
    source = original()
    observed = runner.ObservedScores.from_original(source)
    row = source['pools']['aspect'][0]
    key = digest(['synthetic query', *[row[k] for k in
        ('chunk_id', 'document_id', 'version_id', 'source_sha256', 'title', 'text')]])
    assert observed.scores[key] == 1
    assert observed.observed_pairs == 2


def test_replay_sort_and_deep_clone_preserve_input_and_scope():
    source = original()
    rows = source['pools']['aspect']
    before = deepcopy(rows)
    observed = runner.ObservedScores.from_original(source)
    ranked = runner.replay_rank('synthetic query', rows, observed)
    assert [r['chunk_id'] for r in ranked] == ['b', 'a']
    assert rows == before
    assert [{k: v for k, v in r.items() if k != 'rerank_score'} for r in ranked] == [
        {k: v for k, v in r.items() if k != 'rerank_score'} for r in reversed(rows)]
    ranked[0]['metadata']['nested'].append(2)
    assert rows == before
    assert observed.costs() == dict(logical_ranker_calls=1, logical_ranker_pairs=2,
                                  cache_hits=2, physical_ranker_calls=0,
                                  physical_ranker_pairs=0, search_calls=0,
                                  embedding_calls=0, policy_calls=0, generation_calls=0)


@pytest.mark.parametrize('field', ['chunk_id', 'document_id', 'version_id',
                                  'source_sha256', 'title', 'text'])
def test_changed_identity_cannot_reuse_observed_score(field):
    source = original()
    observed = runner.ObservedScores.from_original(source)
    rows = deepcopy(source['pools']['aspect'])
    rows[0][field] += '-changed'
    with pytest.raises(ValueError, match='observed'):
        runner.replay_rank('synthetic query', rows, observed)
    assert observed.costs()['physical_ranker_calls'] == 0


def test_new_query_never_reuses_candidate_score():
    source = original()
    observed = runner.ObservedScores.from_original(source)
    with pytest.raises(ValueError, match='observed'):
        runner.replay_rank('new query', source['pools']['aspect'], observed)


def test_even_empty_replay_input_cannot_hide_an_unobserved_query():
    observed = runner.ObservedScores.from_original(original())
    with pytest.raises(ValueError, match='observed'):
        runner.replay_rank('new query', [], observed)


@pytest.mark.parametrize('score', [None, True, float('nan'), float('inf'), -float('inf'), '1'])
def test_missing_nonfinite_scores_are_rejected(score):
    source = original()
    source['ranking_traces'][0]['aspect_ranking'][0]['scores'][0]['rerank_score'] = score
    with pytest.raises(ValueError):
        runner.ObservedScores.from_original(source)


@pytest.mark.parametrize('query', ['', ' ', None, 1])
def test_empty_or_nonstring_query_is_rejected(query):
    source = original()
    source['ranking_traces'][0]['aspect_ranking'][0]['query'] = query
    with pytest.raises(ValueError):
        runner.ObservedScores.from_original(source)


@pytest.mark.parametrize('mutation', ['duplicate', 'unknown', 'missing', 'pairs', 'order',
                                     'total_pairs', 'calls', 'pool_conflict'])
def test_malformed_observation_is_rejected(mutation):
    source = original()
    trace = source['ranking_traces'][0]
    aspect = trace['aspect_ranking'][0]
    if mutation == 'duplicate':
        aspect['scores'][1] = deepcopy(aspect['scores'][0])
    elif mutation == 'unknown':
        aspect['scores'][0]['chunk_id'] = 'unknown'
    elif mutation == 'missing':
        del aspect['scores'][0]['rerank_score']
    elif mutation == 'pairs':
        aspect['pairs'] += 1
    elif mutation == 'order':
        aspect['scores'].reverse()
    elif mutation == 'total_pairs':
        trace['ranker_pairs'] += 1
    elif mutation == 'calls':
        trace['ranker_calls'] += 1
    else:
        source['pools']['aspect'].append(dict(source['pools']['aspect'][0], text='tampered'))
    with pytest.raises(ValueError):
        runner.ObservedScores.from_original(source)


def test_same_query_conflicting_repeated_observation_is_rejected():
    source = original()
    trace = source['ranking_traces'][0]
    trace['aspect_ranking'].append(deepcopy(trace['aspect_ranking'][0]))
    trace['aspect_ranking'][1]['scores'][0]['rerank_score'] = 3
    trace.update(ranker_pairs=4, ranker_calls=2)
    with pytest.raises(ValueError, match='conflict'):
        runner.ObservedScores.from_original(source)


def test_query_local_scores_and_tie_break_never_cross_compare():
    source = original()
    trace = source['ranking_traces'][0]
    trace['aspect_ranking'].append(dict(query='another query', pairs=2,
        scores=[dict(chunk_id='a', rerank_score=-100), dict(chunk_id='b', rerank_score=-100)]))
    trace.update(ranker_pairs=4, ranker_calls=2)
    observed = runner.ObservedScores.from_original(source)
    assert [r['chunk_id'] for r in runner.replay_rank(
        'another query', source['pools']['aspect'], observed)] == ['b', 'a']
    assert observed.observed_pairs == 4
    assert observed.costs()['logical_ranker_pairs'] == 2


def test_missing_pair_retains_failed_attempt_without_physical_dispatch():
    source = original()
    source['ranking_traces'][0]['aspect_ranking'][0]['scores'].pop()
    source['ranking_traces'][0]['aspect_ranking'][0]['pairs'] = 1
    source['ranking_traces'][0]['ranker_pairs'] = 1
    observed = runner.ObservedScores.from_original(source)
    with pytest.raises(ValueError, match='observed'):
        runner.replay_rank('synthetic query', source['pools']['aspect'], observed)
    assert observed.events[-1]['status'] == 'failed'
    assert observed.costs()['logical_ranker_pairs'] == 2
    assert observed.costs()['cache_hits'] == 1
    assert observed.costs()['physical_ranker_pairs'] == 0


def test_public_trace_drops_source_prose_recursively():
    trace = {'aspect_ranking': [{'query': 'PRIVATE QUERY', 'scores': []}],
             'aspect_floor': [{'query': 'PRIVATE QUERY', 'reason': 'PRIVATE SOURCE'}],
             'rejected': [{'chunk_id': 'a', 'reason': 'PRIVATE SOURCE'}],
             'selected_ids': ['a'], 'required_ids': ['a'], 'context_tokens': 101,
             'coverage_type': 'relevance_ranking_only', 'text': 'PRIVATE SOURCE'}
    public = runner.public_trace(trace)
    assert 'PRIVATE' not in json.dumps(public)
    assert public['trace_sha256'] == digest(trace)
    assert public['selected_ids'] == ['a']


def test_replay_rejects_duplicate_rows_without_mutation():
    source = original()
    observed = runner.ObservedScores.from_original(source)
    rows = [source['pools']['aspect'][0]] * 2
    with pytest.raises(ValueError):
        runner.replay_rank('synthetic query', rows, observed)


def test_original_lane_occurrences_share_score_without_dropping_pool_rows():
    source = original()
    source['pools']['aspect'].append(dict(source['pools']['aspect'][0],
        lane_type='source', lane_sha256=digest('other lane'), rank=9, parent_rank=9))
    before = deepcopy(source)
    observed = runner.ObservedScores.from_original(source)
    assert observed.observed_pairs == 2
    assert len(observed.scores) == 2
    assert source == before
    alternate = [source['pools']['aspect'][-1]]
    ranked = runner.replay_rank('synthetic query', alternate, observed)
    assert ranked[0]['lane_type'] == 'source'
    assert ranked[0]['lane_sha256'] == digest('other lane')
    assert ranked[0]['rerank_score'] == 1


def native(ids):
    return dict(task_id='t', eligible=True, basis='source_atom_exact_proxy', error=None,
                phases={'context': {'matched_reference_ids': ids}})


def choice_report(external, guarded_native=('a',), errors=False):
    rows = {p: [native(['a'])] for p in runner.PROFILES}
    rows['guarded'] = [native(list(guarded_native))]
    if errors:
        rows['guarded'][0]['error'] = {'type': 'Failure'}
    return dict(rows=rows, summaries={p: {'by_basis': {'upstream_word4_proxy': {
        'phases': {'context': {'delivered': external if p == 'guarded' else 4}}}}}
        for p in runner.PROFILES})


def test_adoption_is_strict_external_gain_and_per_reference_native_no_loss():
    assert runner.choose(choice_report(4)) == 'baseline'
    assert runner.choose(choice_report(5)) == 'guarded'
    assert runner.choose(choice_report(5, ('b', 'c'))) == 'baseline'
    assert runner.choose(choice_report(5, errors=True)) == 'baseline'


def public_original():
    source = original()
    source.update(task_id='t', question='synthetic query', trace={'status': 'completed'})
    # Complete rows, including original real lane and allocation annotations.
    for row in source['pools']['aspect']:
        row.update(retrieval_rank=row['rank'], bm25_rank=None, dense_rank=None,
                   fusion_score=None)
    source['pools']['baseline'] = deepcopy(source['pools']['aspect'])
    source['contexts'] = {'baseline': [deepcopy(source['pools']['aspect'][1])],
                          'aspect': [deepcopy(source['pools']['aspect'][1])]}
    public = {p: {'telemetry': {**runner.original_public_phases(source, p),
        'context_tokens': runner.replay.evidence_tokens(source['contexts'][p])}}
        for p in ('baseline', 'aspect')}
    return source, public


def validate_source(source, public, cost=None):
    return runner.validate_original(source, 't', 'synthetic query', public,
        cost if cost is not None else {'rankings': [{'trace_sha256': digest(source['ranking_traces'][0])}]},
        {'status': source['trace']['status']},
        runner.supplement.public_rows(source['pools']['aspect']),
        runner.supplement.public_rows(source['contexts']['baseline']))


@pytest.mark.parametrize('cost', [{'rankings': []}, {'rankings': [{}]},
                                {'rankings': [{'trace_sha256': '0' * 64}]}])
def test_original_score_cache_requires_prior_public_trace_binding(cost):
    source, public = public_original()
    with pytest.raises(ValueError, match='provenance'):
        validate_source(source, public, cost)


def test_changed_private_score_even_with_self_registered_bytes_is_not_original_provenance():
    source, public = public_original()
    cost = {'rankings': [{'trace_sha256': digest(source['ranking_traces'][0])}]}
    source['ranking_traces'][0]['aspect_ranking'][0]['scores'][0]['rerank_score'] = 3
    with pytest.raises(ValueError, match='provenance'):
        validate_source(source, public, cost)


@pytest.mark.parametrize('mutation', ['question', 'id', 'pool', 'allocation', 'tokens'])
def test_original_private_identity_and_safe_telemetry_are_bound(mutation):
    source, public = public_original()
    if mutation == 'question':
        source['question'] = 'changed'
    elif mutation == 'id':
        source['task_id'] = 'changed'
    elif mutation == 'pool':
        source['pools']['baseline'][0]['rank'] += 1
    elif mutation == 'allocation':
        source['contexts']['aspect'][0]['lane_sha256'] = digest('changed lane')
    else:
        public['aspect']['telemetry']['context_tokens'] += 1
    with pytest.raises(ValueError):
        validate_source(source, public)


def test_original_fallback_requires_recorded_known_reason_and_same_context():
    source, public = public_original()
    source['ranking_traces'] = []
    source['trace']['status'] = runner.FALLBACKS[0]
    observed = validate_source(source, public, {'rankings': []})
    assert observed.observed_pairs == 0
    source['trace']['status'] = 'unknown reason'
    with pytest.raises(ValueError, match='fallback'):
        validate_source(source, public, {'rankings': []})


def fixture_profiles():
    from app.evidence_focus import select_aspect_evidence
    from app.supplement_contract import build_supplement_contract
    question = "Describe Ari's memoir and roof commitment."
    contract = build_supplement_contract(question, [])
    rows = [candidate('a'), candidate('b', 2)]
    rows[0]['text'] = "Ari's memoir offers personal accounts."
    rows[1]['text'] = 'Ari made a roof commitment.'
    caps = dict(limit=8, token_budget=5000, document_quota=None, source_quota=2)
    chosen = select_aspect_evidence(contract, rows,
        ranker=lambda q, rs: [dict(r, rerank_score=-r['parent_rank']) for r in rs], **caps)
    source = {'pools': {'aspect': rows}, 'ranking_traces': [chosen.trace]}
    return question, contract, rows, caps, chosen, runner.ObservedScores.from_original(source)


def test_profiles_reproduce_old_allocation_and_leave_inputs_unchanged():
    question, contract, rows, caps, chosen, observed = fixture_profiles()
    before = deepcopy(rows)
    contexts, traces, costs, errors = runner.run_profiles(question, rows, [], contract,
        caps, observed, chosen.evidence, chosen.trace)
    assert errors == {}
    assert contexts['rank_floor'] == chosen.evidence
    assert rows == before
    assert set(contexts) == set(runner.PROFILES)
    assert traces['guarded']['coverage_type'].startswith('structural_literal')
    assert costs['rank_floor']['logical_ranker_pairs'] > 0
    assert observed.costs()['physical_ranker_calls'] == 0


def test_rank_floor_allocation_mismatch_blocks_guarded_observation():
    question, contract, rows, caps, chosen, observed = fixture_profiles()
    changed = deepcopy(chosen.evidence)
    changed[0]['lane_sha256'] = digest('new lane')
    contexts, traces, costs, errors = runner.run_profiles(question, rows, [], contract,
        caps, observed, changed, chosen.trace)
    assert contexts['rank_floor'] is None and contexts['guarded'] is None
    assert errors['guarded']['type'] == 'UnverifiedRankFloor'
    assert 'guarded' not in costs
    assert traces['guarded']['status'] == 'not_observed'


def test_missing_pair_fails_observation_and_does_not_call_new_ranker():
    question, contract, rows, caps, chosen, observed = fixture_profiles()
    observed.scores.clear()
    contexts, traces, costs, errors = runner.run_profiles(question, rows, [], contract,
        caps, observed, chosen.evidence, chosen.trace)
    assert errors['rank_floor']['type'] == 'ValueError'
    assert costs['rank_floor']['physical_ranker_pairs'] == 0
    assert observed.events[-1]['status'] == 'failed'


def test_source_fallback_is_reproduced_without_rank_calls():
    source, _ = public_original()
    baseline = source['contexts']['baseline']
    question = 'synthetic query'
    contract = SimpleNamespace(question=question, source_requests=[SimpleNamespace(
        resolution='unresolved', document_versions=())], authorized_bindings=[], reason=None)
    observed = runner.ObservedScores()
    contexts, traces, costs, errors = runner.run_profiles(question, source['pools']['aspect'],
        baseline, contract, {}, observed, baseline, None, runner.FALLBACKS[0])
    assert not errors
    assert contexts['guarded'] == baseline
    assert observed.events == []
    assert traces['rank_floor']['status'] == runner.FALLBACKS[0]


@pytest.mark.parametrize('message,registered,retained', [
    ('Too many question aspects (maximum 8)', True, True),
    ('Invalid aspect query length', True, True),
    ('Invalid aspect query length', False, False),
    ('Unexpected selector error', True, False)])
def test_only_registered_unsupported_aspect_fallback_is_retained(monkeypatch, message, registered, retained):
    question, contract, rows, caps, chosen, observed = fixture_profiles()
    def fail(*args, **kwargs):
        raise ValueError(message)
    monkeypatch.setattr(runner, 'select_aspect_evidence', fail)
    baseline = chosen.evidence
    contexts, traces, costs, errors = runner.run_profiles(question, rows, baseline, contract,
        caps, observed, baseline, None, runner.FALLBACKS[2] if registered else None)
    assert (contexts['rank_floor'] == baseline) is retained
    assert bool(errors) is not retained
    assert observed.events == []


def registered_args(tmp_path, monkeypatch):
    root = tmp_path / '.runtime'
    root.mkdir()
    contexts = root / 'original'
    contexts.mkdir()
    tasks = [dict(id=f't{i}', goal='SYNTHETIC QUESTION', user='synthetic') for i in range(47)]
    args = SimpleNamespace(context_root=str(contexts), registration=str(tmp_path / 'registration.json'),
        output=str(tmp_path / 'result.json'), private_root=str(root / 'new'))
    for name in ('suite', 'parent', 'baseline', 'repair', 'overlay', 'manifest'):
        path = tmp_path / (name + '.json')
        setattr(args, name, str(path))
        path.write_text(json.dumps({'split': 'dev', 'tasks': tasks} if name == 'suite' else {}))
    selection = tmp_path / 'selection.json'
    selection.write_text('{}')
    monkeypatch.setattr(runner, 'SELECTION', selection)
    monkeypatch.setattr(runner, 'private_root', lambda path: Path(path).resolve())
    monkeypatch.setattr(runner, 'code_fingerprints', lambda: {'code.py': 'a' * 64})
    monkeypatch.setattr(runner, 'settings', lambda: SimpleNamespace(model_dump=lambda **kwargs: {'flag': True}))
    for task in tasks:
        (contexts / (task['id'] + '.json')).write_text(json.dumps({'task_id': task['id'], 'text': 'PRIVATE SOURCE'}))
    return args, tasks


def test_registration_freezes_each_private_file_code_settings_and_caps(tmp_path, monkeypatch):
    args, tasks = registered_args(tmp_path, monkeypatch)
    declaration = runner.registration_for(args)
    assert declaration['profiles'] == list(runner.PROFILES)
    assert declaration['selected_task_ids'] == [t['id'] for t in tasks]
    assert len(declaration['original_private_bytes_sha256']) == 47
    assert all(v == 0 for v in declaration['budget'].values())
    assert declaration['caps']['document_quota'] is None
    assert declaration['settings_sha256'] == digest({'flag': True})
    assert 'PRIVATE SOURCE' not in json.dumps(declaration)
    path = Path(args.context_root) / 't0.json'
    path.write_text(path.read_text() + '\n')
    changed = runner.registration_for(args)
    assert changed['original_private_bytes_sha256']['t0'] != declaration['original_private_bytes_sha256']['t0']
    assert changed['original_private_bytes_sha256']['t1'] == declaration['original_private_bytes_sha256']['t1']


def test_registration_refuses_missing_private_task_and_non47_suite(tmp_path, monkeypatch):
    args, tasks = registered_args(tmp_path, monkeypatch)
    path = Path(args.context_root) / 't0.json'
    path.unlink()
    with pytest.raises(ValueError, match='private'):
        runner.registration_for(args)
    path.write_text('{}')
    Path(args.suite).write_text(json.dumps({'split': 'dev', 'tasks': tasks[:46]}))
    with pytest.raises(ValueError, match='Dev47'):
        runner.registration_for(args)


def test_registration_refuses_private_symlink_escape(tmp_path, monkeypatch):
    args, _ = registered_args(tmp_path, monkeypatch)
    path = Path(args.context_root) / 't0.json'
    path.unlink()
    outside = tmp_path / 'outside.json'
    outside.write_text('{}')
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='escaping'):
        runner.registration_for(args)


def repair_fixture():
    ids = [f't{i}' for i in range(47)]
    suite = {'tasks': [{'id': tid} for tid in ids]}
    manifest = {'source_free': True}
    models = {'embedding': {'name': 'synthetic'}, 'reranker': {'name': 'cached'}}
    parent = {'identity': {'corpus_sha256': 'corpus', 'source_index_sha256': 'index', 'models': models}}
    seed_rows = [{'task_id': tid, 'error': None, 'telemetry': {
        'candidates': [], 'admitted': [], 'context': [], 'context_tokens': 0}} for tid in ids]
    baseline = {'rows': {'relevant_doc': seed_rows}}
    safe = [dict(row, telemetry=dict(row['telemetry'], candidate_count=0, admitted_count=0, context_count=0))
            for row in seed_rows]
    repair = {'version': 'external-repair-dev-v1', 'valid': True, 'status': 'complete',
        'split': 'dev', 'subset': False, 'error': None, 'drift': {'inputs': False},
        'selected_task_ids': ids, 'run_id': 'a' * 32,
        'rows': {p: deepcopy(safe) for p in ('baseline', 'aspect', 'control', 'treatment')},
        'traces': [{'task_id': tid, 'errors': []} for tid in ids],
        'costs': [{'task_id': tid} for tid in ids], 'identity': {
            'parent_sha256': digest(parent), 'annotation_manifest': manifest,
            'corpus_sha256': 'corpus', 'index_sha256': 'index',
            'models': {'embedding': {'name': 'synthetic', 'status': 'cached'}},
            'reranker': models['reranker']}}
    return repair, parent, baseline, suite, manifest


def test_repair_baseline_binds_shared_telemetry_and_checks_recorded_counts():
    fixture = repair_fixture()
    runner.validate_repair(*fixture)
    fixture[0]['rows']['baseline'][0]['telemetry']['candidate_count'] = 1
    with pytest.raises(ValueError):
        runner.validate_repair(*fixture)


@pytest.mark.parametrize('mutation', ['drift', 'missing_drift', 'null_drift', 'failed',
                                     'parent', 'corpus', 'index', 'models', 'subset',
                                     'rows', 'cost_order', 'baseline_context'])
def test_repair_rejects_incomplete_conflicting_or_drifted_identity(mutation):
    fixture = repair_fixture()
    repair = fixture[0]
    if mutation == 'drift':
        repair['drift']['inputs'] = True
    elif mutation == 'missing_drift':
        repair['drift'] = {}
    elif mutation == 'null_drift':
        repair['drift']['inputs'] = None
    elif mutation == 'failed':
        repair['status'] = 'failed'
    elif mutation == 'parent':
        repair['identity']['parent_sha256'] = 'changed'
    elif mutation in ('corpus', 'index'):
        repair['identity'][mutation + '_sha256'] = 'changed'
    elif mutation == 'models':
        repair['identity']['reranker'] = {'name': 'changed'}
    elif mutation == 'subset':
        repair['subset'] = True
    elif mutation == 'rows':
        repair['rows']['aspect'].pop()
    elif mutation == 'cost_order':
        repair['costs'].reverse()
    else:
        repair['rows']['baseline'][0]['telemetry']['context'] = [{'chunk_id': 'changed'}]
    with pytest.raises(ValueError):
        runner.validate_repair(*fixture)


def execution_fixture(tmp_path, monkeypatch):
    """Synthetic complete 47 inputs with API/DB clients replaced by local fakes."""
    from app import clients, db as database
    args, tasks = registered_args(tmp_path, monkeypatch)
    question, contract, pool, caps, chosen, _ = fixture_profiles()
    for task in tasks:
        task.update(goal=question, evaluator_only='SYNTHETIC GOLD DO NOT PASS TO POLICY')
    suite = {'split': 'dev', 'tasks': tasks}
    parent = {'identity': {'corpus': {}, 'source_index': {'index': 'frozen'},
        'corpus_sha256': digest({}), 'source_index_sha256': digest({'index': 'frozen'}),
        'models': {'embedding': {'name': 'cached'}, 'reranker': {'name': 'cached'}}}}
    originals = []
    baseline_rows, aspect_rows = [], []
    costs, traces = [], []
    for task in tasks:
        source = dict(task_id=task['id'], question=question, ranking_traces=[chosen.trace],
            pools={p: deepcopy(pool) for p in ('baseline', 'aspect')},
            contexts={'baseline': [], 'aspect': chosen.evidence}, trace={'status': 'completed'})
        Path(args.context_root, task['id'] + '.json').write_text(json.dumps(source))
        public = {p: {'task_id': task['id'], 'error': None, 'telemetry': {
            **runner.original_public_phases(source, p),
            'context_tokens': runner.replay.evidence_tokens(source['contexts'][p]),
            'candidate_count': len(pool), 'admitted_count': len(source['contexts'][p]),
            'context_count': len(source['contexts'][p])}} for p in ('baseline', 'aspect')}
        baseline_rows.append(public['baseline'])
        aspect_rows.append(public['aspect'])
        costs.append({'task_id': task['id'], 'rankings': [{'trace_sha256': digest(chosen.trace)}]})
        traces.append({'task_id': task['id'], 'status': 'completed', 'errors': []})
        originals.append({'telemetry': public['baseline']['telemetry']})
    baseline = {'rows': {'relevant_doc': baseline_rows}}
    repair = {'version': 'external-repair-dev-v1', 'run_id': 'b' * 32,
        'valid': True, 'status': 'complete', 'split': 'dev', 'subset': False,
        'error': None, 'drift': {'inputs': False}, 'selected_task_ids': [t['id'] for t in tasks],
        'rows': {'baseline': baseline_rows, 'aspect': aspect_rows,
                 'control': aspect_rows, 'treatment': aspect_rows}, 'costs': costs, 'traces': traces,
        'identity': {'parent_sha256': digest(parent), 'annotation_manifest': {},
                     'corpus_sha256': parent['identity']['corpus_sha256'],
                     'index_sha256': parent['identity']['source_index_sha256'],
                     'models': {'embedding': {'name': 'cached'}}, 'reranker': {'name': 'cached'}}}
    for name, value in (('suite', suite), ('parent', parent), ('baseline', baseline), ('repair', repair)):
        Path(getattr(args, name)).write_text(json.dumps(value))
    monkeypatch.setattr(runner, 'validate_suite', lambda *args: tasks)
    monkeypatch.setattr(runner.replay, 'validate_parent', lambda *args: originals)
    monkeypatch.setattr(runner, 'validate_seed', lambda *args: 'relevant_doc')
    monkeypatch.setattr(runner, 'validate_annotation', lambda *args: {
        'task_records': {t['id']: {} for t in tasks}})
    monkeypatch.setattr(runner.replay, 'rebind_rows', lambda db, user, rows, corpus: deepcopy(rows))
    monkeypatch.setattr(runner.replay, 'rehydrate_task', lambda *args: {
        'candidates': deepcopy(pool), 'admitted': [], 'context': []})
    monkeypatch.setattr(runner.replay, 'policy_caps', lambda *args: deepcopy(caps))
    monkeypatch.setattr(runner, 'contract_for', lambda *args: contract)
    monkeypatch.setattr(runner, 'active_scope', lambda db: ({'synthetic': object()}, [], {}))
    monkeypatch.setattr(runner, 'index_ledger', lambda *args: {'index': 'frozen'})
    monkeypatch.setattr(runner, 'summarize_revised', runner.summarize)
    monkeypatch.setattr(runner, 'missing_revised', lambda task, record, error: runner.missing_row(task, error))
    class FakeDB:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def execute(self, query):
            assert str(query) == 'SET TRANSACTION READ ONLY'
    monkeypatch.setattr(database, 'SessionLocal', FakeDB)
    class LedgerOnlySearch:
        def __init__(self, **kwargs):
            assert kwargs == {'index': 'frozen'}
        def request(self, *args, **kwargs):
            pytest.fail('No search or model request allowed in replay fixture')
    monkeypatch.setattr(clients, 'Search', LedgerOnlySearch)
    calls = []
    real_profiles = runner.run_profiles
    def tracked_profiles(*args, **kwargs):
        calls.append(args[0])
        assert isinstance(args[0], str)
        assert all('evaluator_only' not in row for row in args[1])
        return real_profiles(*args, **kwargs)
    monkeypatch.setattr(runner, 'run_profiles', tracked_profiles)
    def evaluate(db, user, task, raw, corpus, record):
        assert len(calls) == 47, 'Evaluator starts only after all policy runtime ends'
        return runner.diagnose(task, raw['telemetry']) | {'error': raw['error'], 'telemetry': raw['telemetry']}
    monkeypatch.setattr(runner, 'rescore', evaluate)
    Path(args.registration).write_text(json.dumps(runner.registration_for(args)))
    return args, tasks, calls, tracked_profiles


def test_execute_full47_all_profiles_cached_with_evaluator_after_runtime(tmp_path, monkeypatch):
    args, tasks, calls, _ = execution_fixture(tmp_path, monkeypatch)
    report = runner.execute(args, {'run_id': 'fresh'})
    assert report['valid'] is True
    assert report['status'] == 'complete'
    assert len(calls) == 47
    assert report['selected_task_ids'] == [t['id'] for t in tasks]
    assert all(len(rows) == 47 for rows in report['rows'].values())
    assert report['strict_accuracy'] is None and report['semantic_recall'] is None
    assert all(c['physical_ranker_calls'] == 0 and c['search_calls'] == 0 for c in report['costs'])
    assert all(c['logical_ranker_pairs'] > 0 for c in report['costs'])
    assert report['costs'][0]['prior_collection']['observed_pairs'] > 0
    assert 'SYNTHETIC GOLD' not in Path(args.output).read_text()
    assert "Ari's memoir" not in Path(args.output).read_text()
    assert Path(args.private_root, 'fresh', 't0.json').exists()


def test_execute_failed_task_keeps_all47_rows_and_later_observations(tmp_path, monkeypatch):
    args, tasks, calls, profiles = execution_fixture(tmp_path, monkeypatch)
    def fail_one(*args, **kwargs):
        if len(calls) == 2:
            calls.append(args[0])
            raise ValueError('Synthetic operational failure')
        return profiles(*args, **kwargs)
    monkeypatch.setattr(runner, 'run_profiles', fail_one)
    report = runner.execute(args, {'run_id': 'failed'})
    assert report['valid'] is False and len(calls) == 47
    for rows in report['rows'].values():
        assert len(rows) == 47
        assert rows[2]['error'] is not None
        assert rows[2]['telemetry'] is None or rows[2]['telemetry']['context'] is None
        assert rows[3]['error'] is None
    assert report['traces'][2]['status'] == 'failed'
    assert report['costs'][2]['physical_ranker_calls'] == 0


def test_execute_interrupted_saves_private_partial_and_null_full47(tmp_path, monkeypatch):
    args, tasks, calls, profiles = execution_fixture(tmp_path, monkeypatch)
    def interrupt(*args, **kwargs):
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return profiles(*args, **kwargs)
    monkeypatch.setattr(runner, 'run_profiles', interrupt)
    report = runner.execute(args, {'run_id': 'interrupted'})
    assert report['valid'] is False and report['status'] == 'interrupted'
    assert all(len(rows) == 47 for rows in report['rows'].values())
    assert all(r['error'] is not None for rows in report['rows'].values() for r in rows)
    assert Path(args.private_root, 'interrupted', 't2.json').exists()
    assert report['costs'][2]['status'] == 'interrupted'
    assert report['costs'][-1]['logical_ranker_pairs'] is None
    assert report['costs'][-1]['physical_ranker_calls'] == 0


def test_preflight_missing_other_input_retains_known_full47_denominator(tmp_path, monkeypatch):
    args, tasks, calls, _ = execution_fixture(tmp_path, monkeypatch)
    Path(args.repair).unlink()
    report = runner.execute(args, {'run_id': 'preflight'})
    assert report['valid'] is False and calls == []
    assert report['selected_task_ids'] == [t['id'] for t in tasks]
    assert all(len(rows) == 47 for rows in report['rows'].values())
    assert all(r['error'] is not None for rows in report['rows'].values() for r in rows)


def test_preflight_unvalidated_suite_cannot_supply_accepted_dev_positions(tmp_path, monkeypatch):
    args, tasks, calls, _ = execution_fixture(tmp_path, monkeypatch)
    def reject(*args):
        raise ValueError('Synthetic untrusted suite')
    monkeypatch.setattr(runner, 'validate_suite', reject)
    report = runner.execute(args, {'run_id': 'untrusted'})
    assert report['valid'] is False and calls == []
    assert report['selected_task_ids'] == []
    assert all(rows == [] for rows in report['rows'].values())


def eligible_task(task_id='eligible'):
    return {'id': task_id, 'goal': 'SYNTHETIC QUESTION', 'gold_evidence': [{
        'id': 'synthetic-ref', 'document_id': 'doc', 'source_sha256': 'hash',
        'quote': 'A synthetic reference span.'}]}


def test_summary_unobserved_eligible_delivery_is_null_not_zero():
    row = runner.missing_row(eligible_task(), {'type': 'NotObserved', 'stage': 'runtime'})
    summary = runner.selection_summaries([row])
    context = summary['by_basis']['source_span_char4_proxy']['phases']['context']
    assert context['measured_tasks'] == 0 and context['missing_tasks'] == 1
    assert context['reference_denominator'] is None
    assert context['delivered'] is None and context['all_delivered_tasks'] is None
    assert context['proxy_coverage'] is None
    assert context['measurement_status'] == 'pending'
    category = next(iter(summary['by_basis']['source_span_char4_proxy']['by_category'].values()))
    assert category['phases']['context']['delivered'] is None


def test_partial_summary_reports_only_observed_subtotal_and_pending_full_rate():
    task = eligible_task('observed')
    rows = [runner.diagnose(task, {p: [] for p in ('candidates', 'admitted', 'context')}) |
            {'error': None}, runner.missing_row(eligible_task('missing'), {'type': 'NotObserved'})]
    summary = runner.selection_summaries(rows)
    context = summary['by_basis']['source_span_char4_proxy']['phases']['context']
    assert context['measured_tasks'] == 1 and context['missing_tasks'] == 1
    assert context['measurement_status'] == 'partial'
    assert context['proxy_coverage'] is None and context['delivered'] is None
    assert context['observed_subtotal']['delivered'] == 0
    assert context['observed_subtotal']['reference_denominator'] == 1
    assert context['observed_subtotal']['proxy_coverage'] == 0.0
    complete = runner.selection_summaries(rows[:1])['by_basis']['source_span_char4_proxy']['phases']['context']
    assert complete['delivered'] == 0 and complete['reference_denominator'] == 1
    assert complete['proxy_coverage'] == 0.0


def test_revised_missing_sensitivity_totals_are_null_with_observed_zero_preserved(monkeypatch):
    aggregate = {'by_basis': {'synthetic': {'phases': {}, 'sensitivity_including_unclear': {
        'context': {'reference_denominator': 1, 'delivered': 0, 'proxy_coverage': 0.0,
                    'measured_tasks': 1, 'missing_tasks': 1},
        'candidates': {'reference_denominator': 0, 'delivered': 0, 'proxy_coverage': None,
                       'measured_tasks': 0, 'missing_tasks': 2}}}}}
    monkeypatch.setattr(runner, 'summarize_revised', lambda rows: deepcopy(aggregate))
    summary = runner.selection_summaries([], revised=True)['by_basis']['synthetic']
    context = summary['sensitivity_including_unclear']['context']
    assert context['delivered'] is None and context['proxy_coverage'] is None
    assert context['observed_subtotal']['delivered'] == 0
    assert context['measurement_status'] == 'partial'
    candidates = summary['sensitivity_including_unclear']['candidates']
    assert candidates['reference_denominator'] is None and candidates['delivered'] is None
    assert candidates['measurement_status'] == 'pending'


def test_interrupted_task_retains_trace_error_and_existing_private_allocations(tmp_path, monkeypatch):
    args, tasks, calls, profiles = execution_fixture(tmp_path, monkeypatch)
    def interrupt_after_allocation(*params, **kwargs):
        result = profiles(*params, **kwargs)
        if len(calls) == 3:
            raise KeyboardInterrupt()
        return result
    monkeypatch.setattr(runner, 'run_profiles', interrupt_after_allocation)
    report = runner.execute(args, {'run_id': 'partial'})
    trace = report['traces'][2]
    assert trace['status'] == 'interrupted'
    assert trace['errors'] == [{'type': 'KeyboardInterrupt', 'stage': 'selection_runtime'}]
    saved = json.loads(Path(args.private_root, 'partial', 't2.json').read_text())
    assert saved['contexts']['rank_floor'] and saved['contexts']['guarded']
    assert saved['ranker_events']
    assert saved['profile_costs']['guarded']['logical_ranker_pairs'] > 0
    assert report['costs'][2]['profiles']['guarded'] == saved['profile_costs']['guarded']
    assert report['costs'][2]['logical_ranker_pairs'] > 0
    assert report['costs'][2]['status'] == 'interrupted'


@pytest.mark.parametrize('target,field', [('private', 'original_private'), ('registration', 'registration'),
                                       ('suite', 'inputs')])
def test_execute_checks_frozen_bytes_again_after_runtime(tmp_path, monkeypatch, target, field):
    args, tasks, calls, profiles = execution_fixture(tmp_path, monkeypatch)
    def mutate(*params, **kwargs):
        if len(calls) == 2:
            path = Path(args.context_root, 't0.json') if target == 'private' else Path(getattr(args, target))
            path.write_text(path.read_text() + '\n')
        return profiles(*params, **kwargs)
    monkeypatch.setattr(runner, 'run_profiles', mutate)
    report = runner.execute(args, {'run_id': 'drift'})
    assert report['valid'] is False and report['status'] == 'invalid_drift'
    assert report['drift'][field] is True
    assert all(len(rows) == 47 for rows in report['rows'].values())


def test_output_and_reservation_are_exclusive_even_for_failed_reports(tmp_path, monkeypatch):
    from scripts import run_retrieval_quality_dev as common
    monkeypatch.setattr(common, 'RUNTIME', tmp_path / 'runtime')
    output = tmp_path / 'result.json'
    runner.reserve_output(output)
    with pytest.raises(ValueError, match='reserved'):
        runner.reserve_output(output)
    runner._publish(output, {'status': 'failed'})
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        runner._publish(output, {'status': 'complete'})
    with pytest.raises(ValueError, match='exists'):
        runner.reserve_output(output)
    assert output.read_bytes() == before


def fresh_cache():
    source = original()
    cache = runner.ObservedScores.from_original(source)
    return source['pools']['aspect'], cache


def fake_fresh(query, rows):
    return sorted([dict(deepcopy(row), rerank_score=10 - row['rank']) for row in rows],
                  key=lambda row: -row['rerank_score'])


def test_fresh_optin_only_fills_missing_pairs_once_and_preserves_known_scores():
    rows, cache = fresh_cache()
    known = deepcopy(cache.scores)
    missing = candidate('c', 3)
    calls = []
    def scorer(query, supplied):
        calls.append((query, deepcopy(supplied)))
        return fake_fresh(query, supplied)
    ranked = runner.replay_rank('synthetic query', [*rows, missing], cache,
                                fresh_ranking=True, ranker=scorer)
    runner.replay_rank('synthetic query', [*rows, missing], cache,
                       fresh_ranking=True, ranker=scorer)
    assert len(calls) == 1 and calls[0] == ('synthetic query', [missing])
    assert all(cache.scores[key] == value for key, value in known.items())
    assert {row['chunk_id']: row['rerank_score'] for row in ranked} == {'a': 1, 'b': 2, 'c': 7}
    assert cache.costs()['physical_ranker_calls'] == 1
    assert cache.costs()['physical_ranker_pairs'] == 1
    assert cache.costs()['cache_hits'] == 5


@pytest.mark.parametrize('changed', ['query', 'source_sha256', 'text', 'title', 'version_id'])
def test_fresh_identity_requires_exact_query_and_source_bytes(changed):
    rows, cache = fresh_cache()
    row = deepcopy(rows[0])
    query = 'synthetic query'
    if changed == 'query':
        query += ' changed'
    else:
        row[changed] += ' changed'
    ranked = runner.replay_rank(query, [row], cache, fresh_ranking=True, ranker=fake_fresh)
    assert ranked[0]['rerank_score'] == 8
    assert cache.costs()['physical_ranker_pairs'] == 1
    assert cache.scores[runner.score_key('synthetic query', rows[0])] == 1


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'nan', 'mutated', 'input_mutated', 'order'])
def test_fresh_invalid_output_never_enters_cache_and_keeps_failure_event(bad):
    rows, cache = fresh_cache()
    before = deepcopy(cache.scores)
    def scorer(query, supplied):
        output = fake_fresh(query, supplied)
        if bad == 'missing':
            output.pop()
        elif bad == 'duplicate':
            output[1] = deepcopy(output[0])
        elif bad == 'nan':
            output[0]['rerank_score'] = float('nan')
        elif bad == 'mutated':
            output[0]['text'] += ' tampered'
        elif bad == 'input_mutated':
            supplied[0]['metadata']['nested'].append(5)
        else:
            output.reverse()
        return output
    with pytest.raises(ValueError, match='ranker'):
        runner.replay_rank('fresh query', rows, cache, fresh_ranking=True, ranker=scorer)
    assert cache.scores == before
    costs = cache.costs()
    assert costs['physical_ranker_calls'] == 1 and costs['physical_ranker_pairs'] == 2
    assert costs['fresh_failed'] == 1 and costs['fresh_completed'] == 0
    assert costs['fresh_latency_ms'] is None and costs['fresh_token_usage'] is None
    assert cache.events[-1]['status'] == 'failed'
    assert cache.events[-1]['wall_ms'] >= 0
    assert cache.provider_events[-1]['query'] == 'fresh query'
    assert cache.provider_events[-1]['input_rows'] == rows
    assert cache.provider_events[-1]['status'] == 'failed'


@pytest.mark.parametrize('budget', ['calls', 'pairs'])
def test_fresh_budget_is_reserved_before_dispatch_and_exhaustion_prevents_call(budget):
    rows, cache = fresh_cache()
    if budget == 'calls':
        cache.max_new_calls = 1
    else:
        cache.max_new_pairs = 2
    runner.replay_rank('fresh query', rows, cache, fresh_ranking=True, ranker=fake_fresh)
    with pytest.raises(ValueError, match='budget'):
        runner.replay_rank('another fresh query', rows, cache, fresh_ranking=True,
                           ranker=lambda *args: pytest.fail('exhausted dispatch'))
    assert cache.costs()['physical_ranker_calls'] == 1
    assert cache.costs()['physical_ranker_pairs'] == 2


def test_fresh_keyboardinterrupt_preserves_attempt_budget_and_private_partial():
    rows, cache = fresh_cache()
    cache.max_new_calls = 1
    def interrupt(query, supplied):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        runner.replay_rank('fresh query', rows, cache, fresh_ranking=True, ranker=interrupt)
    assert cache.costs()['physical_ranker_calls'] == 1
    assert cache.costs()['physical_ranker_pairs'] == 2
    assert cache.events[-1]['status'] == 'failed'
    assert cache.events[-1]['error'] == {'type': 'KeyboardInterrupt'}
    assert cache.provider_events[-1]['status'] == 'failed'
    assert cache.provider_events[-1]['output_rows'] is None
    with pytest.raises(ValueError, match='budget'):
        runner.replay_rank('retry query', rows, cache, fresh_ranking=True, ranker=fake_fresh)


def test_fresh_public_events_expose_hashes_and_private_events_keep_actual_scores():
    rows, cache = fresh_cache()
    runner.replay_rank('PRIVATE QUERY', rows, cache, fresh_ranking=True, ranker=fake_fresh)
    public = cache.costs()
    assert 'PRIVATE' not in json.dumps(public)
    assert 'Synthetic' not in json.dumps(public)
    event = public['fresh_events'][0]
    assert event['query_sha256'] == digest('PRIVATE QUERY')
    assert event['new_pair_key_sha256'] == [runner.score_key('PRIVATE QUERY', row) for row in rows]
    assert cache.provider_events[0]['output_rows'][0]['rerank_score'] == 9
    assert cache.provider_events[0]['input_rows'] == rows


def fake_identity(*args, **kwargs):
    assert kwargs == {'rerank': True}
    return {'reranker': {'name': 'fake-bge', 'status': 'active', 'revision': 'local-revision',
                        'snapshot_path_sha256': 'local-snapshot', 'digest': 'model-digest',
                        'batch': 8, 'max_tokens': 512,
                        'files': {'config.json': 'cfg', 'tokenizer.json': 'tok', 'weights.bin': 'w'}},
            'libraries': {'transformers': 'fake', 'torch': 'fake'}}


def enable_fresh_fixture(args, monkeypatch):
    args.fresh_ranking = True
    monkeypatch.setattr(runner, 'model_identity', fake_identity)
    monkeypatch.setattr(runner, 'rerank', lambda query, rows, **kwargs: fake_fresh(query, rows))
    # Cached and fresh synthetic scores must describe the same model/runtime.
    models = fake_identity(rerank=True)
    parent = json.loads(Path(args.parent).read_bytes())
    repair = json.loads(Path(args.repair).read_bytes())
    parent['identity']['models'].update(reranker=deepcopy(models['reranker']),
                                       libraries=deepcopy(models['libraries']))
    repair['identity'].update(reranker=deepcopy(models['reranker']), parent_sha256=digest(parent))
    Path(args.parent).write_text(json.dumps(parent))
    Path(args.repair).write_text(json.dumps(repair))
    Path(args.registration).write_text(json.dumps(runner.registration_for(args)))


def test_fresh_registration_v2_freezes_model_libraries_and_original_candidate_budget(tmp_path, monkeypatch):
    args, tasks, _, _ = execution_fixture(tmp_path, monkeypatch)
    enable_fresh_fixture(args, monkeypatch)
    registration = runner.registration_for(args)
    assert registration['version'] == 'guarded-selection-input-v2'
    assert registration['budget']['physical_ranker_calls'] == 8
    assert registration['budget']['physical_ranker_pairs_by_task'] == {t['id']: 16 for t in tasks}
    assert registration['reranker_identity']['revision'] == 'local-revision'
    assert registration['reranker_identity']['libraries'] == {'transformers': 'fake', 'torch': 'fake'}
    assert all(registration['budget'][key] == 0 for key in
               ('search_calls', 'embedding_calls', 'policy_calls', 'generation_calls'))
    assert 'SYNTHETIC QUESTION' not in json.dumps(registration)


def test_fresh_execution_with_zero_dispatch_preserves_full47_and_new_identity(tmp_path, monkeypatch):
    args, tasks, calls, _ = execution_fixture(tmp_path, monkeypatch)
    enable_fresh_fixture(args, monkeypatch)
    report = runner.execute(args, {'run_id': 'fresh-v2'})
    assert report['valid'] is True and report['status'] == 'complete'
    assert report['version'] == 'guarded-selection-dev-v2'
    assert report['drift']['reranker_identity'] is False
    assert len(calls) == 47
    assert all(c['physical_ranker_calls'] == 0 and c['fresh_token_usage'] == 0 for c in report['costs'])
    assert all(c['profiles']['guarded']['physical_ranker_calls'] == 0 for c in report['costs'])


@pytest.mark.parametrize('drift', ['model', 'corpus', 'index'])
def test_fresh_execution_readonly_identity_drift_invalidates_full47(tmp_path, monkeypatch, drift):
    args, tasks, calls, profiles = execution_fixture(tmp_path, monkeypatch)
    enable_fresh_fixture(args, monkeypatch)
    if drift == 'model':
        def changed(*params, **kwargs):
            result = profiles(*params, **kwargs)
            monkeypatch.setattr(runner, 'model_identity', lambda *args, **kwargs: {
                'reranker': {'name': 'drifted'}, 'libraries': {}})
            return result
        monkeypatch.setattr(runner, 'run_profiles', changed)
    elif drift == 'corpus':
        count = []
        def scope(db):
            count.append(True)
            return {'synthetic': object()}, [], {} if len(count) == 1 else {'changed': True}
        monkeypatch.setattr(runner, 'active_scope', scope)
    else:
        count = []
        def index(*args):
            count.append(True)
            return {'index': 'frozen' if len(count) == 1 else 'changed'}
        monkeypatch.setattr(runner, 'index_ledger', index)
    report = runner.execute(args, {'run_id': 'drift-v2'})
    assert report['valid'] is False and report['status'] == 'invalid_drift'
    assert report['drift']['reranker_identity' if drift == 'model' else drift] is True
    assert all(len(rows) == 47 for rows in report['rows'].values())


def test_fresh_guarded_dispatch_follows_strict_rank_floor_and_shares_task_cache(monkeypatch):
    question, contract, rows, caps, chosen, observed = fixture_profiles()
    original_selector = runner.select_aspect_evidence
    calls = []
    before = deepcopy(rows)
    private = {}
    def scorer(query, supplied, **kwargs):
        assert query == 'fresh synthetic aspect'
        assert kwargs == {'window': len(supplied)}
        calls.append(deepcopy(supplied))
        return fake_fresh(query, supplied)
    def selector(contract, candidates, **kwargs):
        if kwargs['selection_policy'] == 'guarded':
            assert private['contexts']['rank_floor'] == chosen.evidence
            kwargs['ranker']('fresh synthetic aspect', candidates)
            kwargs['ranker']('fresh synthetic aspect', candidates)
        return original_selector(contract, candidates, **kwargs)
    monkeypatch.setattr(runner, 'rerank', scorer)
    monkeypatch.setattr(runner, 'select_aspect_evidence', selector)
    contexts, traces, costs, errors = runner.run_profiles(question, rows, [], contract,
        caps, observed, chosen.evidence, chosen.trace, private=private, fresh_ranking=True)
    assert not errors and len(calls) == 1
    assert costs['rank_floor']['physical_ranker_calls'] == 0
    assert costs['guarded']['physical_ranker_calls'] == 1
    assert costs['guarded']['physical_ranker_pairs'] == len(rows)
    assert traces['guarded']['ranking_usage']['fresh_completed'] == 1
    assert len(private['provider_events']) == 1 and private['provider_events'][0]['input_rows'] == rows
    assert contexts['rank_floor'] == chosen.evidence and rows == before


def test_fresh_optin_cannot_rescue_missing_rank_floor_observations(monkeypatch):
    question, contract, rows, caps, chosen, observed = fixture_profiles()
    observed.scores.clear()
    monkeypatch.setattr(runner, 'rerank', lambda *args, **kwargs: pytest.fail('untrusted rank floor'))
    contexts, traces, costs, errors = runner.run_profiles(question, rows, [], contract,
        caps, observed, chosen.evidence, chosen.trace, fresh_ranking=True)
    assert contexts['guarded'] is None and errors['guarded']['type'] == 'UnverifiedRankFloor'
    assert observed.costs()['physical_ranker_calls'] == 0


def test_fresh_default_budget_caps_eight_calls_and_eight_times_unique_original_pool():
    rows, cache = fresh_cache()
    for index in range(8):
        runner.replay_rank(f'fresh query {index}', rows, cache,
                           fresh_ranking=True, ranker=fake_fresh)
    assert cache.new_calls == 8 and cache.new_pairs == 16
    assert cache.max_new_pairs == 16
    with pytest.raises(ValueError, match='budget'):
        runner.replay_rank('fresh query 9', rows[:1], cache, fresh_ranking=True,
                           ranker=lambda *args: pytest.fail('ninth dispatch'))


@pytest.mark.parametrize('field,value', [('source_sha256', 'tampered'), ('title', 'tampered'),
    ('document_id', 'tampered'), ('lane_type', 'tampered'), ('lane_sha256', 'tampered'),
    ('admitted', False), ('metadata', {}), ('locator', {})])
def test_fresh_rejects_output_identity_scope_and_lane_mutations(field, value):
    rows, cache = fresh_cache()
    def scorer(query, supplied):
        output = fake_fresh(query, supplied)
        output[0][field] = value
        return output
    with pytest.raises(ValueError, match='ranker'):
        runner.replay_rank('fresh query', rows, cache, fresh_ranking=True, ranker=scorer)
    assert len(cache.scores) == 2


def test_fresh_original_multiple_lane_occurrences_keep_original_pool_and_unique_pair_cap():
    source = original()
    source['pools']['aspect'].append(dict(deepcopy(source['pools']['aspect'][0]),
        lane_type='source', lane_sha256=digest('other lane')))
    before = deepcopy(source)
    cache = runner.ObservedScores.from_original(source)
    ranked = runner.replay_rank('fresh query', [source['pools']['aspect'][-1]], cache,
                                fresh_ranking=True, ranker=fake_fresh)
    assert cache.max_new_pairs == 16
    assert source == before and len(source['pools']['aspect']) == 3
    assert ranked[0]['lane_type'] == 'source'
    ranked2 = runner.replay_rank('fresh query', source['pools']['aspect'][:1], cache,
                                 fresh_ranking=True, ranker=lambda *args: pytest.fail('same source bytes'))
    assert ranked2[0]['lane_type'] == 'global' and cache.new_calls == 1


def test_fresh_execute_private_partial_and_public_failure_cost_survive_interrupt(tmp_path, monkeypatch):
    args, tasks, calls, _ = execution_fixture(tmp_path, monkeypatch)
    enable_fresh_fixture(args, monkeypatch)
    original_selector = runner.select_aspect_evidence
    def interrupt(query, rows, **kwargs):
        raise KeyboardInterrupt()
    def selector(contract, candidates, **kwargs):
        if kwargs['selection_policy'] == 'guarded':
            kwargs['ranker']('PRIVATE INTERRUPTED QUERY', candidates)
        return original_selector(contract, candidates, **kwargs)
    monkeypatch.setattr(runner, 'rerank', interrupt)
    monkeypatch.setattr(runner, 'select_aspect_evidence', selector)
    report = runner.execute(args, {'run_id': 'fresh-interrupted'})
    assert report['status'] == 'interrupted' and report['valid'] is False
    assert all(len(rows) == 47 for rows in report['rows'].values())
    cost = report['costs'][0]
    assert cost['physical_ranker_calls'] == 1 and cost['physical_ranker_pairs'] == 2
    assert cost['fresh_failed'] == 1 and cost['fresh_token_usage'] is None
    assert cost['profiles']['guarded']['status'] == 'interrupted'
    assert 'PRIVATE INTERRUPTED QUERY' not in Path(args.output).read_text()
    private = json.loads(Path(args.private_root, 'fresh-interrupted', 't0.json').read_text())
    assert private['provider_events'][0]['query'] == 'PRIVATE INTERRUPTED QUERY'
    assert private['provider_events'][0]['status'] == 'failed'
    assert private['contexts']['rank_floor']
    assert report['costs'][-1]['physical_ranker_calls'] == 0


MODEL_COMPATIBILITY_FIELDS = ('name', 'status', 'revision', 'snapshot_path_sha256',
                              'files', 'digest', 'batch', 'max_tokens')


def alter_cached_identity(args, field, target='both'):
    parent = json.loads(Path(args.parent).read_bytes())
    repair = json.loads(Path(args.repair).read_bytes())
    if field == 'libraries':
        parent['identity']['models']['libraries']['transformers'] = 'different-library'
    else:
        value = {'weights.bin': 'different-weights'} if field == 'files' else 'different-' + field
        if target in ('parent', 'both'):
            parent['identity']['models']['reranker'][field] = value
        if target in ('repair', 'both'):
            repair['identity']['reranker'][field] = value
    repair['identity']['parent_sha256'] = digest(parent)
    Path(args.parent).write_text(json.dumps(parent))
    Path(args.repair).write_text(json.dumps(repair))


@pytest.mark.parametrize('field,target', [(field, target) for field in MODEL_COMPATIBILITY_FIELDS
    for target in ('parent', 'repair')] + [('libraries', 'parent')])
def test_fresh_registration_rejects_incompatible_cached_identity_before_inference(
        tmp_path, monkeypatch, field, target):
    args, _, _, _ = execution_fixture(tmp_path, monkeypatch)
    enable_fresh_fixture(args, monkeypatch)
    frozen = Path(args.registration).read_bytes()
    alter_cached_identity(args, field, target)
    monkeypatch.setattr(runner, 'rerank', lambda *args, **kwargs: pytest.fail('incompatible inference'))
    with pytest.raises(ValueError, match='cached.*identity|identity.*cached'):
        runner.registration_for(args)
    assert Path(args.registration).read_bytes() == frozen


@pytest.mark.parametrize('field', [*MODEL_COMPATIBILITY_FIELDS, 'libraries'])
def test_fresh_execute_rejects_cached_active_mismatch_with_full47_null_and_zero_dispatch(
        tmp_path, monkeypatch, field):
    args, tasks, calls, _ = execution_fixture(tmp_path, monkeypatch)
    enable_fresh_fixture(args, monkeypatch)
    alter_cached_identity(args, field)
    # Reproduce a declaration that binds the current model but, as the previous
    # implementation did, permits incompatible cached scores. Only synthetic
    # input hashes change; the declaration does not borrow the fixed gate.
    declaration = json.loads(Path(args.registration).read_bytes())
    declaration['input_bytes_sha256'] = [runner.sha(path.read_bytes()) for path in runner.paths_for(args)]
    Path(args.registration).write_text(json.dumps(declaration))
    fresh_calls = []
    selector = runner.select_aspect_evidence
    def tracked_scorer(query, rows, **kwargs):
        fresh_calls.append(query)
        return fake_fresh(query, rows)
    def dispatch_guarded(contract, rows, **kwargs):
        if kwargs['selection_policy'] == 'guarded':
            kwargs['ranker']('fresh incompatible query', rows)
        return selector(contract, rows, **kwargs)
    monkeypatch.setattr(runner, 'rerank', tracked_scorer)
    monkeypatch.setattr(runner, 'select_aspect_evidence', dispatch_guarded)
    report = runner.execute(args, {'run_id': 'cached-mismatch'})
    assert report['valid'] is False and report['status'] == 'failed'
    assert report['selected_task_ids'] == [task['id'] for task in tasks]
    assert calls == [] and fresh_calls == []
    assert all(len(rows) == 47 for rows in report['rows'].values())
    assert all(row['error'] is not None and row['telemetry']['context'] is None
               for rows in report['rows'].values() for row in rows)
    assert all(cost['physical_ranker_calls'] == 0 and cost['physical_ranker_pairs'] == 0
               for cost in report['costs'])


def test_fresh_execute_compatible_cached_model_allows_all47_dispatches(tmp_path, monkeypatch):
    args, tasks, calls, _ = execution_fixture(tmp_path, monkeypatch)
    enable_fresh_fixture(args, monkeypatch)
    selector = runner.select_aspect_evidence
    fresh_calls = []
    def scorer(query, rows, **kwargs):
        fresh_calls.append(query)
        return fake_fresh(query, rows)
    def dispatch_guarded(contract, rows, **kwargs):
        if kwargs['selection_policy'] == 'guarded':
            kwargs['ranker']('fresh compatible query', rows)
        return selector(contract, rows, **kwargs)
    monkeypatch.setattr(runner, 'rerank', scorer)
    monkeypatch.setattr(runner, 'select_aspect_evidence', dispatch_guarded)
    report = runner.execute(args, {'run_id': 'cached-compatible'})
    assert report['valid'] is True and report['status'] == 'complete'
    assert len(fresh_calls) == len(calls) == 47
    assert all(cost['physical_ranker_calls'] == 1 and cost['physical_ranker_pairs'] == 2
               for cost in report['costs'])
    assert all(cost['prior_collection']['unique_cached_pairs'] == 4 for cost in report['costs'])
