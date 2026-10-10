import copy
import json
from types import SimpleNamespace

import pytest

from scripts import run_evidence_supplement_dev as runner
from scripts import run_evidence_selection_dev as replay
from scripts import run_retrieval_quality_dev as retrieval
from scripts.research_review import digest
import test_evidence_selection_dev as replay_tests

source_cache_fixture = replay_tests.frozen


@pytest.fixture
def seeded(source_cache_fixture, tmp_path):
    frozen = source_cache_fixture
    identity = copy.deepcopy(frozen.parent['identity'])
    identity.update(parent_bytes_sha256=retrieval.sha(frozen.pp.read_bytes()),
        cached_models=identity.pop('models'), method='cached_candidate_replay_v1',
        parent_profile='depth50_rerank', stage='selection',
        profiles=list(replay.SELECTION_PROFILES),
        profile_declarations=replay.profile_declarations(replay.SELECTION_PROFILES),
        effective_retrieval_configuration=replay.effective_configuration(frozen.parent))
    seed = dict(version='evidence-selection-dev-v1', stage='selection', split='dev',
        subset=False, status='complete', valid=True, error=None, drift={},
        selected_task_ids=frozen.parent['selected_task_ids'], identity=identity,
        execution_errors={n: {'tasks': 0} for n in replay.SELECTION_PROFILES},
        rows={n: copy.deepcopy(frozen.parent['rows']['depth50_rerank'])
              for n in replay.SELECTION_PROFILES})
    path = tmp_path / 'seed.json'
    path.write_text(json.dumps(seed))
    frozen.seed, frozen.seed_path = seed, path
    return frozen


@pytest.mark.parametrize('change', ['failed', 'partial', 'profile', 'parent', 'refs',
                                  'contexts', 'models', 'configuration', 'duplicate'])
def test_frozen_seed_guards(seeded, change):
    seed = seeded.seed
    if change == 'failed':
        seed['valid'] = False
    elif change == 'partial':
        seed['rows']['legacy_rerank'].pop()
    elif change == 'profile':
        seed['identity']['profiles'] = ['replace_strict']
    elif change == 'parent':
        seed['identity']['parent_bytes_sha256'] = 'wrong'
    elif change == 'refs':
        seed['rows']['legacy_rerank'][0]['reference_fingerprint'] = 'wrong'
    elif change == 'contexts':
        seed['rows']['legacy_rerank'][0]['telemetry']['context'][0]['chunk_id'] = 'other'
    elif change == 'models':
        seed['identity']['cached_models']['private'] = 'raw source text'
    elif change == 'configuration':
        seed['identity']['effective_retrieval_configuration']['top_k'] = 9
    else:
        seed['rows']['legacy_rerank'][1] = copy.deepcopy(seed['rows']['legacy_rerank'][0])
    with pytest.raises(ValueError):
        runner.validate_seed(seed, seeded.parent, seeded.suite,
                             retrieval.sha(seeded.pp.read_bytes()), 'legacy_rerank')


def test_valid_seed_retains_all47(seeded):
    rows = runner.validate_seed(seeded.seed, seeded.parent, seeded.suite,
        retrieval.sha(seeded.pp.read_bytes()), 'legacy_rerank')
    assert len(rows) == 47
    with pytest.raises(ValueError):
        runner.validate_seed(seeded.seed, seeded.parent, seeded.suite,
            retrieval.sha(seeded.pp.read_bytes()), 'replace_strict')


def test_actual_policy_identity_queries_separate_endpoint_and_hashes(monkeypatch):
    from app.config import Settings
    cfg = Settings(_env_file=None, ollama_url='http://embed', embed_model='encoder',
                   agent_policy_url='http://policy', agent_policy_model='planner')
    seen = []
    def get(url, **kwargs):
        seen.append(url)
        assert kwargs['trust_env'] is False
        data = {'version': '0.11.10'} if url.endswith('/version') else {'models': [
            {'name': 'encoder' if url.startswith('http://embed') else 'planner',
             'digest': digest(url)}]}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr(runner.httpx, 'get', get)
    identity = runner.active_model_identity(cfg)
    assert set(seen) == {'http://embed/api/tags', 'http://embed/api/version',
                        'http://policy/api/tags', 'http://policy/api/version'}
    assert identity['policy']['digest'] == digest('http://policy/api/tags')
    assert identity['policy']['status'] == 'active'
    assert 'http://' not in json.dumps(identity)


@pytest.mark.parametrize('value', [None, 'raw secret', True, {'raw': 'source'}])
def test_model_identity_requires_real_digest(monkeypatch, value):
    from app.config import Settings
    cfg = Settings(_env_file=None)
    monkeypatch.setattr(runner.httpx, 'get', lambda url, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {'version': '0.11.10'}
        if url.endswith('/version') else {'models': [
            {'name': cfg.embed_model, 'digest': value},
            {'name': cfg.agent_policy_model, 'digest': value}]}))
    with pytest.raises(ValueError):
        runner.active_model_identity(cfg)


def test_metering_hashes_payload_and_does_not_invent_unknown_tokens(monkeypatch):
    from app.clients import Models
    monkeypatch.setattr(Models, '_post', lambda *args, **kwargs:
                        {'message': {'content': 'private output'}})
    model = runner.MeteredModels()
    model._post('/api/chat', {'messages': [{'content': 'private source'}]})
    event = model.events[0]
    assert event['prompt_tokens'] is None and event['completion_tokens'] is None
    assert 'private' not in json.dumps(model.events)
    summary = runner.event_summary(model.events)
    assert summary['prompt_tokens'] is None and summary['unknown_prompt_tokens'] == 1
    assert runner.event_summary([])['prompt_tokens'] == 0


def test_failed_backend_attempts_count_without_exporting_query():
    def fail(*args, **kwargs):
        raise RuntimeError('private query credentials')
    search = runner.MeteredSearch(SimpleNamespace(index='index', request=fail))
    with pytest.raises(RuntimeError):
        search.request('POST', '/index/_search', json={'query': 'private query'})
    assert search.events[0]['status'] == 'failed'
    assert runner.event_summary(search.events)['attempted'] == 1
    assert 'private query' not in json.dumps(search.events)


def test_source_scope_validation_is_current_and_exact(monkeypatch):
    docs = ['original']
    monkeypatch.setattr(runner, 'readable_documents', lambda *args: list(docs))
    def routes(query, documents, *, strict_dates):
        assert strict_dates is False
        did = documents[0] if 'new source' not in query else 'other'
        return [dict(key=(did,), versions=[did + '-v'], bounded=False)]
    monkeypatch.setattr(runner, 'route_sources', routes)
    check, frozen = runner.scope_validator(None, None, 'original question')
    assert check('original question', 'original question\nbridge')
    assert not check('original question', 'original question\nnew source')
    docs[:] = ['changed-version']
    assert not check('original question', 'original question\nbridge')
    assert 'source' not in json.dumps(frozen)


def collector_setup(monkeypatch, seeded):
    monkeypatch.setattr(replay, 'require_chunk', lambda *args, **kwargs:
        (seeded.chunk, seeded.version, seeded.doc))
    monkeypatch.setattr(runner, 'scope_validator', lambda *args: (lambda o, q: True, []))
    monkeypatch.setattr(replay, 'policy_caps', lambda *args: dict(limit=8,
        token_budget=5000, document_quota=None, source_quota=None))
    return copy.deepcopy(seeded.parent['rows']['depth50_rerank'][0])


def test_collector_passes_no_oracle_and_rotates_matched_pair(monkeypatch, seeded, tmp_path):
    parent_row = collector_setup(monkeypatch, seeded)
    task = seeded.suite['tasks'][0]
    expected_order = []
    def supplement(question, seed, candidates, **kwargs):
        assert question == task['goal'] and all('gold' not in k for r in candidates for k in r)
        assert not {'task', 'category', 'expected_behavior', 'gold_evidence'} & kwargs.keys()
        expected_order.append(kwargs['order'])
        return SimpleNamespace(arms={n: dict(status='skipped', evidence=seed,
            candidates=None, query=None) for n in ('control', 'bridge')},
            trace={'status': 'gate_false', 'gates': {}, 'errors': [], 'arms': {},
                   'calls': {'policy': {'attempted': 0}, 'search': {'attempted': 0}}})
    monkeypatch.setattr(runner, 'run_supplement', supplement)
    for offset in (0, 1):
        result = runner.collect_pair(None, None, task, parent_row, parent_row,
            seeded.corpus, runner.supplement_configuration(None),
            SimpleNamespace(index='index'), tmp_path / str(offset), offset)
        assert all(result['rows'][n]['phases'] == parent_row['phases']
                   for n in runner.PROFILES)
        assert result['rows']['bridge']['usage']['credited_prior_retrieval_tools'] == 1
    assert expected_order == [('control', 'bridge'), ('bridge', 'control')]


def test_history_never_dispatches_and_no_reference_tasks_are_retained(monkeypatch, seeded, tmp_path):
    parent = collector_setup(monkeypatch, seeded)
    monkeypatch.setattr(runner, 'run_supplement', lambda *args, **kwargs:
                        pytest.fail('historical dispatch'))
    task = dict(seeded.suite['tasks'][0], requires_version=True)
    parent.update(runner.diagnose(task, seeded.phases))
    result = runner.collect_pair(None, None, task, parent, parent, seeded.corpus,
        runner.supplement_configuration(None), SimpleNamespace(index='i'), tmp_path, 0)
    assert result['trace']['status'] == 'history_not_applicable'
    assert all(r['reason'] == 'requires_version' for r in result['rows'].values())


def test_no_reference_annotation_never_routes_the_runtime_gate(monkeypatch, seeded, tmp_path):
    parent = collector_setup(monkeypatch, seeded)
    task = dict(seeded.suite['tasks'][0], gold_evidence=[])
    parent.update(runner.diagnose(task, seeded.phases))
    calls = []
    def supplement(question, seed, candidates, **kwargs):
        calls.append(question)
        return SimpleNamespace(arms={n: dict(status='skipped', evidence=seed,
            candidates=None, query=None) for n in ('control', 'bridge')},
            trace={'status': 'gate_false', 'gates': {}, 'errors': [], 'arms': {}, 'calls': {}})
    monkeypatch.setattr(runner, 'run_supplement', supplement)
    result = runner.collect_pair(None, None, task, parent, parent, seeded.corpus,
        runner.supplement_configuration(None), SimpleNamespace(index='i'), tmp_path, 0)
    assert calls == [task['goal']]
    assert all(r['reason'] == 'no_references' for r in result['rows'].values())


def test_operational_failure_keeps_null_not_empty_evidence(monkeypatch, seeded, tmp_path):
    parent = collector_setup(monkeypatch, seeded)
    def supplement(question, seed, candidates, **kwargs):
        return SimpleNamespace(arms={'control': dict(status='failed', evidence=None,
            candidates=None, query=None), 'bridge': dict(status='skipped', evidence=seed,
            candidates=None, query=None)}, trace={'status': 'partial_failure',
            'gates': {}, 'arms': {}, 'calls': {},
            'errors': [{'arm': 'control', 'stage': 'search', 'type': 'RuntimeError'}]})
    monkeypatch.setattr(runner, 'run_supplement', supplement)
    result = runner.collect_pair(None, None, seeded.suite['tasks'][0], parent, parent,
        seeded.corpus, runner.supplement_configuration(None),
        SimpleNamespace(index='i'), tmp_path, 0)
    assert result['rows']['control']['phases']['context'] is None
    assert result['rows']['control']['error']['stage'] == 'search'
    assert result['rows']['bridge']['error'] is None


def test_missing_required_source_is_rejected_by_collector(monkeypatch, seeded, tmp_path):
    parent = collector_setup(monkeypatch, seeded)
    def supplement(question, seed, candidates, **kwargs):
        return SimpleNamespace(arms={n: dict(status='selected', evidence=seed,
            candidates=candidates, query='private query') for n in ('control', 'bridge')},
            trace={'status': 'completed', 'gates': {}, 'errors': [], 'arms': {},
                   'calls': {}, 'bridge_chunk_id': 'not-in-context'})
    monkeypatch.setattr(runner, 'run_supplement', supplement)
    result = runner.collect_pair(None, None, seeded.suite['tasks'][0], parent, parent,
        seeded.corpus, runner.supplement_configuration(None),
        SimpleNamespace(index='i'), tmp_path, 0)
    assert all(r['phases']['context'] is None for n, r in result['rows'].items() if n != 'seed')


def test_graph_level_failure_cannot_publish_already_selected_contexts_as_success(monkeypatch, seeded, tmp_path):
    parent = collector_setup(monkeypatch, seeded)
    def supplement(question, seed, candidates, **kwargs):
        return SimpleNamespace(arms={n: dict(status='selected', evidence=seed,
            candidates=candidates, query='private query') for n in ('control', 'bridge')},
            trace={'status': 'operational_failure', 'gates': {}, 'arms': {}, 'calls': {},
                   'errors': [{'arm': None, 'stage': 'graph', 'type': 'BudgetExceeded'}]})
    monkeypatch.setattr(runner, 'run_supplement', supplement)
    result = runner.collect_pair(None, None, seeded.suite['tasks'][0], parent, parent,
        seeded.corpus, runner.supplement_configuration(None), SimpleNamespace(index='i'), tmp_path, 0)
    assert all(result['rows'][n]['phases']['context'] is None for n in ('control', 'bridge'))
    assert all(result['rows'][n]['error']['stage'] == 'graph' for n in ('control', 'bridge'))


def test_configuration_never_enables_more_model_work():
    from app.config import Settings
    cfg = runner.supplement_configuration(Settings(_env_file=None, passage_rerank=True,
        source_query_plan=True, top_k=1, context_token_budget=1))
    assert cfg.top_k == 8 and cfg.context_token_budget == 5000
    assert cfg.retrieval_candidate_depth == 50 and cfg.agent_task_timeout_seconds == 180
    assert cfg.agent_policy_max_calls == 2 and not cfg.passage_rerank
    assert not cfg.source_query_plan and cfg.rewrite_mode == 'off'


def execute_setup(monkeypatch, seeded, tmp_path):
    import app.clients
    import app.db
    class DB:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, statement):
            assert str(statement) == 'SET TRANSACTION READ ONLY'
    monkeypatch.setattr(app.db, 'SessionLocal', DB)
    monkeypatch.setattr(app.clients, 'Search', lambda **kwargs: SimpleNamespace(index='source'))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'SELECTION', seeded.sel)
    monkeypatch.setattr(runner, 'RUNTIME', tmp_path / '.runtime/supplement')
    reg = tmp_path / 'registration.json'
    reg.write_text('{}')
    monkeypatch.setattr(runner, 'REGISTRATION', reg)
    monkeypatch.setattr(runner, 'active_scope', lambda db: ({'lt-eng': object()}, [], seeded.corpus))
    monkeypatch.setattr(runner, 'index_ledger', lambda *args: seeded.index)
    monkeypatch.setattr(runner, 'code_fingerprints', lambda: {'code': 'frozen'})
    embedding = seeded.parent['identity']['models']['embedding']
    active = {'embedding': dict(embedding, status='active'),
              'policy': {'name': 'planner', 'digest': digest('actual')}, 'libraries': {}}
    monkeypatch.setattr(runner, 'active_model_identity', lambda cfg: copy.deepcopy(active))
    collector_setup(monkeypatch, seeded)
    def supplement(question, seed, candidates, **kwargs):
        return SimpleNamespace(arms={n: dict(status='skipped', evidence=seed,
            candidates=None, query=None) for n in ('control', 'bridge')},
            trace={'status': 'gate_false', 'gates': {}, 'errors': [], 'arms': {}, 'calls': {}})
    monkeypatch.setattr(runner, 'run_supplement', supplement)
    args = SimpleNamespace(parent=seeded.pp, seed_report=seeded.seed_path,
        seed_profile='legacy_rerank', suite=seeded.sp, output=tmp_path / 'out.json')
    return args, {'registration_bytes_sha256': retrieval.sha(reg.read_bytes())}


@pytest.mark.parametrize('outcome', ['complete', 'partial', 'code', 'suite', 'registration', 'models'])
def test_full47_completion_failure_and_drift_are_honest(monkeypatch, seeded, tmp_path, outcome):
    args, registration = execute_setup(monkeypatch, seeded, tmp_path)
    actual, calls = runner.collect_pair, []
    def collect(*pos, **kwargs):
        calls.append(pos[2]['id'])
        result = actual(*pos, **kwargs)
        if outcome == 'partial' and len(calls) == 3:
            result['rows']['control'] = retrieval.missing_row(pos[2],
                {'type': 'RuntimeError', 'stage': 'search'})
        if outcome == 'suite':
            seeded.sp.write_text('{}')
        elif outcome == 'registration':
            runner.REGISTRATION.write_text('private raw text')
        return result
    monkeypatch.setattr(runner, 'collect_pair', collect)
    if outcome == 'code':
        monkeypatch.setattr(runner, 'code_fingerprints', lambda: {'code': str(len(calls))})
    elif outcome == 'models':
        embedding = dict(seeded.parent['identity']['models']['embedding'], status='active')
        monkeypatch.setattr(runner, 'active_model_identity', lambda cfg:
            {'embedding': embedding, 'policy': {'digest': str(len(calls))}})
    report = runner.execute(args, seeded.suite, seeded.selection, seeded.suite['tasks'],
        seeded.parent, seeded.seed, registration, retrieval.reserve_output(args.output))
    assert len(calls) == 47 and all(len(rs) == 47 for rs in report['rows'].values())
    assert report['valid'] == (outcome == 'complete')
    assert report['gate_outcomes']['gate_false'] == 47
    assert len(report['supplement_traces']) == 47
    assert report['pairs']['bridge_vs_control']['total_pairs'] == 47
    serialized = args.output.read_text()
    assert all(secret not in serialized for secret in
        ('private title', 'question 0', 'alpha beta gamma delta', 'private raw text'))
    if outcome == 'partial':
        assert report['summaries']['control']['missing_tasks'] == 1
        assert report['pairs']['bridge_vs_control']['missing_pairs'] == 1
    assert report['strict_task_success'] is None and report['semantic_recall'] is None


def test_late_legacy_mismatch_prevents_any_model_dispatch(monkeypatch, seeded, tmp_path):
    args, registration = execute_setup(monkeypatch, seeded, tmp_path)
    actual = replay.verify_legacy
    def verify(task, row, phases):
        if task['id'] == 'T46':
            raise ValueError('private failure')
        return actual(task, row, phases)
    monkeypatch.setattr(replay, 'verify_legacy', verify)
    monkeypatch.setattr(runner, 'collect_pair', lambda *args: pytest.fail('dispatch before verification'))
    monkeypatch.setattr(runner, 'active_model_identity', lambda cfg: pytest.fail('model identity before verification'))
    report = runner.execute(args, seeded.suite, seeded.selection, seeded.suite['tasks'],
        seeded.parent, seeded.seed, registration, retrieval.reserve_output(args.output))
    assert not report['valid'] and report['gate_outcomes']['not_attempted'] == 47
    assert all(s['missing_tasks'] == 47 for s in report['summaries'].values())
    assert 'private failure' not in args.output.read_text()


@pytest.mark.parametrize('seed_profile', ['legacy_rerank', 'relevant_doc'])
def test_real_graph_metering_and_equal_per_arm_search_credit(monkeypatch, seeded, tmp_path, seed_profile):
    from app.clients import Models
    parent = collector_setup(monkeypatch, seeded)
    if seed_profile != 'legacy_rerank':
        monkeypatch.setattr('scripts.run_revised_evidence_dev.select_relevance_context',
                            lambda *args: parent)
    monkeypatch.setattr(retrieval, 'require_chunk', lambda *args, **kwargs:
        (seeded.chunk, seeded.version, seeded.doc))
    calls = []
    plan = {'steps': [
        {'id': 's1', 'purpose': 'Find entity', 'query': 'alpha',
         'extract': {'name': 'entity', 'kind': 'entity', 'description': 'alpha'}},
        {'id': 's2', 'purpose': 'Find more', 'query': '{s1.entity} more', 'depends_on': ['s1']}]}
    def post(model, path, body, **kwargs):
        calls.append(path)
        if path == '/api/embed':
            return {'embeddings': [[0.0] * runner.settings().embed_dimension],
                    'prompt_eval_count': 20, 'total_duration': 1000000}
        return {'message': {'content': json.dumps(plan) if body.get('format') else 'alpha'},
                'prompt_eval_count': 100, 'eval_count': 10, 'total_duration': 2000000}
    monkeypatch.setattr(Models, '_post', post)
    def retrieve(db, user, query, *, cfg, top_k, models, search):
        with pytest.raises(AttributeError):
            models.generate
        assert cfg.retrieval_candidate_depth == 50 and not cfg.passage_rerank
        models.embed([query])
        search.request('POST', '/source/_search', json={'query': query})
        candidate = dict(parent['telemetry']['candidates'][0], lane='global', rerank_score=None)
        return SimpleNamespace(candidates=[candidate], blocked_reason=None, embed_ms=1, retrieval_ms=2)
    monkeypatch.setattr(runner, 'retrieve_authorized', retrieve)
    source = SimpleNamespace(index='source', request=lambda *args, **kwargs: {})
    pair = runner.collect_pair(None, None, seeded.suite['tasks'][0], parent, parent,
        seeded.corpus, runner.supplement_configuration(None), source, tmp_path, 0,
        seed_profile=seed_profile)
    assert pair['trace']['status'] == 'completed'
    assert calls.count('/api/chat') == 2 and calls.count('/api/embed') == 2
    assert pair['physical_cost']['policy']['attempted'] == 2
    for name in ('control', 'bridge'):
        usage = pair['rows'][name]['usage']
        assert usage['policy']['attempted'] == 2 and usage['policy']['prompt_tokens'] == 200
        assert usage['additional_retrieval_tools'] == usage['backend']['attempted'] == 1
        assert usage['embedding']['attempted'] == 1
        assert pair['trace']['bridge_chunk_id'] in [
            r['chunk_id'] for r in pair['rows'][name]['telemetry']['context']]
    assert 'alpha beta gamma delta' not in json.dumps(pair)


def test_invalid_registration_and_existing_output_prevent_execution(monkeypatch, seeded, tmp_path):
    args, registration = execute_setup(monkeypatch, seeded, tmp_path)
    monkeypatch.setattr(runner, 'FROZEN_PARENT_SHA', retrieval.sha(seeded.pp.read_bytes()))
    monkeypatch.setattr(runner, 'FROZEN_SEED_SHA', retrieval.sha(seeded.seed_path.read_bytes()))
    monkeypatch.setattr(runner, 'execute', lambda *args: pytest.fail('invalid dispatch'))
    cli = ['--suite', str(seeded.sp), '--parent', str(seeded.pp), '--seed-report',
           str(seeded.seed_path), '--seed-profile', 'legacy_rerank', '--output', str(args.output)]
    with pytest.raises(ValueError):
        runner.main(cli)
    monkeypatch.setattr(runner, 'load_registration', lambda args: registration)
    args.output.write_text('original')
    with pytest.raises(ValueError):
        runner.main(cli)
    assert args.output.read_text() == 'original'


def test_posthoc_candidate_missing_never_changes_runtime_inputs(seeded):
    parent = copy.deepcopy(seeded.parent['rows']['depth50_rerank'])
    parent[0]['losses']['candidate_missing'] = 1
    rows = {n: seeded.seed['rows']['legacy_rerank'] for n in runner.PROFILES}
    diagnostic = runner.posthoc_missing(rows, parent)
    assert diagnostic['task_ids'] == ['T0']
    assert diagnostic['summaries']['bridge']['tasks'] == 1


@pytest.mark.parametrize('interrupt_stage', ['policy', 'search'])
@pytest.mark.parametrize('exception', [KeyboardInterrupt, SystemExit])
def test_interrupted_current_task_retains_observed_events_and_stops_later_tasks(
        monkeypatch, seeded, tmp_path, interrupt_stage, exception):
    from app.clients import Models
    from agent.tools import SearchArgs
    args, registration = execute_setup(monkeypatch, seeded, tmp_path)
    def post(*args, **kwargs):
        if interrupt_stage == 'policy':
            raise exception('private interrupted source')
        return {'prompt_eval_count': 12, 'eval_count': 3,
                'embeddings': [[0.0] * runner.settings().embed_dimension]}
    monkeypatch.setattr(Models, '_post', post)
    def retrieve(db, user, query, *, models, search, **kwargs):
        models.embed([query])
        search.request('POST', '/source/_search', json={'query': query})
    monkeypatch.setattr(runner, 'retrieve_authorized', retrieve)
    class Source:
        index = 'source'
        def request(self, *args, **kwargs):
            raise exception('private interrupted search')
    import app.clients
    monkeypatch.setattr(app.clients, 'Search', lambda **kwargs: Source())
    def supplement(question, seed, candidates, *, models, search, **kwargs):
        if interrupt_stage == 'policy':
            models._post('/api/chat', {'messages': [{'content': 'private source'}]})
        else:
            search(SearchArgs(query='private query', top_k=8))
        pytest.fail('interruption must leave graph')
    monkeypatch.setattr(runner, 'run_supplement', supplement)
    report = runner.execute(args, seeded.suite, seeded.selection, seeded.suite['tasks'],
        seeded.parent, seeded.seed, registration, retrieval.reserve_output(args.output))
    assert report['status'] == 'interrupted' and not report['valid']
    assert report['supplement_traces'][0]['status'] == 'interrupted'
    assert report['gate_outcomes']['not_attempted'] == 46
    assert report['cost_observation_counts'] == {'observed': 1, 'not_observed': 46}
    cost = report['physical_costs'][0]
    role = 'policy' if interrupt_stage == 'policy' else 'backend'
    assert cost[role]['attempted'] == cost[role]['failed'] == 1
    if interrupt_stage == 'policy':
        assert cost['policy_events'][0]['prompt_tokens'] is None
        assert cost['policy_events'][0]['request_sha256']
    else:
        assert cost['search_attempts'] == 1 and cost['search_events'][0]['status'] == 'failed'
        for name in ('control', 'bridge'):
            assert report['rows'][name][0]['usage']['additional_retrieval_tools'] is None
            assert report['rows'][name][0]['usage']['search_attribution'] == 'unknown'
        assert 'arm' not in cost['search_events'][0]
    assert report['physical_costs'][1]['search_attempts'] is None
    assert 'private' not in args.output.read_text()


@pytest.mark.parametrize('mode,expected_requests', [('hybrid', 2), ('dense', 1), ('bm25', 1)])
def test_metered_backend_runs_official_authorized_retrieval_without_service(
        mode, expected_requests):
    from app.config import Settings
    from app.retrieval import retrieve_authorized
    doc = SimpleNamespace(id='doc', active_version_id='version', title='Private document', metadata_json={})
    version = SimpleNamespace(id='version')
    chunk = SimpleNamespace(id='chunk', text='A relevant fact.', locator={})
    requests = []
    def request(method, path, **kwargs):
        requests.append(kwargs['json'])
        assert method == 'POST' and path == '/source/_search'
        return {'hits': {'hits': [{'_id': 'chunk', '_score': 0.9}]}}
    metered = runner.MeteredSearch(SimpleNamespace(index='source', request=request))
    cfg = runner.supplement_configuration(Settings(_env_file=None)).model_copy(
        update={'retrieval_mode': mode, 'source_routing': False})
    found = retrieve_authorized(None, SimpleNamespace(tenant_id='tenant'), 'Explain the relevant fact',
        cfg=cfg, models=SimpleNamespace(embed=lambda qs: [[0.0] for q in qs]), search=metered,
        readable_documents_fn=lambda *args: [doc],
        require_chunk_fn=lambda *args, **kwargs: (chunk, version, doc))
    assert len(found.evidence) == 1 and found.evidence[0]['chunk_id'] == 'chunk'
    assert len(requests) == len(metered.events) == expected_requests
    assert all(e['status'] == 'ok' for e in metered.events)
    assert 'Private document' not in json.dumps(metered.events)
    assert all(body['size'] == 50 for body in requests)
