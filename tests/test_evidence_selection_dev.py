import copy
import json
from types import SimpleNamespace

import pytest

from app.chunking import token_count
from scripts import run_evidence_selection_dev as runner
from scripts import run_retrieval_quality_dev as retrieval
from scripts.research_review import digest
from scripts.retrieval_quality_metrics import diagnose
from test_retrieval_quality_dev import package


@pytest.fixture
def frozen(tmp_path):
    suite, selection = package(tmp_path)
    sp, sel = tmp_path / 'tasks.json', tmp_path / 'selection.json'
    sp.write_text(json.dumps(suite))
    sel.write_text(json.dumps(selection))
    source_sha = suite['tasks'][0]['gold_evidence'][0]['source_sha256']
    chunk = SimpleNamespace(id='c', text='alpha beta gamma delta', locator={})
    version = SimpleNamespace(id='v', content_hash=source_sha)
    doc = SimpleNamespace(id='doc', title='private title', metadata_json={})
    corpus = {'documents': [dict(document_id='doc', active_version_id='v',
        source_sha256=source_sha, document_sha256=digest([doc.title, {}]),
        chunks=[dict(id='c', text_sha256=digest(chunk.text), locator_sha256=digest({}))])],
        'users': []}
    row = dict(chunk_id='c', document_id='doc', version_id='v', source_sha256=source_sha,
        rank=1, retrieval_rank=1, rerank_score=0.9, admitted=True, excluded_because=None,
        lane_type='global', lane_sha256=digest('global'))
    bound = row | dict(text=chunk.text, title=doc.title)
    phases = {p: [bound] for p in ('candidates', 'admitted', 'context')}
    telemetry = {p: [row] for p in phases} | {p.rstrip('s') + '_count': 1 for p in phases}
    telemetry.update(candidate_count=1, admitted_count=1, context_count=1,
        context_tokens=token_count(chunk.text) + token_count(doc.title) + 100,
        wall_ms=10, embed_ms=1, retrieval_ms=8, context_ms=1, excluded_counts={})
    cfg = retrieval.configuration(None, 'depth50_rerank')
    index = {'index': 'source', 'chunks': [dict(id='c', source_sha256=digest('index'))]}
    ids = [t['id'] for t in suite['tasks']]
    identity = dict(suite_sha256=digest(suite), selection_sha256=digest(selection),
        suite_bytes_sha256=retrieval.sha(sp.read_bytes()),
        selection_bytes_sha256=retrieval.sha(sel.read_bytes()), selected_task_ids=ids,
        corpus=corpus, corpus_sha256=digest(corpus), source_index=index,
        source_index_sha256=digest(index), models={
        'embedding': {'name': cfg.embed_model, 'digest': digest('cached embedding'),
            'runtime': {'version': '0.11.10'}, 'endpoint_sha256': digest('endpoint')},
        'generation': {'name': 'unused-model', 'status': 'unused', 'digest': None},
        'policy': {'name': 'unused-model', 'status': 'unused', 'digest': None},
        'reranker': {'name': cfg.rerank_model, 'status': 'active', 'revision': 'a' * 40,
            'snapshot_path_sha256': digest('snapshot'), 'digest': digest('cached reranker'),
            'files': {name: digest(name) for name in ('config.json', 'model.safetensors',
                'tokenizer.json', 'tokenizer_config.json')},
            'batch': cfg.rerank_batch, 'max_tokens': cfg.rerank_max_tokens},
        'libraries': {name: '1.0.0' for name in ('httpx', 'pydantic', 'sqlalchemy',
            'torch', 'transformers', 'tokenizers')}}, profiles=[dict(name='depth50_rerank',
        overrides=retrieval.PROFILES['depth50_rerank'], effective_configuration={
        k: getattr(cfg, k) for k in retrieval.CONFIG_KEYS})])
    parent = dict(valid=True, status='complete', split='dev', subset=False,
        error=None, selected_task_ids=ids, identity=identity, execution_errors={
        'depth50_rerank': {'tasks': 0}}, rows={'depth50_rerank': [
        diagnose(t, phases) | dict(telemetry=copy.deepcopy(telemetry), error=None)
        for t in suite['tasks']]})
    pp = tmp_path / 'parent.json'
    pp.write_text(json.dumps(parent))
    return SimpleNamespace(suite=suite, selection=selection, sp=sp, sel=sel, parent=parent,
        pp=pp, corpus=corpus, chunk=chunk, version=version, doc=doc, phases=phases, index=index)


def validate(f):
    return runner.validate_parent(f.parent, f.suite, f.selection, f.sp.read_bytes(), f.sel.read_bytes())


@pytest.mark.parametrize('mutation', ['partial', 'failed', 'profile', 'duplicate', 'source',
    'suite_sha', 'lane', 'telemetry', 'configuration', 'corpus_sha', 'index_sha'])
def test_parent_rejects_invalid_before_clients(frozen, mutation):
    p = frozen.parent
    r = p['rows']['depth50_rerank'][0]
    if mutation == 'partial':
        p['rows']['depth50_rerank'].pop()
    elif mutation == 'failed':
        p['valid'] = False
    elif mutation == 'profile':
        p['identity']['profiles'][0]['name'] = 'baseline'
    elif mutation == 'duplicate':
        p['rows']['depth50_rerank'][1] = copy.deepcopy(r)
    elif mutation == 'source':
        r['telemetry']['candidates'][0]['source_sha256'] = 'wrong'
    elif mutation == 'suite_sha':
        p['identity']['suite_bytes_sha256'] = 'wrong'
    elif mutation == 'lane':
        r['telemetry']['candidates'][0]['lane_type'] = 'unknown'
    elif mutation == 'telemetry':
        r['telemetry']['context'] = None
    elif mutation == 'configuration':
        p['identity']['profiles'][0]['effective_configuration']['top_k'] = 9
    else:
        p['identity']['source_index_sha256' if mutation == 'index_sha' else 'corpus_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        validate(frozen)


def test_parent_accepts_same_identity_duplicate_candidates(frozen):
    r = frozen.parent['rows']['depth50_rerank'][0]
    r['telemetry']['candidates'].append(copy.deepcopy(r['telemetry']['candidates'][0]))
    r['telemetry']['candidate_count'] = 2
    assert len(validate(frozen)) == 47


@pytest.mark.parametrize('mutation', ['row_identity', 'text', 'title', 'authorization'])
def test_rebind_refuses_identity_or_authorization_changes(monkeypatch, frozen, mutation):
    rows = copy.deepcopy(frozen.parent['rows']['depth50_rerank'][0]['telemetry']['candidates'])
    def authorize(*args, **kw):
        assert kw['active_only'] is True
        if mutation == 'authorization':
            raise PermissionError('private')
        return frozen.chunk, frozen.version, frozen.doc
    monkeypatch.setattr(runner, 'require_chunk', authorize)
    if mutation == 'row_identity':
        rows[0]['document_id'] = 'other'
    elif mutation == 'text':
        frozen.chunk.text = 'changed'
    elif mutation == 'title':
        frozen.doc.title = 'changed'
    with pytest.raises((ValueError, PermissionError)):
        runner.rebind_rows(None, None, rows, frozen.corpus)


def test_legacy_requires_exact_all_phase_reproduction(monkeypatch, frozen):
    monkeypatch.setattr(runner, 'require_chunk', lambda *a, **kw:
        (frozen.chunk, frozen.version, frozen.doc))
    row = frozen.parent['rows']['depth50_rerank'][0]
    phases = runner.rehydrate_task(None, None, row, frozen.corpus)
    runner.verify_legacy(frozen.suite['tasks'][0], row, phases)
    row['phases']['candidates']['first_supporting_rank'] = 42
    with pytest.raises(ValueError):
        runner.verify_legacy(frozen.suite['tasks'][0], row, phases)


def test_caps_use_route_groups_including_empty_hit_groups(monkeypatch):
    monkeypatch.setattr(runner, 'readable_documents', lambda *a: ['doc'])
    def routes(question, documents, *, strict_dates):
        assert question == 'question' and documents == ['doc'] and strict_dates is False
        return [{'versions': []}] * 4
    monkeypatch.setattr(runner, 'route_sources', routes)
    assert runner.policy_caps(None, None, 'question') == dict(limit=8, token_budget=5000,
        document_quota=2, source_quota=2)


def test_policy_input_is_gold_free_and_replacement_seed_shared(monkeypatch, frozen):
    candidates = frozen.phases['candidates']
    seed = SimpleNamespace(evidence=candidates, context_tokens=110, trace={})
    seen = []
    def greedy(question, rows, **kw):
        assert question == 'question' and rows is candidates
        assert 'required_ids' not in kw
        return seed
    def replace(question, rows, actual_seed, **kw):
        assert question == 'question' and rows is candidates and actual_seed is seed.evidence
        assert kw['max_replacements'] == 8 and 'required_ids' not in kw
        seen.append((kw['document_quota'], kw['source_quota']))
        return seed
    monkeypatch.setattr(runner, 'select_complementary', greedy)
    monkeypatch.setattr(runner, 'replace_weak_evidence', replace)
    caps = dict(limit=8, token_budget=5000, document_quota=2, source_quota=3)
    for profile in runner.QUOTA_PROFILES[1:]:
        runner.select_profile('question', candidates, profile, caps, seed=seed)
    assert seen == [(2, 3), (None, 3), (2, None), (None, None)]


def setup_execute(monkeypatch, f, tmp_path):
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
    monkeypatch.setattr(app.clients, 'Search', lambda **kw: SimpleNamespace(index='source'))
    monkeypatch.setattr(app.clients, 'Models', lambda: pytest.fail('model client created'))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'SELECTION', f.sel)
    monkeypatch.setattr(runner, 'RUNTIME', tmp_path / '.runtime/selection')
    monkeypatch.setattr(runner, 'code_fingerprints', lambda: {'code': 'frozen'})
    monkeypatch.setattr(runner, 'active_scope', lambda db: ({'lt-eng': object()}, [], f.corpus))
    monkeypatch.setattr(runner, 'index_ledger', lambda *a: f.index)
    monkeypatch.setattr(runner, 'require_chunk', lambda *a, **kw: (f.chunk, f.version, f.doc))
    monkeypatch.setattr(runner, 'policy_caps', lambda *a: dict(limit=8, token_budget=5000,
        document_quota=None, source_quota=None))
    return SimpleNamespace(stage='selection', output=tmp_path / 'out.json', suite=f.sp, parent=f.pp)


@pytest.mark.parametrize('outcome', ['complete', 'drift', 'failure', 'coverage_drift', 'parent_drift'])
def test_execute_retains_all47_and_text_free_artifact(monkeypatch, frozen, tmp_path, outcome):
    args = setup_execute(monkeypatch, frozen, tmp_path)
    calls = []
    actual = runner.select_complementary
    def select(*a, **kw):
        calls.append(1)
        if outcome == 'failure':
            raise RuntimeError('private gold credential')
        result = actual(*a, **kw)
        if outcome == 'parent_drift':
            frozen.pp.write_text('{}')
        return result
    monkeypatch.setattr(runner, 'select_complementary', select)
    if outcome == 'drift':
        monkeypatch.setattr(runner, 'code_fingerprints', lambda: {'code': str(len(calls))})
    if outcome == 'coverage_drift':
        monkeypatch.setattr(runner, 'settings', lambda: SimpleNamespace(answer_quality_enabled=bool(calls)))
    reservation = retrieval.reserve_output(args.output)
    report = runner.execute(args, frozen.suite, frozen.selection, frozen.suite['tasks'],
        frozen.parent, reservation)
    assert report['valid'] == (outcome == 'complete')
    assert (report['status'] == 'complete') == report['valid']
    assert all(len(rows) == 47 for rows in report['rows'].values())
    assert report['pairs']['complementary_strict']['total_pairs'] == 47
    serialized = args.output.read_text()
    assert all(secret not in serialized for secret in ('private title', 'question 0',
        'alpha beta gamma delta', 'private gold credential'))
    if outcome == 'failure':
        assert report['execution_errors']['complementary_strict']['tasks'] == 47
    for rows in report['rows'].values():
        assert all(r['telemetry']['context_tokens'] is None or
            r['telemetry']['context_tokens'] <= 5000 for r in rows)


def test_invalid_main_and_existing_output_never_create_clients(monkeypatch, frozen, tmp_path):
    monkeypatch.setattr(runner, 'SELECTION', frozen.sel)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'execute', lambda *a: pytest.fail('guard bypassed'))
    args = ['--stage', 'selection', '--suite', str(frozen.sp), '--parent', str(frozen.pp),
        '--output', str(tmp_path / 'out.json')]
    (tmp_path / 'out.json').write_text('original')
    with pytest.raises(ValueError):
        runner.main(args)
    frozen.parent['status'] = 'failed'
    frozen.pp.write_text(json.dumps(frozen.parent))
    with pytest.raises(ValueError):
        runner.main(args)


@pytest.mark.parametrize('mutation', ['corpus', 'index', 'late_legacy'])
def test_all_legacy_and_live_identity_checked_before_any_policy(monkeypatch, frozen, tmp_path, mutation):
    args = setup_execute(monkeypatch, frozen, tmp_path)
    if mutation == 'corpus':
        changed = dict(frozen.corpus, users=['changed'])
        monkeypatch.setattr(runner, 'active_scope', lambda db: ({'lt-eng': object()}, [], changed))
    elif mutation == 'index':
        monkeypatch.setattr(runner, 'index_ledger', lambda *a: dict(frozen.index, changed=True))
    else:
        frozen.parent['rows']['depth50_rerank'][-1]['phases']['context']['delivered'] = 0
        frozen.pp.write_text(json.dumps(frozen.parent))
    monkeypatch.setattr(runner, 'select_complementary', lambda *a, **kw:
        pytest.fail('policy ran before full legacy/live verification'))
    report = runner.execute(args, frozen.suite, frozen.selection, frozen.suite['tasks'],
        frozen.parent, retrieval.reserve_output(args.output))
    assert not report['valid'] and report['identity'] is None
    assert all(len(rows) == 47 and all(r['error'] for r in rows)
        for rows in report['rows'].values())


def test_rebind_whitelist_and_public_lane_hash_are_preserved(monkeypatch, frozen):
    monkeypatch.setattr(runner, 'require_chunk', lambda *a, **kw:
        (frozen.chunk, frozen.version, frozen.doc))
    cached = frozen.parent['rows']['depth50_rerank'][0]['telemetry']['candidates'][0]
    cached.update(gold_evidence=['secret'], expected_behavior='secret', category='secret')
    rows = runner.rebind_rows(None, None, [cached], frozen.corpus)
    assert not {'gold_evidence', 'expected_behavior', 'category'} & rows[0].keys()
    public = runner.public_telemetry(rows)[0]
    assert public['lane_sha256'] == cached['lane_sha256'] == digest('global')
    assert public['parent_rank'] == 1 and 'text' not in public


def test_new_context_ranks_do_not_mutate_parent_rank(monkeypatch, frozen):
    monkeypatch.setattr(runner, 'require_chunk', lambda *a, **kw:
        (frozen.chunk, frozen.version, frozen.doc))
    parent_row = frozen.parent['rows']['depth50_rerank'][0]
    phases = runner.rehydrate_task(None, None, parent_row, frozen.corpus)
    candidates = phases['candidates']
    candidates[0]['rank'] = candidates[0]['parent_rank'] = 7
    result = runner.select_complementary('alpha', candidates, document_quota=None)
    row = runner.replay_row(None, None, frozen.suite['tasks'][0], parent_row,
        phases, result, frozen.corpus)
    assert row['telemetry']['context'][0]['rank'] == 1
    assert row['telemetry']['context'][0]['parent_rank'] == 7
    assert candidates[0]['rank'] == candidates[0]['parent_rank'] == 7


def test_gold_free_policy_spy_and_identity_precedes_first_call(monkeypatch, frozen, tmp_path):
    args = setup_execute(monkeypatch, frozen, tmp_path)
    cached = frozen.parent['rows']['depth50_rerank'][0]['telemetry']['candidates'][0]
    cached['gold_evidence'] = ['secret']
    frozen.pp.write_text(json.dumps(frozen.parent))
    actual = runner.select_complementary
    def spy(question, candidates, **kwargs):
        identity = json.loads(next(runner.RUNTIME.glob('*/replay-identity.json')).read_text())
        assert identity['profiles'] == list(runner.SELECTION_PROFILES)
        assert identity['answer_quality_enabled'] is runner.settings().answer_quality_enabled
        assert identity['effective_retrieval_configuration']['answer_quality_enabled'] is False
        assert set(kwargs) == {'limit', 'token_budget', 'document_quota', 'source_quota'}
        assert all(set(c) <= set(runner.ROW_FIELDS) | {'text', 'title', 'metadata', 'locator'}
                   for c in candidates)
        assert isinstance(question, str) and 'gold_evidence' not in json.dumps(candidates)
        return actual(question, candidates, **kwargs)
    monkeypatch.setattr(runner, 'select_complementary', spy)
    report = runner.execute(args, frozen.suite, frozen.selection, frozen.suite['tasks'],
        frozen.parent, retrieval.reserve_output(args.output))
    assert report['valid']


def test_quotas_replay_pairs_both_controls_and_reproduces_diagnostics(monkeypatch, frozen, tmp_path):
    args = setup_execute(monkeypatch, frozen, tmp_path)
    args.stage = 'quotas'
    report = runner.execute(args, frozen.suite, frozen.selection, frozen.suite['tasks'],
        frozen.parent, retrieval.reserve_output(args.output))
    assert report['valid'] and list(report['rows']) == list(runner.QUOTA_PROFILES)
    for profile, controls in report['pairs'].items():
        assert set(controls) == {'against_replace_strict', 'against_greedy'}
        assert all(control['total_pairs'] == 47 for control in controls.values())
        assert controls['against_greedy']['complete_pairs'] == 47
        for row, baseline in zip(report['rows'][profile], report['rows']['complementary_strict']):
            assert row['phases'] == baseline['phases']
            assert row['telemetry']['context'] == baseline['telemetry']['context']
    assert report['selection_summaries']['replace_both']['losses']['candidate_missing'] == 0


def test_current_authorization_is_checked_before_each_policy(monkeypatch, frozen, tmp_path):
    args = setup_execute(monkeypatch, frozen, tmp_path)
    calls = []
    authorize_calls = []
    def authorize(*a, **kw):
        authorize_calls.append(1)
        if len(authorize_calls) > 47 * 3 + 4:
            raise PermissionError('revoked')
        return frozen.chunk, frozen.version, frozen.doc
    monkeypatch.setattr(runner, 'require_chunk', authorize)
    monkeypatch.setattr(runner, 'select_complementary', lambda *a, **kw: calls.append(1))
    report = runner.execute(args, frozen.suite, frozen.selection, frozen.suite['tasks'],
        frozen.parent, retrieval.reserve_output(args.output))
    assert not calls and not report['valid']
    assert report['execution_errors']['complementary_strict']['tasks'] == 47


@pytest.mark.parametrize('outcome', ['interrupt', 'raw_source', 'suite', 'selection'])
def test_interrupt_or_input_drift_is_retained(monkeypatch, frozen, tmp_path, outcome):
    args = setup_execute(monkeypatch, frozen, tmp_path)
    actual = runner.select_complementary
    def select(*a, **kw):
        if outcome == 'interrupt':
            raise KeyboardInterrupt()
        if outcome == 'raw_source':
            (tmp_path / 'source.txt').write_text('changed raw bytes')
        elif outcome == 'suite':
            frozen.sp.write_text('{}')
        else:
            frozen.sel.write_text('{}')
        return actual(*a, **kw)
    monkeypatch.setattr(runner, 'select_complementary', select)
    report = runner.execute(args, frozen.suite, frozen.selection, frozen.suite['tasks'],
        frozen.parent, retrieval.reserve_output(args.output))
    assert not report['valid']
    assert all(len(rows) == 47 for rows in report['rows'].values())
    if outcome == 'interrupt':
        assert report['status'] == 'interrupted'
        assert report['summaries']['complementary_strict']['missing_tasks'] == 47
    else:
        assert report['drift']['raw_sources' if outcome == 'raw_source' else outcome]


@pytest.mark.parametrize('mutation', ['parent_rank', 'excluded_counts', 'excluded_reason'])
def test_unknown_parent_policy_inputs_are_rejected(frozen, mutation):
    telemetry = frozen.parent['rows']['depth50_rerank'][0]['telemetry']
    if mutation == 'parent_rank':
        telemetry['candidates'][0]['parent_rank'] = 99
    elif mutation == 'excluded_counts':
        telemetry['excluded_counts'] = {'private question text': 1}
    else:
        telemetry['candidates'][0]['excluded_because'] = 'private question text'
    with pytest.raises(ValueError):
        validate(frozen)


def test_history_and_no_reference_tasks_remain_in_all_profiles(monkeypatch, frozen, tmp_path):
    args = setup_execute(monkeypatch, frozen, tmp_path)
    for index, task in enumerate(frozen.suite['tasks']):
        if index < 5:
            task['requires_version'] = True
        elif index < 11:
            task['gold_evidence'] = []
        frozen.parent['rows']['depth50_rerank'][index].update(diagnose(task, frozen.phases))
        frozen.selection['items'][index]['annotation_sha256'] = digest(task)
    frozen.sp.write_text(json.dumps(frozen.suite))
    frozen.sel.write_text(json.dumps(frozen.selection))
    identity = frozen.parent['identity']
    identity.update(suite_sha256=digest(frozen.suite), selection_sha256=digest(frozen.selection),
        suite_bytes_sha256=retrieval.sha(frozen.sp.read_bytes()),
        selection_bytes_sha256=retrieval.sha(frozen.sel.read_bytes()))
    frozen.pp.write_text(json.dumps(frozen.parent))
    report = runner.execute(args, frozen.suite, frozen.selection, frozen.suite['tasks'],
        frozen.parent, retrieval.reserve_output(args.output))
    assert report['valid']
    for profile in runner.SELECTION_PROFILES:
        summary = report['summaries'][profile]
        assert summary['eligible_tasks'] == summary['measured_tasks'] == 36
        assert summary['not_applicable_tasks'] == 11
        assert report['null_evidence_counts'][profile]['tasks'] == 6
        assert all(r['reason'] == 'requires_version' for r in report['rows'][profile][:5])
        assert all(r['reason'] == 'no_references' for r in report['rows'][profile][5:11])
    assert report['pairs']['complementary_strict']['not_applicable_pairs'] == 11


def test_legacy_background_redundancy_uses_same_metric_as_complementary(frozen, monkeypatch):
    row = frozen.phases['candidates'][0]
    candidates = [dict(row, chunk_id='a', rank=1), dict(row, chunk_id='b', rank=2),
        dict(row, chunk_id='c', rank=3, text='epsilon zeta theta iota')]
    baseline = candidates[:2]
    result = runner.select_complementary('alpha beta gamma delta epsilon zeta theta iota',
        candidates, limit=2, document_quota=None)
    assert runner.context_redundancy(baseline) == 1.0
    assert runner.context_redundancy(result.evidence) == 0.0
    assert -result.trace['objective'][2] == 0.0
    monkeypatch.setattr(runner, 'require_chunk', lambda *a, **kw:
        (frozen.chunk, frozen.version, frozen.doc))
    parent_row = frozen.parent['rows']['depth50_rerank'][0]
    phases = runner.rehydrate_task(None, None, parent_row, frozen.corpus)
    legacy = runner.select_profile('question', phases['candidates'], 'legacy_rerank', {},
        legacy=phases['context'])
    public = runner.replay_row(None, None, frozen.suite['tasks'][0], parent_row, phases,
        legacy, frozen.corpus, legacy=True)
    assert public['selection_statistics']['mean_pair_jaccard'] == 0.0


def test_repeated_fresh_replay_has_identical_paired_rows(monkeypatch, frozen, tmp_path):
    args = setup_execute(monkeypatch, frozen, tmp_path)
    reports = []
    for name in ('first', 'second'):
        args.output = tmp_path / f'{name}.json'
        reports.append(runner.execute(args, frozen.suite, frozen.selection, frozen.suite['tasks'],
            frozen.parent, retrieval.reserve_output(args.output)))
    assert all(report['valid'] for report in reports)
    assert reports[0]['pairs'] == reports[1]['pairs']
    assert reports[0]['selection_summaries'] == reports[1]['selection_summaries']
    for profile in runner.SELECTION_PROFILES:
        assert [r['telemetry']['context'] for r in reports[0]['rows'][profile]] == [
            r['telemetry']['context'] for r in reports[1]['rows'][profile]]


BAD_TELEMETRY = [
    ('fusion_score', 'SYNTHETIC_RAW_TEXT'), ('fusion_score', {'raw': 'SYNTHETIC_RAW_TEXT'}),
    ('fusion_score', float('nan')), ('fusion_score', float('inf')), ('fusion_score', True),
    ('rerank_score', 'SYNTHETIC_RAW_TEXT'), ('rerank_score', float('-inf')),
    ('rerank_score', False), ('retrieval_rank', {'raw': 'SYNTHETIC_RAW_TEXT'}),
    ('retrieval_rank', 'SYNTHETIC_RAW_TEXT'), ('retrieval_rank', 0),
    ('retrieval_rank', 1.5), ('retrieval_rank', True), ('bm25_rank', -1),
    ('dense_rank', False), ('rank', None), ('rank', True),
    ('parent_rank', 'SYNTHETIC_RAW_TEXT'), ('parent_rank', 0), ('parent_rank', None),
    ('admitted', 'SYNTHETIC_RAW_TEXT'), ('admitted', None), ('admitted', 1),
    ('excluded_because', {'raw': 'SYNTHETIC_RAW_TEXT'}),
]


@pytest.mark.parametrize('field,value', BAD_TELEMETRY)
def test_malformed_retained_field_rejected_by_parent_guard(frozen, field, value):
    row = frozen.parent['rows']['depth50_rerank'][0]['telemetry']['candidates'][0]
    row[field] = value
    with pytest.raises(ValueError) as error:
        validate(frozen)
    assert 'SYNTHETIC_RAW_TEXT' not in str(error.value)


@pytest.mark.parametrize('field,value', BAD_TELEMETRY)
def test_malformed_retained_field_cannot_cross_public_or_rebind_boundary(monkeypatch, frozen,
                                                                       field, value):
    row = copy.deepcopy(frozen.parent['rows']['depth50_rerank'][0]['telemetry']['candidates'][0])
    row[field] = value
    monkeypatch.setattr(runner, 'require_chunk', lambda *a, **kw:
        pytest.fail('malformed row reached authorization'))
    for boundary in (lambda: runner.public_telemetry([row]),
                     lambda: runner.rebind_rows(None, None, [row], frozen.corpus)):
        with pytest.raises(ValueError) as error:
            boundary()
        assert 'SYNTHETIC_RAW_TEXT' not in str(error.value)


def test_valid_nullable_ranks_and_scores_remain_null_at_boundary(monkeypatch, frozen):
    row = frozen.parent['rows']['depth50_rerank'][0]['telemetry']['candidates'][0]
    row.update(retrieval_rank=None, bm25_rank=None, dense_rank=None, fusion_score=None)
    assert len(validate(frozen)) == 47
    monkeypatch.setattr(runner, 'require_chunk', lambda *a, **kw:
        (frozen.chunk, frozen.version, frozen.doc))
    rebound = runner.rebind_rows(None, None, [row], frozen.corpus)
    public = runner.public_telemetry(rebound)[0]
    assert all(public[key] is None for key in
        ('retrieval_rank', 'bm25_rank', 'dense_rank', 'fusion_score'))
    # Generic public/rebinding helpers can also represent uncached supplement
    # scores, while the frozen depth50 parent requires its actual rerank score.
    row['rerank_score'] = None
    assert runner.public_telemetry([row])[0]['rerank_score'] is None
    assert runner.rebind_rows(None, None, [row], frozen.corpus)[0]['rerank_score'] is None
    with pytest.raises(ValueError):
        validate(frozen)


def test_new_selection_admission_flags_describe_current_phase(monkeypatch, frozen):
    monkeypatch.setattr(runner, 'require_chunk', lambda *a, **kw:
        (frozen.chunk, frozen.version, frozen.doc))
    parent_row = frozen.parent['rows']['depth50_rerank'][0]
    phases = runner.rehydrate_task(None, None, parent_row, frozen.corpus)
    phases['candidates'][0].update(admitted=False, excluded_because='top_k_full')
    result = runner.select_complementary('alpha beta', phases['candidates'], document_quota=None)
    row = runner.replay_row(None, None, frozen.suite['tasks'][0], parent_row,
        phases, result, frozen.corpus)
    assert row['telemetry']['candidates'][0]['admitted'] is False
    assert row['telemetry']['candidates'][0]['excluded_because'] == 'top_k_full'
    for phase in ('admitted', 'context'):
        assert row['telemetry'][phase][0]['admitted'] is True
        assert row['telemetry'][phase][0]['excluded_because'] is None
    assert phases['candidates'][0]['admitted'] is False


@pytest.mark.parametrize('mutation', ['extra', 'embedding_extra', 'runtime_extra', 'reranker_extra',
    'libraries_extra', 'digest', 'endpoint', 'snapshot', 'file_hash', 'revision', 'batch',
    'model_name', 'model_name_raw', 'version_raw', 'generation_status', 'generation_digest'])
def test_cached_model_identity_cannot_export_extra_or_malformed_values(frozen, mutation):
    models = frozen.parent['identity']['models']
    sentinel = 'SYNTHETIC_RAW_TEXT'
    if mutation == 'extra':
        models['unexpected_raw_text'] = sentinel
    elif mutation == 'embedding_extra':
        models['embedding']['unexpected_raw_text'] = sentinel
    elif mutation == 'runtime_extra':
        models['embedding']['runtime']['unexpected_raw_text'] = sentinel
    elif mutation == 'reranker_extra':
        models['reranker']['unexpected_raw_text'] = sentinel
    elif mutation == 'libraries_extra':
        models['libraries']['unexpected_raw_text'] = sentinel
    elif mutation == 'digest':
        models['embedding']['digest'] = sentinel
    elif mutation == 'endpoint':
        models['embedding']['endpoint_sha256'] = sentinel
    elif mutation == 'snapshot':
        models['reranker']['snapshot_path_sha256'] = sentinel
    elif mutation == 'file_hash':
        models['reranker']['files']['config.json'] = sentinel
    elif mutation == 'revision':
        models['reranker']['revision'] = {'raw': sentinel}
    elif mutation == 'batch':
        models['reranker']['batch'] = True
    elif mutation == 'model_name':
        models['embedding']['name'] = 'different-model'
    elif mutation == 'model_name_raw':
        models['policy']['name'] = 'raw source text with whitespace'
    elif mutation == 'version_raw':
        models['libraries']['httpx'] = 'raw source text with whitespace'
    elif mutation == 'generation_status':
        models['generation']['status'] = sentinel
    else:
        models['generation']['digest'] = sentinel
    with pytest.raises(ValueError) as error:
        validate(frozen)
    assert sentinel not in str(error.value)


def test_model_identity_public_boundary_rejects_raw_extra(frozen):
    models = frozen.parent['identity']['models']
    models['unexpected_raw_text'] = 'SYNTHETIC_RAW_TEXT'
    with pytest.raises(ValueError) as error:
        runner.cached_model_identity(models, runner._configuration(frozen.parent))
    assert 'SYNTHETIC_RAW_TEXT' not in str(error.value)


@pytest.mark.parametrize('mutation', ['telemetry', 'models'])
def test_invalid_public_input_keeps_failed_artifact_free_of_raw_sentinel(monkeypatch, frozen,
                                                                        tmp_path, mutation):
    args = setup_execute(monkeypatch, frozen, tmp_path)
    sentinel = 'SYNTHETIC_RAW_TEXT'
    if mutation == 'telemetry':
        frozen.parent['rows']['depth50_rerank'][0]['telemetry']['candidates'][0]['fusion_score'] = sentinel
    else:
        frozen.parent['identity']['models']['unexpected_raw_text'] = sentinel
    frozen.pp.write_text(json.dumps(frozen.parent))
    import app.clients
    monkeypatch.setattr(app.clients, 'Search', lambda **kw: pytest.fail('invalid cache reached clients'))
    report = runner.execute(args, frozen.suite, frozen.selection, frozen.suite['tasks'],
        frozen.parent, retrieval.reserve_output(args.output))
    assert report['identity'] is None and not report['valid']
    assert all(len(rows) == 47 for rows in report['rows'].values())
    assert sentinel not in args.output.read_text()


def test_valid_cached_model_projection_preserves_provenance_and_nullable_versions(frozen):
    models = frozen.parent['identity']['models']
    models['libraries']['torch'] = None
    models['reranker']['revision'] = None
    projected = runner.cached_model_identity(models, runner._configuration(frozen.parent))
    assert projected == models and projected is not models
    projected['libraries']['httpx'] = '2.0.0'
    assert models['libraries']['httpx'] == '1.0.0'
    assert len(validate(frozen)) == 47
