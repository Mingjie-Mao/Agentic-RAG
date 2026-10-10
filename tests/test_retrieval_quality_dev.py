import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import run_retrieval_quality_dev as runner
from scripts.research_review import digest


def package(tmp_path):
    raw = tmp_path / 'source.txt'
    raw.write_text('alpha beta gamma delta')
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    tasks = [dict(id=f'T{i}', goal=f'question {i}', user='lt-eng', category='native',
                  requires_version=False, gold_evidence=[dict(id='fact', document_id='doc',
                  source_path='source.txt', source_sha256=sha, quote='alpha beta gamma delta')])
             for i in range(47)]
    suite = dict(split='dev', tasks=tasks)
    selection = dict(split='dev', count=47, items=[dict(id=t['id'],
                     query_sha256=hashlib.sha256(t['goal'].encode()).hexdigest(),
                     annotation_sha256=digest(t)) for t in tasks])
    return suite, selection


@pytest.mark.parametrize('mutation', ['core', 'duplicate', 'query', 'annotation', 'source', 'selection'])
def test_guards_full_frozen_dev(tmp_path, mutation):
    suite, selection = package(tmp_path)
    if mutation == 'core':
        suite['split'] = 'core'
    elif mutation == 'duplicate':
        suite['tasks'][1] = copy.deepcopy(suite['tasks'][0])
    elif mutation == 'query':
        suite['tasks'][0]['goal'] = 'changed'
    elif mutation == 'annotation':
        suite['tasks'][0]['category'] = 'changed'
    elif mutation == 'source':
        (tmp_path / 'source.txt').write_text('changed')
    else:
        selection['count'] = 46
    with pytest.raises(ValueError):
        runner.validate_suite(suite, selection, tmp_path)


def test_valid_package_and_labeled_subset(tmp_path):
    suite, selection = package(tmp_path)
    assert len(runner.validate_suite(suite, selection, tmp_path)) == 47
    assert runner.select_tasks(suite['tasks'], 'T3,T1')[0]['id'] == 'T3'
    for value in ('T3,T3', 'unknown'):
        with pytest.raises(ValueError):
            runner.select_tasks(suite['tasks'], value)


@pytest.mark.parametrize('value', ['unknown', 'baseline,baseline', '', 'baseline,'])
def test_profiles_reject_invalid(value):
    with pytest.raises(ValueError):
        runner.parse_profiles(value)


def test_configuration_ignores_application_experimental_defaults():
    from app.config import Settings
    cfg = runner.configuration(Settings(top_k=1, source_query_plan=True,
              passage_window_enabled=True, passage_rerank=True, source_facet_queries=True), 'baseline')
    assert (cfg.top_k, cfg.context_token_budget, cfg.retrieval_mode, cfg.source_routing) == (8, 5000, 'hybrid', True)
    assert not any((cfg.source_query_plan, cfg.passage_window_enabled, cfg.passage_rerank,
                    cfg.source_facet_queries, cfg.task_contract_enabled))
    assert runner.configuration(cfg, 'depth100').retrieval_candidate_depth == 100


def test_collection_uses_goal_only_and_reauthorizes(monkeypatch, tmp_path):
    suite, _ = package(tmp_path)
    task = suite['tasks'][0]
    checked = []
    chunk = SimpleNamespace(id='c', text='alpha beta gamma delta', locator={})
    version = SimpleNamespace(id='v', content_hash=task['gold_evidence'][0]['source_sha256'])
    doc = SimpleNamespace(id='doc', title='private title', metadata_json={})
    def authorize(db, user, cid, *, active_only):
        checked.append((cid, active_only))
        return chunk, version, doc
    def retrieve(db, user, goal, *, cfg, models, search):
        assert goal == task['goal']
        assert models.embed(['x']) == [[1.0]]
        with pytest.raises(AttributeError):
            models.generate
        return SimpleNamespace(candidates=[dict(chunk_id='c', rank=2, retrieval_rank=5,
            rerank_score=0.2, excluded_because=None, lane='global')], evidence=[dict(chunk_id='c')],
            context_tokens=120, embed_ms=1, retrieval_ms=2)
    def passages(db, user, goal, evidence, cfg, *, historical):
        assert goal == task['goal'] and historical is False
        assert evidence[0]['source_sha256'] == version.content_hash
        return evidence, {'context_tokens': 125}
    monkeypatch.setattr(runner, 'retrieve_authorized', retrieve)
    monkeypatch.setattr(runner, 'require_chunk', authorize)
    monkeypatch.setattr(runner, 'prepare_passages', passages)
    cfg = runner.configuration(None, 'depth50_window')
    row = runner.collect_task(None, None, task, cfg, SimpleNamespace(embed=lambda x: [[1.0]]), None)
    assert len(checked) == 3 and all(active for _, active in checked)
    assert row['phases']['candidates']['first_supporting_rank'] == 2
    assert row['telemetry']['candidates'][0]['retrieval_rank'] == 5
    assert row['telemetry']['admitted'][0]['rank'] == 1
    serialized = json.dumps(row)
    assert task['goal'] not in serialized and 'private title' not in serialized
    assert chunk.text not in serialized and 'gold_evidence' not in serialized


def test_failure_retains_nulls_and_not_applicable(monkeypatch, tmp_path):
    suite, _ = package(tmp_path)
    def fail(*a, **kw):
        raise RuntimeError('password secret source text')
    monkeypatch.setattr(runner, 'retrieve_authorized', fail)
    cfg = runner.configuration(None, 'baseline')
    row = runner.collect_task(None, None, suite['tasks'][0], cfg, None, None)
    assert row['phases'] == dict(candidates=None, admitted=None, context=None)
    assert row['telemetry']['context_tokens'] is None
    assert 'secret' not in json.dumps(row)
    history = dict(suite['tasks'][0], requires_version=True)
    assert runner.collect_task(None, None, history, cfg, None, None)['reason'] == 'requires_version'
    null = dict(suite['tasks'][0], gold_evidence=[])
    assert runner.collect_task(None, None, null, cfg, None, None)['reason'] == 'no_references'
    temporal = dict(suite['tasks'][0], category='temporal_version')
    assert runner.collect_task(None, None, temporal, cfg, None, None)['eligible']


def test_existing_artifact_or_reservation_rejected(tmp_path):
    output = tmp_path / 'result.json'
    runner.reserve_output(output)
    with pytest.raises(ValueError):
        runner.reserve_output(output)
    output.write_text('original')
    with pytest.raises(ValueError):
        runner.reserve_output(output)
    assert output.read_text() == 'original'


def test_invalid_main_never_constructs_clients(monkeypatch, tmp_path):
    suite, selection = package(tmp_path)
    suite['split'] = 'core'
    sp, sel = tmp_path / 'tasks.json', tmp_path / 'selection.json'
    sp.write_text(json.dumps(suite))
    sel.write_text(json.dumps(selection))
    monkeypatch.setattr(runner, 'SELECTION', sel)
    monkeypatch.setattr(runner, 'execute', lambda *a, **kw: pytest.fail('requests before guard'))
    with pytest.raises(ValueError):
        runner.main(['--suite', str(sp), '--output', str(tmp_path / 'out.json')])


def test_model_identity_ignores_unused_endpoints_and_hashes_encoder(monkeypatch, tmp_path):
    from app.config import Settings
    calls = []
    def get(url, **kw):
        calls.append(url)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda:
            {'models': [{'name': 'embed', 'digest': 'actual-embed-digest'}]}
            if url.endswith('/tags') else {'version': 'runtime-v1'})
    monkeypatch.setattr(runner.httpx, 'get', get)
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    from huggingface_hub import constants
    monkeypatch.setattr(constants, 'HF_HUB_CACHE', str(tmp_path / 'hub'))
    repo = tmp_path / 'hub/models--org--reranker'
    (repo / 'refs').mkdir(parents=True)
    (repo / 'refs/main').write_text('revision-1')
    snapshot = repo / 'snapshots/revision-1'
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{}')
    (snapshot / 'tokenizer.json').write_text('{}')
    (snapshot / 'tokenizer_config.json').write_text('{}')
    (snapshot / 'model.safetensors').write_bytes(b'weights')
    cfg = Settings(embed_model='embed', rerank_model='org/reranker',
                   agent_policy_url='http://unused', chat_model='not-installed')
    identity = runner.model_identity(cfg, rerank=True)
    assert all(url.startswith(cfg.ollama_url) for url in calls)
    assert identity['embedding']['digest'] == 'actual-embed-digest'
    assert identity['generation']['status'] == identity['policy']['status'] == 'unused'
    assert identity['reranker']['revision'] == 'revision-1'
    assert identity['reranker']['digest']
    assert identity['reranker']['files']['model.safetensors'] == hashlib.sha256(b'weights').hexdigest()
    assert identity['reranker']['files']['tokenizer.json'] == hashlib.sha256(b'{}').hexdigest()
    assert identity['reranker']['files']['tokenizer_config.json'] == hashlib.sha256(b'{}').hexdigest()


def test_context_preserves_authorized_substring_and_rejects_fabricated_text(monkeypatch):
    chunk = SimpleNamespace(id='c', text='prefix actual prepared passage suffix', locator={})
    version = SimpleNamespace(id='v', content_hash='sha')
    doc = SimpleNamespace(id='doc', title='title', metadata_json={})
    monkeypatch.setattr(runner, 'require_chunk', lambda *a, **kw: (chunk, version, doc))
    row = runner._bound(None, None, [{'chunk_id': 'c', 'text': 'actual prepared passage'}],
                        sequential=True, preserve_text=True)[0]
    assert row['text'] == 'actual prepared passage'
    with pytest.raises(ValueError):
        runner._bound(None, None, [{'chunk_id': 'c', 'text': 'fabricated gold sentence'}],
                      preserve_text=True)


def test_failed_initial_identity_retains_all_selected_rows_and_local_reservation(monkeypatch, tmp_path):
    suite, selection = package(tmp_path)
    monkeypatch.setattr(runner, 'RUNTIME', tmp_path / '.runtime/retrieval-quality')
    import app.clients
    monkeypatch.setattr(app.clients, 'Search', lambda *a, **kw: (_ for _ in ()).throw(
                       RuntimeError('private credential')))
    output = tmp_path / 'artifacts/out.json'
    reservation = runner.reserve_output(output)
    assert list(output.parent.iterdir()) == []
    args = SimpleNamespace(output=output, suite=tmp_path / 'tasks.json', task_ids=None)
    report = runner.execute(args, suite, selection, suite['tasks'], ['baseline', 'depth50'], reservation)
    assert report['status'] == 'failed' and not report['valid']
    assert all(len(rows) == 47 for rows in report['rows'].values())
    assert report['summaries']['baseline']['missing_tasks'] == 47
    assert report['pairs']['depth50']['missing_pairs'] == 47
    assert report['execution_errors']['baseline']['tasks'] == 47
    assert 'credential' not in output.read_text()
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        runner._publish(output, {'replacement': True})
    assert output.read_bytes() == original


def test_null_evidence_counts_missing_is_not_zero(tmp_path):
    suite, _ = package(tmp_path)
    missing = runner.missing_row(dict(suite['tasks'][0], gold_evidence=[]), None)
    counts = runner.null_evidence_counts([missing])
    assert counts['tasks'] == 1
    assert counts['candidate'] == dict(measured_tasks=0, missing_tasks=1, evidence_count=None)


@pytest.mark.parametrize('outcome', ['complete', 'drift', 'interrupt'])
def test_execute_interleaves_profiles_and_retains_failures(monkeypatch, tmp_path, outcome):
    import app.clients
    import app.db
    suite, selection = package(tmp_path)
    sp, sel = tmp_path / 'tasks.json', tmp_path / 'selection.json'
    sp.write_text(json.dumps(suite))
    sel.write_text(json.dumps(selection))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'RUNTIME', tmp_path / '.runtime/retrieval-quality')
    monkeypatch.setattr(runner, 'SELECTION', sel)
    class DB:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def execute(self, statement):
            assert str(statement) == 'SET TRANSACTION READ ONLY'
    monkeypatch.setattr(app.db, 'SessionLocal', DB)
    monkeypatch.setattr(app.clients, 'Models', lambda: SimpleNamespace(embed=lambda x: [[1.0]]))
    monkeypatch.setattr(app.clients, 'Search', lambda **kw: SimpleNamespace(index='production'))
    code_calls = []
    def codes():
        code_calls.append(1)
        return {'file': 'changed' if outcome == 'drift' and len(code_calls) > 1 else 'sha'}
    monkeypatch.setattr(runner, 'code_fingerprints', codes)
    monkeypatch.setattr(runner, 'model_identity', lambda cfg, **kw: {'embedding': 'frozen'})
    version = SimpleNamespace(id='v', content_hash=suite['tasks'][0]['gold_evidence'][0]['source_sha256'])
    doc = SimpleNamespace(id='doc', title='private title', metadata_json={})
    chunk = SimpleNamespace(id='c', text='alpha beta gamma delta', locator={})
    scope = {'documents': [{'document_id': 'doc', 'source_sha256': version.content_hash}], 'users': []}
    monkeypatch.setattr(runner, 'active_scope', lambda db: ({'lt-eng': object()}, [], scope))
    monkeypatch.setattr(runner, 'index_ledger', lambda *a: {'index': 'production', 'hash': 'frozen'})
    monkeypatch.setattr(runner, 'require_chunk', lambda *a, **kw: (chunk, version, doc))
    calls = []
    def retrieve(db, user, goal, **kw):
        calls.append((goal, kw['cfg'].retrieval_candidate_depth))
        if outcome == 'interrupt' and len(calls) == 3:
            raise KeyboardInterrupt()
        return SimpleNamespace(candidates=[dict(chunk_id='c', rank=1, retrieval_rank=1,
                               admitted=True)], evidence=[dict(chunk_id='c', text=chunk.text)],
                               context_tokens=120, embed_ms=1, retrieval_ms=2)
    monkeypatch.setattr(runner, 'retrieve_authorized', retrieve)
    output = tmp_path / 'artifacts/result.json'
    args = SimpleNamespace(output=output, suite=sp, task_ids='T0,T1,T2')
    report = runner.execute(args, suite, selection, suite['tasks'][:3], ['baseline', 'depth100'],
                            runner.reserve_output(output))
    assert calls[:3] == [('question 0', 0), ('question 0', 100), ('question 1', 100)]
    assert all(len(rows) == 3 for rows in report['rows'].values())
    if outcome == 'interrupt':
        assert report['status'] == 'interrupted'
        assert report['pairs']['depth100']['missing_pairs'] == 2
        checkpoint = next((runner.RUNTIME).glob('*/checkpoint.json'))
        data = json.loads(checkpoint.read_text())
        assert data['pending'] and len(data['completed']) == 2
    elif outcome == 'drift':
        assert not report['valid'] and report['status'] == 'invalid_drift'
        assert report['drift']['code']
    else:
        assert report['valid'] and report['status'] == 'complete'
        assert len(calls) == 6
        assert report['pairs']['depth100']['complete_pairs'] == 3
    assert report['strict_task_success'] is report['semantic_recall'] is None
    assert 'private title' not in output.read_text() and 'alpha beta' not in output.read_text()


def test_valid_cli_paths_validate_before_execution(monkeypatch, tmp_path):
    suite, selection = package(tmp_path)
    sp, sel = tmp_path / 'valid-tasks.json', tmp_path / 'selection.json'
    sp.write_text(json.dumps(suite))
    sel.write_text(json.dumps(selection))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'SELECTION', sel)
    monkeypatch.setattr(runner, 'RUNTIME', tmp_path / '.runtime/retrieval-quality')
    called = []
    def execute(args, loaded, frozen, tasks, profiles, reservation):
        called.append((args, loaded, frozen, tasks, profiles, reservation))
        return {'status': 'complete', 'valid': True}
    monkeypatch.setattr(runner, 'execute', execute)
    output = tmp_path / 'artifacts/new-quality.json'
    assert runner.main(['--suite', str(sp), '--output', str(output),
                        '--profiles', 'baseline,depth50', '--task-ids', 'T2,T0']) == 0
    assert len(called) == 1 and called[0][1] == suite and called[0][2] == selection
    assert [t['id'] for t in called[0][3]] == ['T2', 'T0']
    assert called[0][4] == ['baseline', 'depth50']
    assert called[0][5]['run_id']


@pytest.mark.parametrize('artifacts', [[], ['tokenizer_config.json'], ['tokenizer.json'],
                                     ['tokenizer_config.json', 'merges.txt']])
def test_reranker_missing_tokenizer_artifacts_rejected(monkeypatch, tmp_path, artifacts):
    from app.config import Settings
    monkeypatch.setattr(runner.httpx, 'get', lambda url, **kw: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {'models': [{'name': 'embed', 'digest': 'sha'}]}
        if url.endswith('/tags') else {'version': 'v1'}))
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    from huggingface_hub import constants
    monkeypatch.setattr(constants, 'HF_HUB_CACHE', str(tmp_path / 'hub'))
    repo = tmp_path / 'hub/models--org--reranker'
    (repo / 'refs').mkdir(parents=True)
    (repo / 'refs/main').write_text('revision')
    snapshot = repo / 'snapshots/revision'
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{}')
    (snapshot / 'model.safetensors').write_bytes(b'weights')
    for name in artifacts:
        (snapshot / name).write_text('{}')
    with pytest.raises(ValueError, match='tokenizer'):
        runner.model_identity(Settings(embed_model='embed', rerank_model='org/reranker'), rerank=True)


@pytest.mark.parametrize('cache_mode', ['HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'app_default',
                                        'already_imported_constants'])
def test_reranker_fingerprint_uses_loader_cache(monkeypatch, tmp_path, cache_mode):
    import os
    from pathlib import Path
    import subprocess
    import sys

    project = Path(runner.__file__).resolve().parents[1]
    application_home = tmp_path / '.runtime/huggingface'
    if cache_mode == 'app_default':
        actual_cache = application_home / 'hub'
    elif cache_mode == 'already_imported_constants':
        actual_cache = tmp_path / 'early-cache/huggingface/hub'
    else:
        actual_cache = tmp_path / 'override-cache'
    def cached_revision(cache, revision, marker):
        repo = cache / 'models--org--reranker'
        (repo / 'refs').mkdir(parents=True, exist_ok=True)
        (repo / 'refs/main').write_text(revision)
        snapshot = repo / 'snapshots' / revision
        snapshot.mkdir(parents=True)
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
            (snapshot / name).write_text('{}')
        (snapshot / 'model.safetensors').write_bytes(marker.encode())
    if cache_mode != 'app_default':
        cached_revision(application_home / 'hub', 'a' * 40, 'wrong-registered-weights')
    cached_revision(actual_cache, 'c' * 40, 'changed-executed-weights')
    cached_revision(actual_cache, 'b' * 40, 'actual-executed-weights')
    env = dict(os.environ)
    for name in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        env.pop(name, None)
    env['XDG_CACHE_HOME'] = str(tmp_path / 'early-cache')
    if cache_mode in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        env['HF_HOME'] = str(application_home)
        env[cache_mode] = str(actual_cache)
    script = '''
import sys, json
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
if sys.argv[2] == 'already_imported_constants':
    import huggingface_hub.constants
    import os
    os.environ['HF_HOME'] = sys.argv[3]
from scripts import run_retrieval_quality_dev as runner
from app.config import Settings
runner.httpx.get = lambda url, **kw: SimpleNamespace(raise_for_status=lambda: None,
    json=lambda: {'models': [{'name': 'embed', 'digest': 'sha'}]}
    if url.endswith('/tags') else {'version': 'v1'})
identity = runner.model_identity(Settings(embed_model='embed', rerank_model='org/reranker'), rerank=True)
from transformers.utils.hub import cached_file
actual = cached_file('org/reranker', 'config.json', local_files_only=True)
from pathlib import Path
(Path(actual).parents[2] / 'refs/main').write_text('c' * 40)
after_drift = runner.model_identity(Settings(embed_model='embed', rerank_model='org/reranker'), rerank=True)
print(json.dumps({'identity': identity['reranker'], 'actual_config': actual,
                  'after_drift': after_drift['reranker']}))
'''
    result = subprocess.run([sys.executable, '-c', script, str(project), cache_mode, str(application_home)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    assert value['identity']['revision'] == Path(value['actual_config']).parent.name == 'b' * 40
    assert value['identity']['files']['model.safetensors'] == hashlib.sha256(
        b'actual-executed-weights').hexdigest()
    assert value['after_drift']['revision'] == 'c' * 40
    assert value['after_drift']['digest'] != value['identity']['digest']
    assert value['after_drift']['files']['model.safetensors'] == hashlib.sha256(
        b'changed-executed-weights').hexdigest()
