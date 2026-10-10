"""Frozen Dev-only retrieval component diagnostics, without answer generation.

Run as ``python -m scripts.run_retrieval_quality_dev --output artifacts/new.json``.
Raw task/source text is used in memory and never serialized into derived artifacts.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from sqlalchemy import text

from app.config import Settings, settings
from app.passages import prepare_passages
from app.retrieval import retrieve_authorized
from app.security import require_chunk
from scripts.benchmark_runtime import CheckpointStore, atomic_json, checkpoint_lock
from scripts.research_review import digest
from scripts.retrieval_quality_index import (DEV_USERS, active_scope, build_keyword_index,
    index_ledger, verify_keyword_index)
from scripts.retrieval_quality_metrics import diagnose, pair_profiles, summarize

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime/retrieval-quality'
SELECTION = ROOT / 'benchmarks/enterprise_rag/v1/dev/selection.json'
PROFILES = {'baseline': {}, 'depth50': {'retrieval_candidate_depth': 50},
    'depth100': {'retrieval_candidate_depth': 100},
    'depth50_window': {'retrieval_candidate_depth': 50, 'passage_window_enabled': True},
    'depth50_rerank': {'retrieval_candidate_depth': 50, 'passage_rerank': True},
    'keyword_context': {}}
CONFIG_KEYS = ('top_k', 'context_token_budget', 'retrieval_mode', 'source_routing',
    'rewrite_mode', 'rerank_mode', 'retrieval_candidate_depth', 'passage_window_enabled',
    'passage_rerank', 'passage_rerank_depth', 'passage_expand_documents', 'document_quota',
    'passage_window_extra', 'passage_scan_limit', 'source_query_plan', 'source_clause_queries',
    'source_focus_queries', 'source_facet_queries', 'source_facet_document_queries',
    'article_first_lanes', 'task_contract_enabled', 'embed_model', 'embed_dimension',
    'rerank_model', 'rerank_batch', 'rerank_max_tokens', 'model_timeout_seconds')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_suite(suite, selection, root=ROOT):
    tasks, items = suite.get('tasks', []), selection.get('items', [])
    if (suite.get('split') != 'dev' or selection.get('split') != 'dev'
            or selection.get('count') != 47 or len(tasks) != 47 or len(items) != 47):
        raise ValueError('exact frozen 47-task Dev package required')
    if len({t['id'] for t in tasks}) != 47 or len({t['id'] for t in items}) != 47:
        raise ValueError('duplicate Dev IDs')
    actual = {t['id']: {'id': t['id'], 'query_sha256': sha(t['goal'].encode()),
                        'annotation_sha256': digest(t)} for t in tasks}
    if actual != {t['id']: t for t in items}:
        raise ValueError('Dev query/annotation selection differs')
    checked = set()
    for task in tasks:
        if task.get('user') not in DEV_USERS:
            raise ValueError('unknown Dev user')
        for ref in task.get('gold_evidence', []):
            path = (Path(root) / ref['source_path']).resolve()
            if not path.is_relative_to(Path(root).resolve()):
                raise ValueError('source path outside repository')
            binding = (path, ref['source_sha256'])
            if binding not in checked and sha(path.read_bytes()) != ref['source_sha256']:
                raise ValueError('gold source bytes changed')
            checked.add(binding)
    return tasks


def parse_profiles(value):
    names = value.split(',')
    if len(set(names)) != len(names) or any(name not in PROFILES for name in names):
        raise ValueError('unknown or duplicate profile name')
    return names


def select_tasks(tasks, value):
    if value is None:
        return tasks
    ids = value.split(',')
    by_id = {t['id']: t for t in tasks}
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
        raise ValueError('unknown or duplicate subset task IDs')
    return [by_id[tid] for tid in ids]


def configuration(base, profile):
    base = base or Settings(_env_file=None)
    # Explicitly disable all experimental boolean settings, even when .env enables
    # them. Numeric retrieval settings are also fixed rather than inherited.
    safe = {name: False for name, field in Settings.model_fields.items() if field.annotation is bool}
    safe.update(top_k=8, context_token_budget=5000, retrieval_mode='hybrid', source_routing=True,
        rewrite_mode='off', rerank_mode='off', retrieval_candidate_depth=0,
        passage_rerank_depth=24, passage_expand_documents=0, document_quota=2,
        passage_window_extra=2, passage_scan_limit=64, min_similarity=0.35)
    return base.model_copy(update=safe | PROFILES[profile])


class EmbeddingOnly:
    """Retrieval receives one model capability; planner/generator access is impossible."""
    __slots__ = ('embed',)

    def __init__(self, models):
        self.embed = models.embed


def _bound(db, user, rows, *, sequential=False, preserve_text=False):
    bound = []
    for position, row in enumerate(rows, 1):
        chunk, version, doc = require_chunk(db, user, row['chunk_id'], active_only=True)
        prepared_text = row.get('text') if preserve_text else chunk.text
        if not isinstance(prepared_text, str) or not prepared_text or prepared_text not in chunk.text:
            raise ValueError('prepared context text is not a source substring')
        bound.append(dict(row, chunk_id=chunk.id, version_id=version.id, document_id=doc.id,
            source_sha256=version.content_hash, text=prepared_text, title=doc.title,
            metadata=doc.metadata_json, locator=chunk.locator,
            **({'rank': position} if sequential else {})))
    return bound


def _telemetry(rows):
    fields = ('chunk_id', 'document_id', 'version_id', 'source_sha256', 'rank',
              'retrieval_rank', 'rerank_score', 'excluded_because', 'admitted',
              'bm25_rank', 'dense_rank', 'fusion_score')
    return [{k: row.get(k) for k in fields} | {
        # Source lane names contain publication text; keep only a hash and type.
        'lane_type': 'source' if str(row.get('lane', '')).startswith('source:') else row.get('lane'),
        'lane_sha256': digest(row['lane']) if row.get('lane') else None} for row in rows]


def collect_task(db, user, task, cfg, models, search):
    phases = dict(candidates=None, admitted=None, context=None)
    telemetry = dict(wall_ms=None, embed_ms=None, retrieval_ms=None, context_ms=None,
        context_tokens=None, candidate_count=None, admitted_count=None, context_count=None,
        excluded_counts=None, candidates=None, admitted=None, context=None)
    started, stage, error = time.monotonic(), 'retrieval', None
    try:
        found = retrieve_authorized(db, user, task['goal'], cfg=cfg,
            models=EmbeddingOnly(models) if models is not None else None, search=search)
        telemetry.update(embed_ms=found.embed_ms, retrieval_ms=found.retrieval_ms)
        stage = 'candidate_authorization'
        phases['candidates'] = _bound(db, user, found.candidates)
        telemetry.update(candidate_count=len(phases['candidates']),
            candidates=_telemetry(phases['candidates']),
            excluded_counts=dict(Counter(r.get('excluded_because') for r in found.candidates
                                         if r.get('excluded_because'))))
        stage = 'admission_authorization'
        ranks = {r['chunk_id']: r for r in found.candidates if r.get('admitted')}
        admitted = [{**ranks.get(r['chunk_id'], {}), **r} for r in found.evidence]
        phases['admitted'] = _bound(db, user, admitted, sequential=True)
        telemetry.update(admitted_count=len(phases['admitted']), admitted=_telemetry(phases['admitted']))
        stage = 'context'
        context_started = time.monotonic()
        if cfg.passage_window_enabled:
            context, report = prepare_passages(db, user, task['goal'], phases['admitted'], cfg,
                                              historical=False)
            tokens = report['context_tokens']
        else:
            context, tokens = phases['admitted'], found.context_tokens
        stage = 'context_authorization'
        phases['context'] = _bound(db, user, context, sequential=True, preserve_text=True)
        telemetry.update(context_count=len(phases['context']), context=_telemetry(phases['context']),
            context_tokens=tokens, context_ms=(time.monotonic() - context_started) * 1000)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'stage': stage}
    telemetry['wall_ms'] = (time.monotonic() - started) * 1000
    return diagnose(task, phases) | {'telemetry': telemetry, 'error': error}


def missing_row(task, error):
    telemetry = {k: None for k in ('wall_ms', 'embed_ms', 'retrieval_ms', 'context_ms',
        'context_tokens', 'candidate_count', 'admitted_count', 'context_count', 'excluded_counts',
        'candidates', 'admitted', 'context')}
    return diagnose(task, {}) | {'telemetry': telemetry, 'error': error or {
        'type': 'Unavailable', 'stage': 'not_attempted'}}


def null_evidence_counts(rows):
    nulls = [r for r in rows if r['reason'] == 'no_references']
    result = {'tasks': len(nulls)}
    for phase in ('candidate', 'admitted', 'context'):
        measured = [r['telemetry'][phase + '_count'] for r in nulls
                    if r['telemetry'][phase + '_count'] is not None]
        result[phase] = {'measured_tasks': len(measured), 'missing_tasks': len(nulls) - len(measured),
                         'evidence_count': sum(measured) if measured else None}
    return result


def reserve_output(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation = RUNTIME / 'reservations' / sha(str(output.resolve()).encode()) / 'reservation.json'
    reservation.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('output already exists')
    try:
        with reservation.open('x') as stream:
            json.dump({'output': str(output.resolve()), 'run_id': uuid.uuid4().hex}, stream)
    except FileExistsError as exc:
        raise ValueError('output already reserved') from exc
    return json.loads(reservation.read_text())


def _publish(output, report):
    # Exclusive final creation; atomic_json is used only for mutable local runtime
    # checkpoints. Existing historical output is never replaced, including races.
    with Path(output).open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def code_fingerprints():
    paths = [p for folder in ('app', 'agent', 'scripts') for p in (ROOT / folder).rglob('*.py')]
    paths += [ROOT / 'pyproject.toml', ROOT / 'AGENTS.md']
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted(paths) if p.is_file()}


def model_identity(cfg, *, rerank=False):
    from importlib.metadata import PackageNotFoundError, version

    tags = httpx.get(cfg.ollama_url + '/api/tags', timeout=20, trust_env=False)
    tags.raise_for_status()
    runtime = httpx.get(cfg.ollama_url + '/api/version', timeout=20, trust_env=False)
    runtime.raise_for_status()
    model = next((r for r in tags.json()['models'] if r['name'] == cfg.embed_model), None)
    if model is None or not model.get('digest'):
        raise ValueError('embedding model identity unavailable')
    records = {'embedding': {'name': cfg.embed_model, 'digest': model['digest'],
        'runtime': runtime.json(), 'endpoint_sha256': digest(cfg.ollama_url)},
        'generation': {'name': cfg.chat_model, 'status': 'unused', 'digest': None},
        'policy': {'name': cfg.agent_policy_model, 'status': 'unused', 'digest': None},
        'reranker': {'name': cfg.rerank_model, 'status': 'unused', 'digest': None}}
    if rerank:
        # Initialize the application's HF_HOME default before Transformers; its
        # official resolver also honours already-imported constants and cache overrides.
        import app.rerank  # noqa: F401
        from transformers.utils.hub import cached_file

        config_path = cached_file(cfg.rerank_model, 'config.json', local_files_only=True)
        snapshot = Path(config_path).parent
        revision = snapshot.name if snapshot.parent.name == 'snapshots' else None
        files = {}
        for path in sorted(snapshot.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.safetensors', '.bin', '.model', '.txt'}:
                with path.open('rb') as stream:
                    files[str(path.relative_to(snapshot))] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if not any(n.endswith(('.safetensors', '.bin')) for n in files) or 'config.json' not in files:
            raise ValueError('reranker weights/config unavailable')
        vocabulary = any(n in files for n in ('tokenizer.json', 'vocab.txt', 'tokenizer.model',
            'sentencepiece.bpe.model', 'spiece.model')) or {'vocab.json', 'merges.txt'} <= files.keys()
        if not vocabulary or 'tokenizer_config.json' not in files:
            raise ValueError('reranker tokenizer vocabulary/config unavailable')
        records['reranker'] = {'name': cfg.rerank_model, 'status': 'active',
            'revision': revision, 'snapshot_path_sha256': digest(str(snapshot.absolute())),
            'files': files, 'digest': digest([str(snapshot.absolute()), revision, files]),
            'batch': cfg.rerank_batch, 'max_tokens': cfg.rerank_max_tokens}
    libraries = {}
    for name in ('httpx', 'pydantic', 'sqlalchemy', 'torch', 'transformers', 'tokenizers'):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    return records | {'libraries': libraries}


def _validate_active_gold(tasks, scope):
    hashes = {r['document_id']: r['source_sha256'] for r in scope['documents']}
    for task in tasks:
        if task.get('requires_version'):
            continue
        for ref in task.get('gold_evidence', []):
            if hashes.get(ref['document_id']) != ref['source_sha256']:
                raise ValueError('active source differs from Dev reference')


def execute(args, suite, selection, tasks, profiles, reservation):
    from app.clients import Models, Search
    from app.db import SessionLocal

    output, run_id = Path(args.output), reservation['run_id']
    checkpoint = RUNTIME / run_id / 'checkpoint.json'
    report = {'version': 'retrieval-quality-dev-v1', 'split': 'dev', 'status': 'failed',
        'valid': False, 'run_id': run_id, 'strict_task_success': None, 'semantic_recall': None,
        'subset': args.task_ids is not None, 'selected_task_ids': [t['id'] for t in tasks],
        'rows': {name: [] for name in profiles}, 'identity': None, 'error': None}
    stage = 'initial_identity'
    try:
        cfg = settings()
        configs = {name: configuration(cfg, name) for name in profiles}
        code = code_fingerprints()
        source = Search()
        keyword = Search(index=f'{source.index}-rq-{run_id}') if 'keyword_context' in profiles else None
        with SessionLocal() as db, checkpoint_lock(checkpoint):
            db.execute(text('SET TRANSACTION READ ONLY'))
            users, records, scope = active_scope(db)
            _validate_active_gold(suite['tasks'], scope)
            original_index = index_ledger(source, records)
            identity = {'suite_sha256': digest(suite), 'selection_sha256': digest(selection),
                'suite_bytes_sha256': sha(Path(args.suite).read_bytes()),
                'selection_bytes_sha256': sha(SELECTION.read_bytes()), 'code': code,
                'selected_task_ids': report['selected_task_ids'], 'profiles': [
                    {'name': n, 'overrides': PROFILES[n], 'effective_configuration': {
                        k: getattr(configs[n], k) for k in CONFIG_KEYS}} for n in profiles],
                'models': model_identity(cfg, rerank='depth50_rerank' in profiles), 'corpus': scope, 'corpus_sha256': digest(scope),
                'source_index': original_index, 'source_index_sha256': digest(original_index),
                'runtime': {'python': platform.python_version(), 'platform': platform.platform()},
                'keyword_target': keyword.index if keyword else None}
            report['identity'] = identity
            store = CheckpointStore(checkpoint, identity)
            index_record_path = checkpoint.with_name('keyword-index.json')
            if keyword:
                stage = 'keyword_index'
                index_record = {'status': 'building', 'source_index': source.index,
                                'target_index': keyword.index, 'run_id': run_id}
                atomic_json(index_record_path, index_record)
                def record_index_progress(progress):
                    index_record.update(progress)
                    atomic_json(index_record_path, index_record)
                try:
                    index_record = build_keyword_index(source, keyword, records,
                                                       on_progress=record_index_progress)
                    index_record = verify_keyword_index(source, keyword, records, index_record)
                except BaseException as exc:
                    atomic_json(index_record_path, index_record | {'status': 'interrupted' if isinstance(
                        exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                        'error': {'type': type(exc).__name__, 'stage': stage}})
                    raise
                atomic_json(index_record_path, index_record)
                report['keyword_index'] = index_record
                # Index identity is persisted before the first retrieval.
                atomic_json(checkpoint.with_name('retrieval-identity.json'), identity | {
                    'keyword_index_identity': index_record['index_identity']})
            models = Models()
            stage = 'retrieval'
            for offset, task in enumerate(tasks):
                order = profiles[offset % len(profiles):] + profiles[:offset % len(profiles)]
                for profile in order:
                    store.begin(profile, task['id'], uuid.uuid4().hex)
                    row = collect_task(db, users[task['user']], task, configs[profile], models,
                                       keyword if profile == 'keyword_context' else source)
                    store.finish(profile, task['id'], row)
                    report['rows'][profile].append(row)
            stage = 'drift_check'
            _, final_records, final_scope = active_scope(db)
            final_index = index_ledger(source, final_records)
            drift = {'code': code_fingerprints() != code, 'corpus': final_scope != scope,
                'source_index': final_index != original_index,
                'suite': sha(Path(args.suite).read_bytes()) != identity['suite_bytes_sha256'],
                'selection': sha(SELECTION.read_bytes()) != identity['selection_bytes_sha256'],
                'models': model_identity(cfg, rerank='depth50_rerank' in profiles) != identity['models']}
            try:
                validate_suite(suite, selection, ROOT)
                drift['raw_sources'] = False
            except (ValueError, OSError):
                drift['raw_sources'] = True
            report['drift'] = drift
            if keyword:
                verified = verify_keyword_index(source, keyword, final_records, report['keyword_index'])
                drift['keyword_index'] = verified['index_identity'] != report['keyword_index']['index_identity']
            report.update(drift=drift, valid=not any(drift.values()),
                          status='invalid_drift' if any(drift.values()) else 'complete')
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'stage': stage}
        report['status'] = ('interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else
                            'invalid_verification' if stage == 'drift_check' else 'failed')
        report['valid'] = False
    finally:
        for name, rows in report['rows'].items():
            completed = {r['task_id']: r for r in rows}
            report['rows'][name] = [completed.get(t['id']) or missing_row(t, report['error'])
                                    for t in tasks]
        report['summaries'] = {name: summarize(rows) for name, rows in report['rows'].items()}
        report['null_evidence_counts'] = {name: null_evidence_counts(rows)
                                          for name, rows in report['rows'].items()}
        report['execution_errors'] = {name: {'tasks': sum(r['error'] is not None for r in rows),
            'not_applicable_tasks': sum(r['error'] is not None and not r['eligible'] for r in rows),
            'by_stage': dict(Counter(r['error']['stage'] for r in rows if r['error']))}
            for name, rows in report['rows'].items()}
        report['pairs'] = {name: pair_profiles(report['rows']['baseline'], rows)
            for name, rows in report['rows'].items() if name != 'baseline'
            and 'baseline' in report['rows'] and len(rows) == len(report['rows']['baseline'])}
        _publish(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--suite', default=str(ROOT / '.runtime/benchmark-package/v1/dev/tasks.json'))
    parser.add_argument('--profiles', default=','.join(PROFILES))
    parser.add_argument('--task-ids', help='explicitly labeled Dev subset, comma-separated IDs')
    args = parser.parse_args(argv)
    suite, selection = json.loads(Path(args.suite).read_text()), json.loads(SELECTION.read_text())
    all_tasks = validate_suite(suite, selection, ROOT)
    profiles = parse_profiles(args.profiles)
    tasks = select_tasks(all_tasks, args.task_ids)
    reservation = reserve_output(args.output)
    report = execute(args, suite, selection, tasks, profiles, reservation)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'status': report['status'],
                      'valid': report['valid']}))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
