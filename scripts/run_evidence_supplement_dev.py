"""Frozen Dev matched supplement; raw evidence stays in memory and costs are observed."""

import argparse
from collections import Counter
from copy import deepcopy
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
import time
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from sqlalchemy import text

from agent.evidence_supplement import run_supplement
from app.clients import DependencyError, Models, Search
from app.config import settings
from app.execution_budget import ExecutionBudget
from app.retrieval import retrieve_authorized, route_sources, routing_goal
from app.security import readable_documents
from scripts import run_evidence_selection_dev as replay
from scripts.benchmark_runtime import CheckpointStore, atomic_json, checkpoint_lock
from scripts.research_review import digest
from scripts.retrieval_quality_index import active_scope, index_ledger
from scripts.retrieval_quality_metrics import diagnose, pair_profiles, summarize
from scripts.run_retrieval_quality_dev import (
    CONFIG_KEYS, ROOT, SELECTION, EmbeddingOnly, _bound, _publish, code_fingerprints,
    configuration, missing_row, null_evidence_counts, reserve_output, sha, validate_suite,
)

RUNTIME = ROOT / '.runtime/evidence-supplement'
REGISTRATION = ROOT / '.runtime/retrieval-selection-20261009/supplement-seed-registration.json'
PROFILES = ('seed', 'control', 'bridge')
FROZEN_PARENT_SHA = '3ebdd372139b0e24a7237b9db99475a5306a717d4b114236f9b67b905a7c74e6'
FROZEN_SEED_SHA = '3d0f0f1c43766723863e93da4845985800e0ee6d870e04237c2283b2f2840da9'


def supplement_configuration(base):
    return configuration(base, 'depth50').model_copy(update={
        'agent_task_timeout_seconds': 180, 'agent_policy_max_calls': 2})


def live_configuration():
    cfg = settings()
    return {k: getattr(cfg, k) for k in ('embed_model', 'embed_dimension',
        'agent_policy_model', 'model_timeout_seconds', 'answer_quality_enabled')} | {
        'embedding_endpoint_sha256': digest(cfg.ollama_url),
        'policy_endpoint_sha256': digest(cfg.agent_policy_url or cfg.ollama_url)}


def validate_seed(seed, parent, suite, parent_sha, profile):
    """Only the measured immutable legacy control can initialize this experiment."""
    try:
        if (profile != 'legacy_rerank' or seed.get('version') != 'evidence-selection-dev-v1'
                or seed.get('valid') is not True or seed.get('status') != 'complete'
                or seed.get('split') != 'dev' or seed.get('stage') != 'selection'
                or seed.get('subset') is not False or seed.get('error') is not None
                or any(seed.get('drift', {}).values())):
            raise ValueError('complete frozen legacy selection seed required')
        ids = [t['id'] for t in suite['tasks']]
        identity, expected = seed['identity'], parent['identity']
        if (len(ids) != 47 or len(set(ids)) != 47 or seed['selected_task_ids'] != ids
                or identity['selected_task_ids'] != ids
                or identity['parent_bytes_sha256'] != parent_sha
                or identity['parent_profile'] != replay.PARENT_PROFILE
                or identity['method'] != 'cached_candidate_replay_v1'
                or identity['stage'] != 'selection'
                or identity['profiles'] != list(replay.SELECTION_PROFILES)
                or identity['profile_declarations'] != replay.profile_declarations(replay.SELECTION_PROFILES)):
            raise ValueError('seed declaration or input identity differs')
        for key in ('suite_sha256', 'selection_sha256', 'suite_bytes_sha256',
                    'selection_bytes_sha256', 'corpus', 'corpus_sha256',
                    'source_index', 'source_index_sha256'):
            if identity[key] != expected[key]:
                raise ValueError('seed source/package differs')
        cfg = replay.effective_configuration(parent)
        if (identity['effective_retrieval_configuration'] != cfg
                or replay.cached_model_identity(identity['cached_models'], cfg) != expected['models']):
            raise ValueError('seed configuration/model identity differs')
        for name in replay.SELECTION_PROFILES:
            rows = seed['rows'][name]
            if ([r['task_id'] for r in rows] != ids or seed['execution_errors'][name]['tasks'] != 0
                    or any(r['error'] is not None for r in rows)):
                raise ValueError('seed rows incomplete or failed')
        rows = seed['rows'][profile]
        for old, current in zip(parent['rows'][replay.PARENT_PROFILE], rows, strict=True):
            for key in ('task_id', 'basis', 'category', 'eligible', 'status', 'cause',
                        'reason', 'reference_count', 'reference_fingerprint', 'phases', 'losses'):
                if current[key] != old[key]:
                    raise ValueError('legacy seed diagnostics differ')
            for phase in ('candidates', 'admitted', 'context'):
                if len(current['telemetry'][phase]) != len(old['telemetry'][phase]):
                    raise ValueError('legacy seed evidence count differs')
                for a, b in zip(old['telemetry'][phase], current['telemetry'][phase], strict=True):
                    replay.validate_telemetry_row(b)
                    if any(b.get(k) != value for k, value in a.items()):
                        raise ValueError('legacy seed evidence identity/order differs')
            if current['telemetry']['context_tokens'] != old['telemetry']['context_tokens']:
                raise ValueError('legacy seed budget differs')
        return rows
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('malformed seed artifact') from exc


def active_model_identity(cfg):
    """Resolve actual endpoints and digests, without assuming policy and embed share a server."""
    endpoints, result = {}, {}
    for role, endpoint, name in (
            ('embedding', cfg.ollama_url, cfg.embed_model),
            ('policy', cfg.agent_policy_url or cfg.ollama_url, cfg.agent_policy_model)):
        if endpoint not in endpoints:
            tags = httpx.get(endpoint + '/api/tags', timeout=20, trust_env=False)
            tags.raise_for_status()
            runtime = httpx.get(endpoint + '/api/version', timeout=20, trust_env=False)
            runtime.raise_for_status()
            data, server = tags.json(), runtime.json()
            if not isinstance(data.get('models'), list) or not replay._version(server.get('version')):
                raise ValueError('active endpoint identity unavailable')
            endpoints[endpoint] = (data['models'], {'version': server['version']})
        models, runtime = endpoints[endpoint]
        matches = [m for m in models if m.get('name') == name]
        if (len(matches) != 1 or not replay._identifier(name)
                or not replay._sha256(matches[0].get('digest'))):
            raise ValueError('actual active model digest unavailable')
        result[role] = {'name': name, 'digest': matches[0]['digest'], 'runtime': runtime,
                       'endpoint_sha256': digest(endpoint), 'status': 'active'}
    result['libraries'] = {n: version(n) for n in ('httpx', 'pydantic', 'sqlalchemy', 'langgraph')}
    result['generation'] = {'status': 'unused', 'digest': None}
    result['supplement_reranker'] = {'status': 'unused', 'digest': None}
    return result


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


class MeteredModels(Models):
    """Expose no prompt, vector, answer or endpoint in experiment telemetry."""
    def __init__(self):
        super().__init__()
        self.events = []

    def _post(self, path, body, base=None):
        started = time.monotonic()
        event = {'role': 'embedding' if path == '/api/embed' else 'policy',
            'request_sha256': digest([path, body]), 'status': 'failed',
            'prompt_tokens': None, 'completion_tokens': None, 'server_ms': None,
            'input_count': len(body.get('input', [])) if path == '/api/embed' else None}
        try:
            result = super()._post(path, body, base=base)
            event.update(status='ok', prompt_tokens=_number(result.get('prompt_eval_count')),
                completion_tokens=_number(result.get('eval_count')),
                server_ms=(_number(result.get('total_duration')) / 1e6
                           if _number(result.get('total_duration')) is not None else None))
            return result
        finally:
            event['wall_ms'] = (time.monotonic() - started) * 1000
            self.events.append(event)


class MeteredSearch(Search):
    def __init__(self, source):
        self.source, self.index, self.events = source, source.index, []

    def request(self, method, path, **kwargs):
        started = time.monotonic()
        event = {'request_sha256': digest([method, path, kwargs]), 'status': 'failed'}
        try:
            result = self.source.request(method, path, **kwargs)
            event['status'] = 'ok'
            return result
        finally:
            event['wall_ms'] = (time.monotonic() - started) * 1000
            self.events.append(event)


def event_summary(events):
    result = {'attempted': len(events), 'succeeded': sum(e['status'] == 'ok' for e in events),
        'failed': sum(e['status'] != 'ok' for e in events),
        'wall_ms': sum(e['wall_ms'] for e in events)}
    for key in ('prompt_tokens', 'completion_tokens', 'server_ms'):
        values = [e.get(key) for e in events]
        unknown = sum(v is None for v in values)
        result[key] = None if unknown else sum(values)
        result['known_' + key] = sum(v for v in values if v is not None)
        result['unknown_' + key] = unknown
    return result


def scope_validator(db, user, question):
    def scopes(query):
        groups = route_sources(query, readable_documents(db, user), strict_dates=False)
        return sorted((tuple(sorted(g['key'])), tuple(sorted(g['versions'])),
                       bool(g.get('bounded'))) for g in groups)
    frozen = scopes(question)
    def check(original, full_query):
        return original == question and scopes(original) == frozen and scopes(full_query) == frozen
    return check, frozen


def live_candidates(db, user, found, corpus):
    rows = []
    for row in _bound(db, user, found.candidates):
        lane = row.get('lane')
        clean = {k: row.get(k) for k in replay.ROW_FIELDS if k not in ('lane_type', 'lane_sha256')}
        clean.update(rank=row['rank'], parent_rank=row['rank'],
            lane_type='source' if str(lane).startswith('source:') else lane,
            lane_sha256=digest(lane) if lane else None)
        rows.append(clean)
    return replay.rebind_rows(db, user, rows, corpus)


def public_rows(rows):
    projected = replay.public_telemetry(rows)
    for raw, item in zip(rows, projected, strict=True):
        if 'original_parent_rank' in raw:
            value = raw['original_parent_rank']
            if type(value) is not int or value < 1 or raw.get('supplement_origin') not in (
                    'original', 'supplement'):
                raise ValueError('invalid supplemental rank provenance')
            item.update(original_parent_rank=value, supplement_origin=raw['supplement_origin'])
    return projected


def score_context(db, user, task, pool, evidence, corpus):
    rebound = replay.rebind_rows(db, user, evidence, corpus)
    tokens = replay.evidence_tokens(rebound)
    replay.validate_context(rebound, tokens)
    chosen = [{**raw, **bound, 'rank': i, 'admitted': True, 'excluded_because': None}
              for i, (raw, bound) in enumerate(zip(evidence, rebound, strict=True), 1)]
    bound_pool = replay.rebind_rows(db, user, pool, corpus)
    candidates = [dict(raw, **bound) for raw, bound in zip(pool, bound_pool, strict=True)]
    phases = {'candidates': candidates, 'admitted': chosen, 'context': chosen}
    return diagnose(task, phases) | {'error': None, 'telemetry': {
        'context_tokens': tokens, 'candidate_count': len(candidates), 'admitted_count': len(chosen),
        'context_count': len(chosen), **{p: public_rows(rs) for p, rs in phases.items()}}}


def collect_pair(db, user, task, parent_row, seed_row, corpus, cfg, source, runtime, offset,
                 *, seed_profile='legacy_rerank'):
    rows, stage, interruption = {}, 'seed_reauthorization', None
    models, backend = MeteredModels(), MeteredSearch(source)
    search_usage = {}
    started = time.monotonic()
    trace = {'status': 'operational_failure', 'errors': []}
    try:
        phases = replay.rehydrate_task(db, user, parent_row, corpus)
        replay.verify_legacy(task, parent_row, phases)
        seed = replay.rebind_rows(db, user, seed_row['telemetry']['context'], corpus)
        replay.validate_context(seed, seed_row['telemetry']['context_tokens'])
        caps = replay.policy_caps(db, user, task['goal'])
        original_candidates = phases['candidates']
        if seed_profile == 'legacy_rerank':
            if [r['chunk_id'] for r in seed] != [r['chunk_id'] for r in phases['context']]:
                raise ValueError('frozen seed differs from parent legacy context')
        else:
            from app.evidence_selection import rank_relevance_candidates
            from scripts.run_revised_evidence_dev import RELEVANCE_PROFILES, select_relevance_context
            if seed_profile not in RELEVANCE_PROFILES:
                raise ValueError('unregistered seed profile')
            expected = select_relevance_context(db, user, task, parent_row, corpus, seed_profile)
            if (expected['telemetry']['context'] != seed_row['telemetry']['context']
                    or expected['telemetry']['context_tokens'] != seed_row['telemetry']['context_tokens']):
                raise ValueError('frozen relevance seed differs from reproduced selection')
            original_candidates = rank_relevance_candidates(original_candidates)[0]
            if seed_profile in ('relevant_doc', 'relevant_both'):
                caps['document_quota'] = None
            if seed_profile in ('relevant_source', 'relevant_both'):
                caps['source_quota'] = None
        rows['seed'] = score_context(db, user, task, phases['candidates'], seed, corpus)
        if task.get('requires_version'):
            trace = {'status': 'history_not_applicable', 'gates': {}, 'errors': [], 'arms': {}, 'calls': {}}
            for name in ('control', 'bridge'):
                rows[name] = deepcopy(rows['seed'])
        else:
            stage = 'supplement'
            check_scope, original_scopes = scope_validator(db, user, task['goal'])
            order = ('control', 'bridge') if offset % 2 == 0 else ('bridge', 'control')
            budget = ExecutionBudget(cfg, saved={'limits': {'policy': 2, 'search': 2,
                'judge': 0, 'generation': 0, 'judge_tokens': 0}},
                persist=lambda snap: atomic_json(Path(runtime) / 'budget.json', snap))
            def search(args):
                mstart, sstart, began = len(models.events), len(backend.events), time.monotonic()
                record = {'status': 'failed', 'query_sha256': sha(args.query.encode())}
                try:
                    with routing_goal(task['goal'], fuse=False):
                        found = retrieve_authorized(db, user, args.query, top_k=args.top_k, cfg=cfg,
                            models=EmbeddingOnly(models), search=backend)
                    if found.blocked_reason:
                        raise DependencyError('supplemental query blocked', stage='retrieval_scope')
                    result = live_candidates(db, user, found, corpus)
                    record.update(status='ok', candidate_count=len(result),
                                  embed_ms=found.embed_ms, retrieval_ms=found.retrieval_ms)
                    return result
                finally:
                    record.update(wall_ms=(time.monotonic() - began) * 1000,
                        embedding_events=deepcopy(models.events[mstart:]),
                        backend_events=deepcopy(backend.events[sstart:]))
                    search_usage[sha(args.query.encode())] = record
            result = run_supplement(task['goal'], seed, original_candidates, models=models,
                reauthorize=lambda rs: replay.rebind_rows(db, user, rs, corpus), search=search,
                validate_scope=check_scope, budget=budget, caps=caps,
                profile=seed_profile, order=order)
            trace = deepcopy(result.trace)
            trace['original_scope_sha256'] = digest(original_scopes)
            for name in ('control', 'bridge'):
                arm = result.arms[name]
                try:
                    shared_error = next((e for e in trace['errors'] if e.get('arm') is None), None)
                    if arm['evidence'] is None or shared_error is not None:
                        error = next((e for e in trace['errors'] if e.get('arm') in (name, None)),
                                     {'type': 'SupplementFailure', 'stage': 'supplement'})
                        rows[name] = missing_row(task, {k: error[k] for k in ('type', 'stage')})
                    else:
                        if arm['status'] == 'selected':
                            if trace.get('bridge_chunk_id') not in {r['chunk_id'] for r in arm['evidence']}:
                                raise ValueError('required bridge source missing from context')
                            bound = next(r for r in arm['evidence'] if r['chunk_id'] == trace['bridge_chunk_id'])
                            if bound['source_sha256'] != trace.get('bridge_source_sha256'):
                                raise ValueError('required bridge source hash differs')
                        rows[name] = score_context(db, user, task,
                            arm['candidates'] if arm['candidates'] is not None else phases['candidates'],
                            arm['evidence'], corpus)
                except Exception as exc:
                    rows[name] = missing_row(task, {'type': type(exc).__name__, 'stage': 'context_binding'})
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interruption = type(exc).__name__
            trace['status'] = 'interrupted'
        for name in PROFILES:
            rows.setdefault(name, missing_row(task, {'type': type(exc).__name__, 'stage': stage}))
        trace['errors'].append({'type': type(exc).__name__, 'stage': stage, 'arm': None})
    policy_events = [e for e in models.events if e['role'] == 'policy']
    for name, row in rows.items():
        query_hash = trace.get('query_sha256', {}).get(name)
        own = search_usage.get(query_hash)
        unattributed = name != 'seed' and interruption is not None and bool(search_usage) and own is None
        row['usage'] = {'credited_prior_retrieval_tools': 1,
            'shared_policy_cost_charged_in_full': name != 'seed',
            'policy': event_summary(policy_events if name != 'seed' else []),
            'search_attribution': 'unknown' if unattributed else 'observed',
            'additional_retrieval_tools': None if unattributed else 1 if own is not None else 0,
            'embedding': None if unattributed else event_summary(own['embedding_events'] if own else []),
            'backend': None if unattributed else event_summary(own['backend_events'] if own else []),
            'search': deepcopy(own), 'prior_cost_source': 'cached_parent_not_fresh_measurement'}
    return {'rows': rows, 'trace': {'task_id': task['id'], **trace}, 'interruption': interruption,
        'physical_cost': {'task_id': task['id'], 'status': 'observed',
            'policy_events': deepcopy(policy_events), 'policy': event_summary(policy_events),
            'embedding': event_summary([e for e in models.events if e['role'] == 'embedding']),
            'backend': event_summary(backend.events), 'search_attempts': len(search_usage),
            'search_events': deepcopy(list(search_usage.values())),
            'wall_ms': (time.monotonic() - started) * 1000}}


def load_registration(args):
    raw = REGISTRATION.read_bytes()
    record = json.loads(raw)
    expected = {'seed_profile': 'legacy_rerank', 'seed_report_bytes_sha256': FROZEN_SEED_SHA,
        'parent_bytes_sha256': FROZEN_PARENT_SHA, 'final_evidence_limit': 8,
        'context_token_budget': 5000, 'supplement_tool_attempts_per_arm': 1,
        'shared_policy_calls_max': 2, 'deadline_seconds': 180,
        'control_search_order': 'task_index_parity', 'union_rank_order': 'old_new_round_robin'}
    if (any(record.get(k) != v for k, v in expected.items())
            or args.seed_profile != 'legacy_rerank'
            or sha(Path(args.parent).read_bytes()) != FROZEN_PARENT_SHA
            or sha(Path(args.seed_report).read_bytes()) != FROZEN_SEED_SHA):
        raise ValueError('frozen supplement registration differs')
    return expected | {'registration_bytes_sha256': sha(raw)}


def posthoc_missing(rows, parent_rows):
    # This label is used only after execution; never supplied to the graph gate.
    ids = {r['task_id'] for r in parent_rows if r['eligible'] and r['losses']['candidate_missing'] > 0}
    return {'task_ids': sorted(ids), 'summaries': {
        n: summarize([r for r in rs if r['task_id'] in ids]) for n, rs in rows.items()}}


def execute(args, suite, selection, tasks, parent, seed, registration, reservation):
    from app.clients import Search
    from app.db import SessionLocal

    runtime = RUNTIME / reservation['run_id']
    checkpoint = runtime / 'checkpoint.json'
    report = {'version': 'evidence-supplement-dev-v1', 'split': 'dev', 'subset': False,
        'status': 'failed', 'valid': False, 'run_id': reservation['run_id'], 'identity': None,
        'strict_task_success': None, 'semantic_recall': None, 'error': None,
        'selected_task_ids': [t['id'] for t in tasks], 'rows': {n: [] for n in PROFILES},
        'supplement_traces': [], 'physical_costs': []}
    stage = 'initial_identity'
    try:
        validate_suite(suite, selection, ROOT)
        if tasks != suite['tasks'] or len(tasks) != 47:
            raise ValueError('exact frozen full Dev required')
        parent_bytes, seed_bytes = Path(args.parent).read_bytes(), Path(args.seed_report).read_bytes()
        if json.loads(parent_bytes) != parent or json.loads(seed_bytes) != seed:
            raise ValueError('frozen bytes/object differ')
        parent_rows = replay.validate_parent(parent, suite, selection,
            Path(args.suite).read_bytes(), SELECTION.read_bytes())
        seed_rows = validate_seed(seed, parent, suite, sha(parent_bytes), args.seed_profile)
        cfg, live = supplement_configuration(settings()), live_configuration()
        old_cfg = replay.effective_configuration(parent)
        if cfg.embed_model != old_cfg['embed_model'] or cfg.embed_dimension != old_cfg['embed_dimension']:
            raise ValueError('active embedding configuration differs from cached index')
        code, source = code_fingerprints(), Search(index=parent['identity']['source_index']['index'])
        with SessionLocal() as db, checkpoint_lock(checkpoint):
            db.execute(text('SET TRANSACTION READ ONLY'))
            users, records, scope = active_scope(db)
            original_index = index_ledger(source, records)
            if scope != parent['identity']['corpus'] or original_index != parent['identity']['source_index']:
                raise ValueError('live corpus/index differs from frozen parent')
            stage = 'legacy_reproduction'
            for task, row, seed_row in zip(tasks, parent_rows, seed_rows, strict=True):
                phases = replay.rehydrate_task(db, users[task['user']], row, scope)
                replay.verify_legacy(task, row, phases)
                bound = replay.rebind_rows(db, users[task['user']], seed_row['telemetry']['context'], scope)
                replay.validate_context(bound, seed_row['telemetry']['context_tokens'])
            stage = 'model_identity'
            active_models = active_model_identity(settings())
            embedding = {k: v for k, v in active_models['embedding'].items() if k != 'status'}
            if embedding != parent['identity']['models']['embedding']:
                raise ValueError('active embedding model/runtime differs from cached parent')
            identity = {k: deepcopy(parent['identity'][k]) for k in (
                'suite_sha256', 'selection_sha256', 'suite_bytes_sha256', 'selection_bytes_sha256',
                'corpus_sha256', 'source_index_sha256')}
            identity.update(method='matched_grounded_supplement_v1', registration=registration,
                parent_bytes_sha256=sha(parent_bytes), seed_report_bytes_sha256=sha(seed_bytes),
                seed_profile='legacy_rerank', active_models=active_models, live_configuration=live,
                cached_models=replay.cached_model_identity(parent['identity']['models'], old_cfg),
                source_index_name=source.index, code=code,
                supplement_configuration={k: getattr(cfg, k) for k in CONFIG_KEYS},
                policy_configuration={'policy_calls_max': 2, 'output_caps': [300, 60],
                    'search_attempts_per_arm': 1, 'deadline_seconds': 180, 'num_ctx': 8192,
                    'temperature': 0, 'model_seed': 42, 'history_dispatch': False,
                    'no_reference_dispatch': 'normal_non_oracle_gate',
                    'scope': 'same_frozen_original_source_version_keys', 'context': [8, 5000],
                    'union': 'old_new_round_robin', 'selection': 'rank_only_with_required_bridge_source',
                    'shared_policy_cost': 'charged_in_full_to_each_arm', 'profiles': list(PROFILES)},
                runtime={'python': platform.python_version(), 'platform': platform.platform()})
            report['identity'] = identity
            atomic_json(runtime / 'supplement-identity.json', identity)
            store = CheckpointStore(checkpoint, identity)
            stage = 'supplement'
            for offset, (task, parent_row, seed_row) in enumerate(zip(tasks, parent_rows, seed_rows, strict=True)):
                for name in PROFILES:
                    store.begin(name, task['id'], uuid.uuid4().hex)
                pair = collect_pair(db, users[task['user']], task, parent_row, seed_row, scope,
                                    cfg, source, runtime / task['id'], offset)
                for name in PROFILES:
                    store.finish(name, task['id'], pair['rows'][name])
                    report['rows'][name].append(pair['rows'][name])
                report['supplement_traces'].append(pair['trace'])
                report['physical_costs'].append(pair['physical_cost'])
                print(json.dumps({'task_id': task['id'], 'outcome': pair['trace']['status']}), flush=True)
                if pair.get('interruption'):
                    if pair['interruption'] == 'SystemExit':
                        raise SystemExit()
                    raise KeyboardInterrupt()
            stage = 'drift_check'
            _, final_records, final_scope = active_scope(db)
            drift = {'parent': sha(Path(args.parent).read_bytes()) != identity['parent_bytes_sha256'],
                'seed': sha(Path(args.seed_report).read_bytes()) != identity['seed_report_bytes_sha256'],
                'registration': sha(REGISTRATION.read_bytes()) != registration['registration_bytes_sha256'],
                'suite': sha(Path(args.suite).read_bytes()) != identity['suite_bytes_sha256'],
                'selection': sha(SELECTION.read_bytes()) != identity['selection_bytes_sha256'],
                'code': code_fingerprints() != code, 'corpus': final_scope != scope,
                'source_index': index_ledger(source, final_records) != original_index,
                'models': active_model_identity(settings()) != active_models,
                'live_configuration': live_configuration() != live}
            try:
                validate_suite(suite, selection, ROOT)
                drift['raw_sources'] = False
            except (ValueError, OSError):
                drift['raw_sources'] = True
            report['drift'] = drift
            errors = (any(r['error'] is not None for rs in report['rows'].values() for r in rs)
                      or any(t.get('errors') for t in report['supplement_traces']))
            report.update(valid=not any(drift.values()) and not errors,
                status='invalid_drift' if any(drift.values()) else 'failed' if errors else 'complete')
    except BaseException as exc:
        report.update(valid=False, error={'type': type(exc).__name__, 'stage': stage},
            status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed')
    finally:
        for name, rows in report['rows'].items():
            done = {r['task_id']: r for r in rows}
            report['rows'][name] = [done.get(t['id']) or missing_row(t, report['error']) for t in tasks]
        completed = {t['task_id']: t for t in report['supplement_traces']}
        report['supplement_traces'] = [completed.get(t['id']) or {
            'task_id': t['id'], 'status': 'not_attempted', 'errors': [], 'gates': {}, 'calls': {}}
            for t in tasks]
        observed = {c['task_id']: c for c in report['physical_costs']}
        report['physical_costs'] = [observed.get(t['id']) or {
            'task_id': t['id'], 'status': 'not_observed', 'policy_events': None,
            'policy': None, 'embedding': None, 'backend': None,
            'search_attempts': None, 'search_events': None, 'wall_ms': None} for t in tasks]
        report['cost_observation_counts'] = dict(Counter(c['status'] for c in report['physical_costs']))
        report['summaries'] = {n: summarize(rs) for n, rs in report['rows'].items()}
        report['null_evidence_counts'] = {n: null_evidence_counts(rs) for n, rs in report['rows'].items()}
        report['execution_errors'] = {n: {'tasks': sum(r['error'] is not None for r in rs),
            'by_stage': dict(Counter(r['error']['stage'] for r in rs if r['error']))}
            for n, rs in report['rows'].items()}
        report['gate_outcomes'] = dict(Counter(t['status'] for t in report['supplement_traces']))
        report['pairs'] = {'control_vs_seed': pair_profiles(report['rows']['seed'], report['rows']['control']),
            'bridge_vs_seed': pair_profiles(report['rows']['seed'], report['rows']['bridge']),
            'bridge_vs_control': pair_profiles(report['rows']['control'], report['rows']['bridge'])}
        report['posthoc_prior_candidate_missing'] = posthoc_missing(report['rows'], parent['rows'][replay.PARENT_PROFILE])
        _publish(args.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--seed-report', required=True)
    parser.add_argument('--seed-profile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--suite', default=str(ROOT / '.runtime/benchmark-package/v1/dev/tasks.json'))
    args = parser.parse_args(argv)
    registration = load_registration(args)
    suite, selection = json.loads(Path(args.suite).read_bytes()), json.loads(SELECTION.read_bytes())
    tasks = validate_suite(suite, selection, ROOT)
    parent, seed = json.loads(Path(args.parent).read_bytes()), json.loads(Path(args.seed_report).read_bytes())
    replay.validate_parent(parent, suite, selection, Path(args.suite).read_bytes(), SELECTION.read_bytes())
    validate_seed(seed, parent, suite, sha(Path(args.parent).read_bytes()), args.seed_profile)
    report = execute(args, suite, selection, tasks, parent, seed, registration, reserve_output(args.output))
    print(json.dumps({'output': str(Path(args.output).resolve()), 'status': report['status'],
                      'valid': report['valid']}))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
