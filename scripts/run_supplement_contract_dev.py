"""Registered Dev-only source contracts, slot selection and bounded matched repair."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from agent.supplement_repair import run_repair
from app.clients import DependencyError
from app.config import settings
from app.execution_budget import ExecutionBudget
from app.evidence_selection import rank_relevance_candidates, select_slot_evidence
from app.models import DocumentVersion
from app.retrieval import retrieve_authorized, route_sources, routing_goal
from app.security import readable_documents
from app.supplement_contract import (assess_slot_presence, build_supplement_contract,
                                    diagnose_slot_loss, public_presence)
from scripts import run_evidence_selection_dev as replay
from scripts import run_evidence_supplement_dev as supplement
from scripts.benchmark_runtime import CheckpointStore, atomic_json, checkpoint_lock
from scripts.dev_evidence_annotations import summarize_revised, validate_annotation
from scripts.research_review import digest
from scripts.retrieval_quality_index import active_scope, index_ledger
from scripts.run_retrieval_quality_dev import (
    ROOT, SELECTION, EmbeddingOnly, _publish, code_fingerprints, missing_row,
    reserve_output, sha, validate_suite,
)
from scripts.run_revised_evidence_dev import missing_revised, select_relevance_context
from scripts.run_reviewed_supplement_dev import LIMITS, rescore, validate_seed as validate_baseline

PROFILES = ('seed', 'control', 'treatment')
MATRIX = ('relevant_doc', 'slot_source', 'slot_literal')


def contract_for(db, user, question):
    docs = []
    for d in readable_documents(db, user):
        version = db.get(DocumentVersion, d.active_version_id, populate_existing=True) if d.active_version_id else None
        if version is not None:
            docs.append(SimpleNamespace(id=d.id, active_version_id=version.id, title=d.title,
                metadata_json=deepcopy(d.metadata_json), source_sha256=version.content_hash))
    return build_supplement_contract(question, docs)


def scope_keys(query, documents, cfg):
    documents = list(documents)
    versions = {d.id: d.active_version_id for d in documents}
    return [frozenset((did, versions[did]) for did in g['key'])
            for g in route_sources(query, documents, strict_dates=cfg.focused_generation_enabled)]


def repair_scope_validator(db, user, question, cfg):
    frozen_docs = readable_documents(db, user)
    frozen = scope_keys(question, frozen_docs, cfg)
    allowed = frozenset().union(*frozen)
    frozen_contract = contract_for(db, user, question)
    def check(original, query):
        current = readable_documents(db, user)
        qcontract = contract_for(db, user, query)
        qscope = scope_keys(query, current, cfg)
        if (original != question or scope_keys(original, current, cfg) != frozen
                or contract_for(db, user, original) != frozen_contract
                or any(r.resolution != 'supported' for r in qcontract.source_requests)):
            return False
        # A query can omit or narrow original publisher scopes, never add one.
        contract_allowed = frozenset(pair for r in frozen_contract.source_requests for pair in r.document_versions)
        return (all(group <= allowed for group in qscope) and
                all(set(r.document_versions) <= contract_allowed for r in qcontract.source_requests))
    return check, [[list(pair) for pair in sorted(group)] for group in frozen]


def select_context(db, user, task, parent_row, corpus, profile):
    if profile == 'relevant_doc':
        result = select_relevance_context(db, user, task, parent_row, corpus, profile)
        contract = contract_for(db, user, task['goal'])
        rows = replay.rebind_rows(db, user, result['telemetry']['candidates'], corpus)
        context = replay.rebind_rows(db, user, result['telemetry']['context'], corpus)
        candidate_presence = assess_slot_presence(contract, rows)
        selected_presence = assess_slot_presence(contract, context)
        result['structural'] = {'contract': contract.public_summary(),
            'candidates': public_presence(candidate_presence), 'context': public_presence(selected_presence),
            'loss': diagnose_slot_loss(candidate_presence, selected_presence)}
        return result
    if profile not in MATRIX:
        raise ValueError('Unregistered seed profile')
    phases = replay.rehydrate_task(db, user, parent_row, corpus)
    replay.verify_legacy(task, parent_row, phases)
    rows = rank_relevance_candidates(phases['candidates'])[0]
    caps = replay.policy_caps(db, user, task['goal'])
    if profile == 'slot_literal':
        caps['document_quota'] = None
    contract = contract_for(db, user, task['goal'])
    start = time.monotonic()
    selected = select_slot_evidence(contract, rows, strategy='source' if profile == 'slot_source' else 'literal', **caps)
    result = replay.replay_row(db, user, task, parent_row, phases, selected, corpus,
                              elapsed_ms=(time.monotonic()-start)*1000)
    result['structural'] = {'contract': contract.public_summary(),
        'candidates': public_presence(assess_slot_presence(contract, rows)),
        'context': public_presence(assess_slot_presence(contract, selected.evidence)),
        'loss': diagnose_slot_loss(assess_slot_presence(contract, rows),
            assess_slot_presence(contract, selected.evidence), feasible_slots={f['slot_id']
                for f in selected.trace['slot_floor'] if f['status'] == 'reserved' and f['basis'] == 'coherent_literal'})}
    return result


def choose_matrix_seed(report):
    if report['valid'] is not True:
        raise ValueError('Valid complete replay required')
    baseline = report['rows']['relevant_doc']
    choices = []
    for i, name in enumerate(MATRIX):
        rows = report['rows'][name]
        if len(rows) != 47 or any(r['error'] for r in rows):
            continue
        if any(a['eligible'] and a['basis'] == 'source_atom_exact_proxy' and not
               set(a['phases']['context']['matched_reference_ids']) <=
               set(b['phases']['context']['matched_reference_ids'])
               for a, b in zip(baseline, rows, strict=True)):
            continue
        groups = report['summaries'][name]['by_basis']
        choices.append(((groups['upstream_word4_proxy']['phases']['context']['delivered'],
                         groups['source_atom_exact_proxy']['phases']['context']['delivered'], -i), name))
    return max(choices)[1] if choices else 'relevant_doc'


def verify_bridge_binding(trace, evidence):
    if trace.get('bridge_chunk_sha256') and not any(
        sha(r['chunk_id'].encode()) == trace['bridge_chunk_sha256']
        and r['source_sha256'] == trace['bridge_source_sha256'] for r in evidence):
        raise ValueError('Required literal bridge source missing')


def collect_pair(db, user, task, parent_row, seed_row, corpus, cfg, source, runtime, offset,
                 *, seed_profile='relevant_doc'):
    rows, stage, interruption = {}, 'seed_reauthorization', None
    models, backend = supplement.MeteredModels(), supplement.MeteredSearch(source)
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
        expected = select_context(db, user, task, parent_row, corpus, seed_profile)
        if expected['telemetry']['context'] != seed_row['telemetry']['context']:
            raise ValueError('frozen contract seed differs from reproduction')
        original_candidates = rank_relevance_candidates(original_candidates)[0]
        if seed_profile != 'slot_source':
            caps['document_quota'] = None
        contract = contract_for(db, user, task['goal'])
        rows['seed'] = supplement.score_context(db, user, task, phases['candidates'], seed, corpus)
        if task.get('requires_version'):
            trace = {'status': 'history_not_applicable', 'gates': {}, 'errors': [], 'arms': {}, 'calls': {}}
            for name in ('control', 'treatment'):
                rows[name] = deepcopy(rows['seed'])
        else:
            stage = 'supplement'
            check_scope, original_scopes = repair_scope_validator(db, user, task['goal'], cfg)
            order = ('control', 'treatment') if offset % 2 == 0 else ('treatment', 'control')
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
                    result = supplement.live_candidates(db, user, found, corpus)
                    record.update(status='ok', candidate_count=len(result),
                                  embed_ms=found.embed_ms, retrieval_ms=found.retrieval_ms)
                    return result
                finally:
                    record.update(wall_ms=(time.monotonic() - began) * 1000,
                        embedding_events=deepcopy(models.events[mstart:]),
                        backend_events=deepcopy(backend.events[sstart:]))
                    search_usage[sha(args.query.encode())] = record
            result = run_repair(contract, seed, original_candidates, models=models,
                reauthorize=lambda rs: replay.rebind_rows(db, user, rs, corpus), search=search,
                validate_scope=check_scope, budget=budget, caps=caps,
                order=order)
            trace = deepcopy(result.trace)
            trace['original_scope_sha256'] = digest(original_scopes)
            for name in ('control', 'treatment'):
                arm = result.arms[name]
                try:
                    shared_error = next((e for e in trace['errors'] if e.get('arm') is None), None)
                    if arm['evidence'] is None or shared_error is not None:
                        error = next((e for e in trace['errors'] if e.get('arm') in (name, None)),
                                     {'type': 'SupplementFailure', 'stage': 'supplement'})
                        rows[name] = missing_row(task, {k: error[k] for k in ('type', 'stage')})
                    else:
                        if arm['status'] == 'selected':
                            verify_bridge_binding(trace, arm['evidence'])
                        rows[name] = supplement.score_context(db, user, task,
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
            'policy': supplement.event_summary(policy_events if name != 'seed' else []),
            'search_attribution': 'unknown' if unattributed else 'observed',
            'additional_retrieval_tools': None if unattributed else 1 if own is not None else 0,
            'embedding': None if unattributed else supplement.event_summary(own['embedding_events'] if own else []),
            'backend': None if unattributed else supplement.event_summary(own['backend_events'] if own else []),
            'search': deepcopy(own), 'prior_cost_source': 'cached_parent_not_fresh_measurement'}
    return {'rows': rows, 'trace': {'task_id': task['id'], **trace}, 'interruption': interruption,
        'physical_cost': {'task_id': task['id'], 'status': 'observed',
            'policy_events': deepcopy(policy_events), 'policy': supplement.event_summary(policy_events),
            'embedding': supplement.event_summary([e for e in models.events if e['role'] == 'embedding']),
            'backend': supplement.event_summary(backend.events), 'search_attempts': len(search_usage),
            'search_events': deepcopy(list(search_usage.values())),
            'wall_ms': (time.monotonic() - started) * 1000}}



def input_paths(args):
    paths = [Path(args.suite), SELECTION, Path(args.parent), Path(args.baseline),
             Path(args.overlay), Path(args.manifest)]
    if args.stage == 'live':
        paths.append(Path(args.offline_seed))
    return paths


def registration_for(args):
    registration = {'version': 'supplement-contract-input-v1', 'stage': args.stage, 'split': 'dev',
        'input_bytes_sha256': [sha(p.read_bytes()) for p in input_paths(args)],
        'matrix': list(MATRIX), 'limits': LIMITS,
        'seed_choice': 'per_task_native_no_loss_then_upstream_delivery_then_matrix_order',
        'union': 'old_new_round_robin_rank_only', 'control_order': 'original_task_index_parity',
        'annotation_runtime_access': 'evaluator_only', 'answer_accuracy': 'pending',
        'independent_code_review': 'unavailable_account_usage_limit'}
    if args.stage == 'live':
        registration['selection_focus'] = 'shared_model_query_scoped_lexical_only_no_semantic_certification'
        registration['transport'] = 'prompt_grounded_schema_v3_program_owned_bridge_placeholder'
    return registration


def validate_matrix_seed(seed, tasks, parent, manifest):
    if (seed.get('valid') is not True or seed.get('status') != 'complete'
            or seed.get('stage') != 'offline' or seed.get('version') != 'supplement-contract-dev-v1'
            or seed.get('selected_task_ids') != [t['id'] for t in tasks]
            or seed['identity']['parent_sha256'] != digest(parent)
            or seed['identity']['annotation_manifest'] != manifest
            or seed['chosen_seed_profile'] != choose_matrix_seed(seed)
            or any(seed['drift'].values())):
        raise ValueError('Incomplete or mismatched matrix seed')
    for name in MATRIX:
        if [r['task_id'] for r in seed['rows'][name]] != [t['id'] for t in tasks] or any(r['error'] for r in seed['rows'][name]):
            raise ValueError('Matrix seed rows incomplete')
    return seed['chosen_seed_profile']


def execute(args, reservation):
    from app.clients import Search
    from app.db import SessionLocal
    paths = input_paths(args)
    frozen = [p.read_bytes() for p in paths]
    suite, selection, parent, baseline, manifest = [json.loads(frozen[i]) for i in (0, 1, 2, 3, 5)]
    tasks = validate_suite(suite, selection, ROOT)
    originals = replay.validate_parent(parent, suite, selection, frozen[0], frozen[1])
    annotations = validate_annotation(suite, frozen[4], manifest, ROOT, frozen[0])
    if validate_baseline(baseline, parent, suite, manifest) != 'relevant_doc':
        raise ValueError('Expected frozen relevant_doc baseline')
    reg_bytes = Path(args.registration).read_bytes()
    registration = json.loads(reg_bytes)
    if registration != registration_for(args):
        raise ValueError('Frozen input registration differs')
    seed = json.loads(frozen[6]) if args.stage == 'live' else None
    seed_profile = validate_matrix_seed(seed, tasks, parent, manifest) if seed else None
    profiles = MATRIX if args.stage == 'offline' else PROFILES
    runtime = ROOT / '.runtime/supplement-repair-20261010' / reservation['run_id']
    runtime.mkdir()
    identity = {'registration': registration, 'input_registration_bytes_sha256': sha(reg_bytes),
        'code': code_fingerprints(), 'parent_sha256': digest(parent),
        'annotation_manifest': manifest, 'active_models': None, 'seed_profile': seed_profile,
        'corpus_sha256': parent['identity']['corpus_sha256'],
        'source_index_sha256': parent['identity']['source_index_sha256'],
        'live_configuration': supplement.live_configuration()}
    cfg = supplement.supplement_configuration(settings())
    identity['supplement_configuration'] = {k: getattr(cfg, k) for k in supplement.CONFIG_KEYS}
    atomic_json(runtime / 'registration.json', identity)
    report = {'version': 'supplement-contract-dev-v1', 'stage': args.stage, 'split': 'dev', 'subset': False,
        'run_id': reservation['run_id'], 'valid': False, 'status': 'failed', 'error': None,
        'strict_task_success': None, 'semantic_recall': None, 'selected_task_ids': [t['id'] for t in tasks],
        'rows': {p: [] for p in profiles}, 'identity': identity, 'supplement_traces': [], 'physical_costs': []}
    stage = 'source_identity'
    try:
        source = Search(index=parent['identity']['source_index']['index'])
        with SessionLocal() as db, checkpoint_lock(runtime / 'checkpoint.json'):
            db.execute(text('SET TRANSACTION READ ONLY'))
            users, records, corpus = active_scope(db)
            index = index_ledger(source, records)
            if corpus != parent['identity']['corpus'] or index != parent['identity']['source_index']:
                raise ValueError('Live corpus/index differs from parent')
            stage = 'baseline_reproduction'
            for task, original, prior in zip(tasks, originals, baseline['rows']['relevant_doc'], strict=True):
                replay.verify_legacy(task, original, replay.rehydrate_task(db, users[task['user']], original, corpus))
                expected = select_context(db, users[task['user']], task, original, corpus, 'relevant_doc')
                if expected['telemetry']['context'] != prior['telemetry']['context']:
                    raise ValueError('Frozen relevant_doc reproduction differs')
            if args.stage == 'live':
                stage = 'live_seed_reproduction'
                for task, original, prior in zip(tasks, originals, seed['rows'][seed_profile], strict=True):
                    expected = select_context(db, users[task['user']], task, original, corpus, seed_profile)
                    if expected['telemetry']['context'] != prior['telemetry']['context']:
                        raise ValueError('Frozen matrix seed reproduction differs')
                stage = 'model_identity'
                models = supplement.active_model_identity(settings())
                if {k: v for k, v in models['embedding'].items() if k != 'status'} != parent['identity']['models']['embedding']:
                    raise ValueError('Embedding identity changed')
                identity['active_models'] = models
                atomic_json(runtime / 'identity.json', identity)
            store = CheckpointStore(runtime / 'checkpoint.json', identity)
            stage = args.stage
            for offset, (task, original) in enumerate(zip(tasks, originals, strict=True)):
                record = annotations['task_records'][task['id']]
                if args.stage == 'offline':
                    for name in MATRIX:
                        store.begin(name, task['id'], uuid.uuid4().hex)
                        try:
                            raw = select_context(db, users[task['user']], task, original, corpus, name)
                            row = rescore(db, users[task['user']], task, raw, corpus, record)
                            row['structural'] = raw.get('structural')
                        except Exception as exc:
                            row = missing_revised(task, record, {'type': type(exc).__name__, 'stage': stage})
                        store.finish(name, task['id'], row)
                        report['rows'][name].append(row)
                else:
                    for name in PROFILES:
                        store.begin(name, task['id'], uuid.uuid4().hex)
                    pair = collect_pair(db, users[task['user']], task, original,
                        seed['rows'][seed_profile][offset], corpus, cfg, source, runtime / task['id'], offset,
                        seed_profile=seed_profile)
                    for name in PROFILES:
                        try:
                            row = rescore(db, users[task['user']], task, pair['rows'][name], corpus, record)
                        except Exception as exc:
                            row = missing_revised(task, record, {'type': type(exc).__name__, 'stage': 'evaluation'})
                        store.finish(name, task['id'], row)
                        report['rows'][name].append(row)
                    report['supplement_traces'].append(pair['trace'])
                    report['physical_costs'].append(pair['physical_cost'])
                    print(json.dumps({'task_id': task['id'], 'status': pair['trace']['status'],
                                      'rejections': pair['trace'].get('rejections', [])}), flush=True)
                    if pair['interruption']:
                        raise KeyboardInterrupt()
            stage = 'drift_check'
            _, final_records, final_corpus = active_scope(db)
            report['drift'] = {'inputs': any(p.read_bytes() != b for p, b in zip(paths, frozen, strict=True)),
                'code': code_fingerprints() != identity['code'], 'corpus': final_corpus != corpus,
                'index': index_ledger(source, final_records) != index,
                'registration': Path(args.registration).read_bytes() != reg_bytes,
                'configuration': supplement.live_configuration() != identity['live_configuration'],
                'supplement_configuration': {k: getattr(supplement.supplement_configuration(settings()), k)
                    for k in supplement.CONFIG_KEYS} != identity['supplement_configuration'],
                'models': supplement.active_model_identity(settings()) != identity['active_models'] if args.stage == 'live' else False}
            errors = any(r['error'] for rs in report['rows'].values() for r in rs)
            errors = errors or any(t.get('errors') for t in report['supplement_traces'])
            report.update(valid=not errors and not any(report['drift'].values()),
                status='invalid_drift' if any(report['drift'].values()) else 'failed' if errors else 'complete')
    except BaseException as exc:
        report.update(valid=False, status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                      error={'type': type(exc).__name__, 'stage': stage})
    finally:
        for name in profiles:
            seen = {r['task_id']: r for r in report['rows'][name]}
            report['rows'][name] = [seen.get(t['id']) or missing_revised(t, annotations['task_records'][t['id']],
                report['error'] or {'type': 'NotObserved', 'stage': stage}) for t in tasks]
        if args.stage == 'offline':
            for offset, base in enumerate(report['rows']['relevant_doc']):
                structural = base.get('structural')
                if structural:
                    feasible = {slot['slot_id'] for p in MATRIX[1:]
                        for slot in (report['rows'][p][offset].get('structural') or {}).get('context', {}).get('slots', [])
                        if slot['state'] == 'supported'}
                    structural['loss'] = diagnose_slot_loss(structural['candidates'], structural['context'],
                                                           feasible_slots=feasible)
        report['summaries'] = {p: summarize_revised(rs) for p, rs in report['rows'].items()}
        report['execution_errors'] = {p: sum(r['error'] is not None for r in rs) for p, rs in report['rows'].items()}
        if args.stage == 'offline' and report['valid']:
            try:
                report['chosen_seed_profile'] = choose_matrix_seed(report)
            except Exception as exc:
                report.update(valid=False, status='failed', error={'stage': 'seed_choice', 'type': type(exc).__name__})
        if args.stage == 'live':
            traces = {r['task_id']: r for r in report['supplement_traces']}
            costs = {r['task_id']: r for r in report['physical_costs']}
            report['supplement_traces'] = [traces.get(t['id']) or dict(task_id=t['id'], status='not_attempted',
                errors=[], rejections=[]) for t in tasks]
            report['physical_costs'] = [costs.get(t['id']) or dict(task_id=t['id'], status='not_observed',
                policy=None, embedding=None, backend=None, search_attempts=None, wall_ms=None) for t in tasks]
            report['outcomes'] = dict(Counter(t['status'] for t in report['supplement_traces']))
            report['routes'] = dict(Counter(t.get('mode', 'not_observed') for t in report['supplement_traces']))
            report['rejection_counts'] = dict(Counter(c for t in report['supplement_traces']
                for r in t.get('rejections', []) for c in r['codes']))
            report['matched_dispatch_pairs'] = sum(t['status'] == 'completed' for t in report['supplement_traces'])
            report['treatment_efficacy'] = 'dev_delivery_proxy_only' if report['valid'] and report['matched_dispatch_pairs'] else 'pending'
        _publish(args.output, report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('offline', 'live'), required=True)
    for name in ('parent', 'baseline', 'overlay', 'manifest', 'registration', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--offline-seed')
    p.add_argument('--suite', default=str(ROOT / '.runtime/benchmark-package/v1/dev/tasks.json'))
    args = p.parse_args(argv)
    reservation = reserve_output(args.output)
    try:
        report = execute(args, reservation)
    except BaseException as exc:
        report = {'version': 'supplement-contract-dev-v1', 'stage': args.stage, 'split': 'dev', 'valid': False,
            'status': 'failed', 'run_id': reservation['run_id'], 'rows': None, 'summaries': None,
            'strict_task_success': None, 'error': {'stage': 'preflight', 'type': type(exc).__name__}}
        _publish(args.output, report)
    print(json.dumps({'status': report['status'], 'valid': report['valid'], 'output': args.output}))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
