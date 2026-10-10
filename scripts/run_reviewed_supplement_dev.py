"""Registered full Dev supplement with evaluator-only revised references."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.config import settings
from scripts import run_evidence_selection_dev as replay
from scripts import run_evidence_supplement_dev as supplement
from scripts.benchmark_runtime import CheckpointStore, atomic_json, checkpoint_lock
from scripts.dev_evidence_annotations import diagnose_revised, summarize_revised, validate_annotation
from scripts.research_review import digest
from scripts.retrieval_quality_index import active_scope, index_ledger
from scripts.retrieval_quality_metrics import pair_profiles
from scripts.run_retrieval_quality_dev import (
    ROOT, SELECTION, _publish, code_fingerprints, reserve_output, sha, validate_suite,
)
from scripts.run_revised_evidence_dev import choose_seed, missing_revised, select_relevance_context

LIMITS = {'context_chunks': 8, 'context_tokens': 5000, 'shared_policy_calls': 2,
          'policy_output_tokens': [300, 60], 'search_per_arm': 1, 'deadline_seconds': 180}


def validate_registration(registration, inputs, profile):
    expected = {'version': 'reviewed-supplement-input-registration-v1', 'split': 'dev',
                'input_bytes_sha256': list(map(sha, inputs)), 'seed_profile': profile,
                'limits': LIMITS, 'union': 'old_new_round_robin_rank_only',
                'control_order': 'original_task_index_parity'}
    if registration != expected:
        raise ValueError('frozen supplement registration differs')


def validate_seed(seed, parent, suite, manifest):
    if (seed.get('valid') is not True or seed.get('status') != 'complete'
            or seed.get('version') != 'revised-evidence-dev-v1'
            or seed['identity'].get('stage') != 'relevance'
            or seed['identity'].get('parent_sha256') != digest(parent)
            or seed['identity'].get('annotation_manifest') != manifest
            or any(seed.get('drift', {}).values())
            or seed.get('chosen_seed_profile') != choose_seed(seed)):
        raise ValueError('registered reviewed seed identity differs')
    ids = [t['id'] for t in suite['tasks']]
    if len(ids) != 47 or seed['selected_task_ids'] != ids:
        raise ValueError('full original Dev required')
    for rows in seed['rows'].values():
        if [r['task_id'] for r in rows] != ids or any(r.get('error') for r in rows):
            raise ValueError('seed rows incomplete')
    return seed['chosen_seed_profile']


def rescore(db, user, task, row, corpus, record):
    if row['error'] is not None or row.get('telemetry') is None:
        return missing_revised(task, record, row['error']) | {'usage': row.get('usage')}
    phases = {p: replay.rebind_rows(db, user, row['telemetry'][p], corpus)
              for p in ('candidates', 'admitted', 'context')}
    replay.validate_context(phases['context'], row['telemetry']['context_tokens'])
    return diagnose_revised(task, phases, record) | {
        'error': None, 'telemetry': row['telemetry'], 'usage': row.get('usage')}


def execute(args, reservation):
    from app.clients import Search
    from app.db import SessionLocal

    paths = [Path(args.suite), SELECTION, Path(args.parent), Path(args.seed),
             Path(args.overlay), Path(args.manifest)]
    inputs = [p.read_bytes() for p in paths]
    suite, selection, parent, seed, manifest = [json.loads(inputs[i]) for i in (0, 1, 2, 3, 5)]
    tasks = validate_suite(suite, selection, ROOT)
    parent_rows = replay.validate_parent(parent, suite, selection, inputs[0], inputs[1])
    annotated = validate_annotation(suite, inputs[4], manifest, ROOT, inputs[0])
    profile = validate_seed(seed, parent, suite, manifest)
    input_registration_bytes = Path(args.registration).read_bytes()
    validate_registration(json.loads(input_registration_bytes), inputs, profile)
    cfg = supplement.supplement_configuration(settings())
    live_cfg = supplement.live_configuration()
    cached_cfg = replay.effective_configuration(parent)
    if cfg.embed_model != cached_cfg['embed_model'] or cfg.embed_dimension != cached_cfg['embed_dimension']:
        raise ValueError('embedding configuration differs')
    runtime = ROOT / '.runtime/dev-evidence-repair-20261009' / reservation['run_id']
    runtime.mkdir()
    registration = {'version': 'reviewed-supplement-registration-v1', 'split': 'dev',
        'input_bytes_sha256': list(map(sha, inputs)), 'seed_profile': profile,
        'original_order': 'cached_original_query_relevance_occurrences' if profile != 'legacy_rerank'
                          else 'recorded_parent_occurrences',
        'union': 'old_new_round_robin_rank_only', 'control_order': 'original_task_index_parity',
        'limits': LIMITS, 'input_registration_bytes_sha256': sha(input_registration_bytes),
        'annotation_runtime_access': 'evaluator_only', 'code': code_fingerprints()}
    atomic_json(runtime / 'registration.json', registration)
    report = {'version': 'reviewed-supplement-dev-v1', 'split': 'dev', 'subset': False,
        'run_id': reservation['run_id'], 'valid': False, 'status': 'failed', 'error': None,
        'strict_task_success': None, 'semantic_recall': None,
        'selected_task_ids': [t['id'] for t in tasks], 'rows': {p: [] for p in supplement.PROFILES},
        'supplement_traces': [], 'physical_costs': [], 'identity': {
            'registration': registration, 'annotation_manifest': manifest,
            'live_configuration': live_cfg, 'active_models': None,
            'supplement_configuration': {k: getattr(cfg, k) for k in supplement.CONFIG_KEYS},
            'corpus_sha256': parent['identity']['corpus_sha256'],
            'source_index_sha256': parent['identity']['source_index_sha256']}}
    stage = 'source_identity'
    try:
        source = Search(index=parent['identity']['source_index']['index'])
        with SessionLocal() as db, checkpoint_lock(runtime / 'checkpoint.json'):
            db.execute(text('SET TRANSACTION READ ONLY'))
            users, records, corpus = active_scope(db)
            index = index_ledger(source, records)
            if corpus != parent['identity']['corpus'] or index != parent['identity']['source_index']:
                raise ValueError('live corpus/index differs')
            stage = 'seed_reproduction'
            for t, original, selected in zip(tasks, parent_rows, seed['rows'][profile], strict=True):
                replay.verify_legacy(t, original, replay.rehydrate_task(db, users[t['user']], original, corpus))
                bound = replay.rebind_rows(db, users[t['user']], selected['telemetry']['context'], corpus)
                replay.validate_context(bound, selected['telemetry']['context_tokens'])
                reproduced = select_relevance_context(db, users[t['user']], t, original, corpus, profile)
                if (reproduced['telemetry']['context'] != selected['telemetry']['context']
                        or reproduced['telemetry']['context_tokens'] != selected['telemetry']['context_tokens']):
                    raise ValueError('full frozen seed reproduction differs')
            stage = 'model_identity'
            models = supplement.active_model_identity(settings())
            if {k: v for k, v in models['embedding'].items() if k != 'status'} != parent['identity']['models']['embedding']:
                raise ValueError('active embedding digest/runtime differs')
            report['identity']['active_models'] = models
            store = CheckpointStore(runtime / 'checkpoint.json', report['identity'])
            atomic_json(runtime / 'identity.json', report['identity'])
            stage = 'supplement'
            for offset, (t, original, selected) in enumerate(zip(tasks, parent_rows, seed['rows'][profile], strict=True)):
                for arm in supplement.PROFILES:
                    store.begin(arm, t['id'], uuid.uuid4().hex)
                pair = supplement.collect_pair(db, users[t['user']], t, original, selected,
                    corpus, cfg, source, runtime / t['id'], offset, seed_profile=profile)
                for arm in supplement.PROFILES:
                    try:
                        row = rescore(db, users[t['user']], t, pair['rows'][arm], corpus,
                                      annotated['task_records'][t['id']])
                    except Exception as exc:
                        row = missing_revised(t, annotated['task_records'][t['id']],
                            {'type': type(exc).__name__, 'stage': 'reviewed_evaluation'})
                    store.finish(arm, t['id'], row)
                    report['rows'][arm].append(row)
                report['supplement_traces'].append(pair['trace'])
                report['physical_costs'].append(pair['physical_cost'])
                print(json.dumps({'task_id': t['id'], 'status': pair['trace']['status'],
                                  'rejections': pair['trace'].get('rejections', [])}), flush=True)
                if pair['interruption']:
                    raise KeyboardInterrupt()
            stage = 'drift_check'
            _, end_records, end_corpus = active_scope(db)
            report['drift'] = {'inputs': any(p.read_bytes() != b for p, b in zip(paths, inputs, strict=True)),
                'code': code_fingerprints() != registration['code'], 'corpus': end_corpus != corpus,
                'index': index_ledger(source, end_records) != index,
                'models': supplement.active_model_identity(settings()) != models,
                'configuration': supplement.live_configuration() != live_cfg,
                'supplement_configuration': {k: getattr(supplement.supplement_configuration(settings()), k)
                    for k in supplement.CONFIG_KEYS} != report['identity']['supplement_configuration'],
                'registration': json.loads((runtime / 'registration.json').read_bytes()) != registration}
            report['drift']['input_registration'] = Path(args.registration).read_bytes() != input_registration_bytes
            validate_suite(suite, selection, ROOT)
            validate_annotation(suite, paths[4].read_bytes(), manifest, ROOT, paths[0].read_bytes())
            errors = any(r['error'] for rs in report['rows'].values() for r in rs)
            errors = errors or any(t.get('errors') for t in report['supplement_traces'])
            report.update(valid=not errors and not any(report['drift'].values()),
                status='invalid_drift' if any(report['drift'].values()) else 'failed' if errors else 'complete')
    except BaseException as exc:
        report.update(error={'type': type(exc).__name__, 'stage': stage}, valid=False,
                      status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed')
    finally:
        for arm, rows in report['rows'].items():
            seen = {r['task_id']: r for r in rows}
            report['rows'][arm] = [seen.get(t['id']) or missing_revised(t,
                annotated['task_records'][t['id']], report['error'] or {'type': 'NotObserved', 'stage': stage})
                for t in tasks]
        traces = {r['task_id']: r for r in report['supplement_traces']}
        costs = {r['task_id']: r for r in report['physical_costs']}
        report['supplement_traces'] = [traces.get(t['id']) or dict(task_id=t['id'], status='not_attempted',
            errors=[], rejections=[]) for t in tasks]
        report['physical_costs'] = [costs.get(t['id']) or dict(task_id=t['id'], status='not_observed',
            policy=None, embedding=None, backend=None, search_attempts=None, wall_ms=None) for t in tasks]
        report['summaries'] = {p: summarize_revised(rs) for p, rs in report['rows'].items()}
        report['gate_outcomes'] = dict(Counter(t['status'] for t in report['supplement_traces']))
        report['rejection_counts'] = dict(Counter(code for t in report['supplement_traces']
            for r in t.get('rejections', []) for code in r['codes']))
        report['cost_observation_counts'] = dict(Counter(c['status'] for c in report['physical_costs']))
        report['execution_errors'] = {p: sum(r['error'] is not None for r in rs) for p, rs in report['rows'].items()}
        report['pairs'] = {'bridge_vs_control': pair_profiles(report['rows']['control'], report['rows']['bridge'])}
        report['matched_dispatch_pairs'] = sum(t['status'] == 'completed' for t in report['supplement_traces'])
        report['treatment_efficacy'] = 'dev_delivery_proxy_only' if (
            report['valid'] and report['matched_dispatch_pairs']) else 'pending'
        _publish(args.output, report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'seed', 'overlay', 'manifest', 'registration', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--suite', default=str(ROOT / '.runtime/benchmark-package/v1/dev/tasks.json'))
    args = p.parse_args(argv)
    reservation = reserve_output(args.output)
    try:
        report = execute(args, reservation)
    except BaseException as exc:
        report = {'version': 'reviewed-supplement-dev-v1', 'split': 'dev', 'valid': False,
                  'status': 'failed', 'error': {'stage': 'preflight', 'type': type(exc).__name__},
                  'run_id': reservation['run_id'], 'rows': None, 'summaries': None}
        _publish(args.output, report)
    print(json.dumps({'status': report['status'], 'valid': report['valid'], 'output': args.output}))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
