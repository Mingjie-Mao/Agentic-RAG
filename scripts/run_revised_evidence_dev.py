"""Read-only full Dev replay with a separately frozen source annotation overlay.

Historical contexts are reauthorized, their legacy diagnostics reproduced, then
rescored. Gold is available only to this evaluator, never to a selection policy.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.evidence_selection import SelectionResult, select_relevant_evidence
from app.retrieval import route_sources
from app.security import readable_documents

from scripts.dev_evidence_annotations import diagnose_revised, summarize_revised, validate_annotation
from scripts.research_review import digest
from scripts.retrieval_quality_index import active_scope, index_ledger
from scripts.run_evidence_selection_dev import (
    PARENT_PROFILE, rehydrate_task, rebind_rows, validate_parent,
    verify_legacy, public_telemetry,
    policy_caps, replay_row,
)
from scripts.run_retrieval_quality_dev import (
    ROOT, SELECTION, _publish, code_fingerprints, reserve_output, sha, validate_suite,
)

RELEVANCE_PROFILES = ('legacy_rerank', 'relevant_strict', 'relevant_doc',
                      'relevant_source', 'relevant_both')


def relevance_declarations():
    return [{'name': p, 'method': 'recorded_parent_context' if p == 'legacy_rerank'
             else 'cached_original_question_relevance_v1',
             'document_cap_relaxed': p in ('relevant_doc', 'relevant_both'),
             'source_cap_relaxed': p in ('relevant_source', 'relevant_both'),
             'source_floor': 'one_feasible_best_per_original_routed_group',
             'limit': 8, 'token_budget': 5000, 'new_model_calls': 0}
            for p in RELEVANCE_PROFILES]


def select_relevance_context(db, user, task, parent_row, corpus, profile):
    phases = rehydrate_task(db, user, parent_row, corpus)
    verify_legacy(task, parent_row, phases)
    caps = policy_caps(db, user, task['goal'])
    if profile in ('relevant_doc', 'relevant_both'):
        caps['document_quota'] = None
    if profile in ('relevant_source', 'relevant_both'):
        caps['source_quota'] = None
    groups = route_sources(task['goal'], readable_documents(db, user), strict_dates=False)
    source_groups = {digest([g['mention'], g['key']]): g['key'] for g in groups}
    started = time.monotonic()
    if profile == 'legacy_rerank':
        from scripts.run_evidence_selection_dev import evidence_tokens
        result = SelectionResult(phases['context'], evidence_tokens(phases['context']),
                                 {'method': 'recorded_parent_context'})
    else:
        result = select_relevant_evidence(phases['candidates'], **caps, source_groups=source_groups)
    row = replay_row(db, user, task, parent_row, phases, result, corpus,
                     legacy=profile == 'legacy_rerank',
                     elapsed_ms=(time.monotonic() - started) * 1000)
    # These extra fields are constructed by the pure strategy, with no raw text.
    row['selection_trace'].update({k: deepcopy(result.trace[k]) for k in
                                  ('source_floor', 'relevance_order') if k in result.trace})
    return row


def choose_seed(report):
    """Dev-only rule registered before replay, including per-reference no-loss."""
    if not report['valid'] or report['identity']['profiles'] != list(RELEVANCE_PROFILES):
        raise ValueError('complete registered relevance comparison required')
    baseline = report['rows']['legacy_rerank']
    qualifying = []
    for index, profile in enumerate(RELEVANCE_PROFILES):
        rows = report['rows'][profile]
        if len(rows) != 47 or any(r['error'] is not None for r in rows):
            continue
        if any(a['eligible'] and a['basis'] == 'source_atom_exact_proxy'
               and not set(a['phases']['context']['matched_reference_ids']) <=
                       set(b['phases']['context']['matched_reference_ids'])
               for a, b in zip(baseline, rows, strict=True)):
            continue
        groups = report['summaries'][profile]['by_basis']
        up = groups['upstream_word4_proxy']['phases']['context']['delivered']
        native = groups['source_atom_exact_proxy']['phases']['context']['delivered']
        relaxations = 2 if profile == 'relevant_both' else int(profile in ('relevant_doc', 'relevant_source'))
        qualifying.append(((up, native, -relaxations, -index), profile))
    return max(qualifying)[1] if qualifying else 'legacy_rerank'


def cached_contexts(parent, histories):
    """Fail closed on incomplete or source/protocol-mismatched old contexts."""
    contexts = {'legacy_rerank': parent['rows'][PARENT_PROFILE]}
    ids = parent['selected_task_ids']
    for history in histories:
        if (history.get('valid') is not True or history.get('status') != 'complete'
                or history.get('split') != 'dev' or history.get('subset') is not False
                or history.get('error') is not None or any(history.get('drift', {}).values())
                or history.get('selected_task_ids') != ids
                or history['identity'].get('parent_sha256') != digest(parent)):
            raise ValueError('historical replay identity differs')
        for profile, rows in history['rows'].items():
            if (len(rows) != 47 or [r['task_id'] for r in rows] != ids
                    or any(r.get('error') is not None for r in rows)):
                raise ValueError('historical contexts incomplete')
            if profile in contexts:
                previous = contexts[profile]
                def normalized(rows):
                    return [dict(r, parent_rank=r.get('parent_rank', r['rank'])) for r in rows]
                if any(normalized(a['telemetry']['context']) != normalized(b['telemetry']['context'])
                       for a, b in zip(previous, rows, strict=True)):
                    raise ValueError('duplicate historical profile differs')
            else:
                contexts[profile] = rows
    return contexts


def score_context(db, user, task, row, parent_row, corpus, record):
    phases = rehydrate_task(db, user, row, corpus)
    verify_legacy(task, row, phases)
    candidates = rebind_rows(db, user, parent_row['telemetry']['candidates'], corpus)
    if phases['candidates'] != candidates:
        raise ValueError('historical candidate pool differs')
    result = diagnose_revised(task, phases, record)
    result.update(error=None, telemetry={
        **{p: public_telemetry(v) for p, v in phases.items()},
        'context_tokens': row['telemetry']['context_tokens']})
    return result


def missing_revised(task, record, error):
    return diagnose_revised(task, {}, record) | {'error': error, 'telemetry': None}


def execute(args, reservation):
    from app.clients import Search
    from app.db import SessionLocal

    paths = [Path(args.suite), SELECTION, Path(args.parent), Path(args.overlay),
             Path(args.manifest), *map(Path, args.history)]
    frozen = {str(p): p.read_bytes() for p in paths}
    suite, selection, parent, manifest = [json.loads(frozen[str(p)])
        for p in (paths[0], paths[1], paths[2], paths[4])]
    tasks = validate_suite(suite, selection, ROOT)
    parent_rows = validate_parent(parent, suite, selection, frozen[str(paths[0])], frozen[str(SELECTION)])
    annotations = validate_annotation(suite, frozen[str(paths[3])], manifest, ROOT, frozen[str(paths[0])])
    sources = cached_contexts(parent, [json.loads(frozen[str(p)]) for p in paths[5:]])
    if args.stage == 'relevance':
        sources = {p: parent_rows for p in RELEVANCE_PROFILES}
    report = {'version': 'revised-evidence-dev-v1', 'split': 'dev', 'subset': False,
        'run_id': reservation['run_id'], 'status': 'failed', 'valid': False,
        'selected_task_ids': [t['id'] for t in tasks], 'error': None,
        'strict_task_success': None, 'semantic_recall': None,
        'rows': {p: [] for p in sources}, 'identity': {
            'input_bytes_sha256': [sha(frozen[str(p)]) for p in paths],
            'annotation_manifest': deepcopy(manifest), 'parent_sha256': digest(parent),
            'profiles': list(sources), 'policy_model_calls': 0, 'embedding_calls': 0,
            'new_retrieval_calls': 0, 'code': code_fingerprints(),
            'corpus_sha256': parent['identity']['corpus_sha256'],
            'source_index_sha256': parent['identity']['source_index_sha256']}}
    report['identity']['stage'] = args.stage
    report['identity']['declarations'] = relevance_declarations() if args.stage == 'relevance' else None
    registration_path = ROOT / '.runtime/dev-evidence-repair-20261009' / (reservation['run_id'] + '-registration.json')
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    with registration_path.open('x') as stream:
        json.dump(report['identity'], stream, indent=2)
    stage = 'source_identity'
    try:
        source = Search(index=parent['identity']['source_index']['index'])
        with SessionLocal() as db:
            db.execute(text('SET TRANSACTION READ ONLY'))
            users, records, corpus = active_scope(db)
            index = index_ledger(source, records)
            if (corpus != parent['identity']['corpus'] or index != parent['identity']['source_index']):
                raise ValueError('live corpus/index differs from frozen parent')
            stage = 'legacy_reproduction'
            for task, row in zip(tasks, parent_rows, strict=True):
                verify_legacy(task, row, rehydrate_task(db, users[task['user']], row, corpus))
            stage = 'revised_evaluation'
            for profile, rows in sources.items():
                for task, row, original in zip(tasks, rows, parent_rows, strict=True):
                    record = annotations['task_records'][task['id']]
                    try:
                        if args.stage == 'relevance':
                            row = select_relevance_context(db, users[task['user']], task,
                                                           original, corpus, profile)
                        result = score_context(db, users[task['user']], task, row,
                                               original, corpus, record)
                        if args.stage == 'relevance':
                            result.update(selection_trace=row['selection_trace'],
                                          selection_statistics=row['selection_statistics'],
                                          selection_ms=row['telemetry']['selection_ms'])
                    except Exception as exc:
                        result = missing_revised(task, record,
                            {'type': type(exc).__name__, 'stage': stage})
                    report['rows'][profile].append(result)
            stage = 'drift_check'
            _, final_records, final_corpus = active_scope(db)
            report['drift'] = {'inputs': any(p.read_bytes() != frozen[str(p)] for p in paths),
                'code': code_fingerprints() != report['identity']['code'],
                'corpus': final_corpus != corpus, 'index': index_ledger(source, final_records) != index}
            validate_suite(suite, selection, ROOT)
            validate_annotation(suite, paths[3].read_bytes(), manifest, ROOT, paths[0].read_bytes())
            errors = any(r['error'] is not None for rows in report['rows'].values() for r in rows)
            report.update(valid=not errors and not any(report['drift'].values()),
                          status='invalid_drift' if any(report['drift'].values()) else
                          'failed' if errors else 'complete')
    except BaseException as exc:
        report.update(error={'type': type(exc).__name__, 'stage': stage}, valid=False,
            status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed')
    finally:
        for profile, rows in report['rows'].items():
            by_id = {r['task_id']: r for r in rows}
            report['rows'][profile] = [by_id.get(t['id']) or missing_revised(t,
                annotations['task_records'][t['id']], report['error'] or
                {'type': 'NotObserved', 'stage': stage}) for t in tasks]
        report['summaries'] = {p: summarize_revised(rows) for p, rows in report['rows'].items()}
        report['execution_errors'] = {p: dict(tasks=sum(r['error'] is not None for r in rows),
            by_stage=dict(Counter(r['error']['stage'] for r in rows if r['error'])))
            for p, rows in report['rows'].items()}
        if args.stage == 'relevance' and report['valid']:
            report['chosen_seed_profile'] = choose_seed(report)
        _publish(args.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--stage', choices=('audit', 'relevance'), default='audit')
    parser.add_argument('--history', action='append', default=[])
    parser.add_argument('--overlay', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--suite', default=str(ROOT / '.runtime/benchmark-package/v1/dev/tasks.json'))
    args = parser.parse_args(argv)
    reservation = reserve_output(args.output)
    try:
        report = execute(args, reservation)
    except BaseException as exc:
        report = {'version': 'revised-evidence-dev-v1', 'split': 'dev', 'subset': False,
                  'run_id': reservation['run_id'], 'status': 'failed', 'valid': False,
                  'error': {'type': type(exc).__name__, 'stage': 'preflight'},
                  'rows': None, 'summaries': None, 'strict_task_success': None,
                  'semantic_recall': None}
        _publish(args.output, report)
    print(json.dumps({'status': report['status'], 'valid': report['valid'],
                      'output': str(Path(args.output).resolve())}))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
