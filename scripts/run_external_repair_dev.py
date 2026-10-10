"""Readonly, registered Dev47 aspect selection and matched supplementation.

Runtime adapters receive only the question and authorized source rows. Labels
are read exclusively by the post-execution evaluator. Private raw records stay
under .runtime; public results contain only hashes, IDs and measurements.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from agent.supplement_proposal import FocusProposal, UnknownProposal, make_proposal
from agent.supplement_repair import run_repair
from app.evidence_focus import question_aspects, select_aspect_evidence
from app.execution_budget import ExecutionBudget, use_budget
from app.retrieval import retrieve_authorized, routing_goal
from app.security import readable_documents
from scripts import run_evidence_selection_dev as replay
from scripts import run_evidence_supplement_dev as supplement
from scripts.benchmark_runtime import atomic_json
from scripts.dev_evidence_annotations import summarize_revised, validate_annotation
from scripts.research_review import digest
from scripts.retrieval_quality_index import active_scope, index_ledger
from scripts.retrieval_quality_metrics import diagnose, summarize
from scripts.run_retrieval_quality_dev import (
    ROOT, SELECTION, EmbeddingOnly, _publish, code_fingerprints, missing_row, model_identity, reserve_output, sha, validate_suite,
)
from scripts.run_revised_evidence_dev import missing_revised
from scripts.run_reviewed_supplement_dev import rescore, validate_seed
from scripts.run_supplement_contract_dev import contract_for, select_context, verify_bridge_binding

PROFILES = ('baseline', 'aspect', 'control', 'treatment')


def public_trace(trace):
    return {k: deepcopy(trace[k]) for k in ('ranker_calls', 'ranker_pairs', 'ranker_wall_ms',
        'coverage_type', 'scope_rejected_count') if k in trace} | {'trace_sha256': digest(trace)}


def focus_proposal(models, contract, seed, needed):
    if contract.mode == 'entity_bridge':
        return make_proposal(models, contract, seed, needed)
    indices = {s.slot_id: i for i, s in enumerate(contract.slots)}
    choices = []
    stop = {'the', 'and', 'that', 'with', 'from', 'about', 'according', 'report', 'reported', 'discuss'}
    for offset, aspect in enumerate(question_aspects(contract)):
        i = indices[aspect.slot_id]
        if i not in needed or not 2 <= len(aspect.query) <= 200:
            continue
        words = set(re.findall(r'\w+', aspect.query.casefold())) - stop
        scoped = [r for r in seed if aspect.document_versions is None or
                  (r['document_id'], r['version_id']) in aspect.document_versions]
        present = max((len(words & set(re.findall(r'\w+', r['text'].casefold()))) / max(1, len(words))
                       for r in scoped), default=0)
        choices.append(((present, -offset), i, aspect.query))
    if not choices:
        return UnknownProposal(mode='unknown', reason='no_grounded_query')
    _, i, query = min(choices)
    return FocusProposal(mode='independent', target_slot=i, query=query)


def paths_for(args):
    return [Path(args.suite), SELECTION, Path(args.parent), Path(args.baseline),
            Path(args.overlay), Path(args.manifest)]


def private_root(path):
    root = Path(path).resolve()
    if not root.is_relative_to((ROOT / '.runtime').resolve()):
        raise ValueError('Raw records require the ignored project .runtime directory')
    return root


def registration_for(args):
    return {'version': 'external-repair-input-v1', 'split': 'dev',
        'input_bytes_sha256': [sha(p.read_bytes()) for p in paths_for(args)],
        'profiles': list(PROFILES), 'query_policy': 'least_query_word_coverage_late_tie_no_semantic_claim',
        'ranking': 'BGE_all_eligible_pairs_per_aspect_no_cross_query_score_comparison',
        'limits': {'chunks': 8, 'tokens': 5000, 'aspects': 8, 'search_per_arm': 1,
                   'shared_policy_calls': 2, 'seconds': 180},
        'scope': 'exact_question_source_version_bindings_no_global_lane_widening',
        'union': 'old_new_round_robin_rank_only', 'arm_order': 'task_index_parity',
        'choice': 'native_per_reference_no_loss_then_revised_upstream_delivery_then_profile_order',
        'gold_access': 'post_execution_evaluator_only', 'answer_correctness': None}


def collect(db, user, question, original, prior, corpus, cfg, source, offset, private):
    """Execute one task; no gold/required facts/expected behavior arguments."""
    from app.rerank import rerank
    started = time.monotonic()
    phases = replay.rehydrate_task(db, user, {'telemetry': original}, corpus)
    baseline = replay.rebind_rows(db, user, prior['context'], corpus)
    contract = contract_for(db, user, question)
    caps = replay.policy_caps(db, user, question)
    caps['document_quota'] = None
    models, backend = supplement.MeteredModels(), supplement.MeteredSearch(source)
    rankings, searches, rank_events = [], [], []
    budget = ExecutionBudget(cfg, saved={'limits': {'policy': 2, 'search': 2,
        'judge': 0, 'generation': 0, 'judge_tokens': 0}})
    budget.state['deadline'] = min(budget.state['deadline'], budget.clock()+180)
    private['physical_events'] = models.events
    private['backend_events'] = backend.events
    private['rankings'], private['searches'] = rankings, searches
    private['ranker_events'] = rank_events

    def rank(query, rows):
        began = time.monotonic()
        event = {'query_sha256': sha(query.encode()), 'input_pairs': len(rows), 'status': 'failed'}
        rank_events.append(event)
        try:
            budget.check()
            result = rerank(query, rows, window=len(rows))
            event['dispatch_completed'] = True
            budget.check()
            event['status'] = 'ok'
            return result
        finally:
            event['wall_ms'] = (time.monotonic()-began)*1000

    def selector(c, rows, *, focus_queries=None, **kwargs):
        with use_budget(budget):
            result = select_aspect_evidence(c, rows, ranker=rank, **kwargs)
        rankings.append(public_trace(result.trace))
        private.setdefault('ranking_traces', []).append(deepcopy(result.trace))
        result.trace['slot_floor'] = result.trace['aspect_floor']
        return result

    if contract.reason == 'unsupported_history':
        selected = baseline
        selection_status = 'history_baseline_retained'
    else:
        try:
            selected = selector(contract, phases['candidates'], **caps).evidence
            selection_status = 'selected'
        except ValueError as exc:
            # Unsupported question decomposition is explicit, never silently
            # truncated. No absent measurement is represented by a zero.
            if str(exc) not in ('Too many question aspects (maximum 8)', 'Invalid aspect query length'):
                raise
            selected, selection_status = baseline, 'unsupported_aspects_baseline_retained'
    contexts = {'baseline': baseline, 'aspect': selected}
    pools = {'baseline': phases['candidates'], 'aspect': phases['candidates']}
    allowed = frozenset(pair for r in contract.source_requests for pair in r.document_versions)

    def scoped_docs(_db, _user):
        docs = readable_documents(_db, _user)
        if contract.source_requests:
            return [d for d in docs if (d.id, d.active_version_id) in allowed]
        return docs

    def check(original_question, query):
        if original_question != question or contract_for(db, user, question) != contract:
            return False
        qc = contract_for(db, user, query)
        return all(r.resolution == 'supported' and set(r.document_versions) <= allowed
                   for r in qc.source_requests)

    def search(args):
        m, b, began = len(models.events), len(backend.events), time.monotonic()
        event = {'query_sha256': sha(args.query.encode()), 'status': 'failed'}
        try:
            with routing_goal(question, fuse=False):
                found = retrieve_authorized(db, user, args.query, top_k=args.top_k, cfg=cfg,
                    models=EmbeddingOnly(models), search=backend, readable_documents_fn=scoped_docs)
            if found.blocked_reason:
                raise ValueError('Supplement scope blocked')
            rows = supplement.live_candidates(db, user, found, corpus)
            if contract.source_requests and any((r['document_id'], r['version_id']) not in allowed for r in rows):
                raise ValueError('Supplement source widening')
            event.update(status='ok', candidate_count=len(rows))
            return rows
        finally:
            event.update(wall_ms=(time.monotonic()-began)*1000,
                         embedding=supplement.event_summary(models.events[m:]),
                         backend=supplement.event_summary(backend.events[b:]))
            searches.append(event)

    if selection_status != 'selected':
        trace = {'status': selection_status, 'errors': []}
        contexts.update(control=selected, treatment=selected)
        pools.update(control=phases['candidates'], treatment=phases['candidates'])
    else:
        result = run_repair(contract, selected, phases['candidates'], models=models,
            reauthorize=lambda rs: replay.rebind_rows(db, user, rs, corpus), search=search,
            validate_scope=check, budget=budget, caps=caps,
            order=('control', 'treatment') if offset % 2 == 0 else ('treatment', 'control'),
            proposal_provider=focus_proposal, evidence_selector=selector)
        trace = deepcopy(result.trace)
        private['arm_queries'] = {name: result.arms[name].get('query') for name in ('control', 'treatment')}
        for name in ('control', 'treatment'):
            arm = result.arms[name]
            if arm['evidence'] is not None:
                verify_bridge_binding(trace, arm['evidence'])
            contexts[name] = arm['evidence']
            pools[name] = (arm['candidates'] or phases['candidates']) if arm['evidence'] is not None else None
    private.update(contexts=deepcopy(contexts), pools=deepcopy(pools), trace=trace)
    rows = {name: supplement.score_context(db, user, {'id': private['task_id'], 'goal': question},
                         pools[name], contexts[name], corpus) if contexts[name] is not None else
            missing_row({'id': private['task_id']}, {'type': 'RepairFailure', 'stage': 'supplement'})
            for name in PROFILES}
    return rows, trace, {'rankings': rankings, 'ranker_events': rank_events, 'searches': searches,
        'policy': supplement.event_summary([e for e in models.events if e['role'] == 'policy']),
        'embedding': supplement.event_summary([e for e in models.events if e['role'] == 'embedding']),
        'backend': supplement.event_summary(backend.events), 'business_budget': budget.snapshot(),
        'wall_ms': (time.monotonic()-started)*1000}


def choose(report):
    base = report['rows']['baseline']
    candidates = []
    for index, name in enumerate(PROFILES):
        rows = report['rows'][name]
        if any(r['error'] for r in rows):
            continue
        if any(a['eligible'] and a['basis'] == 'source_atom_exact_proxy' and not
            set(a['phases']['context']['matched_reference_ids']) <=
            set(b['phases']['context']['matched_reference_ids']) for a, b in zip(base, rows, strict=True)):
            continue
        summary = report['summaries'][name]['by_basis']
        candidates.append(((summary['upstream_word4_proxy']['phases']['context']['delivered'],
                            summary['source_atom_exact_proxy']['phases']['context']['delivered'], -index), name))
    return max(candidates)[1] if candidates else 'baseline'


def execute(args, reservation):
    from app.clients import Search
    from app.config import settings
    from app.db import SessionLocal
    paths = paths_for(args)
    inputs = [p.read_bytes() for p in paths]
    suite, selection, parent, baseline, manifest = [json.loads(inputs[i]) for i in (0, 1, 2, 3, 5)]
    tasks = validate_suite(suite, selection, ROOT)
    originals = replay.validate_parent(parent, suite, selection, inputs[0], inputs[1])
    annotation = validate_annotation(suite, inputs[4], manifest, ROOT, inputs[0])
    if validate_seed(baseline, parent, suite, manifest) != 'relevant_doc':
        raise ValueError('Expected original relevant_doc baseline')
    reg_bytes = Path(args.registration).read_bytes()
    if json.loads(reg_bytes) != registration_for(args):
        raise ValueError('Registration mismatch')
    runtime = private_root(args.private_root) / reservation['run_id']
    runtime.mkdir()
    report = {'version': 'external-repair-dev-v1', 'split': 'dev', 'subset': False,
        'run_id': reservation['run_id'], 'status': 'failed', 'valid': False,
        'selected_task_ids': [t['id'] for t in tasks], 'rows': {p: [] for p in PROFILES},
        'legacy_rows': {p: [] for p in PROFILES}, 'traces': [], 'costs': [],
        'strict_task_success': None, 'semantic_recall': None, 'error': None}
    cfg = supplement.supplement_configuration(settings())
    settings_hash = digest(settings().model_dump(mode='json'))
    cached_cfg = replay.effective_configuration(parent)
    if cfg.embed_model != cached_cfg['embed_model'] or cfg.embed_dimension != cached_cfg['embed_dimension']:
        raise ValueError('Embedding configuration changed')
    code, configuration = code_fingerprints(), supplement.live_configuration()
    report['identity'] = {'code': code, 'registration': json.loads(reg_bytes),
        'settings_sha256': settings_hash,
        'configuration': configuration, 'parent_sha256': digest(parent),
        'annotation_manifest': manifest, 'private_directory_sha256': digest(str(runtime)),
        'corpus_sha256': parent['identity']['corpus_sha256'],
        'index_sha256': parent['identity']['source_index_sha256']}
    stage = 'source_identity'
    try:
        source = Search(index=parent['identity']['source_index']['index'])
        with SessionLocal() as db:
            db.execute(text('SET TRANSACTION READ ONLY'))
            users, records, corpus = active_scope(db)
            index = index_ledger(source, records)
            if corpus != parent['identity']['corpus'] or index != parent['identity']['source_index']:
                raise ValueError('Corpus/index identity differs')
            active_models = supplement.active_model_identity(settings())
            if {k: v for k, v in active_models['embedding'].items() if k != 'status'} != parent['identity']['models']['embedding']:
                raise ValueError('Embedding identity changed')
            report['identity']['models'] = active_models
            report['identity']['reranker'] = model_identity(settings(), rerank=True)['reranker']
            stage = 'baseline_reproduction'
            for task, original, prior in zip(tasks, originals, baseline['rows']['relevant_doc'], strict=True):
                expected = select_context(db, users[task['user']], task, original, corpus, 'relevant_doc')
                if expected['telemetry']['context'] != prior['telemetry']['context']:
                    raise ValueError('Baseline reproduction differs')
            stage = 'execution'
            for offset, (task, original, prior) in enumerate(zip(tasks, originals, baseline['rows']['relevant_doc'], strict=True)):
                private = {'task_id': task['id'], 'question': task['goal']}
                # Runtime arguments deliberately exclude evaluator-only task data.
                try:
                    raw_rows, trace, cost = collect(db, users[task['user']], task['goal'],
                        original['telemetry'], prior['telemetry'], corpus, cfg, source, offset, private)
                except BaseException:
                    atomic_json(runtime / (task['id']+'.json'), private)
                    events = private.get('physical_events', [])
                    report['costs'].append({'task_id': task['id'], 'status': 'interrupted_execution',
                        'rankings': private.get('rankings'), 'searches': private.get('searches'),
                        'ranker_events': private.get('ranker_events'),
                        'policy': supplement.event_summary([e for e in events if e['role'] == 'policy']),
                        'embedding': supplement.event_summary([e for e in events if e['role'] == 'embedding']),
                        'backend': supplement.event_summary(private.get('backend_events', []))})
                    raise
                atomic_json(runtime / (task['id']+'.json'), private)
                report['traces'].append({'task_id': task['id'], **trace})
                report['costs'].append({'task_id': task['id'], **cost})
                for name in PROFILES:
                    phases = {'candidates': private['pools'][name], 'admitted': private['contexts'][name],
                              'context': private['contexts'][name]}
                    report['legacy_rows'][name].append(diagnose(task, phases))
                    try:
                        evaluated = rescore(db, users[task['user']], task, raw_rows[name],
                                           corpus, annotation['task_records'][task['id']])
                    except Exception as exc:
                        evaluated = missing_revised(task, annotation['task_records'][task['id']],
                            {'type': type(exc).__name__, 'stage': 'evaluation'})
                    report['rows'][name].append(evaluated)
                atomic_json(runtime / 'checkpoint.json', report)
                print(json.dumps({'task_id': task['id'], 'status': trace['status']}), flush=True)
            _, final_records, final_corpus = active_scope(db)
            report['drift'] = {'inputs': any(p.read_bytes() != b for p, b in zip(paths, inputs, strict=True)),
                'code': code_fingerprints() != code, 'configuration': supplement.live_configuration() != configuration,
                'settings': digest(settings().model_dump(mode='json')) != settings_hash,
                'models': supplement.active_model_identity(settings()) != active_models,
                'reranker': model_identity(settings(), rerank=True)['reranker'] != report['identity']['reranker'],
                'corpus': final_corpus != corpus, 'index': index_ledger(source, final_records) != index,
                'registration': Path(args.registration).read_bytes() != reg_bytes}
            errors = any(r['error'] for rs in report['rows'].values() for r in rs)
            errors = errors or any(t.get('errors') for t in report['traces'])
            report.update(valid=not errors and not any(report['drift'].values()),
                          status='invalid_drift' if any(report['drift'].values()) else 'failed' if errors else 'complete')
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error={'type': type(exc).__name__, 'stage': stage})
    finally:
        observed_traces = {r['task_id']: r for r in report['traces']}
        observed_costs = {r['task_id']: r for r in report['costs']}
        report['traces'] = [observed_traces.get(t['id']) or {'task_id': t['id'],
            'status': 'not_observed', 'errors': []} for t in tasks]
        report['costs'] = [observed_costs.get(t['id']) or {'task_id': t['id'],
            'status': 'not_observed', 'rankings': None, 'searches': None, 'policy': None,
            'embedding': None, 'backend': None, 'wall_ms': None} for t in tasks]
        for name in PROFILES:
            measured = {r['task_id']: r for r in report['rows'][name]}
            old = {r['task_id']: r for r in report['legacy_rows'][name]}
            report['rows'][name] = [measured.get(t['id']) or missing_revised(t,
                annotation['task_records'][t['id']], report['error'] or
                {'type': 'NotObserved', 'stage': stage}) for t in tasks]
            report['legacy_rows'][name] = [old.get(t['id']) or diagnose(t, {}) for t in tasks]
        report['summaries'] = {p: summarize_revised(rows) for p, rows in report['rows'].items()}
        report['legacy_summaries'] = {p: summarize(rows) for p, rows in report['legacy_rows'].items()}
        if report['valid']:
            report['chosen_profile'] = choose(report)
        report['outcomes'] = dict(Counter(t['status'] for t in report['traces']))
        _publish(args.output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('suite', 'parent', 'baseline', 'overlay', 'manifest', 'registration', 'output', 'private-root'):
        p.add_argument('--'+key, required=True)
    args = p.parse_args()
    reservation = reserve_output(args.output)
    try:
        report = execute(args, reservation)
    except BaseException as exc:
        report = {'version': 'external-repair-dev-v1', 'split': 'dev',
            'run_id': reservation['run_id'], 'status': 'failed', 'valid': False,
            'error': {'type': type(exc).__name__, 'stage': 'preflight'},
            'rows': None, 'strict_task_success': None, 'semantic_recall': None}
        _publish(args.output, report)
    print(json.dumps({'valid': report['valid'], 'status': report['status']}))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
