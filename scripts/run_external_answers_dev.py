"""Equal-generator Dev47 comparison over reauthorized registered contexts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from app.config import settings
from app.execution_budget import ExecutionBudget, use_budget
from app.qa import answer_verdict, validate_claims
from scripts import run_evidence_selection_dev as replay
from scripts import run_evidence_supplement_dev as supplement
from scripts.benchmark_package_scoring import answer_review_packet, provisional_score
from scripts.benchmark_runtime import atomic_json
from scripts.research_review import digest
from scripts.retrieval_quality_index import active_scope, index_ledger
from scripts.run_external_repair_dev import private_root
from scripts.run_retrieval_quality_dev import ROOT, SELECTION, _publish, code_fingerprints, reserve_output, sha, validate_suite

ARMS = ('baseline', 'treatment')


def generate_answer(question, evidence, *, models, cfg, check_conflict):
    """Only original question/source evidence, never benchmark labels."""
    budget = ExecutionBudget(cfg, saved={'limits': {'generation': 3, 'policy': 0,
        'search': 0, 'judge': 0, 'judge_tokens': 0}})
    budget.state['deadline'] = min(budget.state['deadline'], budget.clock()+180)
    supplied = [dict(r, id=r['chunk_id']) for r in evidence]
    with use_budget(budget):
        generated, usage = models.generate(question, supplied, max_output_tokens=700,
                                           check_conflict=check_conflict)
        claims, status = validate_claims(generated, supplied)
        if status == 'answered' and usage.get('answer_status') == 'conflict':
            status = 'conflict'
        verdict, verdict_usage = answer_verdict(models, question, claims, status, evidence=supplied)
    used = {cid for c in claims for cid in c['evidence_ids']}
    citations = [{k: r[k] for k in ('chunk_id', 'document_id', 'version_id', 'source_sha256',
                                   'title', 'text', 'locator')} for r in evidence if r['chunk_id'] in used]
    return {'status': status, 'claims': claims, 'citations': citations, 'verdict': verdict}, {
        'generation': usage, 'verdict': verdict_usage, 'business_budget': budget.snapshot()}


def registration_for(args):
    tasks = json.loads(Path(args.suite).read_bytes())['tasks']
    contexts = private_root(args.context_root)
    return {'version': 'external-answers-input-v1', 'split': 'dev', 'arms': list(ARMS),
        'inputs_sha256': [sha(Path(p).read_bytes()) for p in
            (args.suite, args.retrieval, args.source_audit, args.parent, SELECTION)],
        'context_bytes_sha256': {t['id']: sha((contexts / (t['id']+'.json')).read_bytes()) for t in tasks},
        'context_profile': 'treatment', 'generator': 'unchanged_Models_generate_and_binding_verdict',
        'generation_limits': {'calls': 3, 'first_output_tokens': 700, 'conflict_tokens': 200,
            'verdict_max_tokens': 500, 'seconds': 180},
        'order': 'task_index_parity', 'review': 'method_blind_named_model_after_generation',
        'strict_accuracy': None, 'semantic_recall': None}


def execute(args, reservation):
    from app.clients import Search
    from app.db import SessionLocal
    inputs = [Path(p) for p in (args.suite, args.retrieval, args.source_audit, args.registration, args.parent)] + [SELECTION]
    before = [p.read_bytes() for p in inputs]
    suite, retrieval = [json.loads(before[i]) for i in (0, 1)]
    tasks = validate_suite(suite, json.loads(before[5]), ROOT)
    if not retrieval.get('valid') or retrieval['selected_task_ids'] != [t['id'] for t in tasks]:
        raise ValueError('Complete valid Dev47 retrieval required')
    if json.loads(before[3]) != registration_for(args):
        raise ValueError('Generation registration differs')
    source_root, output_root = private_root(args.context_root), private_root(args.private_root)
    runtime = output_root / reservation['run_id']
    runtime.mkdir()
    code, configuration = code_fingerprints(), supplement.live_configuration()
    cfg = settings()
    settings_hash = digest(cfg.model_dump(mode='json'))
    if cfg.answer_quality_enabled or cfg.source_facts_enabled:
        raise ValueError('Unexpected extra generation/judge pipeline')
    raw = {'version': 'external-answers-dev-v1', 'run_id': reservation['run_id'], 'split': 'dev',
        'corpus_sha256': retrieval['identity']['corpus_sha256'], 'valid': False, 'status': 'failed',
        'results': {name: [] for name in ARMS}, 'registration': json.loads(before[3]), 'error': None}
    public = {'version': raw['version'], 'split': 'dev', 'run_id': raw['run_id'],
        'strict_task_success': None, 'semantic_recall': None, 'valid': False, 'status': 'failed',
        'rows': [], 'identity': {'code': code, 'configuration': configuration,
                                'settings_sha256': settings_hash,
                                'registration': json.loads(before[3])}, 'error': None}
    stage = 'identity'
    try:
        parent = json.loads(before[4])
        if digest(parent) != retrieval['identity']['parent_sha256']:
            raise ValueError('Registered parent differs')
        source = Search(index=parent['identity']['source_index']['index'])
        with SessionLocal() as db:
            db.execute(text('SET TRANSACTION READ ONLY'))
            users, records, corpus = active_scope(db)
            index = index_ledger(source, records)
            if digest(corpus) != raw['corpus_sha256'] or digest(index) != retrieval['identity']['index_sha256']:
                raise ValueError('Corpus/index changed')
            active_models = supplement.active_model_identity(cfg)
            if active_models != retrieval['identity']['models']:
                raise ValueError('Retrieval model identity changed')
            generator_cfg = cfg.model_copy(update={'agent_policy_model': cfg.chat_model, 'agent_policy_url': ''})
            generator_identity = supplement.active_model_identity(generator_cfg)['policy']
            public['identity']['models'] = active_models
            public['identity']['generator'] = generator_identity
            stage = 'generation'
            for offset, task in enumerate(tasks):
                original = json.loads((source_root / (task['id']+'.json')).read_bytes())
                if original['task_id'] != task['id'] or original['question'] != task['goal']:
                    raise ValueError('Private context task differs')
                for name in ARMS if offset % 2 == 0 else reversed(ARMS):
                    models = supplement.MeteredModels()
                    started, err, usage = time.monotonic(), None, None
                    contexts = original['contexts'][name]
                    if contexts is None:
                        raise ValueError('Missing source context')
                    evidence = replay.rebind_rows(db, users[task['user']], contexts, corpus)
                    expected = retrieval['rows'][name][offset]['telemetry']['context']
                    def bindings(rows):
                        return [[r[k] for k in replay.IDENTITY_FIELDS] for r in rows]
                    if bindings(evidence) != bindings(expected):
                        raise ValueError('Registered selected evidence differs')
                    try:
                        payload, usage = generate_answer(task['goal'], evidence, models=models, cfg=cfg,
                            check_conflict=users[task['user']].tenant_id not in cfg.conflict_check_disabled_tenants)
                        if replay.rebind_rows(db, users[task['user']], evidence, corpus) != evidence:
                            raise ValueError('Post-generation source binding changed')
                    except Exception as exc:
                        err = type(exc).__name__
                        payload = {'status': 'execution_failed', 'claims': [], 'citations': [], 'verdict': None}
                    elapsed = (time.monotonic()-started)*1000
                    # Contexts are replayed, not freshly retrieved here. Do not
                    # manufacture a complete tool trace from final evidence.
                    trace = []
                    row = provisional_score(task, payload, trace, 1, elapsed,
                        execution_error_kind=err, scenario_events={})
                    row.update(task_id=task['id'], run_id=raw['run_id'], retrieved_evidence=evidence,
                        first_retrieval_documents=None, steps=None,
                        trace_complete=False, tool_trace=trace, generation_usage=usage,
                        retrieval_provenance='immutable_registered_context_replay',
                        usage=supplement.event_summary(models.events))
                    raw['results'][name].append(row)
                    public['rows'].append({'task_id': task['id'], 'arm': name, 'status': payload['status'],
                        'answer_sha256': digest(payload), 'error': err, 'latency_ms': elapsed,
                        'physical_generation': supplement.event_summary(models.events)})
                    atomic_json(runtime / 'raw.json', raw)
                    print(json.dumps({'task_id': task['id'], 'arm': name, 'status': payload['status']}), flush=True)
            _, final_records, final_corpus = active_scope(db)
            drift = {'code': code_fingerprints() != code, 'configuration': supplement.live_configuration() != configuration,
                'settings': digest(settings().model_dump(mode='json')) != settings_hash,
                'models': supplement.active_model_identity(cfg) != active_models,
                'generator': supplement.active_model_identity(generator_cfg)['policy'] != generator_identity,
                'corpus': final_corpus != corpus, 'index': index_ledger(source, final_records) != index,
                'inputs': any(p.read_bytes() != b for p, b in zip(inputs, before, strict=True)),
                'context_registration': registration_for(args) != json.loads(before[3])}
            for r in (raw, public):
                r.update(drift=drift, valid=not any(drift.values()),
                         status='invalid_drift' if any(drift.values()) else 'complete')
    except BaseException as exc:
        for r in (raw, public):
            r.update(error={'stage': stage, 'type': type(exc).__name__},
                     status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed')
    finally:
        for name in ARMS:
            observed = {r['task_id']: r for r in raw['results'][name]}
            raw['results'][name] = [observed.get(t['id']) or {'task_id': t['id'],
                'run_id': raw['run_id'], 'not_observed': True, 'status': 'not_observed',
                'answer_payload': {'status': 'not_observed', 'claims': [], 'citations': []},
                'retrieved_evidence': None, 'first_retrieval_documents': None, 'steps': None,
                'latency_ms': None, 'trace_complete': False, 'usage': None,
                'execution_error_kind': 'NotObserved', 'scenario_events': {}}
                for t in tasks]
        atomic_json(runtime / 'raw.json', raw)
        reviewed_suite = deepcopy(suite)
        reviewed_suite['arms'] = list(ARMS)
        atomic_json(runtime / 'suite.json', reviewed_suite)
        if raw['valid']:
            atomic_json(runtime / 'review-packet.json', answer_review_packet(raw, reviewed_suite))
            atomic_json(runtime / 'registered-review-guidance.json', json.loads(before[2]))
        public['private_raw_sha256'] = digest(raw)
        _publish(args.output, public)
    return public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('suite', 'retrieval', 'parent', 'source-audit', 'registration', 'context-root', 'private-root', 'output'):
        p.add_argument('--'+key, required=True)
    args = p.parse_args()
    reservation = reserve_output(args.output)
    try:
        report = execute(args, reservation)
    except BaseException as exc:
        report = {'version': 'external-answers-dev-v1', 'split': 'dev', 'status': 'failed', 'valid': False,
            'strict_task_success': None, 'error': {'stage': 'preflight', 'type': type(exc).__name__}}
        _publish(args.output, report)
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
