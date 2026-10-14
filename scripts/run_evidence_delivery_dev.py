"""One immutable Dev evidence-injection diagnostic, using unchanged Legacy generation.

Original text and responses stay local. This is a replay of frozen generation
contexts, not a fresh end-to-end retrieval run or a deployable gold-assisted arm.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.config import settings
from app.db import SessionLocal
from scripts.benchmark_runtime import atomic_json
from scripts.run_external_answers_dev import generate_answer
from scripts.run_generation_diagnostic_dev import MODELS, ROOT, model_digest, readable_row
from scripts.run_grounding_attribution_dev import CapturedModels, costs, fingerprints

PRIVATE = ROOT / '.runtime/evidence-delivery-20261014/v1/delivery'
INPUT = PRIVATE.parent / 'inputs.json'
PUBLIC = ROOT / 'artifacts/dev-evidence-delivery-20261014-v1.json'
REGISTRATION = ROOT / 'benchmarks/enterprise_rag/v1/dev/evidence-delivery-20261014/registration-v1.json'
ARMS = ('legacy_current', 'necessary_evidence_only')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if any(p.exists() for p in (PRIVATE, PUBLIC, REGISTRATION)):
        raise SystemExit('Immutable diagnostic exists; no overwrite or repeated tuning')
    packet = json.loads(INPUT.read_text())
    cfg = settings()
    assert packet['split'] == 'dev' and len(packet['rows']) == 8
    assert cfg.verdict_protocol == 'legacy' and cfg.task_contract_enabled
    assert not any((cfg.answer_quality_enabled, cfg.answer_extractive_enabled,
                    cfg.answer_contract_enabled, cfg.source_facts_enabled))
    code = fingerprints()
    registration = {
        'version': 'evidence-delivery-dev-v1', 'split': 'dev',
        'task_ids': [r['task_id'] for r in packet['rows']],
        'primary_task_ids': [r['task_id'] for r in packet['rows'] if not r['pending']],
        'pending_task_ids': [r['task_id'] for r in packet['rows'] if r['pending']],
        'controls': ['DD16', 'SF-66b7ed601d85'],
        'arms': list(ARMS), 'order': 'rotated_by_task_index',
        'input_sha256': sha(INPUT), 'code_sha256': code,
        'settings_sha256': hashlib.sha256(cfg.model_dump_json().encode()).hexdigest(),
        'model': {**MODELS['base'], 'digest': model_digest(MODELS['base'])},
        'generation': 'existing_generate_answer_shared_contract_ordinary_legacy',
        'retrieval': 'none; immutable_original_context_replay; all source bytes reauthorized',
        'injection': 'append_confirmed_active_original_chunks_without_removing_any_current_chunk',
        'injection_scope': 'historical_candidate_body_SF64_SF596; same_article_publication_headers; DD14_directory_corpus_original_candidate_unknown',
        'identical_context_controls': ['U16', 'DD17', 'SF-4345a2fc43c9', 'DD16', 'SF-66b7ed601d85'],
        'caps': {'generation_calls': 3, 'seconds': 180, 'embedding': 0, 'search': 0,
                 'first_output_tokens': 700, 'num_ctx': 8192,
                 'diagnostic_row_cap_deviation': '10_rows_for_SF64_SF596; no_silent_replacement'},
        'gold_answers_or_facts_input_to_model': False, 'formal_Core': False,
        'strict_rubric': 'Complete requested facts; each fact has genuine own citation support with correct entity/source/time/condition. Requested verdict agrees with body. Literal transport, semantic support, and strict answer correctness are separate. Pending annotation remains null.',
        'metrics': ['evidence_completeness', 'direction_correct', 'strict_success',
                    'required_fact_expression', 'required_fact_citation_support',
                    'mechanical_pass', 'source_time_numeric_branch_errors',
                    'physical_calls', 'prompt_tokens', 'completion_tokens', 'latency_p50_p95',
                    'insufficient_evidence', 'execution_failed'],
        'review': 'source_inspection_GPT_family_same_session_not_blind_not_independent_human0; deterministic checks separately recorded',
        'decision': 'Diagnostic only. Select at most one generic candidate after review. No deploy or default change.',
    }
    PRIVATE.mkdir(parents=True)
    REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(REGISTRATION, registration)
    atomic_json(PRIVATE / 'registration.json', registration)
    raw = {'registration': registration, 'status': 'running', 'rows': []}
    public = {'version': registration['version'], 'split': 'dev', 'status': 'running',
              'registration_sha256': sha(REGISTRATION), 'review': 'pending',
              'strict_success': None, 'default': 'legacy', 'adopted': False, 'rows': []}

    def save():
        atomic_json(PRIVATE / 'raw.json', raw)
        public['private_raw_sha256'] = sha(PRIVATE / 'raw.json')
        atomic_json(PUBLIC, public)

    save()
    try:
        from app.models import User
        with SessionLocal() as db:
            db.execute(text('SET TRANSACTION READ ONLY'))
            db.execute(text('SET LOCAL statement_timeout = 30000'))
            for offset, item in enumerate(packet['rows']):
                task = item['task']
                user = db.get(User, task['user'])
                for arm in ARMS if offset % 2 == 0 else reversed(ARMS):
                    sources = item['current'] + (item['additions'] if arm == ARMS[1] else [])
                    rows = [readable_row(db, user, r['chunk_id'], expect=r, rank=i)
                            for i, r in enumerate(sources, 1)]
                    models = CapturedModels()
                    payload = {'status': 'execution_failed', 'claims': [], 'verdict': None}
                    usage, error = None, None
                    started = time.monotonic()
                    try:
                        payload, usage = generate_answer(task['goal'], rows, models=models, cfg=cfg,
                            check_conflict=user.tenant_id not in cfg.conflict_check_disabled_tenants,
                            protocol='shared_contract', enforce_zero_embedding=True)
                        for i, r in enumerate(rows, 1):
                            readable_row(db, user, r['chunk_id'], expect=r, rank=i)
                    except Exception as exc:
                        error = type(exc).__name__
                    record = {'task_id': task['id'], 'arm': arm, 'question': task['goal'],
                              'sources': [dict(r, id=f'E{i}') for i, r in enumerate(rows, 1)],
                              'drafts': models.drafts, 'payload': payload, 'usage': usage,
                              'events': models.events, 'error_kind': error,
                              'budget': getattr(models, 'answer_budget', None).snapshot()
                              if getattr(models, 'answer_budget', None) else None,
                              'latency_ms': round((time.monotonic() - started) * 1000, 1)}
                    raw['rows'].append(record)
                    projection = {key: record[key] for key in ('task_id', 'arm', 'error_kind', 'latency_ms')}
                    projection.update(status=payload['status'], cost=costs(models.events),
                        context_chunk_ids=[r['chunk_id'] for r in rows],
                        answer_sha256=hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest(),
                        draft_issues=[d['issues'] for d in models.drafts])
                    public['rows'].append(projection)
                    save()
                    print(json.dumps(projection), flush=True)
        drift = {'code': fingerprints() != code, 'input': sha(INPUT) != registration['input_sha256'],
                 'settings': hashlib.sha256(cfg.model_dump_json().encode()).hexdigest() != registration['settings_sha256'],
                 'model': model_digest(MODELS['base']) != registration['model']['digest']}
        for result in (raw, public):
            result.update(drift=drift, status='invalid_drift' if any(drift.values()) else 'complete')
    except BaseException as exc:
        for result in (raw, public):
            result.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error_kind=type(exc).__name__)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
