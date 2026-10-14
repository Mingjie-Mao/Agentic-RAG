"""Capture four new Dev-only draft replays without exposing rejected text publicly."""
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
from app.qa import claim_validation_issues
from scripts import run_external_answers_dev as generation
from scripts.benchmark_runtime import atomic_json
from scripts.run_generation_diagnostic_dev import (
    MODELS, ROOT, TaggedModels, build_tasks, model_digest, readable_row,
)

TASK_IDS = ('SF-64b127ac198d', 'DD14', 'DD16', 'DD17')
PRIVATE = ROOT / '.runtime/roadmap-0-2-20261013/v1/claim-rejection'
PUBLIC = ROOT / 'artifacts/dev-claim-rejection-20261013-v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def capture_validation(captures):
    original = generation.validate_claims

    def validate(generated, evidence):
        result = original(generated, evidence)
        captures.append({'draft': generated.model_dump(mode='json'), 'evidence': evidence,
                         'issues': claim_validation_issues(generated, evidence),
                         'status': result[1]})
        return result

    return original, validate


def public_row(row):
    return {'task_id': row['task_id'], 'status': row['payload']['status'],
            'original_draft_available': False, 'sample_basis': 'new_generation_replay',
            'draft_sha256': digest(row['captures']), 'payload_sha256': digest(row['payload']),
            'issues': [issue for capture in row['captures'] for issue in capture['issues']],
            'physical_calls': len(row['events']),
            'prompt_tokens': sum(e.get('prompt_tokens') or 0 for e in row['events']),
            'completion_tokens': sum(e.get('completion_tokens') or 0 for e in row['events']),
            'latency_ms': row['latency_ms'], 'error_kind': row['error_kind']}


def main():
    if PRIVATE.exists() or PUBLIC.exists():
        raise SystemExit('Replay already exists; preserve it and use a new run name')
    PRIVATE.mkdir(parents=True)
    cfg = settings()
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ('app', 'agent', 'scripts') for p in sorted((ROOT/directory).glob('*.py'))}
    registration = {'version': 'claim-rejection-dev-v1', 'split': 'dev', 'task_ids': list(TASK_IDS),
                    'code_sha256': code, 'settings_sha256': digest(cfg.model_dump(mode='json')),
                    'model': {**MODELS['base'], 'digest': model_digest(MODELS['base'])},
                    'protocol': 'shared_contract', 'context': 'registered_current_context',
                    'limits': {'generation_calls': 3, 'seconds': 180, 'embedding_calls': 0},
                    'sample_basis': 'new_sample_original_rejected_draft_not_retained'}
    private = {'registration': registration, 'rows': [], 'status': 'running'}
    public = {'version': 'claim-rejection-dev-v1', 'split': 'dev', 'registration': registration,
              'rows': [], 'status': 'running', 'strict_task_success': None}
    atomic_json(PRIVATE/'registration.json', registration)
    atomic_json(PUBLIC, public)
    try:
        with SessionLocal() as db:
            db.execute(text('SET TRANSACTION READ ONLY'))
            db.execute(text('SET LOCAL statement_timeout = 30000'))
            built = {i['task']['id']: i for i in build_tasks(db)}
            for task_id in TASK_IDS:
                item = built[task_id]
                evidence = item['current']
                for row in evidence:
                    readable_row(db, item['user'], row['chunk_id'], expect=row, rank=row['rank'])
                captures, models = [], TaggedModels()
                original, generation.validate_claims = capture_validation(captures)
                started = time.monotonic()
                payload, usage, error = {'status': 'execution_failed'}, None, None
                try:
                    payload, usage = generation.generate_answer(item['task']['goal'], evidence,
                        models=models, cfg=cfg, protocol='shared_contract', enforce_zero_embedding=True,
                        check_conflict=item['user'].tenant_id not in cfg.conflict_check_disabled_tenants)
                    for row in evidence:
                        readable_row(db, item['user'], row['chunk_id'], expect=row, rank=row['rank'])
                except Exception as exc:
                    error = type(exc).__name__
                finally:
                    generation.validate_claims = original
                record = {'task_id': task_id, 'question': item['task']['goal'], 'evidence': evidence,
                          'captures': captures, 'payload': payload, 'usage': usage,
                          'events': models.events, 'error_kind': error,
                          'latency_ms': round((time.monotonic()-started)*1000, 1)}
                private['rows'].append(record)
                public['rows'].append(public_row(record))
                atomic_json(PRIVATE/'raw.json', private)
                atomic_json(PUBLIC, public)
                print(json.dumps(public['rows'][-1]), flush=True)
        drift = {'code': any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h
                             for p, h in code.items()),
                 'settings': digest(cfg.model_dump(mode='json')) != registration['settings_sha256'],
                 'model': model_digest(MODELS['base']) != registration['model']['digest']}
        for result in (private, public):
            result['drift'] = drift
            result['status'] = 'invalid_drift' if any(drift.values()) else 'complete'
    except BaseException as exc:
        for result in (private, public):
            result.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error_kind=type(exc).__name__)
        raise
    finally:
        atomic_json(PRIVATE/'raw.json', private)
        public['private_raw_sha256'] = hashlib.sha256((PRIVATE/'raw.json').read_bytes()).hexdigest()
        atomic_json(PUBLIC, public)


if __name__ == '__main__':
    main()
