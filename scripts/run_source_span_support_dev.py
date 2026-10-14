"""Single Dev trial of the existing original-span renderer, unchanged Legacy judge.

Baseline is the just-completed necessary-evidence diagnostic arm. Candidate uses
fresh generation with exactly the same request; equality of requests and initial
drafts is an attribution gate. No prompt, selector or production defaults change.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from app.answer_quality import bind_source_text
from app.config import settings
from app.db import SessionLocal
from app.execution_budget import ExecutionBudget, use_budget
from app.qa import answer_verdict, claim_validation_issues, validate_claims
from app.task_contract import contract_generate
from scripts.benchmark_runtime import atomic_json
from scripts.run_generation_diagnostic_dev import MODELS, ROOT, model_digest, readable_row
from scripts.run_grounding_attribution_dev import CapturedModels, costs, fingerprints

PARENT = ROOT / '.runtime/evidence-delivery-20261014/v1'
INPUT = PARENT / 'inputs.json'
BASELINE = PARENT / 'delivery/raw.json'
BASELINE_REVIEW = ROOT / 'artifacts/dev-evidence-delivery-20261014-v1.review.json'
PRIVATE = PARENT / 'support'
PUBLIC = ROOT / 'artifacts/dev-source-span-support-20261014-v1.json'
REGISTRATION = ROOT / 'benchmarks/enterprise_rag/v1/dev/evidence-delivery-20261014/source-span-registration-v1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def app_code(code):
    return {p: h for p, h in code.items() if p.startswith(('app/', 'agent/'))}


def initial_drafts(rows):
    return [{'draft': r['draft'], 'issues': r['issues']} for r in rows]


def generation_requests(events):
    return [e['request_sha256'] for e in events if e['operation'] == 'generation']


def main():
    if any(p.exists() for p in (PRIVATE, PUBLIC, REGISTRATION)):
        raise SystemExit('Immutable one-candidate trial exists; no overwrite or tuning rerun')
    base = json.loads(BASELINE.read_text())
    assert base['status'] == 'complete' and not any(base['drift'].values())
    reviewed = json.loads(BASELINE_REVIEW.read_text())
    assert reviewed['status'] == 'complete'
    packet = json.loads(INPUT.read_text())
    cfg = settings()
    assert cfg.verdict_protocol == 'legacy' and not any((cfg.answer_quality_enabled,
        cfg.answer_contract_enabled, cfg.answer_extractive_enabled, cfg.source_facts_enabled))
    code = fingerprints()
    assert app_code(code) == app_code(base['registration']['code_sha256'])
    assert hashlib.sha256(cfg.model_dump_json().encode()).hexdigest() == base['registration']['settings_sha256']
    baseline = {r['task_id']: r for r in base['rows'] if r['arm'] == 'necessary_evidence_only'}
    assert len(baseline) == 8
    registration = {
        'version': 'source-span-support-dev-v1', 'split': 'dev',
        'task_ids': base['registration']['task_ids'],
        'primary_task_ids': base['registration']['primary_task_ids'],
        'pending_task_ids': base['registration']['pending_task_ids'],
        'input_sha256': sha(INPUT), 'baseline_raw_sha256': sha(BASELINE),
        'baseline_review_sha256': sha(BASELINE_REVIEW), 'code_sha256': code,
        'settings_sha256': base['registration']['settings_sha256'],
        'model': base['registration']['model'],
        'candidate': 'unchanged_original_generation_then_existing_bind_source_text_then_existing_validate_claims_and_Legacy_answer_verdict',
        'baseline': 'fresh_necessary_evidence_only_arm_in_preceding_frozen_diagnostic_same_round',
        'generation_equality_gate': 'exact_generation_request_hashes_and_initial_draft_payloads_must_match_per_task; actual_latency_not_equated',
        'application_changes': 'none; reuse existing functions; no new prompts or business validators',
        'context': 'same diagnostic complete contexts; no additional source injection in candidate',
        'render_failure': 'verification_failed; no fallback_to_unproved_paraphrase_or_truncation',
        'limits': {'generation_calls': 3, 'seconds': 180, 'embedding': 0, 'search': 0, 'judge': 0, 'first_output_tokens': 700},
        'adoption_gates': ['net_primary_strict_gain_ge_1', 'zero_strict_regression',
            'zero_required_fact_expression_or_support_loss',
            'zero_new_source_time_number_branch_or_safety_error',
            'all_generation_equality_and_drift_checks_pass',
            'candidate_total_tokens_le_1.25x_baseline', 'candidate_p95_le_1.5x_baseline'],
        'failure_or_pending_gate': 'not_adopted; keep_negative_result; no_second_candidate',
        'no_default_switch_in_this_round': True, 'formal_Core': False,
        'strict_rubric': base['registration']['strict_rubric'],
        'review': base['registration']['review'],
        'metrics': base['registration']['metrics'],
        'latency_limit': 'sequential_phases_local_cache_and_background_load_not_isolated; observed_cost_not_benchmark_speedup',
        'test_log_sha256': sha(PARENT / 'targeted-tests-v2.log'),
    }
    assert model_digest(MODELS['base']) == registration['model']['digest']
    PRIVATE.mkdir(parents=True)
    atomic_json(REGISTRATION, registration)
    atomic_json(PRIVATE / 'registration.json', registration)
    raw = {'registration': registration, 'status': 'running', 'rows': []}
    public = {'version': registration['version'], 'split': 'dev', 'status': 'running',
        'registration_sha256': sha(REGISTRATION), 'review': 'pending', 'strict_success': None,
        'default': 'legacy', 'adopted': False, 'rows': []}

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
            for item in packet['rows']:
                task = item['task']
                user = db.get(User, task['user'])
                sources = item['current'] + item['additions']
                rows = [readable_row(db, user, r['chunk_id'], expect=r, rank=i)
                        for i, r in enumerate(sources, 1)]
                assert all(all(r[k] == original[k] for k in ('title', 'metadata', 'locator'))
                           for r, original in zip(rows, sources, strict=True))
                supplied = [dict(r, id=f'E{i}') for i, r in enumerate(rows, 1)]
                models = CapturedModels()
                budget = ExecutionBudget(cfg, saved={'limits': {
                    'generation': 3, 'policy': 0, 'judge': 0, 'embedding': 0, 'search': 0}})
                budget.state['deadline'] = min(budget.state['deadline'], budget.clock() + 180)
                generated, rendered, usage, render_trace, error = None, None, None, None, None
                payload = {'status': 'execution_failed', 'claims': [], 'verdict': None}
                started = time.monotonic()
                try:
                    with use_budget(budget):
                        generated, usage = contract_generate(models, task['goal'], supplied,
                            ordinary_policy='legacy', options={'max_output_tokens': 700,
                                'check_conflict': user.tenant_id not in cfg.conflict_check_disabled_tenants})
                        rendered, render_trace = bind_source_text(generated, supplied)
                        if render_trace['status'] != 'literal_source_claims':
                            claims, status = [], 'verification_failed'
                        else:
                            claims, status = validate_claims(rendered, supplied)
                        if status == 'answered' and usage.get('answer_status') == 'conflict':
                            status = 'conflict'
                        verdict, extra = answer_verdict(models, task['goal'], claims, status, evidence=supplied)
                        usage['verdict'] = extra
                        payload = {'status': status, 'claims': claims, 'verdict': verdict}
                        for i, r in enumerate(rows, 1):
                            rebound = readable_row(db, user, r['chunk_id'], expect=r, rank=i)
                            assert all(rebound[k] == r[k] for k in ('title', 'metadata', 'locator'))
                except Exception as exc:
                    error = type(exc).__name__
                parent = baseline[task['id']]
                equality = {'generation_requests': generation_requests(models.events) == generation_requests(parent['events']),
                            'initial_drafts': initial_drafts(models.drafts) == initial_drafts(parent['drafts'])}
                record = {'task_id': task['id'], 'arm': 'original_source_spans', 'question': task['goal'],
                    'sources': supplied, 'drafts': models.drafts,
                    'rendered_draft': rendered.model_dump(mode='json') if rendered else None,
                    'render_trace': render_trace,
                    'rendered_issues': claim_validation_issues(rendered, supplied) if rendered else None,
                    'payload': payload, 'events': models.events, 'usage': usage,
                    'budget': budget.snapshot(), 'error_kind': error, 'equality_to_baseline': equality,
                    'latency_ms': round((time.monotonic() - started) * 1000, 1)}
                raw['rows'].append(record)
                projection = {k: record[k] for k in ('task_id', 'arm', 'error_kind', 'latency_ms', 'equality_to_baseline')}
                projection.update(status=payload['status'], cost=costs(models.events), render_status=(render_trace or {}).get('status'),
                    rendered_issues=record['rendered_issues'],
                    answer_sha256=hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest())
                public['rows'].append(projection)
                save()
                print(json.dumps(projection), flush=True)
        drift = {'code': fingerprints() != code, 'input': sha(INPUT) != registration['input_sha256'],
            'baseline': sha(BASELINE) != registration['baseline_raw_sha256'],
            'baseline_review': sha(BASELINE_REVIEW) != registration['baseline_review_sha256'],
            'settings': hashlib.sha256(cfg.model_dump_json().encode()).hexdigest() != registration['settings_sha256'],
            'model': model_digest(MODELS['base']) != registration['model']['digest']}
        for result in (raw, public):
            result.update(drift=drift, status='invalid_drift' if any(drift.values()) else 'complete')
    except BaseException as exc:
        for result in (raw, public):
            result.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error_kind=type(exc).__name__)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
