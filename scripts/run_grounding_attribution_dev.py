"""One Dev-only grounding trial, fixed source contexts and unchanged Legacy verdict.

Both arms use the existing bounded LangGraph recovery. This is a generation /
citation component comparison, not a retrieval change or a formal benchmark run.
All questions, answers, source text and rejected drafts remain local.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.answer_quality import recover_answer
from app.config import settings
from app.db import SessionLocal
from app.execution_budget import ExecutionBudget, use_budget
from app.qa import answer_verdict, claim_validation_issues, validate_claims
from scripts.benchmark_runtime import atomic_json
from scripts.run_generation_diagnostic_dev import MODELS, ROOT, TaggedModels, model_digest, readable_row

PRIVATE = ROOT / '.runtime/grounding-attribution-20261014/v1/trial'
PUBLIC = ROOT / 'artifacts/dev-grounding-trial-20261014-v1.json'
REGISTRATION = ROOT / 'benchmarks/enterprise_rag/v1/dev/grounding-attribution-20261014/registration-v1.json'
INPUT = ROOT / '.runtime/grounding-attribution-20261014/v1/inputs.json'
PENDING = {'U22', 'SF-3e8219f58317', 'SF-59679085cc4a'}
ARMS = ('existing_recovery', 'grounding_feedback')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints():
    return {str(p.relative_to(ROOT)): sha(p) for directory in ('app', 'agent', 'scripts')
            for p in sorted((ROOT / directory).glob('*.py'))}


def costs(events):
    def total(key):
        return sum(e[key] for e in events) if all(e.get(key) is not None for e in events) else None
    return {'physical_calls': len(events), 'prompt_tokens': total('prompt_tokens'),
            'completion_tokens': total('completion_tokens')}


class CapturedModels(TaggedModels):
    def __init__(self):
        super().__init__()
        self.drafts = []

    def generate(self, *args, **kwargs):
        generated, usage = super().generate(*args, **kwargs)
        self.drafts.append({'draft': generated.model_dump(mode='json'),
                            'issues': claim_validation_issues(generated, args[1]),
                            'repair_context': kwargs.get('repair_context'), 'usage': usage})
        return generated, usage


def main():
    if any(p.exists() for p in (PRIVATE, PUBLIC, REGISTRATION)):
        raise SystemExit('Immutable trial exists; no overwrite or tuning rerun')
    cfg = settings()
    if cfg.verdict_protocol != 'legacy' or any((cfg.answer_quality_enabled,
        cfg.answer_contract_enabled, cfg.answer_extractive_enabled,
        cfg.answer_semantic_audit_enabled, cfg.source_facts_enabled)):
        raise ValueError('Expected unchanged Legacy/default configuration')
    packet = json.loads(INPUT.read_text())
    assert packet['split'] == 'dev' and len(packet['rows']) == 12
    code = fingerprints()
    registration = {
        'version': 'grounding-attribution-dev-v1', 'split': 'dev',
        'task_ids': [r['task_id'] for r in packet['rows']],
        'primary_task_ids': [r['task_id'] for r in packet['rows'] if r['task_id'] not in PENDING],
        'pending_task_ids': sorted(PENDING), 'positive_control': 'SF-66b7ed601d85',
        'input_sha256': sha(INPUT), 'code_sha256': code,
        'settings_sha256': hashlib.sha256(cfg.model_dump_json().encode()).hexdigest(),
        'model': {**MODELS['base'], 'digest': model_digest(MODELS['base'])},
        'arms': list(ARMS), 'order': 'rotated_by_task_index',
        'generation': 'fresh_both_arms_same_current_context_existing_LangGraph_one_repair',
        'candidate': 'grounding_checklist_and_actual_draft_bindings_only',
        'verdict': 'unchanged_legacy_both_arms',
        'retrieval': 'none; same_stored_current_context; reauthorized_before_and_after',
        'gold_input_to_model': False, 'formal_Core': False,
        'limits': {'generation_calls': 3, 'seconds': 180, 'embedding': 0, 'policy': 0, 'search': 0},
        'metrics': ['strict_success', 'required_fact_support', 'own_quote_coverage', 'source_group_coverage',
                    'paired_gain_loss', 'physical_calls', 'tokens', 'latency_p50_p95', 'execution_failures'],
        'strict_rubric': 'All required facts expressed with supporting own quotes; valid source/condition/time attribution; requested verdict agrees with body; material unsupported assertions fail. Insufficiency is not success on a settled gold task.',
        'review': 'source_inspection_GPT_family_same_session_not_independent; deterministic_literal_checks_separate',
        'adoption_rule': 'retain_legacy_default_even_if_local_net_gain; no_more_tuning_in_this_round',
        'pending_annotations': 'unchanged; strict=null; diagnostic_only',
        'population_limit': '12_selected_cases_9_annotation_valid; component_study_not_full_Dev47',
    }
    PRIVATE.mkdir(parents=True)
    REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(REGISTRATION, registration)
    atomic_json(PRIVATE / 'registration.json', registration)
    private = {'registration': registration, 'status': 'running', 'rows': []}
    public = {'version': registration['version'], 'split': 'dev', 'status': 'running',
              'registration_sha256': sha(REGISTRATION), 'rows': [], 'strict_success': None,
              'review': 'pending', 'default': 'legacy', 'adopted': False}

    def save():
        atomic_json(PRIVATE / 'raw.json', private)
        public['private_raw_sha256'] = sha(PRIVATE / 'raw.json')
        atomic_json(PUBLIC, public)

    save()
    try:
        from app.models import User
        with SessionLocal() as db:
            db.execute(text('SET TRANSACTION READ ONLY'))
            db.execute(text('SET LOCAL statement_timeout = 30000'))
            for number, item in enumerate(packet['rows']):
                task = item['task']
                user = db.get(User, task['user'])
                arms = ARMS if number % 2 == 0 else tuple(reversed(ARMS))
                for arm in arms:
                    rows = [readable_row(db, user, r['chunk_id'], expect=r, rank=i)
                            for i, r in enumerate(item['current'], 1)]
                    supplied = [dict(r, id=f'E{i}') for i, r in enumerate(rows, 1)]
                    models = CapturedModels()
                    budget = ExecutionBudget(cfg, saved={'limits': {
                        'generation': 3, 'judge': 0, 'policy': 0, 'embedding': 0, 'search': 0}})
                    budget.state['deadline'] = min(budget.state['deadline'], budget.clock() + 180)
                    payload = {'status': 'execution_failed', 'claims': [], 'verdict': None}
                    generated, usage, error = None, None, None
                    started = time.monotonic()
                    try:
                        with use_budget(budget):
                            generated, usage = recover_answer(models, task['goal'], supplied,
                                options={'max_output_tokens': 700,
                                         'check_conflict': user.tenant_id not in cfg.conflict_check_disabled_tenants},
                                grounding_feedback=arm == 'grounding_feedback')
                            claims, status = validate_claims(generated, supplied)
                            if status == 'answered' and usage.get('answer_status') == 'conflict':
                                status = 'conflict'
                            verdict, verdict_usage = answer_verdict(models, task['goal'], claims, status,
                                                                   evidence=supplied)
                            usage['verdict'] = verdict_usage
                            payload = {'status': status, 'claims': claims, 'verdict': verdict}
                            for i, r in enumerate(rows, 1):
                                readable_row(db, user, r['chunk_id'], expect=r, rank=i)
                    except Exception as exc:
                        error = type(exc).__name__
                    row = {'task_id': task['id'], 'arm': arm, 'question': task['goal'],
                           'sources': supplied, 'drafts': models.drafts, 'payload': payload,
                           'final_draft': generated.model_dump(mode='json') if generated else None,
                           'usage': usage, 'events': models.events, 'budget': budget.snapshot(),
                           'error_kind': error, 'latency_ms': round((time.monotonic() - started) * 1000, 1)}
                    private['rows'].append(row)
                    projection = {k: row[k] for k in ('task_id', 'arm', 'error_kind', 'latency_ms')}
                    projection.update(status=payload['status'], value=(payload['verdict'] or {}).get('value'),
                                      cost=costs(models.events), drafts_sha256=hashlib.sha256(
                                          json.dumps(models.drafts, ensure_ascii=False).encode()).hexdigest(),
                                      issues=claim_validation_issues(generated, supplied) if generated else None)
                    public['rows'].append(projection)
                    save()
                    print(json.dumps(projection), flush=True)
        drift = {'code': fingerprints() != code, 'inputs': sha(INPUT) != registration['input_sha256'],
                 'settings': hashlib.sha256(cfg.model_dump_json().encode()).hexdigest() != registration['settings_sha256'],
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
        save()


if __name__ == '__main__':
    main()
