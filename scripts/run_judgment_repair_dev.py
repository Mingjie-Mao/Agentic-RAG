"""One registered Dev trial: four local repairs and paired source-bound judgments.

Cached drafts are common inputs, not fresh control generation. No retrieval,
held-out data, policy call, production setting or annotation is changed.
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
from app.clients import GeneratedAnswer
from app.config import settings
from app.db import SessionLocal
from app.execution_budget import ExecutionBudget, use_budget
from app.qa import answer_verdict, validate_claims
from app.task_analysis import judgment_intent
from app.verdict import EXPLICIT_NEGATION_PREFIX
from scripts.benchmark_runtime import atomic_json
from scripts.run_generation_diagnostic_dev import (
    DEV47, MODELS, ROOT, TaggedModels, load_tasks, model_digest, readable_row,
)

PRIVATE = ROOT / '.runtime/roadmap-0-2-20261013/v1/judgment-trial'
PUBLIC = ROOT / 'artifacts/dev-judgment-repair-20261013-v1.json'
REGISTRATION = ROOT / 'benchmarks/enterprise_rag/v1/dev/generation-diagnostic-20261013/judgment-repair-registration-v1.json'
RAW_INPUT = ROOT / '.runtime/generation-diagnostic-20261013/v1/raw.json'
REPAIR_INPUT = ROOT / '.runtime/roadmap-0-2-20261013/v1/claim-rejection/raw.json'
PENDING = {'U12', 'U22', 'SF-3e8219f58317', 'SF-56d3160a9c64', 'SF-59679085cc4a',
           'SF-3ea856eeec7b', 'SF-3fb3599dcc99'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for directory in ('app', 'agent', 'scripts')
            for p in sorted((ROOT/directory).glob('*.py'))}


def cost(events):
    def total(key):
        return sum(e[key] for e in events) if all(e.get(key) is not None for e in events) else None
    return {'physical_calls': len(events), 'prompt_tokens': total('prompt_tokens'),
            'completion_tokens': total('completion_tokens')}


class SeededRepairModels(TaggedModels):
    def __init__(self, draft):
        super().__init__()
        self.seed = draft

    def generate(self, *args, **kwargs):
        if self.seed is not None:
            draft, self.seed = self.seed, None
            return draft, {'answer_status': 'answered'}
        return super().generate(*args, **kwargs)


def bind_rows(db, user, rows):
    return [readable_row(db, user, r['chunk_id'], expect=r, rank=i)
            for i, r in enumerate(rows, 1)]


def execute():
    from app.models import User

    if any(p.exists() for p in (PRIVATE, PUBLIC, REGISTRATION)):
        raise SystemExit('Existing trial is immutable; select a new run name')
    PRIVATE.mkdir(parents=True)
    cfg = settings()
    original_protocol = cfg.verdict_protocol
    raw_input = json.loads(RAW_INPUT.read_text())
    repair_input = json.loads(REPAIR_INPUT.read_text())
    if repair_input['status'] != 'complete' or any(repair_input['drift'].values()):
        raise ValueError('Valid four-case capture required')
    tasks = {t['id']: t for t in load_tasks(DEV47)}
    # Select from current-context outcomes, never from answer labels or new results.
    selected = [t for t in tasks.values() if judgment_intent(t['goal'])
                and raw_input['cells'][t['id']+':A']['status'] == 'answered']
    assert len(selected) == 7
    registration = {'version': 'judgment-repair-dev-v1', 'split': 'dev',
        'task_ids': [t['id'] for t in selected],
        'primary_task_ids': [t['id'] for t in selected if t['id'] not in PENDING],
        'pending_task_ids': [t['id'] for t in selected if t['id'] in PENDING],
        'repair_task_ids': [r['task_id'] for r in repair_input['rows']],
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (RAW_INPUT, REPAIR_INPUT, DEV47)},
        'code_sha256': code_hashes(), 'model': {**MODELS['base'], 'digest': model_digest(MODELS['base'])},
        'settings_sha256': hashlib.sha256(cfg.model_dump_json().encode()).hexdigest(),
        'arms': ['legacy', 'structured_bound_quotes'],
        'comparison': 'same_cached_valid_claims; fresh verdict calls both arms; no answer regeneration',
        'repair_comparison': 'same_four_cached_rejected_drafts; existing_LangGraph_one_repair; no_new_retrieval',
        'order': 'arm_order_rotated_by_task_and_question_form',
        'counterfactual': 'explicit_outer_NOT_independent_calls; structured_normalizes_NOT_and_combines_in_code',
        'counterfactual_limit': 'engineered_composition_consistency_is_not_independent_semantic_accuracy',
        'limits': {'judge_physical_calls_per_position': 1, 'repair_physical_generation_calls': 2,
                   'seconds_per_position': 180, 'embedding_calls': 0, 'policy_calls': 0},
        'metrics': ['required_fact_retention', 'direction', 'body_conclusion_consistency',
                    'paired_win_loss', 'counterfactual_flip_and_unknown', 'yes_rate_and_gold_yes_rate',
                    'physical_calls_tokens_latency', 'mechanical_and_safety_regression'],
        'adoption_rule': ['primary_strict_gain_gt_zero', 'zero_correct_to_wrong',
                          'zero_required_fact_loss', 'zero_safety_regression', 'complete_GPT_review'],
        'review': 'GPT_family_same_session_not_independent; randomized_hidden_arm_packet; pending_annotations_unchanged',
        'production_default': 'unchanged_legacy_no_semantic_gate', 'headline_eligible': False}
    atomic_json(REGISTRATION, registration)
    atomic_json(PRIVATE/'registration.json', registration)
    private = {'registration': registration, 'judgments': [], 'repairs': [], 'status': 'running'}
    public = {'version': registration['version'], 'split': 'dev', 'registration': str(REGISTRATION.relative_to(ROOT)),
              'registration_sha256': sha(REGISTRATION), 'judgments': [], 'repairs': [], 'status': 'running',
              'strict_task_success': None, 'review': 'pending', 'adopted': False}

    def save():
        atomic_json(PRIVATE/'raw.json', private)
        public['private_raw_sha256'] = sha(PRIVATE/'raw.json')
        atomic_json(PUBLIC, public)

    save()
    try:
        with SessionLocal() as db:
            db.execute(text('SET TRANSACTION READ ONLY'))
            db.execute(text('SET LOCAL statement_timeout = 30000'))
            for number, task in enumerate(selected):
                user = db.get(User, task['user'])
                payload = raw_input['cells'][task['id']+':A']['answer_payload']
                sources = bind_rows(db, user, payload['citations'])
                supplied = [dict(r, id=r['chunk_id']) for r in sources]
                generated = GeneratedAnswer.model_validate({'answerable': True, 'claims': payload['claims']})
                if validate_claims(generated, supplied)[1] != 'answered':
                    raise ValueError('Cached judgment claims no longer validate')
                for form, question in [('original', task['goal']),
                                       ('negated', EXPLICIT_NEGATION_PREFIX+task['goal'])]:
                    arms = ['legacy', 'structured_bound_quotes']
                    if (number + (form == 'negated')) % 2:
                        arms.reverse()
                    for arm in arms:
                        cfg.verdict_protocol = 'legacy' if arm == 'legacy' else 'structured'
                        models = TaggedModels()
                        budget = ExecutionBudget(cfg, saved={'limits': {
                            'generation': 1, 'judge': 0, 'policy': 0, 'embedding': 0, 'search': 0}})
                        budget.state['deadline'] = min(budget.state['deadline'], budget.clock()+180)
                        started, verdict, usage, error = time.monotonic(), None, None, None
                        try:
                            with use_budget(budget):
                                bind_rows(db, user, sources)
                                verdict, usage = answer_verdict(models, question, payload['claims'],
                                    'answered', evidence=supplied)
                                bind_rows(db, user, sources)
                        except Exception as exc:
                            error = type(exc).__name__
                        row = {'task_id': task['id'], 'form': form, 'arm': arm, 'question': question,
                               'claims': payload['claims'], 'sources': sources, 'verdict': verdict,
                               'usage': usage, 'events': models.events, 'error_kind': error,
                               'latency_ms': round((time.monotonic()-started)*1000, 1),
                               'budget': budget.snapshot()}
                        private['judgments'].append(row)
                        projection = {k: row[k] for k in ('task_id', 'form', 'arm', 'error_kind', 'latency_ms')}
                        projection.update(value=(verdict or {}).get('value'), cost=cost(models.events),
                                          validation_issues=(verdict or {}).get('validation_issues', []))
                        public['judgments'].append(projection)
                        save()
                        print(json.dumps(projection), flush=True)
            cfg.verdict_protocol = 'structured'
            dynamic_tasks = {t['id']: t for t in load_tasks(
                ROOT/'benchmarks/enterprise_rag/v1/dynamic/dev/tasks.json')}
            for capture in repair_input['rows']:
                task = tasks.get(capture['task_id']) or dynamic_tasks[capture['task_id']]
                user = db.get(User, task['user'])
                sources = bind_rows(db, user, capture['evidence'])
                supplied = [dict(r, id=f'E{i}') for i, r in enumerate(sources, 1)]
                models = SeededRepairModels(GeneratedAnswer.model_validate(capture['captures'][0]['draft']))
                budget = ExecutionBudget(cfg, saved={'limits': {
                    'generation': 2, 'judge': 0, 'policy': 0, 'embedding': 0, 'search': 0}})
                budget.state['deadline'] = min(budget.state['deadline'], budget.clock()+180)
                started, error, usage = time.monotonic(), None, None
                payload = {'status': 'execution_failed', 'claims': [], 'verdict': None}
                try:
                    with use_budget(budget):
                        generated, usage = recover_answer(models, task['goal'], supplied,
                            options={'check_conflict': False})
                        claims, status = validate_claims(generated, supplied)
                        verdict, verdict_usage = answer_verdict(models, task['goal'], claims, status,
                                                               evidence=supplied)
                        usage['verdict'] = verdict_usage
                        payload = {'status': status, 'claims': claims, 'verdict': verdict}
                        bind_rows(db, user, sources)
                except Exception as exc:
                    error = type(exc).__name__
                row = {'task_id': task['id'], 'question': task['goal'], 'sources': sources,
                       'payload': payload, 'usage': usage, 'events': models.events,
                       'error_kind': error, 'latency_ms': round((time.monotonic()-started)*1000, 1),
                       'budget': budget.snapshot()}
                private['repairs'].append(row)
                projection = {k: row[k] for k in ('task_id', 'error_kind', 'latency_ms')}
                projection.update(status=payload['status'], value=(payload.get('verdict') or {}).get('value'),
                                  cost=cost(models.events), recovery=(usage or {}).get('answer_recovery'))
                # Recovery carries codes and timings only, never claim text.
                public['repairs'].append(projection)
                save()
                print(json.dumps(projection), flush=True)
        cfg.verdict_protocol = original_protocol
        drift = {'code': code_hashes() != registration['code_sha256'],
                 'inputs': any(sha(ROOT/p) != h for p, h in registration['input_sha256'].items()),
                 'settings': hashlib.sha256(cfg.model_dump_json().encode()).hexdigest() != registration['settings_sha256'],
                 'model': model_digest(MODELS['base']) != registration['model']['digest']}
        for result in (private, public):
            result.update(drift=drift, status='invalid_drift' if any(drift.values()) else 'complete')
    except BaseException as exc:
        for result in (private, public):
            result.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error_kind=type(exc).__name__)
        raise
    finally:
        cfg.verdict_protocol = original_protocol
        save()


if __name__ == '__main__':
    execute()
