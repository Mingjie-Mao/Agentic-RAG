"""Frozen Dev47 answer generation over one authorized active context.

No retrieval, policy inference or historical-version fetch is performed. History
contracts can therefore remain incomplete. All answer/source bytes stay private.
The default three protocols remain v1; --comparison fallback opts into two fresh arms.
"""
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
from scripts import run_evidence_selection_dev as replay
from scripts import run_evidence_supplement_dev as supplement
from scripts.benchmark_package_scoring import answer_review_packet, provisional_score
from scripts.benchmark_runtime import atomic_json
from scripts.research_review import digest
from scripts.retrieval_quality_index import active_scope, index_ledger
from scripts.run_external_answers_dev import generate_answer
from scripts.run_external_repair_dev import private_root
from scripts.run_guarded_selection_dev import PROFILES as SELECTION_PROFILES
from scripts.run_retrieval_quality_dev import (
    ROOT, SELECTION, _publish, code_fingerprints, reserve_output, sha, validate_suite,
)

PROTOCOLS = ('direct', 'shared_contract', 'literal_slots')
FALLBACK_PROTOCOLS = ('shared_contract', 'literal_fallback')
LIMITS = {'generation_calls': 3, 'first_output_tokens': 700, 'conflict_tokens': 200,
          'verdict_max_tokens': 500, 'seconds': 180, 'policy_calls': 0,
          'search_calls': 0, 'embedding_calls': 0, 'judge_calls': 0}


def protocols_for(args):
    comparison = getattr(args, 'comparison', 'frozen')
    if type(comparison) is not str or comparison not in ('frozen', 'fallback'):
        raise ValueError('Invalid generation comparison')
    return FALLBACK_PROTOCOLS if comparison == 'fallback' else PROTOCOLS


def input_paths(args):
    paths = [Path(args.suite), SELECTION, Path(args.selection_result), Path(args.parent),
             Path(args.review_registration)]
    if args.source_audit:
        paths.append(Path(args.source_audit))
    return paths


def context_paths(args, tasks):
    root = private_root(args.context_root)
    paths = []
    for task in tasks:
        path = (root / (task['id'] + '.json')).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError('Missing or escaping private task context')
        paths.append(path)
    return paths


def selected_profile(selection, parent, tasks):
    """Validate prior selection, without requiring equality across stage code."""
    ids = [t['id'] for t in tasks]
    try:
        profile = selection['chosen_profile']
        identity = selection['identity']
        declaration = identity['registration']
        caps = declaration['caps']
        if (selection['version'] != 'guarded-selection-dev-v2'
                or declaration['version'] != 'guarded-selection-input-v2'
                or declaration['profiles'] != list(SELECTION_PROFILES)
                or type(caps['limit']) is not int or caps['limit'] != 8
                or type(caps['token_budget']) is not int or caps['token_budget'] != 5000
                or selection['valid'] is not True or selection['status'] != 'complete'
                or selection['split'] != 'dev' or selection['subset'] is not False
                or selection['error'] is not None or selection['selected_task_ids'] != ids
                or not selection['drift'] or any(v is not False for v in selection['drift'].values())
                or profile not in identity['registration']['profiles']):
            raise ValueError('Complete valid frozen selection required')
        if (identity['parent_sha256'] != digest(parent)
                or identity['corpus_sha256'] != parent['identity']['corpus_sha256']
                or identity['index_sha256'] != parent['identity']['source_index_sha256']
                or digest(parent['identity']['corpus']) != identity['corpus_sha256']
                or digest(parent['identity']['source_index']) != identity['index_sha256']):
            raise ValueError('Selection parent/corpus/index differs')
        if (set(selection['rows']) != set(SELECTION_PROFILES)
                or set(selection['execution_errors']) != set(SELECTION_PROFILES)):
            raise ValueError('Complete declared Stage1 profiles required')
        for name in SELECTION_PROFILES:
            rows = selection['rows'][name]
            if [r['task_id'] for r in rows] != ids or any(r['error'] is not None for r in rows):
                raise ValueError('Complete declared Stage1 profile rows required')
        for error in selection['execution_errors'].values():
            if (error.get('tasks') if isinstance(error, dict) else error) != 0:
                raise ValueError('Selection contains execution errors')
        rows = selection['rows'][profile]
        for row in rows:
            context, tokens = row['telemetry']['context'], row['telemetry']['context_tokens']
            if (not isinstance(context, list) or len(context) > 8
                    or len({r['chunk_id'] for r in context}) != len(context)
                    or type(tokens) is not int or not 0 <= tokens <= 5000):
                raise ValueError('Selected context caps/token telemetry invalid')
            for evidence in context:
                replay.validate_telemetry_row(evidence)
        return profile
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Malformed selection input') from exc


def registration_for(args):
    """Freeze exact task/input/private bytes and actual generator before calls."""
    protocols = protocols_for(args)
    paths = input_paths(args)
    inputs = [p.read_bytes() for p in paths]
    suite, selection, chosen, parent, review = map(json.loads, inputs[:5])
    tasks = validate_suite(suite, selection, ROOT)
    profile = selected_profile(chosen, parent, tasks)
    pending = review.get('pending_task_ids', review.get('pending_annotation_task_ids'))
    ids = [t['id'] for t in tasks]
    if (not isinstance(pending, list) or len(pending) != 7 or len(set(pending)) != 7
            or not set(pending) <= set(ids)):
        raise ValueError('Prior blind registration must supply seven pending task IDs')
    cfg = settings()
    if cfg.answer_quality_enabled or cfg.source_facts_enabled:
        raise ValueError('Unexpected extra generation/judge pipeline')
    generator_cfg = cfg.model_copy(update={'agent_policy_model': cfg.chat_model, 'agent_policy_url': ''})
    actual = supplement.active_model_identity(generator_cfg)
    registration = {'version': 'slot-generation-input-v1', 'split': 'dev', 'subset': False,
        'selected_task_ids': ids, 'input_bytes_sha256': list(map(sha, inputs)),
        'parent_semantic_sha256': digest(parent), 'selection_semantic_sha256': digest(chosen),
        'private_task_bytes_sha256': {t['id']: sha(p.read_bytes())
            for t, p in zip(tasks, context_paths(args, tasks), strict=True)},
        'code': code_fingerprints(), 'settings_sha256': digest(cfg.model_dump(mode='json')),
        'generator_identity': actual['policy'], 'libraries': actual['libraries'],
        'protocols': list(PROTOCOLS), 'shared_context': {'profile': profile,
            'caps': {'limit': 8, 'token_budget': 5000},
            'binding': 'same_ordered_full_source_identities_all_three_arms',
            'aliases': 'server_E1_E2_back_to_source_chunk_ids',
            'historical_fetch': False}, 'limits': deepcopy(LIMITS),
        'order': 'task_index_rotated_protocol_order',
        'adoption_rule': {'candidate': 'literal_slots', 'baseline': 'shared_contract',
            'required': ['full_valid_reviewed_original_primary', 'equal_denominator',
                'no_previously_correct_task_loss', 'no_required_fact_support_loss'],
            'improvement': ['strict_task_success_gain',
                'equal_correctness_and_fewer_actual_generation_calls'],
            'direct_to_shared_contract': 'evaluation_alignment_only',
            'headline': 'pending_frozen_reviewed_core'},
        'review': {'protocol': 'named_method_blind_fact_citation_scope_review_v1',
            'pending_task_ids': pending, 'prior_blind_registration_sha256': sha(inputs[4]),
            'primary': 'original_rubric_with_prior_pending_exclusions',
            'sensitivity': 'separate_source_aware_registered_guidance',
            'identical_visible_answers': 'review_once_copy_identical_grade_via_private_mapping',
            'retrieved_facts': 'one_canonical_blind_evidence_assessment_per_task_shared_all_arms',
            'hidden': ['protocol_names', 'usage', 'task_contract', 'tool_trace'],
            'required': ['named_reviewer', 'model_identity', 'reason', 'independence',
                         'all_material_question_parts', 'citation_entailment', 'condition', 'version']},
        'strict_task_success': None, 'semantic_recall': None}
    if protocols == FALLBACK_PROTOCOLS:
        registration.update(version='slot-generation-input-v2', comparison='fallback',
                            protocols=list(protocols))
        registration['shared_context']['binding'] = 'same_ordered_full_source_identities_all_declared_arms'
        registration['adoption_rule']['candidate'] = 'literal_fallback'
        registration['adoption_rule'].pop('direct_to_shared_contract')
    return registration


class GenerationModels(supplement.MeteredModels):
    """Use existing metering; distinguish generation/conflict/verdict transports."""
    operation = 'generation'

    def decide_verdict(self, *args, **kwargs):
        self.operation = 'verdict'
        try:
            return super().decide_verdict(*args, **kwargs)
        finally:
            self.operation = 'generation'

    def _post(self, path, body, base=None):
        count = len(self.events)
        try:
            return super()._post(path, body, base=base)
        finally:
            for event in self.events[count:]:
                if path == '/api/chat':
                    event['role'] = 'generation'
                    event['operation'] = ('conflict' if self.operation == 'generation'
                        and body.get('options', {}).get('num_predict') == 200 else self.operation)


def cost_projection(models, usage):
    """Budget attempts and physical transport results remain separate observations."""
    events = models.events
    summary = supplement.event_summary(events)
    physical_records = [{key: deepcopy(e.get(key)) for key in (
        'role', 'operation', 'status', 'request_sha256', 'prompt_tokens',
        'completion_tokens', 'server_ms', 'wall_ms')} for e in events]
    budget = deepcopy((usage or {}).get('business_budget'))
    if budget is None:
        return {'physical_generation': summary, 'business_budget': None,
                'physical_call_records': physical_records, 'transport_budget_matched': None}
    calls = budget.setdefault('calls', {})
    generation = calls.setdefault('generation', {'attempted': 0, 'succeeded': 0, 'failed': 0})
    records = budget.get('model_call_records', [])
    matched = (generation['attempted'] == len(events) == len(records)
               and all(e['role'] == 'generation' for e in events)
               and all(r['role'] == 'generation' for r in records)
               and all(calls.get(role, {}).get('attempted', 0) == 0
                       for role in ('policy', 'search', 'embedding', 'judge')))
    # ExecutionBudget internally uses zero for absent counters. Public provider
    # telemetry preserves unknown, even when a transport completed successfully.
    for field in ('prompt_tokens', 'completion_tokens'):
        generation[field] = summary[field]
    for i, record in enumerate(records):
        event = events[i] if i < len(events) else {}
        record.update(operation=event.get('operation'),
            prompt_tokens=event.get('prompt_tokens'), completion_tokens=event.get('completion_tokens'),
            server_ms=event.get('server_ms'))
    return {'physical_generation': summary, 'business_budget': budget,
            'physical_call_records': physical_records, 'transport_budget_matched': matched}


def contract_projection(usage):
    trace = (usage or {}).get('generation', {}).get('task_contract')
    if trace is None:
        return None
    # Only code-like fields; never contract/condition queries, bound values,
    # source quotes or generated prose. The full trace lives in raw.json.
    return {key: deepcopy(trace[key]) for key in ('status', 'slot_states', 'slot_reasons') if key in trace}


def missing(task_id, run_id):
    return {'task_id': task_id, 'run_id': run_id, 'status': 'not_observed', 'not_observed': True,
        'answer_payload': {'status': 'not_observed', 'claims': [], 'citations': [], 'verdict': None},
        'retrieved_evidence': None, 'first_retrieval_documents': None, 'steps': None,
        'latency_ms': None, 'trace_complete': False, 'tool_trace': [], 'usage': None,
        'generation_usage': None, 'execution_error_kind': 'NotObserved', 'scenario_events': {},
        'strict_task_success': None, 'answer_correct': None, 'task_success': None,
        'physical_generation': None, 'physical_call_records': None,
        'business_budget': None, 'transport_budget_matched': None}


def public_row(row):
    usage = row.get('generation_usage') or {}
    result_pipeline = {key: deepcopy(value) for section in ('generation', 'verdict')
        for key, value in usage.get(section, {}).items() if key in (
            'answer_status', 'verdict_status', 'verdict_failure_stage', 'verdict_error_kind')}
    return {'task_id': row['task_id'], 'status': row['status'],
        'answer_sha256': None if row.get('not_observed') else digest(row['answer_payload']),
        'error': row['execution_error_kind'], 'latency_ms': row['latency_ms'],
        'physical_generation': row['physical_generation'], 'business_budget': row['business_budget'],
        'physical_call_records': row.get('physical_call_records'), 'result_pipeline': result_pipeline,
        'transport_budget_matched': row['transport_budget_matched'],
        'operation_status': 'not_observed' if row.get('not_observed') else
            ('execution_failed' if row['execution_error_kind'] else 'completed'),
        'contract': contract_projection(row.get('generation_usage'))}


def execute(args, reservation):
    from app.clients import Search
    from app.db import SessionLocal

    run_id = reservation['run_id']
    protocols = PROTOCOLS
    raw = {'version': 'slot-generation-dev-v1', 'split': 'dev', 'subset': False,
        'run_id': run_id, 'valid': False, 'status': 'failed', 'error': None,
        'results': {p: [] for p in PROTOCOLS}, 'corpus_sha256': None}
    public = {'version': raw['version'], 'split': 'dev', 'subset': False,
        'run_id': run_id, 'valid': False, 'status': 'failed', 'error': None,
        'strict_task_success': None, 'semantic_recall': None, 'selected_task_ids': [],
        'rows': {p: [] for p in PROTOCOLS}, 'identity': None, 'drift': None,
        'context_limitation': 'fixed_active_context_no_historical_version_fetch',
        'retrieval_provenance': 'immutable_registered_context_replay',
        'trace_complete': False, 'steps': None, 'search_statistics': None}
    tasks, runtime, suite, registration = [], None, None, None
    paths, inputs, private_paths, private_bytes = [], [], [], []
    stage, runtime_complete = 'preflight', False
    try:
        paths = input_paths(args)
        # Establish trusted full denominator first, even if subsequent preflight fails.
        inputs = [p.read_bytes() for p in paths[:2]]
        suite, selection = map(json.loads, inputs)
        tasks = validate_suite(suite, selection, ROOT)
        public['selected_task_ids'] = [t['id'] for t in tasks]
        protocols = protocols_for(args)
        raw['results'] = {p: [] for p in protocols}
        public['rows'] = {p: [] for p in protocols}
        if protocols == FALLBACK_PROTOCOLS:
            for result in (raw, public):
                result.update(version='slot-generation-dev-v2', comparison='fallback')
        runtime = private_root(args.private_root) / run_id
        runtime.mkdir(parents=True, exist_ok=False)
        inputs.extend(p.read_bytes() for p in paths[2:])
        chosen, parent = map(json.loads, inputs[2:4])
        profile = selected_profile(chosen, parent, tasks)
        reg_bytes = Path(args.registration).read_bytes()
        registration = json.loads(reg_bytes)
        if registration != registration_for(args):
            raise ValueError('Frozen generation registration differs')
        private_paths = context_paths(args, tasks)
        private_bytes = [p.read_bytes() for p in private_paths]
        if (registration['input_bytes_sha256'] != list(map(sha, inputs))
                or registration['private_task_bytes_sha256'] != {
                    t['id']: sha(b) for t, b in zip(tasks, private_bytes, strict=True)}):
            raise ValueError('Generation input bytes changed before runtime')
        originals = [json.loads(b) for b in private_bytes]
        raw['corpus_sha256'] = chosen['identity']['corpus_sha256']
        raw['registration'] = deepcopy(registration)
        public['identity'] = deepcopy(registration)
        atomic_json(runtime / 'registration.json', registration)
        cfg = settings()
        source = Search(index=parent['identity']['source_index']['index'])
        with SessionLocal() as db:
            db.execute(text('SET TRANSACTION READ ONLY'))
            users, records, corpus = active_scope(db)
            index = index_ledger(source, records)
            if (digest(corpus) != chosen['identity']['corpus_sha256']
                    or digest(index) != chosen['identity']['index_sha256']):
                raise ValueError('Active corpus/index changed')
            stage = 'generation'
            try:
                for offset, (task, original) in enumerate(zip(tasks, originals, strict=True)):
                    if original['task_id'] != task['id'] or original['question'] != task['goal']:
                        raise ValueError('Private task/question differs')
                    context = original['contexts'][profile]
                    telemetry = chosen['rows'][profile][offset]['telemetry']
                    expected = telemetry['context']
                    def bindings(rows):
                        return [[r[k] for k in replay.IDENTITY_FIELDS] for r in rows]
                    if context is None or bindings(context) != bindings(expected):
                        raise ValueError('Selected public context identities/order differ')
                    rotation = offset % len(protocols)
                    order = protocols[rotation:] + protocols[:rotation]
                    for protocol in order:
                        evidence = replay.rebind_rows(db, users[task['user']], context, corpus)
                        if bindings(evidence) != bindings(expected):
                            raise ValueError('Reauthorized selected context differs')
                        # Existing validator recomputes tokens from full authorized
                        # title/text bytes and verifies the frozen prior measurement.
                        # Invalid fixed contexts abort before any model dispatch.
                        replay.validate_context(evidence, telemetry['context_tokens'])
                        models, usage = GenerationModels(), None
                        started, error, interruption = time.monotonic(), None, None
                        payload = {'status': 'execution_failed', 'claims': [], 'citations': [], 'verdict': None}
                        try:
                            budget_options = {'enforce_zero_embedding': True} if protocols == FALLBACK_PROTOCOLS else {}
                            payload, usage = generate_answer(task['goal'], evidence, models=models, cfg=cfg,
                                check_conflict=users[task['user']].tenant_id not in cfg.conflict_check_disabled_tenants,
                                protocol=protocol, **budget_options)
                            if replay.rebind_rows(db, users[task['user']], evidence, corpus) != evidence:
                                raise ValueError('Post-generation ACL/version/SHA changed')
                        except BaseException as exc:
                            error = type(exc).__name__
                            if not isinstance(exc, Exception):
                                interruption = exc
                            if getattr(models, 'answer_budget', None) is not None:
                                usage = (usage or {}) | {'business_budget': models.answer_budget.snapshot()}
                            payload = {'status': 'execution_failed', 'claims': [], 'citations': [], 'verdict': None}
                        finally:
                            cost = cost_projection(models, usage)
                            row = {'task_id': task['id'], 'run_id': run_id, 'status': payload['status'],
                                'answer_payload': payload, 'retrieved_evidence': evidence,
                                'first_retrieval_documents': None, 'steps': None,
                                'latency_ms': (time.monotonic()-started)*1000, 'trace_complete': False,
                                'tool_trace': [], 'scenario_events': {}, 'execution_error_kind': error,
                                'generation_usage': usage, 'usage': cost['physical_generation'], **cost,
                                'strict_task_success': None, 'answer_correct': None, 'task_success': None}
                            raw['results'][protocol].append(row)
                            atomic_json(runtime / 'raw.json', raw)
                        print(json.dumps({'task_id': task['id'], 'protocol': protocol,
                            'status': row['status']}), flush=True)
                        if interruption:
                            raise interruption
                runtime_complete = True
            finally:
                _, final_records, final_corpus = active_scope(db)
                public['drift'] = {'corpus': final_corpus != corpus,
                    'index': index_ledger(source, final_records) != index}
        # No benchmark labels reach any generation call. Gold-derived diagnostics
        # start only after every task/protocol dispatch and authorization check.
        if runtime_complete:
            stage = 'post_runtime_evaluation'
            for protocol in protocols:
                for row in raw['results'][protocol]:
                    task = next(t for t in tasks if t['id'] == row['task_id'])
                    scored = provisional_score(task, row['answer_payload'], [], 0, row['latency_ms'],
                        execution_error_kind=row['execution_error_kind'], scenario_events={})
                    row.update(scored)
                    row.update(steps=None, trace_complete=False, tool_trace=[],
                        usage=row['physical_generation'], early_stop=None, over_planned=None,
                        evidence_sufficient_step=None, extra_calls_after_sufficient_evidence=None,
                        recovered=None, search_statistics=None)
    except BaseException as exc:
        public.update(error={'stage': stage, 'type': type(exc).__name__},
                      status='interrupted' if not isinstance(exc, Exception) else 'failed')
    finally:
        if registration is not None:
            drift = public['drift'] or {'corpus': None, 'index': None}
            checks = {'code': lambda: code_fingerprints() != registration['code'],
                'settings': lambda: digest(settings().model_dump(mode='json')) != registration['settings_sha256'],
                'inputs': lambda: [p.read_bytes() for p in paths] != inputs,
                'private_tasks': lambda: [p.read_bytes() for p in private_paths] != private_bytes,
                'registration': lambda: Path(args.registration).read_bytes() != reg_bytes,
                'generator': lambda: registration_for(args) != registration}
            for key, check in checks.items():
                try:
                    drift[key] = check()
                except Exception:
                    drift[key] = True
            public['drift'] = drift
        for protocol in protocols:
            observed = {r['task_id']: r for r in raw['results'][protocol]}
            raw['results'][protocol] = [observed.get(t['id']) or missing(t['id'], run_id) for t in tasks]
            public['rows'][protocol] = [public_row(r) for r in raw['results'][protocol]]
        errors = any(r['execution_error_kind'] for rows in raw['results'].values() for r in rows)
        costs_match = all(r['transport_budget_matched'] is True for rows in raw['results'].values() for r in rows)
        drift_ok = public['drift'] is not None and all(v is False for v in public['drift'].values())
        public['valid'] = runtime_complete and len(tasks) == 47 and not errors and costs_match and drift_ok and public['error'] is None
        if public['valid']:
            public['status'] = 'complete'
        elif public['status'] != 'interrupted' and any(v is True for v in (public['drift'] or {}).values()):
            public['status'] = 'invalid_drift'
        raw.update(valid=public['valid'], status=public['status'], error=public['error'], drift=public['drift'])
        if runtime is not None:
            atomic_json(runtime / 'raw.json', raw)
            reviewed_suite = deepcopy(suite)
            reviewed_suite['arms'] = list(protocols)
            atomic_json(runtime / 'suite.json', reviewed_suite)
            if public['valid']:
                atomic_json(runtime / 'review-packet.json', answer_review_packet(raw, reviewed_suite))
            public['private_raw_sha256'] = digest(raw)
        _publish(args.output, public)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('suite', 'selection-result', 'parent', 'context-root', 'review-registration',
                'registration', 'private-root', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--source-audit')
    parser.add_argument('--comparison', choices=('frozen', 'fallback'), default='frozen',
                        help='Opt into the fresh shared_contract/literal_fallback comparison')
    args = parser.parse_args()
    reservation = reserve_output(args.output)
    result = execute(args, reservation)
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
