"""Registered full Dev47 selection replay from frozen observed aspect scores.

Three profiles share the original depth50 pool and mandatory source backbone.
Default replay dispatches no inference; --fresh-ranking explicitly fills only
missing guarded query/row scores with the existing reranker. Raw source rows
and queries are retained only under .runtime; public results are source-free.
Register registration_for(args) before execution, using a fresh output path.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.config import settings
from app.evidence_focus import _ranked, select_aspect_evidence
from app.evidence_selection import _rank
from app.rerank import rerank
from scripts import run_evidence_selection_dev as replay
from scripts import run_evidence_supplement_dev as supplement
from scripts.benchmark_runtime import atomic_json
from scripts.dev_evidence_annotations import summarize_revised, validate_annotation
from scripts.research_review import digest
from scripts.retrieval_quality_index import active_scope, index_ledger
from scripts.retrieval_quality_metrics import diagnose, summarize
from scripts.run_external_repair_dev import backbone_ids, private_root
from scripts.run_retrieval_quality_dev import (
    ROOT, SELECTION, _publish, code_fingerprints, missing_row, model_identity,
    reserve_output, sha, validate_suite,
)
from scripts.run_revised_evidence_dev import missing_revised
from scripts.run_reviewed_supplement_dev import rescore, validate_seed
from scripts.run_supplement_contract_dev import contract_for

PROFILES = ('baseline', 'rank_floor', 'guarded')
IDENTITY_FIELDS = ('chunk_id', 'document_id', 'version_id', 'source_sha256', 'title', 'text')
FALLBACKS = ('original_scope_unresolved_baseline_retained', 'history_baseline_retained',
             'unsupported_aspects_baseline_retained')
UNSUPPORTED = ('Too many question aspects (maximum 8)', 'Invalid aspect query length')
MAX_NEW_CALLS = 8


def score_key(query, row):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Invalid observed query')
    if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in IDENTITY_FIELDS):
        raise ValueError('Invalid observed row identity')
    if any(not row[k] for k in IDENTITY_FIELDS[:4]):
        raise ValueError('Missing observed row identity')
    return digest([query, *[row[k] for k in IDENTITY_FIELDS]])


@dataclass
class ObservedScores:
    """Task-local exact observations and explicitly budgeted missing-pair fills."""
    scores: dict = field(default_factory=dict)
    queries: set = field(default_factory=set)
    observed_pairs: int = 0
    observed_calls: int = 0
    observed_unique_pairs: int = 0
    events: list = field(default_factory=list)
    provider_events: list = field(default_factory=list)
    fresh_enabled: bool = False
    max_new_calls: int = MAX_NEW_CALLS
    max_new_pairs: int = 0
    new_calls: int = 0
    new_pairs: int = 0

    @classmethod
    def from_original(cls, original):
        try:
            pool = original['pools']['aspect']
            by_id = {}
            for row in pool:
                score_key('identity validation', row)
                cid = row['chunk_id']
                if cid in by_id and score_key('identity validation', by_id[cid]) != score_key(
                        'identity validation', row):
                    raise ValueError('Conflicting observed pool identity')
                # Real source/global lane occurrences remain in the original
                # pool. Ranking scores belong to source bytes, not cap lanes.
                by_id.setdefault(cid, row)
            trace = original['ranking_traces'][0]
            cache = cls(max_new_pairs=MAX_NEW_CALLS * len(by_id))
            aspects = trace['aspect_ranking']
            if not isinstance(aspects, list):
                raise ValueError('Invalid observed aspect rankings')
            for aspect in aspects:
                query, scores = aspect['query'], aspect['scores']
                if not isinstance(query, str) or not query.strip() or not isinstance(scores, list):
                    raise ValueError('Invalid observed query/scores')
                cache.queries.add(query)
                if type(aspect['pairs']) is not int or aspect['pairs'] != len(scores):
                    raise ValueError('Observed pair count differs')
                seen, previous = set(), math.inf
                for item in scores:
                    cid, value = item['chunk_id'], item['rerank_score']
                    if cid not in by_id or cid in seen:
                        raise ValueError('Unknown or duplicate observed score identity')
                    if type(value) not in (int, float) or not math.isfinite(value) or value > previous:
                        raise ValueError('Invalid observed score/order')
                    previous = value
                    seen.add(cid)
                    key = score_key(query, by_id[cid])
                    if key in cache.scores and cache.scores[key] != value:
                        raise ValueError('conflicting repeated observed score')
                    cache.scores[key] = value
                cache.observed_pairs += len(scores)
                cache.observed_calls += bool(scores)
            if (type(trace['ranker_pairs']) is not int or trace['ranker_pairs'] != cache.observed_pairs
                    or type(trace['ranker_calls']) is not int
                    or trace['ranker_calls'] != cache.observed_calls):
                raise ValueError('Observed call/pair totals differ')
            cache.observed_unique_pairs = len(cache.scores)
            return cache
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError('Malformed observed ranking cache') from exc

    def costs(self):
        return ranking_costs(self.events, fresh=self.fresh_enabled)


def ranking_costs(events, *, fresh=False):
    calls = sum(e['physical_ranker_calls'] for e in events)
    out = dict(logical_ranker_calls=len(events),
               logical_ranker_pairs=sum(e['input_pairs'] for e in events),
               cache_hits=sum(e['cache_hits'] for e in events),
               physical_ranker_calls=calls,
               physical_ranker_pairs=sum(e['physical_ranker_pairs'] for e in events),
               search_calls=0, embedding_calls=0, policy_calls=0, generation_calls=0)
    if fresh:
        actual = [e for e in events if e['physical_ranker_calls']]
        complete = all(e['status'] == 'ok' for e in actual)
        out.update(fresh_attempted=len(actual),
                   fresh_completed=sum(e['status'] == 'ok' for e in actual),
                   fresh_failed=sum(e['status'] == 'failed' for e in actual),
                   fresh_wall_ms=sum(e['wall_ms'] for e in actual),
                   fresh_latency_ms=sum(e['wall_ms'] for e in actual) if complete else None,
                   fresh_token_usage=None if calls else 0,
                   fresh_cost=None if calls else 0,
                   fresh_events=deepcopy(actual))
    return out


def replay_rank(query, rows, observed, *, fresh_ranking=False, ranker=None):
    event = {'query_sha256': digest(query), 'input_pairs': len(rows), 'cache_hits': 0,
             'status': 'failed', 'physical_ranker_calls': 0, 'physical_ranker_pairs': 0}
    observed.events.append(event)
    if not isinstance(query, str) or not query.strip() or (
            not fresh_ranking and query not in observed.queries):
        raise ValueError('Missing exact observed query')
    observed.fresh_enabled |= fresh_ranking
    keys, seen = [], set()
    for row in rows:
        key = score_key(query, row)
        cid = row['chunk_id']
        if cid in seen:
            raise ValueError('Duplicate replay row identity')
        seen.add(cid)
        keys.append(key)
        event['cache_hits'] += key in observed.scores
    missing = [(key, row) for key, row in zip(keys, rows, strict=True)
               if key not in observed.scores]
    if missing:
        if not fresh_ranking:
            raise ValueError('Missing exact observed query/row score')
        if (observed.new_calls >= observed.max_new_calls
                or observed.new_pairs + len(missing) > observed.max_new_pairs):
            raise ValueError('Fresh ranking budget exhausted')
        # Reserve before dispatch; interrupted/invalid calls consume the same
        # budget as successful calls. Scores are published only after _ranked.
        observed.new_calls += 1
        observed.new_pairs += len(missing)
        event.update(physical_ranker_calls=1, physical_ranker_pairs=len(missing),
                     new_pair_key_sha256=[key for key, _ in missing],
                     status='attempted', dispatch_status='attempted', wall_ms=0)
        private = dict(query=query, input_rows=deepcopy([row for _, row in missing]),
                       output_rows=None, status='attempted', dispatch_status='attempted',
                       query_sha256=event['query_sha256'],
                       new_pair_key_sha256=deepcopy(event['new_pair_key_sha256']))
        observed.provider_events.append(private)
        began = time.monotonic()
        def dispatch(q, supplied):
            output = (ranker(q, supplied) if ranker is not None
                      else rerank(q, supplied, window=len(supplied)))
            private['output_rows'] = deepcopy(output)
            return output
        try:
            scored, _ = _ranked(query, [row for _, row in missing], dispatch)
            by_id = dict(scored)
            additions = {key: by_id[row['chunk_id']] for key, row in missing}
            observed.scores.update(additions)
            observed.queries.add(query)
            event['status'] = private['status'] = 'ok'
            event['dispatch_status'] = private['dispatch_status'] = 'completed'
        except BaseException as exc:
            event['status'] = private['status'] = 'failed'
            event['dispatch_status'] = private['dispatch_status'] = 'failed'
            event['error'] = private['error'] = {'type': type(exc).__name__}
            raise
        finally:
            event['wall_ms'] = private['wall_ms'] = (time.monotonic() - began) * 1000
    ranked = [dict(deepcopy(row), rerank_score=observed.scores[key])
              for row, key in zip(rows, keys, strict=True)]
    ranked.sort(key=lambda r: (-r['rerank_score'], _rank(r), r['chunk_id']))
    event['status'] = 'ok'
    return ranked


def public_trace(trace):
    """Only validated selector scalars, IDs and the private trace digest escape."""
    fields = ('selected_ids', 'required_ids', 'context_tokens', 'candidate_count',
              'scope_rejected_count', 'content_rejected_count', 'condition_witness_ids',
              'ranking_usage')
    out = {k: deepcopy(trace[k]) for k in fields if k in trace}
    if 'coverage_type' in trace:
        allowed = {'relevance_ranking_only',
                   'structural_literal_protection_with_unverified_relevance_fill'}
        if trace['coverage_type'] not in allowed:
            raise ValueError('Unknown public coverage type')
        out['coverage_type'] = trace['coverage_type']
    out['trace_sha256'] = digest(trace)
    return out


def paths_for(args):
    return [Path(args.suite), SELECTION, Path(args.parent), Path(args.baseline),
            Path(args.repair), Path(args.overlay), Path(args.manifest)]


def frozen_private_paths(args, ids):
    root = private_root(args.context_root)
    paths = []
    for task_id in ids:
        path = (root / (task_id + '.json')).resolve()
        if path.parent != root or not path.is_file():
            raise ValueError('Missing or escaping original private task record')
        paths.append(path)
    return paths


def registration_for(args):
    """Freeze inputs and identity without search or model/policy inference."""
    paths = paths_for(args)
    inputs = [p.read_bytes() for p in paths]
    suite = json.loads(inputs[0])
    ids = [t['id'] for t in suite['tasks']]
    if len(ids) != 47 or len(set(ids)) != 47 or suite.get('split') != 'dev':
        raise ValueError('Registration requires exact full Dev47')
    originals = frozen_private_paths(args, ids)
    registration = {'version': 'guarded-selection-input-v1', 'split': 'dev', 'subset': False,
        'selected_task_ids': ids, 'input_bytes_sha256': list(map(sha, inputs)),
        'original_private_bytes_sha256': {tid: sha(p.read_bytes())
            for tid, p in zip(ids, originals, strict=True)},
        'profiles': list(PROFILES), 'baseline': 'original_relevant_doc',
        'candidate_pool': 'original_depth50_rerank_same_authorized_occurrences',
        'ranking': 'only_original_ranking_traces_0_exact_observed_query_identity_bytes',
        'score_provenance': 'private_trace_digest_equals_original_public_cost_ranking_digest',
        'backbone': 'first_four_existing_nonmetadata_source_version_witnesses_no_gold',
        'caps': {'limit': 8, 'token_budget': 5000, 'document_quota': None,
                 'source_quota': 'original_question_policy_caps'},
        'budget': {'search_calls': 0, 'embedding_calls': 0, 'policy_calls': 0,
                   'generation_calls': 0, 'physical_ranker_calls': 0, 'physical_ranker_pairs': 0},
        'fallback': 'reproduce_original_source_history_or_registered_unsupported_aspects_only',
        'adoption_rule': 'guarded_only_native_per_reference_no_loss_and_strict_external_delivery_gain_else_baseline',
        'gold_access': 'post_runtime_evaluator_only', 'strict_accuracy': None, 'semantic_recall': None,
        'code': code_fingerprints(), 'settings_sha256': digest(settings().model_dump(mode='json'))}
    if getattr(args, 'fresh_ranking', False):
        reranker_identity = active_reranker_identity()
        cached_reranker = {k: v for k, v in reranker_identity.items() if k != 'libraries'}
        parent, repair = json.loads(inputs[2]), json.loads(inputs[4])
        try:
            if (cached_reranker != parent['identity']['models']['reranker']
                    or cached_reranker != repair['identity']['reranker']
                    or reranker_identity['libraries'] != parent['identity']['models']['libraries']):
                raise ValueError('Active and cached reranker identity differs')
        except (KeyError, TypeError) as exc:
            raise ValueError('Malformed cached reranker identity') from exc
        pair_caps = {}
        for task_id, path in zip(ids, originals, strict=True):
            pool = json.loads(path.read_bytes())['pools']['aspect']
            for row in pool:
                score_key('identity validation', row)
            pair_caps[task_id] = MAX_NEW_CALLS * len({row['chunk_id'] for row in pool})
        registration.update(version='guarded-selection-input-v2',
            ranking='original_exact_observations_then_guarded_only_exact_missing_pair_rerank',
            score_provenance='original_trace_authentication_and_registered_model_private_provider_events',
            reranker_identity=reranker_identity)
        registration['budget'].update(physical_ranker_calls=MAX_NEW_CALLS,
            physical_ranker_pairs=None, physical_ranker_pairs_by_task=pair_caps,
            reservation='per_task_before_dispatch_including_failed_and_interrupted_attempts')
    return registration


def active_reranker_identity():
    # Existing identity API uses read-only localhost runtime metadata and local
    # HF files. It never loads the BGE model or dispatches embedding inference.
    models = model_identity(settings(), rerank=True)
    return dict(deepcopy(models['reranker']), libraries=deepcopy(models['libraries']))


def validate_repair(repair, parent, baseline, suite, manifest):
    try:
        ids = [t['id'] for t in suite['tasks']]
        identity = repair['identity']
        if (repair.get('version') != 'external-repair-dev-v1' or repair.get('valid') is not True
                or repair.get('status') != 'complete' or repair.get('split') != 'dev'
                or repair.get('subset') is not False or repair.get('error') is not None
                or len(ids) != 47 or len(set(ids)) != 47
                or not isinstance(repair.get('drift'), dict) or not repair['drift']
                or any(v is not False for v in repair['drift'].values())
                or repair['selected_task_ids'] != ids):
            raise ValueError('Repair must be complete valid full Dev47 without drift')
        if (identity['parent_sha256'] != digest(parent) or identity['annotation_manifest'] != manifest
                or identity['corpus_sha256'] != parent['identity']['corpus_sha256']
                or identity['index_sha256'] != parent['identity']['source_index_sha256']):
            raise ValueError('Repair parent/corpus/index/manifest identity differs')
        if ({k: v for k, v in identity['models']['embedding'].items() if k != 'status'}
                != parent['identity']['models']['embedding']
                or identity['reranker'] != parent['identity']['models']['reranker']):
            raise ValueError('Recorded ranking/model provenance differs')
        for profile in ('baseline', 'aspect', 'control', 'treatment'):
            rows = repair['rows'][profile]
            if [r['task_id'] for r in rows] != ids or any(r['error'] is not None for r in rows):
                raise ValueError('Repair rows incomplete or failed')
        for name in ('traces', 'costs'):
            if [r['task_id'] for r in repair[name]] != ids:
                raise ValueError('Repair trace/cost order differs')
        if any(t.get('errors') for t in repair['traces']):
            raise ValueError('Repair contains trace errors')
        for old, seed in zip(repair['rows']['baseline'], baseline['rows']['relevant_doc'], strict=True):
            fields = ('candidates', 'admitted', 'context', 'context_tokens')
            if any(old['telemetry'][k] != seed['telemetry'][k] for k in fields):
                raise ValueError('Repair original relevant_doc baseline differs')
            for phase in ('candidates', 'admitted', 'context'):
                count = 'candidate_count' if phase == 'candidates' else phase + '_count'
                if old['telemetry'][count] != len(old['telemetry'][phase]):
                    raise ValueError('Repair baseline telemetry count differs')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Malformed repair artifact') from exc


def original_public_phases(original, profile):
    pool, context = original['pools'][profile], original['contexts'][profile]
    chosen = [dict(r, rank=i, admitted=True, excluded_because=None)
              for i, r in enumerate(context, 1)]
    return {'candidates': supplement.public_rows(pool),
            'admitted': supplement.public_rows(chosen), 'context': supplement.public_rows(chosen)}


def validate_original(original, task_id, question, repair_rows, prior_cost, prior_trace,
                      parent_pool, baseline_context):
    """Authenticate original scores and the exact unsupplemented pool/allocation."""
    try:
        if original['task_id'] != task_id or original['question'] != question:
            raise ValueError('Original private task/question identity differs')
        rankings = original.get('ranking_traces', [])
        public_rankings = prior_cost['rankings']
        if bool(rankings) != bool(public_rankings):
            raise ValueError('Original ranking provenance missing')
        if rankings and digest(rankings[0]) != public_rankings[0]['trace_sha256']:
            raise ValueError('Original ranking provenance hash differs')
        for profile in ('baseline', 'aspect'):
            phases = original_public_phases(original, profile)
            telemetry = repair_rows[profile]['telemetry']
            if any(telemetry[p] != phases[p] for p in phases):
                raise ValueError('Original private pool/context telemetry differs')
            replay.validate_context(original['contexts'][profile], telemetry['context_tokens'])
            if supplement.public_rows(original['pools'][profile]) != parent_pool:
                raise ValueError('Original pool differs from frozen parent depth50 pool')
        if supplement.public_rows(original['contexts']['baseline']) != baseline_context:
            raise ValueError('Original relevant_doc context differs')
        if not rankings:
            status = original['trace']['status']
            if status not in FALLBACKS or prior_trace['status'] != status:
                raise ValueError('Unobserved ranking lacks original fallback reason')
            if original['contexts']['aspect'] != original['contexts']['baseline']:
                raise ValueError('Original fallback context differs from baseline')
        return ObservedScores.from_original(original) if rankings else ObservedScores()
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError('Malformed original private record/provenance') from exc


def rebound_preserving(db, user, rows, corpus):
    bound = replay.rebind_rows(db, user, rows, corpus)
    return [dict(deepcopy(raw), **clean) for raw, clean in zip(rows, bound, strict=True)]


def allocation_identity(rows):
    """All actual allocation fields except the purely presentational E-number."""
    return [{k: deepcopy(v) for k, v in r.items() if k != 'id'} for r in rows]


def run_profiles(question, candidates, baseline, contract, caps, observed,
                 expected_aspect, expected_trace, fallback_status=None, private=None,
                 *, fresh_ranking=False):
    """Gold-free boundary: question, authorized rows, contract/caps and cache only."""
    if contract.question != question:
        raise ValueError('Contract question differs')
    caps = dict(caps, limit=8, token_budget=5000, document_quota=None)
    required = backbone_ids(contract, baseline, candidates)
    if expected_trace is not None and expected_trace.get('required_ids') != sorted(required):
        raise ValueError('Original mandatory backbone differs')
    contexts = {'baseline': deepcopy(baseline)}
    traces, costs, errors = {}, {}, {}
    observed.fresh_enabled |= fresh_ranking
    if private is not None:
        private.update(contexts=contexts, selection_traces=traces, errors=errors,
                       ranker_events=observed.events, profile_costs=costs)
        if fresh_ranking:
            private['provider_events'] = observed.provider_events
    scope_fallback = (FALLBACKS[0] if any(r.resolution != 'supported' for r in contract.source_requests)
                      else FALLBACKS[1] if contract.reason == 'unsupported_history' else None)
    for profile in ('rank_floor', 'guarded'):
        began, start = time.monotonic(), len(observed.events)
        try:
            if scope_fallback:
                if fallback_status != scope_fallback:
                    raise ValueError('Original source/history fallback differs')
                selected, trace = deepcopy(baseline), {'status': scope_fallback}
            else:
                try:
                    result = select_aspect_evidence(contract, deepcopy(candidates),
                        ranker=lambda q, rs: replay_rank(q, rs, observed,
                            fresh_ranking=fresh_ranking and profile == 'guarded'),
                        selection_policy=profile, required_ids=required, **caps)
                except ValueError as exc:
                    if str(exc) not in UNSUPPORTED or fallback_status != FALLBACKS[2]:
                        raise
                    selected, trace = deepcopy(baseline), {'status': FALLBACKS[2]}
                else:
                    if fallback_status is not None:
                        raise ValueError('Original unsupported fallback no longer reproduced')
                    selected, trace = result.evidence, deepcopy(result.trace)
                    trace['status'] = 'selected'
            if profile == 'rank_floor' and allocation_identity(selected) != allocation_identity(expected_aspect):
                raise ValueError('Original rank_floor IDs/order/actual allocation differs')
            contexts[profile], traces[profile] = selected, trace
        except BaseException as exc:
            contexts[profile] = None
            errors[profile] = {'type': type(exc).__name__, 'stage': 'selection'}
            traces[profile] = {'status': 'failed' if isinstance(exc, Exception) else 'interrupted',
                               'error': errors[profile]}
            if not isinstance(exc, Exception):
                raise
        finally:
            events = observed.events[start:]
            costs[profile] = {'status': traces.get(profile, {}).get('status', 'failed')
                if profile in errors else 'observed',
                **ranking_costs(events, fresh=fresh_ranking),
                'wall_ms': (time.monotonic()-began)*1000}
            if fresh_ranking and profile in traces:
                traces[profile]['ranking_usage'] = ranking_costs(events, fresh=True)
        # Guarded observations are meaningful only after validating the exact old
        # policy reproduction. Never continue with an untrusted seed/cache.
        if profile == 'rank_floor' and profile in errors:
            contexts['guarded'] = None
            errors['guarded'] = {'type': 'UnverifiedRankFloor', 'stage': 'selection'}
            traces['guarded'] = {'status': 'not_observed', 'error': errors['guarded']}
            break
    return contexts, traces, costs, errors


def choose(report):
    """Dev delivery proxy adoption only; never claim accuracy or semantics."""
    baseline, guarded = report['rows']['baseline'], report['rows']['guarded']
    if len(baseline) != len(guarded) or any(r.get('error') for r in baseline + guarded):
        return 'baseline'
    for old, new in zip(baseline, guarded, strict=True):
        if old['task_id'] != new['task_id']:
            return 'baseline'
        if old['eligible'] and old['basis'] == 'source_atom_exact_proxy':
            a, b = old['phases']['context'], new['phases']['context']
            if a is None or b is None or not set(a['matched_reference_ids']) <= set(b['matched_reference_ids']):
                return 'baseline'
    def delivery(profile):
        group = report['summaries'][profile]['by_basis'].get('upstream_word4_proxy')
        return group['phases']['context']['delivered'] if group else None
    old, new = delivery('baseline'), delivery('guarded')
    return 'guarded' if old is not None and new is not None and new > old else 'baseline'


def no_dispatch_cost(status='not_observed', *, fresh=False):
    out = {'status': status, 'logical_ranker_calls': None, 'logical_ranker_pairs': None,
            'cache_hits': None, 'physical_ranker_calls': 0, 'physical_ranker_pairs': 0,
            'search_calls': 0, 'embedding_calls': 0, 'policy_calls': 0, 'generation_calls': 0,
            'wall_ms': None}
    if fresh:
        out.update({k: v for k, v in ranking_costs([], fresh=True).items() if k.startswith('fresh_')})
    return out


def selection_summaries(rows, *, revised=False):
    """Project missing delivery to null; observed partial sums stay explicit."""
    result = (summarize_revised if revised else summarize)(rows)

    def project(group):
        for phase in group.get('phases', {}).values():
            measured, missing = phase['measured_tasks'], phase['missing_tasks']
            if missing:
                values = ('reference_denominator', 'delivered', 'all_delivered_tasks', 'proxy_coverage')
                if measured:
                    phase['observed_subtotal'] = {k: phase[k] for k in values if k in phase}
                for key in values:
                    if key in phase:
                        phase[key] = None
                phase['measurement_status'] = 'partial' if measured else 'pending'
            else:
                phase['measurement_status'] = 'complete' if measured else 'not_applicable'
        for phase in group.get('sensitivity_including_unclear', {}).values():
            measured, missing = phase['measured_tasks'], phase['missing_tasks']
            if missing:
                values = ('reference_denominator', 'delivered', 'proxy_coverage')
                if measured:
                    phase['observed_subtotal'] = {k: phase[k] for k in values}
                phase.update({k: None for k in values})
                phase['measurement_status'] = 'partial' if measured else 'pending'
        for category in group.get('by_category', {}).values():
            project(category)

    for group in result['by_basis'].values():
        project(group)
    return result


def execute(args, reservation):
    from app.clients import Search
    from app.db import SessionLocal

    fresh = getattr(args, 'fresh_ranking', False)
    report = {'version': 'guarded-selection-dev-v2' if fresh else 'guarded-selection-dev-v1',
        'split': 'dev', 'subset': False,
        'run_id': reservation['run_id'], 'status': 'failed', 'valid': False, 'error': None,
        'strict_accuracy': None, 'strict_task_success': None, 'semantic_recall': None,
        'selected_task_ids': [], 'rows': {p: [] for p in PROFILES},
        'legacy_rows': {p: [] for p in PROFILES}, 'traces': [], 'costs': [],
        'chosen_profile': None, 'identity': None,
        'drift': {k: None for k in ('inputs', 'original_private', 'code', 'settings',
                                  'registration', 'corpus', 'index')}}
    if fresh:
        report['drift']['reranker_identity'] = None
    tasks, annotation, runtime = [], None, None
    paths, inputs, private_paths, private_bytes = [], [], [], []
    code, settings_hash, reg_bytes = None, None, None
    reranker_identity = None
    observed_runtime = {}
    stage = 'preflight'
    try:
        paths = paths_for(args)
        suite_bytes, selection_bytes = paths[0].read_bytes(), paths[1].read_bytes()
        suite, selection = json.loads(suite_bytes), json.loads(selection_bytes)
        # Establish the trusted full Dev denominator before optional experiment
        # inputs. An invalid suite never supplies accepted task positions.
        tasks = validate_suite(suite, selection, ROOT)
        report['selected_task_ids'] = ids = [t['id'] for t in tasks]
        inputs = [suite_bytes, selection_bytes, *[p.read_bytes() for p in paths[2:]]]
        parent, baseline, repair, manifest = [json.loads(inputs[i]) for i in (2, 3, 4, 6)]
        originals = replay.validate_parent(parent, suite, selection, inputs[0], inputs[1])
        annotation = validate_annotation(suite, inputs[5], manifest, ROOT, inputs[0])
        if validate_seed(baseline, parent, suite, manifest) != 'relevant_doc':
            raise ValueError('Expected original relevant_doc seed')
        validate_repair(repair, parent, baseline, suite, manifest)
        private_paths = frozen_private_paths(args, ids)
        private_bytes = [p.read_bytes() for p in private_paths]
        private_records = [json.loads(b) for b in private_bytes]
        reg_bytes = Path(args.registration).read_bytes()
        registration = json.loads(reg_bytes)
        if registration != registration_for(args):
            raise ValueError('Frozen selection registration differs')
        if (registration['input_bytes_sha256'] != list(map(sha, inputs))
                or registration['original_private_bytes_sha256'] != {
                    tid: sha(b) for tid, b in zip(ids, private_bytes, strict=True)}):
            raise ValueError('Input bytes changed before runtime')
        code, settings_hash = registration['code'], registration['settings_sha256']
        if fresh:
            reranker_identity = registration['reranker_identity']
        runtime = private_root(args.private_root) / reservation['run_id']
        runtime.mkdir(parents=True, exist_ok=False)
        atomic_json(runtime / 'registration.json', registration)
        report['identity'] = {'registration': registration, 'parent_sha256': digest(parent),
            'baseline_sha256': digest(baseline), 'repair_sha256': digest(repair),
            'annotation_manifest': manifest, 'code': code, 'settings_sha256': settings_hash,
            'corpus_sha256': parent['identity']['corpus_sha256'],
            'index_sha256': parent['identity']['source_index_sha256'],
            'cached_models': deepcopy(parent['identity']['models']),
            'prior_collection': {'run_id': repair['run_id'], 'artifact_sha256': sha(inputs[4]),
                                 'models_sha256': digest(repair['identity']['models']),
                                 'reranker_sha256': digest(repair['identity']['reranker'])},
            'private_directory_sha256': digest(str(runtime)),
            'production_strategy_changed': False}
        if fresh:
            report['identity']['reranker_identity'] = deepcopy(reranker_identity)
        atomic_json(runtime / 'identity.json', report['identity'])
        stage = 'source_identity'
        # This Search client is used only by index_ledger, never retrieval.
        source = Search(index=parent['identity']['source_index']['index'])
        with SessionLocal() as db:
            db.execute(text('SET TRANSACTION READ ONLY'))
            users, records, corpus = active_scope(db)
            index = index_ledger(source, records)
            if corpus != parent['identity']['corpus'] or index != parent['identity']['source_index']:
                raise ValueError('Current corpus/index differs from frozen parent')
            try:
                stage = 'selection_runtime'
                for offset, task in enumerate(tasks):
                    task_id, question = task['id'], task['goal']
                    user = users[task['user']]
                    original = private_records[offset]
                    record = {'task_id': task_id, 'question': question,
                              'contexts': {}, 'pools': {}, 'errors': {}}
                    observed = None
                    try:
                        parent_phases = replay.rehydrate_task(db, user, originals[offset], corpus)
                        prior = baseline['rows']['relevant_doc'][offset]['telemetry']
                        recorded_baseline = replay.rebind_rows(db, user, prior['context'], corpus)
                        cache = validate_original(original, task_id, question,
                            {p: repair['rows'][p][offset] for p in ('baseline', 'aspect')},
                            repair['costs'][offset], repair['traces'][offset],
                            supplement.public_rows(parent_phases['candidates']),
                            supplement.public_rows(recorded_baseline))
                        observed = cache
                        cache.fresh_enabled = fresh
                        if fresh:
                            cache.max_new_calls = registration['budget']['physical_ranker_calls']
                            cache.max_new_pairs = registration['budget']['physical_ranker_pairs_by_task'][task_id]
                        candidates = rebound_preserving(db, user, original['pools']['aspect'], corpus)
                        baseline_context = rebound_preserving(db, user, original['contexts']['baseline'], corpus)
                        expected = rebound_preserving(db, user, original['contexts']['aspect'], corpus)
                        contract = contract_for(db, user, question)
                        caps = replay.policy_caps(db, user, question)
                        trace0 = original.get('ranking_traces', [])
                        fallback = original['trace']['status'] if not trace0 else None
                        record['pools'] = {p: deepcopy(candidates) for p in PROFILES}
                        contexts, traces, costs, errors = run_profiles(question, candidates,
                            baseline_context, contract, caps, cache, expected,
                            trace0[0] if trace0 else None, fallback, record,
                            fresh_ranking=fresh)
                        raw_rows = {p: supplement.score_context(db, user, {'id': task_id, 'goal': question},
                            candidates, contexts[p], corpus) if contexts[p] is not None
                            else missing_row({'id': task_id}, errors[p]) for p in PROFILES}
                        observed_runtime[task_id] = {'rows': raw_rows,
                            'phases': {p: {'candidates': candidates, 'admitted': contexts[p],
                                          'context': contexts[p]} for p in PROFILES}}
                        public = {'task_id': task_id, 'status': 'failed' if errors else 'completed',
                            'errors': [{'profile': p, **e} for p, e in errors.items()],
                            'profiles': {p: {'status': t['status'], **public_trace(t)}
                                         for p, t in traces.items()}}
                        report['traces'].append(public)
                        report['costs'].append({'task_id': task_id, 'status': public['status'],
                            **cache.costs(), 'profiles': {'baseline': dict(no_dispatch_cost('observed', fresh=fresh),
                                logical_ranker_calls=0, logical_ranker_pairs=0, cache_hits=0),
                                **{p: costs.get(p, no_dispatch_cost(fresh=fresh)) for p in ('rank_floor', 'guarded')}},
                            'prior_collection': {'trace_sha256': digest(trace0[0]) if trace0 else None,
                                'observed_calls': cache.observed_calls,
                                'observed_pairs': cache.observed_pairs,
                                'unique_cached_pairs': cache.observed_unique_pairs,
                                'physical_usage_attributed_to': 'prior_repair_artifact_only'}})
                    except Exception as exc:
                        error = {'type': type(exc).__name__, 'stage': 'selection_runtime'}
                        record['errors'] = {p: error for p in PROFILES}
                        report['traces'].append({'task_id': task_id, 'status': 'failed',
                            'errors': [{'profile': p, **error} for p in PROFILES]})
                        report['costs'].append({'task_id': task_id, **no_dispatch_cost('failed', fresh=fresh),
                                               **(observed.costs() if observed is not None else {})})
                    except BaseException as exc:
                        error = {'type': type(exc).__name__, 'stage': 'selection_runtime'}
                        record['interruption'] = error
                        report['traces'].append({'task_id': task_id, 'status': 'interrupted',
                            'errors': [error], 'profiles': {p: {'status': t.get('status', 'interrupted'),
                                **public_trace(t)} for p, t in record.get('selection_traces', {}).items()}})
                        report['costs'].append({'task_id': task_id, **no_dispatch_cost('interrupted', fresh=fresh),
                            **(observed.costs() if observed is not None else {}),
                            'profiles': deepcopy(record.get('profile_costs', {}))})
                        raise
                    finally:
                        atomic_json(runtime / (task_id + '.json'), record)
                    atomic_json(runtime / 'checkpoint.json', report)
                    print(json.dumps({'task_id': task_id, 'status': report['traces'][-1]['status']}), flush=True)

                # The entire policy runtime completes before evaluator task gold
                # or revised annotation records enter any scoring call.
                stage = 'post_runtime_evaluation'
                for task in tasks:
                    found = observed_runtime.get(task['id'])
                    if found is None:
                        continue
                    for profile in PROFILES:
                        raw = found['rows'][profile]
                        phases = found['phases'][profile]
                        report['legacy_rows'][profile].append(diagnose(task,
                            phases if phases['context'] is not None else {}))
                        try:
                            row = rescore(db, users[task['user']], task, raw, corpus,
                                          annotation['task_records'][task['id']])
                        except Exception as exc:
                            row = missing_revised(task, annotation['task_records'][task['id']],
                                {'type': type(exc).__name__, 'stage': 'post_runtime_evaluation'})
                        report['rows'][profile].append(row)
            finally:
                _, end_records, end_corpus = active_scope(db)
                report['drift']['corpus'] = end_corpus != corpus
                report['drift']['index'] = index_ledger(source, end_records) != index
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                      error={'type': type(exc).__name__, 'stage': stage})
    finally:
        checks = {
            'inputs': lambda: any(p.read_bytes() != b for p, b in zip(paths, inputs, strict=True)),
            'original_private': lambda: any(p.read_bytes() != b for p, b in
                                            zip(private_paths, private_bytes, strict=True)),
            'code': lambda: code_fingerprints() != code,
            'settings': lambda: digest(settings().model_dump(mode='json')) != settings_hash,
            'registration': lambda: Path(args.registration).read_bytes() != reg_bytes}
        if fresh and reranker_identity is not None:
            checks['reranker_identity'] = lambda: active_reranker_identity() != reranker_identity
        for key, check in checks.items():
            if ((key == 'code' and code is None) or (key == 'settings' and settings_hash is None)
                    or (key == 'registration' and reg_bytes is None)
                    or (key == 'inputs' and len(paths) != len(inputs))
                    or (key == 'original_private' and not private_bytes)):
                continue
            try:
                report['drift'][key] = check()
            except Exception:
                report['drift'][key] = True
        traces = {t['task_id']: t for t in report['traces']}
        costs = {c['task_id']: c for c in report['costs']}
        report['traces'] = [traces.get(t['id']) or {'task_id': t['id'], 'status': 'not_observed',
                           'errors': []} for t in tasks]
        report['costs'] = [costs.get(t['id']) or {'task_id': t['id'], **no_dispatch_cost(fresh=fresh)} for t in tasks]
        for profile in PROFILES:
            seen = {r['task_id']: r for r in report['rows'][profile]}
            old = {r['task_id']: r for r in report['legacy_rows'][profile]}
            report['rows'][profile] = []
            report['legacy_rows'][profile] = []
            for task in tasks:
                error = report['error'] or {'type': 'NotObserved', 'stage': stage}
                if task['id'] in seen:
                    row = seen[task['id']]
                elif annotation is not None:
                    row = missing_revised(task, annotation['task_records'][task['id']], error)
                else:
                    row = missing_row(task, error)
                report['rows'][profile].append(row)
                report['legacy_rows'][profile].append(old.get(task['id']) or diagnose(task, {}))
        report['summaries'] = {p: selection_summaries(rs, revised=annotation is not None)
                               for p, rs in report['rows'].items()}
        report['legacy_summaries'] = {p: selection_summaries(rs)
                                      for p, rs in report['legacy_rows'].items()}
        errors = report['error'] is not None or any(r['error'] for rs in report['rows'].values() for r in rs)
        errors = errors or any(t.get('errors') for t in report['traces'])
        drift_ok = all(value is False for value in report['drift'].values())
        complete = len(tasks) == 47 and all(len(rs) == 47 for rs in report['rows'].values())
        report['valid'] = complete and not errors and drift_ok
        if report['valid']:
            report['status'], report['chosen_profile'] = 'complete', choose(report)
        elif report['status'] != 'interrupted':
            report['status'] = 'invalid_drift' if any(v is True for v in report['drift'].values()) else 'failed'
        report['outcomes'] = dict(Counter(t['status'] for t in report['traces']))
        report['execution_errors'] = {p: sum(r['error'] is not None for r in rs)
                                      for p, rs in report['rows'].items()}
        if runtime is not None:
            atomic_json(runtime / 'checkpoint.json', report)
        _publish(args.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('suite', 'parent', 'baseline', 'repair', 'context-root', 'overlay', 'manifest',
                'registration', 'private-root', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--fresh-ranking', action='store_true',
                        help='Explicit v2 guarded-only missing-score reranker dispatch')
    args = parser.parse_args(argv)
    reservation = reserve_output(args.output)
    report = execute(args, reservation)
    print(json.dumps({'valid': report['valid'], 'status': report['status'], 'output': args.output}))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
