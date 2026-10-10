"""Frozen Dev-only cached evidence selection and quota replay; no model calls.

Run with --stage selection|quotas --parent <retrieval artifact> --output <fresh path>.
Only public IDs, hashes, cached scores and diagnostics are written to artifacts.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from agent.planner import _grams
from app.chunking import token_count
from app.config import Settings, settings
from app.evidence_selection import SelectionResult, replace_weak_evidence, select_complementary
from app.retrieval import route_sources
from app.security import readable_documents, require_chunk
from app.task_analysis import multi_source_intent, needs_document_diversity
from scripts.benchmark_runtime import CheckpointStore, atomic_json, checkpoint_lock
from scripts.research_review import digest
from scripts.retrieval_quality_index import active_scope, index_ledger
from scripts.retrieval_quality_metrics import PHASES, diagnose, pair_profiles, summarize
from scripts.run_retrieval_quality_dev import (
    CONFIG_KEYS, PROFILES, ROOT, SELECTION, _publish, _telemetry, _validate_active_gold,
    code_fingerprints, configuration, missing_row, null_evidence_counts, reserve_output,
    sha, validate_suite,
)

RUNTIME = ROOT / '.runtime/evidence-selection'
PARENT_PROFILE = 'depth50_rerank'
SELECTION_PROFILES = ('legacy_rerank', 'complementary_strict')
QUOTA_PROFILES = ('complementary_strict', 'replace_strict', 'replace_doc',
                  'replace_source', 'replace_both')
IDENTITY_FIELDS = ('chunk_id', 'document_id', 'version_id', 'source_sha256')
ROW_FIELDS = ('chunk_id', 'document_id', 'version_id', 'source_sha256', 'rank',
              'retrieval_rank', 'rerank_score', 'excluded_because', 'admitted',
              'bm25_rank', 'dense_rank', 'fusion_score', 'parent_rank',
              'lane_type', 'lane_sha256')
EXCLUSION_REASONS = frozenset({'boilerplate_only', 'below_min_similarity',
    'outside_requested_publication_dates', 'tenant_scope', 'document_quota',
    'source_quota', 'top_k_full', 'context_budget_exhausted'})


def stage_profiles(stage):
    if stage == 'selection':
        return SELECTION_PROFILES
    if stage == 'quotas':
        return QUOTA_PROFILES
    raise ValueError('unknown replay stage')


def profile_declarations(profiles):
    """Concrete frozen arm definitions, persisted before any selection call."""
    result = []
    for profile in profiles:
        result.append({'name': profile,
            'method': 'recorded_parent_context' if profile == 'legacy_rerank' else
                'complementary_lexical_v1' if profile == 'complementary_strict' else
                'weak_evidence_replacement_lexical_v1',
            'relax_document_quota': profile in ('replace_doc', 'replace_both'),
            'relax_source_quota': profile in ('replace_source', 'replace_both'),
            'seed': 'complementary_strict' if profile.startswith('replace_') else None,
            'max_replacements': 8 if profile.startswith('replace_') else 0})
    return result


def corpus_bindings(corpus):
    """Index frozen source identities without reading or emitting source text."""
    found = {}
    for doc in corpus['documents']:
        for chunk in doc['chunks']:
            if chunk['id'] in found:
                raise ValueError('duplicate corpus chunk identity')
            found[chunk['id']] = (doc, chunk)
    return found


def _lane(row):
    if (row.get('lane_type') not in ('source', 'global')
            or not isinstance(row.get('lane_sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', row['lane_sha256'])):
        raise ValueError('unsupported or missing recorded lane')
    if row['lane_type'] == 'global' and row['lane_sha256'] != digest('global'):
        raise ValueError('global lane hash differs')


def _source_binding(row, bindings):
    validate_telemetry_row(row)
    entry = bindings.get(row['chunk_id'])
    if entry is None:
        raise ValueError('parent row outside frozen corpus')
    doc, chunk = entry
    if (row['document_id'], row['version_id'], row['source_sha256']) != (
            doc['document_id'], doc['active_version_id'], doc['source_sha256']):
        raise ValueError('parent row source identity differs')
    if not chunk.get('text_sha256') or not doc.get('document_sha256'):
        raise ValueError('missing corpus text/document hash')
    return doc, chunk


def validate_telemetry_row(row):
    """Validate every retained scalar before authorization or public projection."""
    _lane(row)
    if any(not isinstance(row.get(k), str) or not row[k] for k in IDENTITY_FIELDS):
        raise ValueError('missing row identity')
    for key in ('rank', 'retrieval_rank', 'bm25_rank', 'dense_rank', 'parent_rank'):
        value = row.get(key)
        nullable = key in ('retrieval_rank', 'bm25_rank', 'dense_rank')
        if key == 'parent_rank' and key not in row:
            continue
        if value is None and nullable:
            continue
        if type(value) is not int or value < 1:
            raise ValueError('invalid telemetry rank field')
    for key in ('rerank_score', 'fusion_score'):
        value = row.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError('invalid telemetry score field')
    if type(row.get('admitted')) is not bool:
        raise ValueError('invalid telemetry admission flag')
    reason = row.get('excluded_because')
    if reason is not None and (not isinstance(reason, str) or reason not in EXCLUSION_REASONS):
        raise ValueError('invalid telemetry exclusion reason')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError('unexpected or missing cached model identity fields')


def _identifier(value):
    return (isinstance(value, str)
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,255}', value) is not None)


def _version(value):
    return (isinstance(value, str)
            and re.fullmatch(r'[0-9][A-Za-z0-9.+_-]{0,63}', value) is not None)


def cached_model_identity(models, cfg):
    """Validate cached provenance recursively before copying it into public identity."""
    _fields(models, ('embedding', 'generation', 'policy', 'reranker', 'libraries'))
    embedding, reranker = models['embedding'], models['reranker']
    _fields(embedding, ('name', 'digest', 'runtime', 'endpoint_sha256'))
    _fields(embedding['runtime'], ('version',))
    if (not _identifier(embedding['name']) or embedding['name'] != cfg['embed_model']
            or not _sha256(embedding['digest']) or not _sha256(embedding['endpoint_sha256'])
            or not _version(embedding['runtime']['version'])):
        raise ValueError('invalid cached embedding identity')
    for key in ('generation', 'policy'):
        entry = models[key]
        _fields(entry, ('name', 'status', 'digest'))
        if not _identifier(entry['name']) or entry['status'] != 'unused' or entry['digest'] is not None:
            raise ValueError('invalid unused cached model identity')
    _fields(reranker, ('name', 'status', 'revision', 'snapshot_path_sha256', 'files',
                      'digest', 'batch', 'max_tokens'))
    revision = reranker['revision']
    if (not _identifier(reranker['name']) or reranker['name'] != cfg['rerank_model']
            or reranker['status'] != 'active' or not _sha256(reranker['digest'])
            or not _sha256(reranker['snapshot_path_sha256'])
            or (revision is not None and (not isinstance(revision, str)
                or re.fullmatch('[0-9a-f]{40,64}', revision) is None))
            or any(type(reranker[key]) is not int or reranker[key] < 1
                   or reranker[key] != cfg[cfg_key] for key, cfg_key in
                   (('batch', 'rerank_batch'), ('max_tokens', 'rerank_max_tokens')))):
        raise ValueError('invalid cached reranker identity')
    files = reranker['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('cached reranker files missing')
    for name, value in files.items():
        if (not isinstance(name, str) or len(name) > 256
                or re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', name) is None
                or any(part in ('.', '..') for part in name.split('/'))
                or not name.endswith(('.json', '.safetensors', '.bin', '.model', '.txt'))
                or not _sha256(value)):
            raise ValueError('invalid cached reranker file identity')
    vocabulary = any(name in files for name in ('tokenizer.json', 'vocab.txt', 'tokenizer.model',
        'sentencepiece.bpe.model', 'spiece.model')) or {'vocab.json', 'merges.txt'} <= files.keys()
    if ('config.json' not in files or 'tokenizer_config.json' not in files or not vocabulary
            or not any(name.endswith(('.safetensors', '.bin')) for name in files)):
        raise ValueError('cached reranker weights/config/tokenizer missing')
    libraries = models['libraries']
    _fields(libraries, ('httpx', 'pydantic', 'sqlalchemy', 'torch', 'transformers', 'tokenizers'))
    if any(value is not None and not _version(value) for value in libraries.values()):
        raise ValueError('invalid cached library version')
    return deepcopy(models)


def validate_parent(parent, suite, selection, suite_bytes, selection_bytes):
    """Validate a complete 47-row frozen depth50_rerank cache before clients."""
    try:
        if (parent.get('valid') is not True or parent.get('status') != 'complete'
                or parent.get('split') != 'dev' or parent.get('subset') is not False
                or parent.get('error') is not None
                or any(parent.get('drift', {}).values())):
            raise ValueError('parent must be a valid complete full Dev run')
        ids = [task['id'] for task in suite['tasks']]
        identity = parent['identity']
        if len(ids) != 47 or len(set(ids)) != 47:
            raise ValueError('exact 47 unique Dev tasks required')
        expected = {'suite_sha256': digest(suite), 'selection_sha256': digest(selection),
            'suite_bytes_sha256': sha(suite_bytes),
            'selection_bytes_sha256': sha(selection_bytes), 'selected_task_ids': ids}
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError('parent suite/selection identity differs')
        if parent.get('selected_task_ids') != ids:
            raise ValueError('parent selected IDs differ')
        for key in ('corpus', 'source_index'):
            if digest(identity[key]) != identity.get(key + '_sha256'):
                raise ValueError('parent corpus/index digest differs')
        profiles = [p for p in identity['profiles'] if p['name'] == PARENT_PROFILE]
        if len(profiles) != 1 or profiles[0]['overrides'] != PROFILES[PARENT_PROFILE]:
            raise ValueError('supported cached retrieval profile required')
        cfg = profiles[0]['effective_configuration']
        fixed = configuration(Settings(_env_file=None, **cfg), PARENT_PROFILE)
        if cfg != {key: getattr(fixed, key) for key in CONFIG_KEYS}:
            raise ValueError('parent retrieval configuration differs')
        cached_model_identity(identity['models'], cfg)
        errors = parent['execution_errors']
        if PARENT_PROFILE not in errors or any(e['tasks'] != 0 for e in errors.values()):
            raise ValueError('parent execution errors')
        rows = parent['rows'][PARENT_PROFILE]
        if len(rows) != 47 or [r['task_id'] for r in rows] != ids:
            raise ValueError('parent rows incomplete, duplicated or reordered')
        bindings = corpus_bindings(identity['corpus'])
        index_ids = [r['id'] for r in identity['source_index']['chunks']]
        if len(index_ids) != len(set(index_ids)) or set(index_ids) != set(bindings):
            raise ValueError('parent index/corpus chunk sets differ')
        _validate_active_gold(suite['tasks'], identity['corpus'])
        for task, row in zip(suite['tasks'], rows, strict=True):
            if row['error'] is not None or row['status'] not in ('measured', 'not_applicable'):
                raise ValueError('parent contains missing/failed diagnostics')
            telemetry = row['telemetry']
            if not isinstance(telemetry['excluded_counts'], dict):
                raise ValueError('parent exclusion telemetry incomplete')
            if (not set(telemetry['excluded_counts']) <= EXCLUSION_REASONS
                    or any(type(n) is not int or n < 0
                           for n in telemetry['excluded_counts'].values())):
                raise ValueError('unsupported parent exclusion telemetry')
            for name in ('wall_ms', 'embed_ms', 'retrieval_ms', 'context_ms', 'context_tokens'):
                if not _finite(telemetry[name]):
                    raise ValueError('parent telemetry incomplete')
            if type(telemetry['context_tokens']) is not int or telemetry['context_tokens'] > 5000:
                raise ValueError('parent context over budget')
            identities = {}
            for phase in PHASES:
                values = telemetry[phase]
                count_key = 'candidate_count' if phase == 'candidates' else phase + '_count'
                if not isinstance(values, list) or type(telemetry[count_key]) is not int:
                    raise ValueError('parent phase telemetry incomplete')
                if len(values) != telemetry[count_key]:
                    raise ValueError('parent telemetry count differs')
                for value in values:
                    _source_binding(value, bindings)
                    if value.get('excluded_because') not in EXCLUSION_REASONS | {None}:
                        raise ValueError('unsupported parent exclusion reason')
                    if 'parent_rank' in value and value['parent_rank'] != value['rank']:
                        raise ValueError('parent rank override differs from recorded rank')
                    key = tuple(value[k] for k in IDENTITY_FIELDS)
                    if value['chunk_id'] in identities and identities[value['chunk_id']] != key:
                        raise ValueError('conflicting duplicate parent identity')
                    identities[value['chunk_id']] = key
                    score = value.get('rerank_score')
                    if type(score) not in (int, float) or not math.isfinite(score):
                        raise ValueError('cached rerank score missing')
                if phase != 'candidates' and (len(values) > 8 or len({
                        v['chunk_id'] for v in values}) != len(values)):
                    raise ValueError('parent context/admission must have at most 8 unique IDs')
            candidate_ids = {v['chunk_id'] for v in telemetry['candidates']}
            if any(v['chunk_id'] not in candidate_ids for phase in ('admitted', 'context')
                   for v in telemetry[phase]):
                raise ValueError('legacy evidence outside cached candidates')
            # References are audited independently from the policy API below.
            expected_diagnostic = diagnose(task, {})
            for key in ('reference_count', 'reference_fingerprint', 'basis', 'category', 'eligible'):
                if row[key] != expected_diagnostic[key]:
                    raise ValueError('parent task/reference identity differs')
        return rows
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('malformed parent artifact') from exc


def rebind_rows(db, user, rows, corpus):
    """Reauthorize each recorded occurrence; never repair a conflicting identity."""
    bindings = corpus_bindings(corpus)
    rebound = []
    for row in rows:
        doc_record, chunk_record = _source_binding(row, bindings)
        chunk, version, doc = require_chunk(db, user, row['chunk_id'], active_only=True)
        if (chunk.id, doc.id, version.id, version.content_hash) != tuple(
                row[k] for k in IDENTITY_FIELDS):
            raise ValueError('reauthorized source identity differs')
        if (digest(chunk.text) != chunk_record['text_sha256']
                or digest([doc.title, doc.metadata_json]) != doc_record['document_sha256']
                or digest(chunk.locator) != chunk_record['locator_sha256']):
            raise ValueError('reauthorized text/title/metadata differs')
        for key, actual in (('text', chunk.text), ('title', doc.title),
                            ('metadata', doc.metadata_json), ('locator', chunk.locator)):
            if key in row and row[key] != actual:
                raise ValueError('in-memory evidence identity differs')
        clean = {key: deepcopy(row[key]) for key in ROW_FIELDS if key in row}
        clean.update(text=chunk.text, title=doc.title, metadata=deepcopy(doc.metadata_json),
                     locator=deepcopy(chunk.locator))
        clean.setdefault('parent_rank', row['rank'])
        rebound.append(clean)
    return rebound


def evidence_tokens(rows):
    return sum(token_count(row['text']) + token_count(row['title']) + 100 for row in rows)


def validate_context(rows, tokens):
    if (len(rows) > 8 or len({r['chunk_id'] for r in rows}) != len(rows)
            or type(tokens) is not int or tokens != evidence_tokens(rows) or tokens > 5000):
        raise ValueError('context IDs or exact token budget invalid')


def rehydrate_task(db, user, parent_row, corpus):
    phases = {phase: rebind_rows(db, user, parent_row['telemetry'][phase], corpus)
              for phase in PHASES}
    validate_context(phases['context'], parent_row['telemetry']['context_tokens'])
    return phases


def verify_legacy(task, parent_row, phases):
    measured = diagnose(task, phases)
    if any(parent_row.get(key) != value for key, value in measured.items()):
        raise ValueError('legacy diagnostic reproduction differs')
    return measured


def policy_caps(db, user, question):
    groups = route_sources(question, readable_documents(db, user), strict_dates=False)
    diversify = bool(groups) or needs_document_diversity(question) or multi_source_intent(question)
    return {'limit': 8, 'token_budget': 5000,
            'document_quota': 2 if diversify else None,
            'source_quota': max(2, (8 - 2) // len(groups)) if groups else None}


def select_profile(question, candidates, profile, caps, *, seed=None, legacy=None):
    """Gold-free policy boundary, reused by later replay/supplement experiments."""
    if profile == 'legacy_rerank':
        if legacy is None:
            raise ValueError('legacy context required')
        return SelectionResult(deepcopy(legacy), evidence_tokens(legacy), {
            'method': 'recorded_parent_context', 'selected_ids': [r['chunk_id'] for r in legacy],
            'context_tokens': evidence_tokens(legacy)})
    if profile == 'complementary_strict':
        return select_complementary(question, candidates, **caps)
    if profile not in QUOTA_PROFILES or seed is None:
        raise ValueError('unknown replacement profile or missing strict seed')
    relaxed = dict(caps)
    if profile in ('replace_doc', 'replace_both'):
        relaxed['document_quota'] = None
    if profile in ('replace_source', 'replace_both'):
        relaxed['source_quota'] = None
    return replace_weak_evidence(question, candidates, seed.evidence, **relaxed, max_replacements=8)


def public_telemetry(rows):
    # The parent's lanes are already hashed. _telemetry handles score/identity
    # fields; restoring both fields avoids rehashing hashes or inventing lanes.
    for row in rows:
        validate_telemetry_row(row)
    clean = _telemetry(rows)
    for output, row in zip(clean, rows, strict=True):
        output.update(lane_type=row['lane_type'], lane_sha256=row['lane_sha256'],
                      parent_rank=row.get('parent_rank', row['rank']))
    return clean


def public_trace(trace):
    """Explicit nested whitelist; no question, facet or source text can escape."""
    fields = ('method', 'coverage_type', 'facet_sha256', 'facet_count', 'candidate_count',
              'selected_ids', 'required_ids', 'context_tokens', 'objective',
              'accepted_replacements', 'max_replacements', 'objective_before', 'objective_after')
    result = {key: deepcopy(trace[key]) for key in fields if key in trace}
    if 'caps' in trace:
        result['caps'] = {k: trace['caps'][k] for k in
            ('limit', 'token_budget', 'document_quota', 'source_quota')}
    if 'cap_flags' in trace:
        result['cap_flags'] = {k: trace['cap_flags'][k] for k in ('limit_full',
            'token_budget_full', 'document_quota_active', 'source_quota_active')}
    if 'rejected' in trace:
        result['rejected'] = [{'chunk_id': r['chunk_id'], 'reasons': list(r['reasons'])}
                              for r in trace['rejected']]
    if 'changes' in trace:
        result['changes'] = [{k: deepcopy(r[k]) for k in
            ('added_ids', 'removed_ids', 'objective', 'context_tokens')} for r in trace['changes']]
    return result


def context_redundancy(rows):
    """Same content Jaccard as the selector, also measurable for legacy context."""
    terms = [_grams(row['text']) for row in rows]
    values = [len(left & right) / len(left | right) if left | right else 0.0
              for index, left in enumerate(terms) for right in terms[index + 1:]]
    return sum(values) / len(values) if values else 0.0


def replay_row(db, user, task, parent_row, phases, result, corpus, *, legacy=False, elapsed_ms=None):
    selected = rebind_rows(db, user, result.evidence, corpus)
    validate_context(selected, result.context_tokens)
    if legacy:
        admitted, context = phases['admitted'], phases['context']
    else:
        admitted = [dict(row, rank=i, admitted=True, excluded_because=None)
                    for i, row in enumerate(selected, 1)]
        context = admitted
    current = dict(candidates=phases['candidates'], admitted=admitted, context=context)
    telemetry = {'wall_ms': None, 'embed_ms': None, 'retrieval_ms': None,
        'context_ms': None, 'selection_ms': elapsed_ms, 'context_tokens': result.context_tokens,
        'candidate_count': len(current['candidates']), 'admitted_count': len(admitted),
        'context_count': len(context), 'excluded_counts': deepcopy(parent_row['telemetry']['excluded_counts']),
        **{phase: public_telemetry(rows) for phase, rows in current.items()},
        'cached_parent_latency_ms': {k: parent_row['telemetry'][k] for k in
            ('wall_ms', 'embed_ms', 'retrieval_ms', 'context_ms')}}
    trace = public_trace(result.trace)
    statistics = {'tokens': result.context_tokens, 'count': len(context),
        'mean_pair_jaccard': context_redundancy(context),
        'accepted_swaps': trace.get('accepted_replacements', 0)}
    return diagnose(task, current) | dict(telemetry=telemetry, error=None,
        selection_trace=trace, selection_statistics=statistics)


def _configuration(parent):
    return next(p['effective_configuration'] for p in parent['identity']['profiles']
                if p['name'] == PARENT_PROFILE)


def effective_configuration(parent):
    """Freeze the explicit retrieval settings as well as coverage's global flag."""
    cfg = configuration(Settings(_env_file=None, **_configuration(parent)), PARENT_PROFILE)
    keys = set(CONFIG_KEYS) | {'min_similarity'} | {
        name for name, field in Settings.model_fields.items() if field.annotation is bool}
    return {key: getattr(cfg, key) for key in sorted(keys)}


def selection_summary(rows):
    measured = [r for r in rows if r.get('selection_statistics') is not None]
    result = {'measured_tasks': len(measured), 'missing_tasks': len(rows) - len(measured)}
    for key in ('tokens', 'count', 'mean_pair_jaccard', 'accepted_swaps'):
        values = [r['selection_statistics'][key] for r in measured
                  if r['selection_statistics'][key] is not None]
        result[key] = {'measured_tasks': len(values),
                       'mean': sum(values) / len(values) if values else None}
    loss_rows = [r for r in rows if r['losses'] is not None]
    result['losses'] = {key: sum(r['losses'][key] for r in loss_rows) if loss_rows else None
        for key in ('candidate_missing', 'admission_loss', 'context_loss')}
    result['loss_measured_tasks'] = len(loss_rows)
    return result


def execute(args, suite, selection, tasks, parent, reservation):
    from app.clients import Search
    from app.db import SessionLocal

    profiles = stage_profiles(args.stage)
    run_id = reservation['run_id']
    checkpoint = RUNTIME / run_id / 'checkpoint.json'
    report = {'version': 'evidence-selection-dev-v1', 'stage': args.stage, 'split': 'dev',
        'subset': False, 'status': 'failed', 'valid': False, 'run_id': run_id,
        'strict_task_success': None, 'semantic_recall': None,
        'selected_task_ids': [t['id'] for t in tasks],
        'rows': {p: [] for p in profiles}, 'identity': None, 'error': None}
    stage = 'initial_identity'
    try:
        validate_suite(suite, selection, ROOT)
        parent_bytes = Path(args.parent).read_bytes()
        if json.loads(parent_bytes) != parent:
            raise ValueError('parent bytes/object differ before execution')
        parent_rows = validate_parent(parent, suite, selection,
            Path(args.suite).read_bytes(), SELECTION.read_bytes())
        if tasks != suite['tasks']:
            raise ValueError('subsets not supported')
        frozen_cfg = effective_configuration(parent)
        coverage_flag = settings().answer_quality_enabled
        code = code_fingerprints()
        source = Search(index=parent['identity']['source_index']['index'])
        with SessionLocal() as db, checkpoint_lock(checkpoint):
            db.execute(text('SET TRANSACTION READ ONLY'))
            users, records, scope = active_scope(db)
            original_index = index_ledger(source, records)
            if (scope != parent['identity']['corpus']
                    or digest(scope) != parent['identity']['corpus_sha256']
                    or original_index != parent['identity']['source_index']
                    or digest(original_index) != parent['identity']['source_index_sha256']):
                raise ValueError('live source corpus/index differs from cache')
            # Every cached phase is reauthorized and every legacy diagnostic
            # reproduced for all tasks before the first comparative policy call.
            stage = 'legacy_reproduction'
            for task, row in zip(tasks, parent_rows, strict=True):
                phases = rehydrate_task(db, users[task['user']], row, scope)
                verify_legacy(task, row, phases)
            identity = {k: deepcopy(parent['identity'][k]) for k in (
                'suite_sha256', 'selection_sha256', 'suite_bytes_sha256', 'selection_bytes_sha256',
                'corpus', 'corpus_sha256', 'source_index', 'source_index_sha256')}
            identity.update(method='cached_candidate_replay_v1', stage=args.stage,
                profiles=list(profiles), selected_task_ids=report['selected_task_ids'],
                profile_declarations=profile_declarations(profiles),
                parent_bytes_sha256=sha(parent_bytes), parent_sha256=digest(parent),
                parent_profile=PARENT_PROFILE,
                cached_models=cached_model_identity(parent['identity']['models'], _configuration(parent)),
                effective_retrieval_configuration=frozen_cfg,
                answer_quality_enabled=coverage_flag, code=code,
                runtime={'python': platform.python_version(), 'platform': platform.platform()},
                policy_configuration={'limit': 8, 'token_budget': 5000, 'strict_dates': False,
                    'document_quota': '2_if_original_diversify_else_null',
                    'source_quota': 'max(2,(8-2)//route_group_count)_else_null',
                    'required_ids': [], 'max_replacements': 8,
                    'replacement_seed': 'complementary_strict',
                    'objective': 'facets,terms,negative_jaccard,negative_parent_rank'})
            report['identity'] = identity
            atomic_json(checkpoint.with_name('replay-identity.json'), identity)
            store = CheckpointStore(checkpoint, identity)
            stage = 'selection'
            for task, parent_row in zip(tasks, parent_rows, strict=True):
                user = users[task['user']]
                seed, seed_error = None, None
                for profile in profiles:
                    store.begin(profile, task['id'], uuid.uuid4().hex)
                    try:
                        phases = rehydrate_task(db, user, parent_row, scope)
                        verify_legacy(task, parent_row, phases)
                        # The boundary receives question + bound candidates only.
                        caps = policy_caps(db, user, task['goal'])
                        if profile.startswith('replace_') and seed_error:
                            raise ValueError('strict greedy seed unavailable')
                        started = time.monotonic()
                        result = select_profile(task['goal'], phases['candidates'], profile, caps,
                            seed=seed, legacy=phases['context'] if profile == 'legacy_rerank' else None)
                        elapsed_ms = (time.monotonic() - started) * 1000
                        if profile == 'complementary_strict':
                            seed = result
                        row = replay_row(db, user, task, parent_row, phases, result, scope,
                            legacy=profile == 'legacy_rerank', elapsed_ms=elapsed_ms)
                    except Exception as exc:
                        if profile == 'complementary_strict':
                            seed_error = type(exc).__name__
                        row = missing_row(task, {'type': type(exc).__name__, 'stage': 'selection'})
                    store.finish(profile, task['id'], row)
                    report['rows'][profile].append(row)
            stage = 'drift_check'
            _, final_records, final_scope = active_scope(db)
            final_index = index_ledger(source, final_records)
            drift = {'parent': sha(Path(args.parent).read_bytes()) != identity['parent_bytes_sha256'],
                'code': code_fingerprints() != code, 'corpus': final_scope != scope,
                'source_index': final_index != original_index,
                'suite': sha(Path(args.suite).read_bytes()) != identity['suite_bytes_sha256'],
                'selection': sha(SELECTION.read_bytes()) != identity['selection_bytes_sha256'],
                'coverage_configuration': settings().answer_quality_enabled != coverage_flag,
                'retrieval_configuration': effective_configuration(parent) != frozen_cfg}
            try:
                validate_suite(suite, selection, ROOT)
                drift['raw_sources'] = False
            except (ValueError, OSError):
                drift['raw_sources'] = True
            report['drift'] = drift
            errors = any(r['error'] is not None for rows in report['rows'].values() for r in rows)
            report.update(valid=not any(drift.values()) and not errors,
                status='invalid_drift' if any(drift.values()) else 'failed' if errors else 'complete')
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'stage': stage}
        report['status'] = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        report['valid'] = False
    finally:
        for profile, rows in report['rows'].items():
            by_id = {r['task_id']: r for r in rows}
            report['rows'][profile] = [by_id.get(t['id']) or missing_row(t, report['error']) for t in tasks]
        report['summaries'] = {p: summarize(rows) for p, rows in report['rows'].items()}
        report['null_evidence_counts'] = {p: null_evidence_counts(rows) for p, rows in report['rows'].items()}
        report['selection_summaries'] = {p: selection_summary(rows)
                                        for p, rows in report['rows'].items()}
        report['execution_errors'] = {p: {'tasks': sum(r['error'] is not None for r in rows),
            'not_applicable_tasks': sum(r['error'] is not None and not r['eligible'] for r in rows),
            'by_stage': dict(Counter(r['error']['stage'] for r in rows if r['error']))}
            for p, rows in report['rows'].items()}
        if args.stage == 'selection':
            report['pairs'] = {'complementary_strict': pair_profiles(
                report['rows']['legacy_rerank'], report['rows']['complementary_strict'])}
        else:
            report['pairs'] = {p: {'against_replace_strict': pair_profiles(
                report['rows']['replace_strict'], rows), 'against_greedy': pair_profiles(
                report['rows']['complementary_strict'], rows)}
                for p, rows in report['rows'].items()}
        _publish(args.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('selection', 'quotas'), required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--suite', default=str(ROOT / '.runtime/benchmark-package/v1/dev/tasks.json'))
    args = parser.parse_args(argv)
    suite = json.loads(Path(args.suite).read_bytes())
    selection = json.loads(SELECTION.read_bytes())
    tasks = validate_suite(suite, selection, ROOT)
    parent = json.loads(Path(args.parent).read_bytes())
    validate_parent(parent, suite, selection, Path(args.suite).read_bytes(), SELECTION.read_bytes())
    reservation = reserve_output(args.output)
    report = execute(args, suite, selection, tasks, parent, reservation)
    print(json.dumps({'output': str(Path(args.output).resolve()),
        'status': report['status'], 'valid': report['valid']}))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
