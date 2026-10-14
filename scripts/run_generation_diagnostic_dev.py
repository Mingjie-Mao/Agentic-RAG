"""2x2 generation-layer diagnostic on Dev only: generator x context.

Cells: A = 7B + current context, B = 7B + gold context, C = candidate + current
context, D = candidate + gold context. The gold context is a diagnostic, not a
method: it places evaluator-only reference chunks first, then fills the same
8-chunk / 5000-token budget from the current context. Only the question and
reauthorized evidence chunks reach a model; answers and fact labels never do.

Diagnostic tier: inputs, model digests and the decision rule are hashed into a
registration before the first dispatch; raw answers and third-party text stay in
ignored ``.runtime``. Public output carries IDs, statuses, literal diagnostics
and costs only. Core, Security, External and Dynamic Test are never read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from sqlalchemy import select

from app.config import settings
from scripts.benchmark_package_scoring import provisional_score
from scripts.benchmark_runtime import atomic_json
from scripts.dev_evidence_annotations import references_for_task
from scripts.retrieval_quality_metrics import _character_grams, phase_coverage
from scripts.run_evidence_selection_dev import evidence_tokens
from scripts.run_external_answers_dev import generate_answer
from scripts.run_slot_generation_dev import GenerationModels

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime/generation-diagnostic-20261013'
DEV47 = ROOT / '.runtime/benchmark-package/v1/dev/tasks.json'
OVERLAY = ROOT / '.runtime/dev-evidence-repair-20261009/accepted-annotation-v1.json'
DEV47_CONTEXTS = ROOT / '.runtime/evidence-first-20261011/policy-contexts-20261012-v1'
DEV47_PROFILE = 'baseline'
DYNAMIC = ROOT / 'benchmarks/enterprise_rag/v1/dynamic/dev/tasks.json'
DYNAMIC_RUN = ROOT / 'artifacts/dynamic-dev-six-arms-20261009-v2.json'
DYNAMIC_ARM = 'rag'
FORBIDDEN_INPUTS = ('core', 'security', 'external/tasks', 'dynamic/test')

MODELS = {
    'base': {'model': 'qwen2.5:7b-instruct', 'url': 'http://127.0.0.1:11436', 'path': 'production_default'},
    'candidate': {'model': 'qwen3.5:9b', 'url': 'http://127.0.0.1:11437', 'path': 'chat_backend_think_off'},
}
CELLS = {'A': ('base', 'current'), 'B': ('base', 'gold'),
         'C': ('candidate', 'current'), 'D': ('candidate', 'gold')}
MAX_ROWS, MAX_TOKENS = 8, 5000
PROTOCOL = 'shared_contract'

DECISION_RULE = {
    'primary_comparison': 'C_vs_A_on_reviewed_dev47_primary_tasks',
    'switch_default_generator_only_if_all': [
        'net_strict_success_gain_gt_0_after_semantic_review',
        'zero_tasks_correct_in_A_wrong_in_C',
        'zero_required_fact_loss_C_vs_A',
        'no_safety_regression_forbidden_output_or_unanswerable_answered',
        'C_p95_latency_le_1.5x_A_p95_latency_same_run',
    ],
    'otherwise': 'keep_qwen2.5_7b_default',
    'descriptive_only': ['B_minus_A_selection_layer_size', 'D_minus_C', 'dynamic_dev_all_cells',
                         'literal_provisional_scores'],
    'gold_context_status': 'diagnostic_not_upper_bound_not_a_method',
    'review': 'blind_model_review_after_all_dispatch_reviewer_identity_recorded_independent_false',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_tasks(path):
    obj = json.loads(path.read_text())
    return obj['tasks'] if isinstance(obj, dict) else obj


def model_digest(spec):
    tags = httpx.get(spec['url'] + '/api/tags', timeout=10, trust_env=False).json()['models']
    found = [m['digest'] for m in tags if m['name'] == spec['model']]
    if len(found) != 1:
        raise ValueError(f"model {spec['model']} not served at {spec['url']}")
    return found[0]


def inputs():
    paths = [DEV47, OVERLAY, DYNAMIC, DYNAMIC_RUN, Path(__file__)]
    paths += sorted(DEV47_CONTEXTS.glob('*.json'))
    for path in paths:
        if any(part in str(path) for part in FORBIDDEN_INPUTS):
            raise ValueError('held-out split must not be read')
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths}


def registration():
    return {
        'version': 'generation-diagnostic-v1', 'tier': 'diagnostic', 'split': 'dev',
        'suites': {'dev47': 47, 'dynamic_dev': 30},
        'cells': {k: {'generator': g, 'context': c} for k, (g, c) in CELLS.items()},
        'models': {k: dict(v, digest=model_digest(v)) for k, v in MODELS.items()},
        'generation': {'protocol': PROTOCOL, 'limits': 'generate_answer_defaults_generation3_180s',
                       'embedding_calls': 0, 'candidate_request_overrides': {'think': False}},
        'contexts': {'current_dev47': f'policy-contexts-20261012-v1/{DEV47_PROFILE}',
                     'current_dynamic': f'dynamic-dev-six-arms-20261009-v2/{DYNAMIC_ARM}.retrieved_evidence',
                     'gold': 'reference_chunks_first_then_current_fill',
                     'caps': {'rows': MAX_ROWS, 'tokens': MAX_TOKENS},
                     'gold_cap_deviation': 'gold_chunks_bounded_by_tokens_only_row_cap_excess_reported'},
        'order': 'cells_rotated_by_task_index',
        'inputs_sha256': inputs(),
        'decision_rule': DECISION_RULE,
    }


def readable_row(db, user, chunk_id, *, expect=None, rank):
    from app.security import require_chunk
    chunk, version, document = require_chunk(db, user, chunk_id, active_only=False)
    row = {'chunk_id': chunk.id, 'document_id': document.id, 'version_id': version.id,
           'source_sha256': version.content_hash, 'rank': rank, 'title': document.title,
           'text': chunk.text, 'metadata': document.metadata_json, 'locator': chunk.locator}
    if expect is not None:
        same = all(expect.get(k) == row[k] for k in ('chunk_id', 'document_id', 'version_id', 'source_sha256'))
        if not same or ('text' in expect and expect['text'] != row['text']):
            raise ValueError('current context identity or text drifted')
    return row


def version_chunks(db, user, reference):
    from app.models import Chunk, DocumentVersion
    version = db.scalars(select(DocumentVersion).where(
        DocumentVersion.document_id == reference['document_id'],
        DocumentVersion.content_hash == reference['source_sha256'])).first()
    if version is None:
        return []
    ids = db.scalars(select(Chunk.id).where(Chunk.version_id == version.id).order_by(Chunk.ordinal)).all()
    return [readable_row(db, user, cid, rank=i) for i, cid in enumerate(ids, 1)]


QUOTE_COVERAGE = 0.9


def delivers(reference, rows):
    if reference['basis'] == 'gold_evidence_quote':
        # Gold context needs (nearly) the whole quoted span, not the 0.5 proxy.
        wanted = _character_grams(reference['text'])
        present = set().union(*(_character_grams(r['text']) for r in rows))
        return bool(wanted) and len(wanted & present) / len(wanted) >= QUOTE_COVERAGE
    return bool(phase_coverage([reference], rows)['matched_reference_ids'])



def reference_rows(reference, chunks):
    """Smallest contiguous chunk window that delivers the reference (proxy)."""
    for width in range(1, len(chunks) + 1):
        for start in range(len(chunks) - width + 1):
            window = chunks[start:start + width]
            if delivers(reference, window):
                return window
    return None


def gold_context(db, user, references, current):
    chosen, unresolved = [], []
    for reference in references:
        window = reference_rows(reference, version_chunks(db, user, reference))
        if window is None:
            unresolved.append(reference['id'])
            continue
        chosen += [r for r in window if r['chunk_id'] not in {c['chunk_id'] for c in chosen}]
    # Gold chunks keep only the token cap: multi-hop gold documents are split into
    # many tiny chunks, and an 8-row cap would drop required evidence. Current-context
    # fill still stops at 8 rows. Any row-cap excess is reported, never hidden.
    rows, truncated = [], False
    for row in chosen:
        if evidence_tokens(rows + [row]) <= MAX_TOKENS:
            rows.append(row)
        else:
            truncated = True
    for row in current:
        if (row['chunk_id'] not in {r['chunk_id'] for r in rows} and len(rows) < MAX_ROWS
                and evidence_tokens(rows + [row]) <= MAX_TOKENS):
            rows.append(row)
    gold_ids = {r['chunk_id'] for r in chosen}
    rows = [dict(r, rank=i) for i, r in enumerate(rows, 1)]
    meta = {'reference_count': len(references), 'unresolved_reference_ids': unresolved,
            'gold_chunks': len(chosen), 'gold_chunks_in_context': sum(r['chunk_id'] in gold_ids for r in rows),
            'gold_truncated': truncated, 'row_cap_exceeded': len(rows) > MAX_ROWS,
            'references_delivered': (sum(delivers(r, rows) for r in references) if references else None)}
    return rows, meta


def references(task, record):
    if record is not None and record['applicability'] in ('active_native', 'active_upstream'):
        return references_for_task(task, record)
    return [{'id': g['id'], 'document_id': g['document_id'], 'source_sha256': g['source_sha256'],
             'text': g['quote'], 'basis': 'gold_evidence_quote'} for g in task.get('gold_evidence', [])]


def candidate_backend(models, spec):
    def chat(body):
        return models._post('/api/chat', {**body, 'model': spec['model'], 'think': False}, base=spec['url'])
    return chat


class TaggedModels(GenerationModels):
    """Record which model and endpoint each physical call actually used."""
    def _post(self, path, body, base=None):
        count = len(self.events)
        try:
            return super()._post(path, body, base=base)
        finally:
            for event in self.events[count:]:
                event['model'] = body.get('model')
                event['endpoint'] = base or settings().ollama_url


def run_cell(task, user, evidence, generator, cfg):
    models = TaggedModels()
    if MODELS[generator]['path'] != 'production_default':
        models.chat_backend = candidate_backend(models, MODELS[generator])
    started, usage, error = time.monotonic(), None, None
    payload = {'status': 'execution_failed', 'claims': [], 'citations': [], 'verdict': None}
    try:
        payload, usage = generate_answer(task['goal'], evidence, models=models, cfg=cfg,
            check_conflict=user.tenant_id not in cfg.conflict_check_disabled_tenants,
            protocol=PROTOCOL, enforce_zero_embedding=True)
    except Exception as exc:  # recorded failure stays in the denominator
        error = type(exc).__name__
    latency = (time.monotonic() - started) * 1000
    used = sorted({(e.get('model'), e.get('endpoint')) for e in models.events}, key=str)
    expected = MODELS[generator]
    wrong_model = [m for m, _ in used if m != expected['model']]
    return {'status': payload['status'], 'answer_payload': payload, 'latency_ms': latency,
            'execution_error_kind': error, 'generation_usage': usage,
            'physical_calls': len(models.events),
            'prompt_tokens': sum(e.get('prompt_tokens') or 0 for e in models.events),
            'completion_tokens': sum(e.get('completion_tokens') or 0 for e in models.events),
            'models_used': [list(u) for u in used], 'model_mismatch': bool(wrong_model)}


def build_tasks(db):
    from app.models import User
    overlay = {r['task_id']: r for r in json.loads(OVERLAY.read_text())['records']}
    dynamic_rows = {r['task_id']: r for r in json.loads(DYNAMIC_RUN.read_text())['results'][DYNAMIC_ARM]}
    plan = []
    for task in load_tasks(DEV47):
        context = json.loads((DEV47_CONTEXTS / f"{task['id']}.json").read_text())
        if context['task_id'] != task['id'] or context['question'] != task['goal']:
            raise ValueError('Dev47 context/task mismatch')
        plan.append(('dev47', task, context['contexts'][DEV47_PROFILE], overlay.get(task['id'])))
    for task in load_tasks(DYNAMIC):
        plan.append(('dynamic_dev', task, dynamic_rows[task['id']]['retrieved_evidence'], None))
    built = []
    for suite, task, rows, record in plan:
        user = db.get(User, task['user'])
        current = [readable_row(db, user, r['chunk_id'], expect=r, rank=i) for i, r in enumerate(rows, 1)]
        refs = references(task, record)
        gold, meta = gold_context(db, user, refs, current)
        meta['applicability'] = record['applicability'] if record else 'dynamic_gold_evidence'
        built.append({'suite': suite, 'task': task, 'user': user, 'current': current,
                      'gold': gold, 'gold_meta': meta,
                      'current_tokens': evidence_tokens(current), 'gold_tokens': evidence_tokens(gold)})
    return built


def public_view(raw, built):
    rows = []
    for item in built:
        tid = item['task']['id']
        cells = {}
        for cell in CELLS:
            r = raw['cells'].get(f'{tid}:{cell}')
            if r is None:
                cells[cell] = None
                continue
            cells[cell] = {k: r.get(k) for k in (
                'status', 'latency_ms', 'execution_error_kind', 'physical_calls', 'prompt_tokens',
                'completion_tokens', 'model_mismatch', 'literal_answer_diagnostic', 'facts', 'status_ok',
                'forbidden_present', 'strict_task_success')}
        rows.append({'task_id': tid, 'suite': item['suite'], 'category': item['task'].get('category'),
                     'dynamic_type': item['task'].get('dynamic_type'),
                     'context': {'current_rows': len(item['current']), 'current_tokens': item['current_tokens'],
                                 'gold_rows': len(item['gold']), 'gold_tokens': item['gold_tokens'],
                                 **item['gold_meta']},
                     'cells': cells})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='v1')
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='smoke: first N tasks only, separate run id')
    parser.add_argument('--tasks', default=None, help='smoke: comma-separated task IDs, separate run id')
    parser.add_argument('--cells', default='ABCD', help='smoke: subset of cells to run')
    args = parser.parse_args()
    from app.db import SessionLocal

    run_dir = RUNTIME / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    smoke = bool(args.limit or args.tasks or args.cells != 'ABCD')
    reg_path = (run_dir / 'registration.json' if smoke else
                ROOT / f'benchmarks/enterprise_rag/v1/dev/generation-diagnostic-20261013/registration-{args.run}.json')
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        if reg['inputs_sha256'] != inputs() or reg['models'] != registration()['models']:
            raise SystemExit('Registered inputs, code or model digests changed; use a new run id')
    else:
        reg = registration()
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(reg_path, reg)
        print(json.dumps({'registered': str(reg_path)}), flush=True)
    if args.register_only:
        return
    raw_path = run_dir / 'raw.json'
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {
        'run': args.run, 'registration_sha256': sha(reg_path.read_bytes()), 'cells': {}, 'contexts': {}}
    cfg = settings()
    with SessionLocal() as db:
        built = build_tasks(db)
        if args.limit:
            built = built[:args.limit]
        if args.tasks:
            wanted = set(args.tasks.split(','))
            built = [b for b in built if b['task']['id'] in wanted]
        for item in built:
            raw['contexts'][item['task']['id']] = {'gold_meta': item['gold_meta'],
                'current': [r['chunk_id'] for r in item['current']], 'gold': [r['chunk_id'] for r in item['gold']]}
        atomic_json(raw_path, raw)
        for index, item in enumerate(built):
            order = list(CELLS)[index % 4:] + list(CELLS)[:index % 4]
            for cell in [c for c in order if c in args.cells]:
                key = f"{item['task']['id']}:{cell}"
                if key in raw['cells']:  # one attempt per cell; failures stay in the denominator
                    continue
                generator, context = CELLS[cell]
                evidence = item[context]
                # Reauthorize immediately before dispatch; drift aborts the run.
                for row in evidence:
                    readable_row(db, item['user'], row['chunk_id'], expect=row, rank=row['rank'])
                result = run_cell(item['task'], item['user'], evidence, generator, cfg)
                raw['cells'][key] = {'task_id': item['task']['id'], 'cell': cell, **result}
                atomic_json(raw_path, raw)
                print(json.dumps({'task': item['task']['id'], 'cell': cell, 'status': result['status'],
                                  'error': result['execution_error_kind'],
                                  'latency_s': round(result['latency_ms'] / 1000, 1),
                                  'model_mismatch': result['model_mismatch']}), flush=True)
    # Labels are used only after every dispatch.
    by_id = {item['task']['id']: item['task'] for item in built}
    for key, row in raw['cells'].items():
        task = by_id.get(row['task_id'])
        if task is not None:
            row.update(provisional_score(task, row['answer_payload'], [], 0, row['latency_ms'],
                                         execution_error_kind=row['execution_error_kind'], scenario_events={}))
    atomic_json(raw_path, raw)
    if smoke:
        return
    public = {'version': 'generation-diagnostic-v1', 'tier': 'diagnostic', 'split': 'dev',
              'run': args.run, 'registration': str(reg_path.relative_to(ROOT)),
              'registration_sha256': raw['registration_sha256'], 'private_raw_sha256': sha(raw_path.read_bytes()),
              'semantic_review': 'pending', 'strict_task_success': None,
              'note': 'literal diagnostics only; gold context is a diagnostic, not an upper bound or method',
              'rows': public_view(raw, built)}
    out = ROOT / f'artifacts/dev-generation-diagnostic-20261013-{args.run}.json'
    if out.exists():
        raise SystemExit(f'{out} exists; historical artifacts are never overwritten')
    atomic_json(out, public)
    print(json.dumps({'public': str(out.relative_to(ROOT))}), flush=True)


if __name__ == '__main__':
    main()
