"""Pure Dev-only source annotation compiler; raw text stays in local overlays.

The caller must hash the original draft bytes before parsing. Loading an overlay
requires its exact bytes, so file integrity is separate from parsed-object hashes.
No annotation is installed into a task, retriever, prompt, index or runtime here.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from scripts.benchmark_package_scoring import validate_reviews
from scripts.research_review import digest
from scripts.retrieval_quality_metrics import (
    SOURCE_ATOM, UPSTREAM, diagnose, phase_coverage, summarize, supporting_facts,
)

VERSION = "dev-evidence-annotation-v1"
MANIFEST_VERSION = "dev-evidence-annotation-manifest-v1"
LABELS = {"keep", "reject", "unclear"}
AUDITS = {"correct", "incorrect", "unclear", "not_applicable"}
APPLICABILITY = {"active_native": 26, "active_upstream": 10,
                 "history_not_applicable": 5, "no_references": 6}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def _id(value):
    _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value),
             "invalid identifier")
    return value


def _reason(row):
    _require(isinstance(row.get("reason"), str) and row["reason"].strip(), "reason required")


def _keys(row, required, optional=()):
    _require(isinstance(row, dict) and set(required) <= row.keys()
             and row.keys() <= set(required) | set(optional), "invalid annotation fields")


def _source(root, path, expected_sha):
    _require(isinstance(path, str) and path and not Path(path).is_absolute()
             and ".." not in Path(path).parts, "invalid source path")
    base = Path(root).resolve()
    target = (base / path).resolve()
    _require(target.is_relative_to(base) and target.is_file(), "source path escape or missing")
    data = target.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == _hash(expected_sha), "source hash changed")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("source must be UTF-8") from exc


def _kind(task):
    if task.get("requires_version"):
        return "history_not_applicable"
    if not task.get("gold_evidence"):
        return "no_references"
    return "active_upstream" if "upstream_gold_evidence" in task else "active_native"


def _suite(suite):
    _require(suite.get("split") == "dev", "only original Dev is supported")
    tasks = suite.get("tasks", [])
    _require(len(tasks) == 47 and len({t['id'] for t in tasks}) == 47, "exactly 47 unique Dev tasks required")
    for task in tasks:
        _id(task['id'])
    _require(Counter(_kind(t) for t in tasks) == Counter(APPLICABILITY), "original Dev eligibility changed")
    return tasks


def _provenance(provenance, *, reviewer):
    _require(isinstance(provenance, dict), "review provenance required")
    _require(provenance.get("reviewer_type") == "model" and provenance.get("model") in {"gpt-6-astra", "GPT-6"}
             and provenance.get("human") is False, "named nonhuman model provenance required")
    _require(type(provenance.get("independent_of_retrieval_results")) is bool
             and type(provenance.get("independent_of_authored_annotation")) is bool,
             "annotation independence must be explicitly disclosed")
    _require(provenance.get("role") == ("source_reviewer" if reviewer else "proposer"),
             "annotation role required")
    if provenance['model'] == 'GPT-6':
        _require(provenance.get('model_identity_precision') == 'family_from_session_instructions'
                 and provenance.get('serving_model_id') is None,
                 "unavailable serving identifier must not be fabricated")
    if reviewer:
        _require(provenance.get("model_family_independent") is False,
                 "same-family review must be disclosed")


def _audit(audit, *, proposed):
    _keys(audit, {"label", "reason"}, {"proposed_answer"} if not proposed else ())
    # Draft v2 carries a proposed answer slot, but substitutions are not permitted.
    _require(audit.get("label") in AUDITS, "invalid answer audit label")
    _reason(audit)
    _require(audit.get("proposed_answer") is None, "answer replacement requires separate adjudication")


def _native(task, record, reviewed, root):
    atoms = record['native_atoms']
    reviews = reviewed['native_atom_reviews']
    _require(isinstance(atoms, list) and isinstance(reviews, list), "native annotations must be lists")
    _require([a['id'] for a in atoms] == [a['id'] for a in reviews]
             and len({a['id'] for a in atoms}) == len(atoms), "native atom review IDs/order differ")
    parents = task.get('gold_evidence', [])
    required = {f['id']: f for f in task.get('required_facts', [])}
    covered, output = set(), []
    for atom, review in zip(atoms, reviews):
        _keys(atom, {'id', 'document_id', 'source_path', 'source_sha256', 'start_char', 'end_char',
                     'quote', 'supports_fact_ids', 'reason'}, {'label'})
        _keys(review, {'id', 'label', 'reason'})
        _reason(atom)
        _reason(review)
        label = atom.get('label', 'keep')
        _require(label in LABELS and label == review['label'], "native labels differ; reproposal required")
        start, end = atom['start_char'], atom['end_char']
        _require(type(start) is int and type(end) is int and 0 <= start < end, "invalid character span")
        parent = [p for p in parents if all(p.get(k) == atom[k] for k in
                  ('document_id', 'source_path', 'source_sha256'))]
        _require(len(parent) == 1, "native atom lacks unique original parent binding")
        parent = parent[0]
        text = _source(root, atom['source_path'], atom['source_sha256'])
        _require(end <= len(text) and atom['quote'] == text[start:end]
                 and isinstance(atom['quote'], str) and atom['quote'].strip(), "atom quote/span mismatch")
        _require(not all(re.fullmatch(r'\s*#{1,6}\s+.+', line)
                         for line in atom['quote'].splitlines() if line.strip()),
                 "structural heading is not a content atom")
        locator = parent.get('locator')
        if locator:
            _require(locator['start_char'] <= start < end <= locator['end_char'], "atom outside parent span")
        _require(atom['id'] == 'NA-' + digest([task['id'], atom['document_id'], atom['source_sha256'],
                                             start, end])[:20], "unstable native atom ID")
        facts = atom['supports_fact_ids']
        _require(isinstance(facts, list) and facts and len(set(facts)) == len(facts)
                 and set(facts) <= required.keys(), "unknown or duplicate required fact support")
        _require(all(not required[f].get('document_ids') or atom['document_id'] in required[f]['document_ids']
                     for f in facts), "fact support document differs")
        covered.update(facts)
        output.append({**deepcopy(atom), 'label': label, 'parent_reference_id': parent.get('id'),
                       'parent_binding_sha256': digest(parent)})
    if record['applicability'] == 'active_native':
        _require(bool(atoms) and covered == required.keys(), "complete required fact/operand support required")
    else:
        _require(not atoms, "native atoms only allowed for active native tasks")
    return output


def _upstream(task, record, reviewed, root):
    decisions, reviews = record['upstream_decisions'], reviewed['upstream_reviews']
    _require(isinstance(decisions, list) and isinstance(reviews, list), "upstream annotations must be lists")
    original = supporting_facts(task) if record['applicability'] == 'active_upstream' else []
    _require(len(decisions) == len(original) == len(reviews), "every original upstream reference requires review")
    output = []
    for index, (decision, review, fact) in enumerate(zip(decisions, reviews, original)):
        _keys(decision, {'reference_id', 'original_index', 'label', 'source_sha256', 'reason',
                         'source_quote_found', 'proposed_replacement'})
        _keys(review, {'reference_id', 'label', 'reason'}, {'source_quote_found'})
        _reason(decision)
        _reason(review)
        _require(decision['reference_id'] == fact['id'] == review['reference_id']
                 and type(decision['original_index']) is int and decision['original_index'] == index,
                 "upstream original reference ID/order changed")
        _require(decision['label'] in LABELS and decision['label'] == review['label'],
                 "upstream labels differ; reproposal required")
        _require(decision['source_sha256'] == fact['source_sha256'], "upstream source binding changed")
        _require(decision['source_quote_found'] is None or type(decision['source_quote_found']) is bool,
                 "source quote state must be boolean or null")
        if 'source_quote_found' in review:
            _require(review['source_quote_found'] is decision['source_quote_found'], "source quote states differ")
        _require(decision['proposed_replacement'] is None, "replacement reference prohibited")
        parents = [p for p in task['gold_evidence'] if p.get('document_id') == fact['document_id']
                   and p.get('source_sha256') == fact['source_sha256']]
        _require(bool(parents), "upstream original source binding required")
        for parent in parents:
            _source(root, parent['source_path'], parent['source_sha256'])
        output.append(deepcopy(decision))
    return output


def compile_annotation(suite, draft, review, root, *, draft_sha256, suite_bytes_sha256):
    """Compile explicitly disclosed source review; hashes refer to original bytes.

    ``draft_sha256`` must be computed by the caller from draft bytes before parsing.
    Differences require a new proposal/review; unclear is valid but remains pending.
    """
    tasks = _suite(suite)
    _require(draft.get('version') == 'dev-evidence-annotation-draft-v2' and draft.get('split') == 'dev',
             "unsupported draft")
    _require(review.get('version') == 'dev-evidence-source-review-v2', "unsupported source review")
    _require(draft.get('suite_bytes_sha256') == review.get('suite_bytes_sha256') == _hash(suite_bytes_sha256),
             "suite bytes hash changed")
    _require(review.get('draft_bytes_sha256') == _hash(draft_sha256), "draft bytes hash changed")
    _provenance(draft.get('provenance'), reviewer=False)
    _provenance(review.get('provenance'), reviewer=True)
    records, reviews = draft.get('records', []), review.get('records', [])
    ids = [t['id'] for t in tasks]
    _require(draft.get('task_ids47') == ids and [r['task_id'] for r in records] == ids
             and [r['task_id'] for r in reviews] == ids, "47 original task IDs/order required")
    # Reuse benchmark review identity/provenance validation, with a small local adapter.
    items = [{'id': t['id'], 'task': t, 'proposed_record': r} for t, r in zip(tasks, records)]
    packet = {'protocol': 'enterprise-benchmark-review-v1', 'kind': 'dev_source_annotation',
              'items': items, 'source_sha256': digest(items), 'corpus_sha256': suite_bytes_sha256}
    provenance = review['provenance']
    adapted = {**packet, 'reviews': [
        {'id': r['task_id'], 'reviewer': provenance['model'], 'reviewer_type': provenance['reviewer_type'],
         'model': provenance['model'], 'independent': provenance['independent_of_authored_annotation'],
         'reason': r.get('reason')} for r in reviews]}
    validate_reviews(packet, adapted)
    compiled = []
    for task, record, reviewed in zip(tasks, records, reviews):
        _keys(record, {'task_id', 'input_sha256', 'applicability', 'native_atoms', 'upstream_decisions',
                       'answer_audit'})
        _keys(reviewed, {'task_id', 'input_sha256', 'verdict', 'reason', 'native_atom_reviews',
                         'upstream_reviews', 'answer_audit'})
        _require(record['input_sha256'] == reviewed['input_sha256'] == digest(task), "task input changed")
        _require(record['applicability'] == _kind(task), "original applicability changed")
        _require(reviewed['verdict'] == 'approved', "source review not approved; adjudication required")
        _audit(record['answer_audit'], proposed=False)
        _audit(reviewed['answer_audit'], proposed=False)
        _require(record['answer_audit']['label'] == reviewed['answer_audit']['label']
                 and record['answer_audit'].get('proposed_answer') == reviewed['answer_audit'].get('proposed_answer'),
                 "answer audit differs; reproposal required")
        compiled.append({**deepcopy(record),
                         'original_reference_count': len(supporting_facts(task))
                         if record['applicability'].startswith('active_') else 0,
                         'basis': UPSTREAM if 'upstream_gold_evidence' in task else SOURCE_ATOM,
                         'annotation_input_sha256': digest([record, reviewed]),
                         'native_atoms': _native(task, record, reviewed, root),
                         'upstream_decisions': _upstream(task, record, reviewed, root)})
    _require(sum(len(r['upstream_decisions']) for r in compiled) == 26, "26 original upstream references required")
    return {'version': VERSION, 'split': 'dev', 'suite_bytes_sha256': suite_bytes_sha256,
            'draft_bytes_sha256': draft_sha256, 'review_input_sha256': digest(review),
            'records': compiled, 'proposal': deepcopy(draft), 'review': deepcopy(review)}


def references_for_task(task, record, include_unclear=False):
    """Evaluator-only retained references; runtime must never receive these records."""
    _require(record['task_id'] == task['id'] and record['input_sha256'] == digest(task)
             and record['applicability'] == _kind(task), "annotation/task mismatch")
    labels = {'keep', 'unclear'} if include_unclear else {'keep'}
    if record['applicability'] == 'active_native':
        return [{'id': a['id'], 'document_id': a['document_id'], 'source_sha256': a['source_sha256'],
                 'text': a['quote'], 'basis': SOURCE_ATOM}
                for a in record['native_atoms'] if a['label'] in labels]
    if record['applicability'] == 'active_upstream':
        decisions = record['upstream_decisions']
        facts = supporting_facts(task)
        _require([d['reference_id'] for d in decisions] == [f['id'] for f in facts], "original references changed")
        return [f for f, d in zip(facts, decisions) if d['label'] in labels]
    return []


def _ledger(overlay):
    ledger = []
    for record in overlay['records']:
        native = record['native_atoms']
        upstream = record['upstream_decisions']
        decisions = native or upstream
        counts = Counter(d['label'] for d in decisions)
        pending = counts['unclear'] > 0
        active = record['applicability'].startswith('active_')
        ledger.append({'task_id': _id(record['task_id']), 'input_sha256': _hash(record['input_sha256']),
                       'applicability': record['applicability'],
                       'basis': record['basis'],
                       'original_reference_count': record['original_reference_count'],
                       'revised_reference_count': len(decisions),
                       'accepted_reference_count': counts['keep'], 'rejected_reference_count': counts['reject'],
                       'unclear_reference_count': counts['unclear'], 'annotation_pending': pending,
                       'fullcompletion_pending': active and (pending or not counts['keep']
                                                            or record['answer_audit']['label'] != 'correct'),
                       'answer_audit_label': record['answer_audit']['label'],
                       'references': [{'id': _id(d['id'] if native else d['reference_id']),
                                       'source_sha256': _hash(d['source_sha256']), 'label': d['label']}
                                      for d in decisions]})
    return ledger


def build_manifest(overlay, overlay_bytes_sha256):
    """Construct a strict public whitelist; never recursively copy review metadata."""
    _require(overlay.get('version') == VERSION and overlay.get('split') == 'dev', "unsupported overlay")
    ledger = _ledger(overlay)
    _require(len(ledger) == 47 and len({r['task_id'] for r in ledger}) == 47, "47 unique records required")
    counts = {k: sum(r[k] for r in ledger) for k in ('original_reference_count', 'revised_reference_count', 'accepted_reference_count',
                                                   'rejected_reference_count', 'unclear_reference_count')}
    counts.update(tasks=47, eligible_tasks=sum(r['applicability'].startswith('active_') for r in ledger),
                  annotation_pending_tasks=sum(r['annotation_pending'] for r in ledger),
                  fullcompletion_pending_tasks=sum(r['fullcompletion_pending'] for r in ledger))
    # Explicit enums/model identity prevent accidental disclosure through model/reason fields.
    for record in ledger:
        _require(record['applicability'] in APPLICABILITY and record['answer_audit_label'] in AUDITS
                 and all(r['label'] in LABELS for r in record['references']), "invalid ledger labels")
    _provenance(overlay['review']['provenance'], reviewer=True)
    return {'version': MANIFEST_VERSION, 'split': 'dev', 'overlay_bytes_sha256': _hash(overlay_bytes_sha256),
            'suite_bytes_sha256': _hash(overlay['suite_bytes_sha256']),
            'draft_bytes_sha256': _hash(overlay['draft_bytes_sha256']),
            'review_input_sha256': _hash(overlay['review_input_sha256']),
            'review_provenance': {k: overlay['review']['provenance'][k] for k in (
                'reviewer_type', 'model', 'human', 'independent_of_authored_annotation',
                'independent_of_retrieval_results', 'model_family_independent')},
            'model_identity_precision': ('family_from_session_instructions'
                if overlay['review']['provenance']['model'] == 'GPT-6' else 'configured_model_alias'),
            'counts': counts, 'ledger': ledger}


def validate_annotation(suite, overlay, manifest, root, suite_bytes):
    """Load exact overlay bytes, revalidate sources/reviews, and check public manifest.

    Returns local task records/references plus a text-free public ledger. This does
    not assert semantic gold, final answer correctness or completion of an audit.
    """
    _require(isinstance(overlay, bytes) and isinstance(suite_bytes, bytes), "exact input bytes required")
    try:
        parsed = json.loads(overlay)
        _require(json.loads(suite_bytes) == suite, "suite object differs from supplied bytes")
        expected = compile_annotation(suite, parsed['proposal'], parsed['review'], root,
                                      draft_sha256=parsed['draft_bytes_sha256'],
                                      suite_bytes_sha256=hashlib.sha256(suite_bytes).hexdigest())
        _require(parsed == expected, "compiled overlay changed")
        public = build_manifest(parsed, hashlib.sha256(overlay).hexdigest())
        _require(manifest == public, "manifest fields or overlay bytes changed")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed annotation") from exc
    records = {r['task_id']: r for r in parsed['records']}
    return {'task_records': records, 'references': {
        t['id']: references_for_task(t, records[t['id']]) for t in suite['tasks']},
        'public_ledger': public['ledger']}


def annotation_metadata(record):
    """The same public census used by the manifest and evaluator."""
    return _ledger({'records': [record]})[0]


def _coverage(references, rows):
    if rows is None:
        return None
    if references:
        return phase_coverage(references, rows)
    # Active audited tasks with zero accepted references stay observable/pending,
    # rather than being turned into an answerable zero-reference success.
    return {'status': 'measured', 'total': 0, 'delivered': 0,
            'matched_reference_ids': [], 'all_delivered': None,
            'first_supporting_rank': None, 'first_supporting_ranks': {}}


def diagnose_revised(task, phases, record):
    """Opt-in evaluator; original task/gold and runtime input remain untouched."""
    result = diagnose(task, phases)
    refs = references_for_task(task, record)
    sensitivity = references_for_task(task, record, include_unclear=True)
    meta = annotation_metadata(record)
    result.update(basis=record['basis'], reference_count=len(refs), annotation=meta,
                  reference_fingerprint=digest(['dev-evidence-reviewed-v1', digest(task),
                      record['annotation_input_sha256'], refs, sensitivity]))
    if not result['eligible']:
        result['sensitivity_phases'] = dict.fromkeys(('candidates', 'admitted', 'context'))
        return result
    scored = {p: _coverage(refs, (phases or {}).get(p))
              for p in ('candidates', 'admitted', 'context')}
    for phase in scored.values():
        if phase is not None:
            phase['all_accepted_references_delivered'] = phase['all_delivered']
            if meta['fullcompletion_pending']:
                phase['all_delivered'] = None
    result.update(phases=scored, sensitivity_phases={
        p: _coverage(sensitivity, (phases or {}).get(p))
        for p in ('candidates', 'admitted', 'context')})
    if any(v is None for v in scored.values()):
        result.update(status='error', reason='missing_phase_telemetry', cause=None, losses=None)
        return result
    sets = [set(scored[p]['matched_reference_ids']) for p in ('candidates', 'admitted', 'context')]
    losses = {'candidate_missing': len(refs) - len(sets[0]),
              'admission_loss': len(sets[0] - sets[1]), 'context_loss': len(sets[1] - sets[2])}
    result.update(status='measured', reason=None, losses=losses,
        cause=next((k for k, v in losses.items() if v),
                   'annotation_pending' if meta['fullcompletion_pending'] else 'proxy_complete'))
    return result


def summarize_revised(rows):
    rows = list(rows)
    result = summarize(rows)
    keys = ('original_reference_count', 'revised_reference_count', 'accepted_reference_count',
            'rejected_reference_count', 'unclear_reference_count')
    for basis, summary in result['by_basis'].items():
        group = [r for r in rows if r['basis'] == basis]
        summary['annotation_counts'] = {k: sum(r['annotation'][k] for r in group) for k in keys}
        summary['annotation_counts'].update(
            annotation_pending_tasks=sum(r['annotation']['annotation_pending'] for r in group),
            fullcompletion_pending_tasks=sum(r['annotation']['fullcompletion_pending'] for r in group))
        for phase in ('candidates', 'admitted', 'context'):
            values = [r['sensitivity_phases'][phase] for r in group if r['eligible']]
            measured = [v for v in values if v is not None]
            denominator = sum(v['total'] for v in measured)
            summary.setdefault('sensitivity_including_unclear', {})[phase] = {
                'reference_denominator': denominator, 'delivered': sum(v['delivered'] for v in measured),
                'proxy_coverage': sum(v['delivered'] for v in measured) / denominator if denominator else None,
                'measured_tasks': len(measured), 'missing_tasks': len(values) - len(measured),
                'basis': 'sensitivity_only_not_accepted_gold'}
    return result
