"""Summarize explicit, local AI judgments using the existing benchmark scorer.

This tool makes no model calls, creates no gold labels and runs no benchmark task.
Original questions/answers/reasons stay in the caller's ignored local directory.
"""
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import statistics

from scripts.benchmark_package_scoring import ANSWER_REVIEW_PROTOCOL, assess, conjunction
from scripts.benchmark_runtime import percentile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reference(reference, packet):
    items = reference['items']
    questions = {q['id']: q['question'] for q in packet['items']}
    sources = {s['source_id']: s['text'] for s in packet['sources']}
    if (len(items) != 150 or len({q['id'] for q in items}) != 150
            or {q['id'] for q in items} != set(questions)
            or reference.get('review_kind') != 'AI-assisted'
            or reference.get('human') is not False or reference.get('independent') is not False):
        raise ValueError('exactly 150 original AI-assisted reference records required')
    for q in items:
        if (q['question'] != questions[q['id']] or not isinstance(q.get('reason'), str) or not q['reason']
                or q['reference_state'] not in {'answered', 'unanswerable', 'unresolved'}
                or not isinstance(q['reviewer'].get('model_identity'), str) or not q['reviewer']['model_identity']
                or q['reviewer'].get('reviewer_type') != 'model'
                or q['reviewer'].get('human') is not False or q['reviewer'].get('independent') is not False
                or q['conclusion'] not in {'yes', 'no', 'unclear', 'not_binary'}
                or (q['reference_state'] == 'unresolved' and q['conclusion'] != 'unclear')):
            raise ValueError('reference identity, state or provenance changed')
        for e in q['original_evidence']:
            text = sources[e['source_id']]
            if (text[e['start_char']:e['end_char']] != e['quote'] or not e['quote']
                    or hashlib.sha256(text.encode()).hexdigest() != e['source_sha256']):
                raise ValueError('reference quote/source binding invalid')
    return True


def citation_verdicts(payload, claim_support):
    """Project explicitly reviewed claim support onto its actual citation uses.

    A citation used correctly once cannot excuse its unsupported use by another
    claim. This is bookkeeping, not automatic semantic inference from strings.
    """
    priority = {'supported': 0, 'unclear': 1, 'partial': 2, 'unsupported': 3, 'contradicted': 4}
    if len(claim_support) != len(payload.get('claims', [])):
        raise ValueError('every actual claim needs a saved citation-support judgment')
    verdicts = []
    for c in payload.get('citations', []):
        uses = [v for claim, v in zip(payload['claims'], claim_support, strict=True)
                if c['chunk_id'] in claim.get('evidence_ids', [])]
        verdicts.append(max(uses, key=priority.__getitem__) if uses else 'unsupported')
    return verdicts


def score_saved_answer(task, row, reference, decision):
    task, row = deepcopy(task), deepcopy(row)
    if row.get('not_observed'):
        result = assess(task, row)
        result['answer']['strict_task_success'] = None
        return result
    payload = row['answer_payload']
    state = reference['reference_state']
    task['required_facts'] = [{'id': 'answer'}]
    task['expected_status'] = ('insufficient_evidence' if state == 'unanswerable' else
                               payload['status'] if state == 'unresolved' else 'answered')
    task['answerable'] = state == 'answered' or bool(payload.get('claims'))
    truth, support = decision['claim_truth_verdicts'], decision['claim_support_verdicts']
    if len(truth) != len(payload.get('claims', [])):
        raise ValueError('every actual claim needs a saved factual judgment')
    priority = {'supported': 0, 'unclear': 1, 'partial': 2, 'unsupported': 3, 'contradicted': 4}
    combined = [max((a, b), key=priority.__getitem__) for a, b in zip(truth, support, strict=True)]
    complete = decision['necessary_facts_complete']
    complete_label = 'unclear' if complete is None else 'supported' if complete else 'partial'
    cited = citation_verdicts(payload, support)
    semantic = decision['semantic_correct']
    no_error = semantic
    if 'verdict_body_inconsistent' in decision['failure_tags']:
        no_error = False
    review = {'scoring_protocol': ANSWER_REVIEW_PROTOCOL,
              'fact_verdicts': {'answer': complete_label}, 'claim_verdicts': combined,
              'citation_verdicts': cited, 'no_critical_error': no_error,
              'version_correct': None if semantic is None and 'date_scope' in decision['failure_tags'] else True,
              'condition_correct': None if semantic is None and 'condition_scope' in decision['failure_tags'] else True,
              'citations_supported': conjunction(None if v == 'unclear' else v == 'supported' for v in support),
              'abstention_correct': None if state == 'unresolved' else
                  payload['status'] == 'insufficient_evidence' if state == 'unanswerable' else
                  payload['status'] in {'answered', 'conflict'}}
    result = assess(task, row, review)
    if state == 'unresolved':
        result['answer']['strict_task_success'] = conjunction([result['answer']['strict_task_success'], None])
    result['AI_review'] = decision
    result['adapted_review'] = review
    result['answer']['citation_chain_verdicts'] = support
    return result


def metric(values):
    known = [v for v in values if v is not None]
    return {'value': statistics.mean(known) if known else None, 'correct': sum(v is True for v in known),
            'resolved': len(known), 'unresolved': len(values) - len(known), 'total': len(values)}


def summarize(records):
    """Keep unknowns out of point estimates and expose all denominators/bounds."""
    result = {}
    for method in sorted({r['method'] for r in records}):
        rows = [r for r in records if r['method'] == method]
        strict = [r['assessment']['answer']['strict_task_success'] for r in rows]
        semantic = [r.get('decision', {}).get('semantic_correct') for r in rows]
        complete = [r.get('decision', {}).get('necessary_facts_complete') for r in rows]
        support = [v for r in rows for v in r.get('decision', {}).get('claim_support_verdicts', [])]
        binary = [r for r in rows if r['reference']['conclusion'] in {'yes', 'no'}
                  and r['reference']['reference_state'] == 'answered']
        yn = [None if r.get('not_observed') else
              (r['answer_payload'].get('verdict') or {}).get('value') == r['reference']['conclusion']
              for r in binary]
        distribution = Counter(r['reference']['conclusion'] for r in binary)
        token_values = [r.get('measured_tokens') for r in rows]
        measured = [n for n in token_values if n is not None]
        latencies = [r['latency_ms'] for r in rows if r.get('latency_ms') is not None]
        result[method] = {'positions': len(rows), 'audited_payloads': sum(not r.get('not_observed') for r in rows),
                         'semantic_accuracy': metric(semantic), 'necessary_fact_completeness': metric(complete),
                         'strict_success': metric(strict),
                         'strict_all_position_bounds': [sum(v is True for v in strict) / len(rows),
                            (sum(v is True for v in strict) + sum(v is None for v in strict)) / len(rows)],
                         'claim_citation_chain_support': metric([None if v == 'unclear' else v == 'supported' for v in support]),
                         'yes_no_accuracy': metric(yn), 'binary_reference_distribution': dict(distribution),
                         'majority_class_baseline': max(distribution.values()) / len(binary) if binary else None,
                         'failure_task_counts': dict(Counter(t for r in rows for t in set(r.get('decision', {}).get('failure_tags', [])))),
                         'tokens': {'total': sum(measured) if len(measured) == len(rows) else None,
                                    'observed_subtotal': sum(measured) if measured else None,
                                    'observed': len(measured), 'unknown': len(rows) - len(measured)},
                         'latency_ms': {'p50': statistics.median(latencies) if latencies else None,
                                        'p95': percentile(latencies, .95), 'observed': len(latencies),
                                        'unknown': len(rows) - len(latencies)}}
    return result
