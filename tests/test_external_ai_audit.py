from copy import deepcopy
import hashlib
import json

import pytest

from scripts.audit_external_ai_results import citation_verdicts, metric, score_saved_answer, validate_reference
from scripts.benchmark_package import external_ai_review_readiness
from scripts.benchmark_package_scoring import ANSWER_REVIEW_PROTOCOL, answer_review_packet, assess
from test_benchmark_package import sample


def test_claim_support_is_bound_to_each_citation_use():
    payload = {'claims': [{'evidence_ids': ['a']}, {'evidence_ids': ['a']}],
               'citations': [{'chunk_id': 'a'}, {'chunk_id': 'unused'}]}
    assert citation_verdicts(payload, ['supported', 'unsupported']) == ['unsupported', 'unsupported']
    with pytest.raises(ValueError):
        citation_verdicts(payload, ['supported'])


def test_unknown_metrics_have_explicit_denominators():
    assert metric([True, False, None]) == {'value': .5, 'correct': 1, 'resolved': 2,
                                         'unresolved': 1, 'total': 3}
    assert metric([None])['value'] is None


def test_v3_semantic_guard_requires_explicit_audit_but_keeps_binding(monkeypatch):
    import app.qa as qa
    task, row, review = sample()
    monkeypatch.setattr(qa, 'claim_validation_issues', lambda *args: [{'codes': ['number_unit']}])
    monkeypatch.setattr(qa, 'validate_claims', lambda *args: ([], 'insufficient_evidence'))
    review['scoring_protocol'] = ANSWER_REVIEW_PROTOCOL
    assert assess(task, row, review)['answer']['strict_task_success'] is True
    legacy = deepcopy(review)
    legacy['scoring_protocol'] = 'enterprise-benchmark-review-v2'
    assert assess(task, row, legacy)['answer']['strict_task_success'] is False
    review['no_critical_error'] = False
    assert assess(task, row, review)['answer']['strict_task_success'] is False
    review['no_critical_error'] = True
    monkeypatch.setattr(qa, 'claim_validation_issues', lambda *args: [{'codes': ['nonliteral_quote']}])
    assert assess(task, row, review)['answer']['strict_task_success'] is False


def test_unresolved_reference_is_not_automatic_failure_or_success():
    task, row, _ = sample()
    reference = {'reference_state': 'unresolved'}
    decision = {'semantic_correct': True, 'necessary_facts_complete': True,
                'claim_truth_verdicts': ['supported'], 'claim_support_verdicts': ['supported'],
                'failure_tags': []}
    assert score_saved_answer(task, row, reference, decision)['answer']['strict_task_success'] is None
    decision['claim_support_verdicts'] = ['unsupported']
    assert score_saved_answer(task, row, reference, decision)['answer']['strict_task_success'] is False


def test_v3_blinds_implementation_message_and_title_without_changing_v2():
    task, row, _ = sample()
    row['answer_payload']['message'] = 'Agent 找到了资料，但最终答案未通过引用校验。'
    row['answer_payload']['citations'][0]['title'] = 'Synthetic · 当前版本'
    raw, suite = {'corpus_sha256': 'x', 'results': {'rag': [row]}}, {'tasks': [task]}
    old = answer_review_packet(raw, suite, protocol='enterprise-benchmark-review-v2')
    new = answer_review_packet(raw, suite)
    assert old['items'][0]['answer']['citations'][0]['title'].endswith('当前版本')
    assert new['items'][0]['answer']['citations'][0]['title'] == 'Synthetic'
    assert new['items'][0]['answer']['message'] != old['items'][0]['answer']['message']
    assert row['answer_payload']['message'].startswith('Agent')


def synthetic_reference():
    source = 'Original synthetic source.'
    items = [{'id': str(i), 'question': str(i), 'reference_state': 'answered',
              'conclusion': 'not_binary', 'reason': 'Explicit source support',
              'reviewer': {'model_identity': 'synthetic', 'reviewer_type': 'model',
                           'human': False, 'independent': False},
              'original_evidence': [{'source_id': 's', 'quote': source, 'start_char': 0,
                  'end_char': len(source), 'source_sha256': hashlib.sha256(source.encode()).hexdigest()}]}
             for i in range(150)]
    reference = {'items': items, 'review_kind': 'AI-assisted', 'human': False, 'independent': False}
    packet = {'items': items, 'sources': [{'source_id': 's', 'text': source}]}
    return reference, packet


def test_reference_rejects_forged_offsets_and_fake_identity():
    ref, packet = synthetic_reference()
    assert validate_reference(ref, packet)
    ref['items'][0]['original_evidence'][0]['end_char'] -= 1
    with pytest.raises(ValueError):
        validate_reference(ref, packet)
    ref, packet = synthetic_reference()
    ref['items'][0]['reviewer']['model_identity'] = True
    with pytest.raises(ValueError):
        validate_reference(ref, packet)


def test_ai_receipt_requires_hash_chain_and_all_answers(tmp_path):
    ref, _ = synthetic_reference()
    (tmp_path / 'reference.json').write_text(json.dumps(ref))
    (tmp_path / 'audit.json').write_text('preserved audit')
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    policy = {'external_review_path': 'reference.json', 'external_summary_path': 'summary.json',
              'scoring_rules_sha256': 'rules'}
    summary = {'reference_sha256': sha(tmp_path / 'reference.json'), 'unresolved_ids': [],
               'source_first_verified': True, 'answer_audit_complete': True,
               'isolated_cross_review_complete': True, 'answer_positions_reviewed': 300,
               'review_kind': 'AI-assisted', 'human': False, 'independent': False,
               'scoring_rules_sha256': 'rules', 'local_receipts': {'audit.json': sha(tmp_path / 'audit.json')}}
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    assert not external_ai_review_readiness(policy, tmp_path)['blocking_reasons']
    (tmp_path / 'audit.json').write_text('changed')
    assert 'external_AI_audit_receipt_chain_invalid' in external_ai_review_readiness(policy, tmp_path)['blocking_reasons']


def test_freeze_carries_local_audit_receipts_and_rejects_tampering(tmp_path, monkeypatch):
    import scripts.run_unseen_benchmark as runner
    from scripts.benchmark_package_scoring import EVALUATION_POLICY, SCORING_RULES
    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    task = save('tasks.json', {'evaluation_protocol': 'enterprise-benchmark-v1'})
    spec = save('documents.json', {'documents': []})
    save(SCORING_RULES, {})
    receipt = save('.runtime/local-originals.json', {'synthetic': True})
    expected = hashlib.sha256(receipt.read_bytes()).hexdigest()
    save('summary.json', {'local_receipts': {'.runtime/local-originals.json': expected}})
    save('reference.json', {})
    save(EVALUATION_POLICY, {'external_review_path': 'reference.json',
                             'external_summary_path': 'summary.json'})
    (tmp_path / 'pyproject.toml').write_text('')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'TASKS', task)
    monkeypatch.setattr(runner, 'SPEC', spec)
    assert runner.inputs()['.runtime/local-originals.json'] == expected
    receipt.write_text('changed')
    with pytest.raises(ValueError, match='receipt changed'):
        runner.inputs()
