from copy import deepcopy

import pytest

from scripts.benchmark_package_scoring import METHODS, answer_review_packet, assess, report, measured_token_total, validate_reviews
from scripts.run_external_answer_review import project_item
from test_benchmark_package import sample


def full_report(usage_rows):
    task, row, _ = sample()
    tasks, rows = [], []
    for i, usage in enumerate(usage_rows):
        task_i, row_i = deepcopy(task), deepcopy(row)
        task_i['id'] = row_i['task_id'] = str(i)
        row_i['usage'] = usage
        tasks.append(task_i)
        rows.append(row_i)
    return report({'corpus_sha256': 'synthetic', 'results': {m: deepcopy(rows) for m in METHODS}},
                  {'version': 'synthetic', 'split': 'dev', 'tasks': tasks, 'arms': list(METHODS)})


@pytest.mark.parametrize('field', ['fact_verdicts', 'claim_verdicts', 'citation_verdicts'])
def test_unresolved_semantics_are_pending_unless_another_requirement_definitely_fails(field):
    task, row, review = sample()
    if isinstance(review[field], dict):
        review[field]['a'] = 'unclear'
    else:
        review[field][0] = 'unclear'
    assert assess(task, row, review)['answer']['strict_task_success'] is None
    review['condition_correct'] = False
    assert assess(task, row, review)['answer']['strict_task_success'] is False


def test_pending_strict_check_is_distinct_from_missing_review_field():
    task, row, review = sample()
    review['condition_correct'] = None
    assert assess(task, row, review)['answer']['strict_task_success'] is None
    del review['condition_correct']
    with pytest.raises(ValueError):
        assess(task, row, review)


@pytest.mark.parametrize('usage', [
    {'execution_budget': {'calls': {'generation': {'attempted': 1, 'failed': 1}}}},
    {'execution_budget': {'calls': {'generation': {'prompt_tokens': None, 'completion_tokens': None}}}},
    {'total_prompt_tokens': 9},
    {'total_prompt_tokens': 9, 'total_completion_tokens': None},
    {'prompt_tokens': 9, 'completion_tokens': 1, 'unknown_prompt_tokens': 1},
    {'execution_budget': {'calls': {'generation': {'attempted': 1, 'failed': 1,
         'prompt_tokens': 0, 'completion_tokens': 0}},
         'model_call_records': [{'role': 'generation', 'status': 'failed'}]}},
])
def test_missing_counters_never_become_zero(usage):
    tokens = full_report([usage])['summary']['rag']['total_tokens']
    assert tokens['sum'] is None and tokens['observed'] == 0


def test_partial_tokens_are_labelled_subtotal_and_explicit_zero_is_retained():
    tokens = full_report([{'prompt_tokens': 10, 'completion_tokens': 2}, {}])['summary']['rag']['total_tokens']
    assert tokens['sum'] is None and tokens['observed_sum'] == 12 and tokens['observed'] == 1
    tokens = full_report([{'prompt_tokens': 0, 'completion_tokens': 0}])['summary']['rag']['total_tokens']
    assert tokens['sum'] == 0 and tokens['observed'] == 1


def test_new_review_projection_hides_nested_implementation_and_keeps_actual_conclusion():
    task, row, _ = sample()
    row['answer_payload']['verdict'] = {'value': 'no', 'claim_index': 0, 'evidence_ids': ['a'],
                                      'method': 'experimental-X', 'scope': 'generator-Y'}
    row['answer_payload']['citations'][0]['usage'] = {'model': 'generator-Y'}
    row['retrieved_evidence'][0].update(method='experimental-X', metadata={'publication': 'Source', 'model': 'generator-Y'})
    raw, suite = {'corpus_sha256': 'x', 'results': {'rag': [row]}}, {'tasks': [task]}
    packet = answer_review_packet(raw, suite)
    assert packet['protocol'] == 'enterprise-benchmark-review-v3-ai-assisted'
    assert packet['items'][0]['answer']['verdict'] == {
        'value': 'no', 'claim_index': 0, 'evidence_ids': ['a']}
    assert 'generator-Y' not in str(packet['items'][0]['answer'])
    assert 'experimental-X' not in str(project_item(packet['items'][0]))
    assert 'experimental-X' not in str(packet)
    assert packet['items'][0]['retrieved_evidence'][0]['metadata'] == {'publication': 'Source'}
    # Old packets remain reproducible for checking historical judgments.
    legacy = answer_review_packet(raw, suite, protocol='enterprise-benchmark-review-v1')
    assert legacy['items'][0]['answer']['verdict']['method'] == 'experimental-X'


def test_supported_equivalent_answer_does_not_require_literal_match():
    task, row, review = sample()
    row['literal_answer_diagnostic'] = False
    assert assess(task, row, review)['answer']['strict_task_success'] is True


def test_known_failed_call_counters_are_preserved_and_empty_measured_budget_is_zero():
    assert measured_token_total({'execution_budget': {'calls': {}, 'model_call_records': []}}) == 0
    assert measured_token_total({'execution_budget': {
        'calls': {'generation': {'attempted': 1, 'failed': 1, 'prompt_tokens': 3, 'completion_tokens': 1}},
        'model_call_records': [{'role': 'generation', 'status': 'failed', 'prompt_tokens': 3, 'completion_tokens': 1}],
    }}) == 4


def test_uncertain_review_cannot_qualify_a_frozen_headline(tmp_path):
    from scripts.research_review import digest
    task, row, decision = sample()
    suite = {'version': 'synthetic', 'split': 'core', 'tasks': [task], 'arms': list(METHODS)}
    frozen = {'synthetic_test': True}
    freeze = tmp_path / 'fake-freeze.json'
    import json
    freeze.write_text(json.dumps(frozen))
    raw = {'corpus_sha256': 'x', 'freeze': str(freeze),
           'registration': {'suite_sha256': digest(suite), 'freeze_sha256': digest(frozen)},
           'results': {m: [deepcopy(row)] for m in METHODS}}
    packet = answer_review_packet(raw, suite)
    decision['fact_verdicts']['a'] = 'unclear'
    packet['reviews'] = [{**deepcopy(decision), 'id': item['id'], 'reviewer': 'Synthetic reviewer',
                         'reviewer_type': 'human', 'independent': True, 'reason': 'Unresolved synthetic fact'}
                        for item in packet['items']]
    result = report(raw, suite, packet)
    assert result['frozen_registration_verified'] is True
    assert result['headline_eligible'] is False


def test_readiness_never_calls_runtime_models_or_index(monkeypatch):
    from scripts.benchmark_package import freeze_readiness
    import scripts.run_unseen_benchmark as runner
    import scripts.benchmark_package_runtime as runtime
    def forbidden(*args, **kwargs):
        raise AssertionError('Readiness must not query services, index or run a question')
    monkeypatch.setattr(runner, 'configuration', forbidden)
    monkeypatch.setattr(runtime, 'indexed_corpus', forbidden)
    monkeypatch.setattr(runtime, 'run_package_arm', forbidden)
    result = freeze_readiness()
    assert result['formal_run_started'] is False and result['freeze_written'] is False
    assert result['formal_core_ready'] is False
    assert result['splits']['core']['count'] == 70
    assert result['human_gold_required'] is False
    assert 'external_independent_human_annotation_and_adjudication_pending' not in result['blockers']
    assert 'formal_environment_index_and_model_digests_not_verified_this_check' in result['blockers']


def test_formal_freeze_inputs_bind_scoring_rules_without_running_tasks(tmp_path, monkeypatch):
    import json
    import scripts.run_unseen_benchmark as runner
    from scripts.benchmark_package_scoring import SCORING_RULES
    from pathlib import Path
    rule = tmp_path / SCORING_RULES
    rule.parent.mkdir(parents=True)
    rule.write_bytes(Path(SCORING_RULES).read_bytes())
    tasks, spec = tmp_path / 'tasks.json', tmp_path / 'documents.json'
    tasks.write_text(json.dumps({'evaluation_protocol': 'enterprise-benchmark-v1'}))
    spec.write_text(json.dumps({'documents': []}))
    (tmp_path / 'pyproject.toml').write_text('')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'TASKS', tasks)
    monkeypatch.setattr(runner, 'SPEC', spec)
    assert runner.inputs()[SCORING_RULES] == runner.sha(rule)


@pytest.mark.parametrize('field', ['reviewer', 'reason', 'model'])
def test_boolean_or_empty_values_are_not_named_review_provenance(field):
    task, row, review = sample()
    packet = answer_review_packet({'corpus_sha256': 'synthetic', 'results': {'rag': [row]}}, {'tasks': [task]})
    packet['reviews'] = [{**review, 'id': packet['items'][0]['id'], 'reviewer': 'Named reviewer',
                          'reviewer_type': 'model', 'model': 'gpt-synthetic', 'independent': False,
                          'reason': 'Synthetic reasoning', field: True}]
    with pytest.raises(ValueError):
        validate_reviews(packet, packet)
