from copy import deepcopy

import pytest

from scripts.benchmark_package_scoring import CHECKS, answer_review_packet, report


def fixture():
    task = dict(id='t', category='ordinary', question='What is the limit?', goal='What is the limit?',
        expected_status='answered', required_facts=[{'id':'limit'}], gold_documents=['doc'],
        expected_behavior={'strict_scoring':False}, answerable=True, scenario_events=[])
    row = dict(task_id='t', run_id='r', status='answered', steps=1, latency_ms=1,
        answer_payload={'status':'answered','claims':[{'text':'Limit is nine units.',
            'evidence_ids':['chunk'],'quotes':['Limit is nine units.']}],
            'citations':[{'document_id':'doc','chunk_id':'chunk','text':'Limit is nine units.'}]},
        retrieved_evidence=[{'document_id':'doc'}], first_retrieval_documents=['doc'],
        trace_complete=True, tool_trace=[], forbidden_present={}, scenario_events={}, execution_error_kind=None)
    suite = dict(version='test-dev', split='dev', tasks=[task], arms=['baseline','treatment'])
    raw = {'corpus_sha256':'corpus','results':{m:[deepcopy(row)] for m in suite['arms']}}
    reviewed = answer_review_packet(raw,suite)
    reviewed['reviews'] = [dict(id=i['id'], reviewer_type='model', reviewer='test-reviewer',
        model='explicit-test-model', independent=False, reason='source-bound fact check',
        fact_verdicts={'limit':'supported'}, retrieved_fact_verdicts={'limit':'supported'},
        citation_verdicts=['supported'], claim_verdicts=['supported'], **dict.fromkeys(CHECKS,True))
        for i in reviewed['items']]
    return suite,raw,reviewed


def test_context_experiment_uses_explicit_comparator_without_claiming_workflow_execution():
    suite,raw,reviewed=fixture()
    result=report(raw,suite,reviewed,baseline_method='baseline')
    assert result['summary']['baseline']['strict_task_success_rate']['value']==1
    assert result['paired']['treatment']['baseline']=='baseline'
    assert result['paired']['treatment']['difference']==0
    assert result['headline_eligible'] is False


def test_pending_annotation_is_null_even_with_positive_model_verdicts():
    suite,raw,reviewed=fixture()
    result=report(raw,suite,reviewed,baseline_method='baseline',pending_task_ids=['t'])
    for name in suite['arms']:
        assert result['details'][name]['t']['answer']['strict_task_success'] is None
        assert result['summary'][name]['strict_task_success_rate']=={'value':None,'applicable':0,'total':1}
    assert result['paired']['treatment']['n']==0
    assert result['paired']['treatment']['difference'] is None


def test_pending_dev_annotation_cannot_exclude_core_test_tasks():
    suite,raw,reviewed=fixture()
    suite['split']='core'
    with pytest.raises(ValueError,match='Dev'):
        report(raw,suite,reviewed,baseline_method='baseline',pending_task_ids=['t'])


def test_unknown_pending_id_and_missing_comparator_fail_closed():
    suite,raw,reviewed=fixture()
    with pytest.raises(ValueError):
        report(raw,suite,reviewed,baseline_method='baseline',pending_task_ids=['unknown'])
    with pytest.raises(ValueError):
        report(raw,suite,reviewed,baseline_method='not_an_arm')


def test_unobserved_answer_and_tokens_remain_null_with_complete_review():
    suite, raw, reviewed = fixture()
    raw['results']['treatment'][0].update(not_observed=True, usage=None)
    result = report(raw, suite, reviewed, baseline_method='baseline')
    assert result['details']['treatment']['t']['answer']['strict_task_success'] is None
    assert result['details']['treatment']['t']['retrieval']['gold_fact_recall'] is None
    assert result['summary']['treatment']['total_tokens']['sum'] is None
    assert result['paired']['treatment']['pending_pairs'] == 1
