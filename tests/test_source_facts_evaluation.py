import sys
from pathlib import Path

from pydantic import ValidationError
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from freeze_source_facts_external import near
from run_source_facts_external import gate, summarize
from app.semantic_review import ReviewResult


def rows(arm,answers,*,refusal=False):
    return [{'id':str(index),'arm':arm,'question_type':'comparison_query',
             'answer_correct':value,'status':'insufficient_evidence' if refusal else 'answered',
             'execution_error':None,'latency_ms':1000,'total_prompt_tokens':100,'total_completion_tokens':30}
            for index,value in enumerate(answers)]


def criteria():
    return {'minimum_answer_gain':2,'maximum_latency_ratio':1.15,'maximum_total_token_ratio':1.15}


def test_equal_accuracy_cannot_enable_a_more_expensive_protocol():
    before=rows('legacy',[True,False,True,False])
    after=rows('source_facts',[True,False,True,False])
    for item in after:
        item['latency_ms']=2000
        item['total_completion_tokens']=200
    result=gate({'legacy':summarize(before),'source_facts':summarize(after)},before+after,criteria())
    assert result['decision']=='keep_default_off'
    assert not result['checks']['correct_gain'] and not result['checks']['p95_budget']
    assert not result['checks']['token_budget'] and not result['default_enabled']


def test_accuracy_gain_cannot_hide_increased_refusals_or_execution_errors():
    before=rows('legacy',[False,False,False,False])
    after=rows('source_facts',[True,True,True,False])
    after[-1]['status']='insufficient_evidence'
    after[-1]['execution_error']='DependencyError'
    result=gate({'legacy':summarize(before),'source_facts':summarize(after)},before+after,criteria())
    assert result['checks']['correct_gain']
    assert not result['checks']['no_additional_false_refusals']
    assert not result['checks']['no_execution_errors']
    assert result['decision']=='keep_default_off'


def test_gate_never_compares_different_denominators():
    before=rows('legacy',[True,False])
    after=rows('source_facts',[True])
    result=gate({'legacy':summarize(before),'source_facts':summarize(after)},before+after,criteria())
    assert result['decision']=='incomplete'


def test_null_error_is_reported_as_error_not_a_successful_refusal():
    data=rows('legacy',[False])
    data[0].update(question_type='null_query',status='execution_failed',execution_error='Timeout')
    result=summarize(data)
    assert result['execution_errors']==1 and result['null_correct_refusals']==0


def test_near_duplicates_are_screened_beyond_exact_hashes():
    assert near('Did the two stores open on the same date?', 'Did both stores open on the same date?')>.8
    assert near('Is the opening date consistent?', 'What is the policy response limit?')<.8


def test_semantic_result_requires_every_dimension_and_valid_labels():
    with pytest.raises(ValidationError):
        ReviewResult.model_validate({'entailment':'supported','relevance':'answers'})
    full={key:'not_applicable' for key in ['subject','modality','negation_scope','attribute','time','conclusion']}
    result=ReviewResult.model_validate({'entailment':'unclear','relevance':'unclear','reason':'Ambiguous source',**full})
    assert result.entailment=='unclear'
    with pytest.raises(ValidationError):
        ReviewResult.model_validate({**result.model_dump(),'entailment':'probably_correct'})
