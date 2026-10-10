import json

import pytest

from agent.evidence_supplement import BridgePlan, _NoBridge, _shape
from test_evidence_supplement import Models, plan
import test_evidence_supplement as supplement_tests

run = supplement_tests.run


@pytest.mark.parametrize('change,code', [
    ('dependency', 'second_dependency'), ('id', 'step_ids'),
    ('extract', 'second_extraction'), ('kind', 'first_extraction_kind'),
    ('foreach', 'foreach'), ('history', 'temporal_filter'),
    ('placeholder', 'placeholder_binding'), ('first_placeholder', 'first_query_placeholder')])
def test_shape_rejection_is_specific_and_has_no_model_values(change, code):
    p = plan()
    if change == 'dependency':
        p['steps'][1]['depends_on'] = ['s9']
    elif change == 'id':
        p['steps'][0]['id'] = 's3'
    elif change == 'extract':
        p['steps'][1]['extract'] = p['steps'][0]['extract']
    elif change == 'kind':
        p['steps'][0]['extract']['kind'] = 'list'
    elif change == 'foreach':
        p['steps'][1]['foreach'] = True
    elif change == 'history':
        p['steps'][1]['as_of'] = 'private-date'
    elif change == 'placeholder':
        p['steps'][1]['query'] = '{s1.private} phone'
    else:
        p['steps'][0]['query'] = '{s1.private}'
    with pytest.raises(_NoBridge) as error:
        _shape(BridgePlan.model_validate(p))
    assert code in error.value.codes
    assert 'private' not in json.dumps(error.value.public())


def test_schema_error_keeps_safe_field_path_and_never_unknown_key(run):
    p = plan()
    p['steps'][0]['private-secret-key'] = 'private-value'
    result, _, _, calls = run(models=Models(planning=p))
    assert not calls
    assert result.trace['rejections'][0]['stage'] == 'schema'
    assert 'private' not in json.dumps(result.trace)


def test_json_failure_is_distinct_from_shape_failure(run):
    result, _, _, calls = run(models=Models(planning='not JSON private'))
    assert not calls
    assert result.trace['rejections'][0]['stage'] == 'json'
    assert 'json_invalid' in result.trace['rejections'][0]['codes']


def test_valid_bridge_has_no_rejections_and_budget_is_unchanged(run):
    result, model, allocation, calls = run()
    assert result.trace['rejections'] == []
    assert len(calls) == 2 and len(model.bodies) == 2
    assert allocation.state['calls']['policy']['attempted'] == 2


def test_unbound_literal_reason_is_visible(run):
    result, _, _, calls = run(models=Models(entity='Imaginary'))
    assert not calls
    assert 'literal_source_missing' in result.trace['rejections'][0]['codes']
