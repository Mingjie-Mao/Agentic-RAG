import json

import pytest

from agent.supplement_proposal import parse_proposal, proposal_schema, program_bridge


def test_independent_has_no_execution_fields():
    p = parse_proposal(json.dumps({'mode': 'independent', 'target_slot': 0, 'query': 'quota Product A'}), 'independent', [0])
    assert p.query == 'quota Product A'
    schema = proposal_schema('independent', [0, 2])
    assert schema['$defs']['FocusProposal']['properties']['target_slot']['enum'] == [0, 2]
    assert 'steps' not in schema['$defs']['FocusProposal']['properties']
    assert parse_proposal('{"mode":"unknown","reason":"no_grounded_query"}', 'independent', [0]).mode == 'unknown'


@pytest.mark.parametrize('extra', ['steps', 'depends_on', 'extract', 'id', 'foreach', 'as_of'])
def test_model_owned_execution_fields_are_rejected(extra):
    with pytest.raises(ValueError):
        parse_proposal(json.dumps({'mode': 'independent', 'target_slot': 0, 'query': 'quota Product A', extra: []}), 'independent', [0])


def test_bridge_structure_is_owned_by_program():
    p = parse_proposal(json.dumps({'mode': 'entity_bridge', 'target_slot': 0,
        'first_query': 'order A carrier', 'entity_description': 'carrier name',
        'followup_prefix': '', 'followup_suffix': ' phone'}), 'unknown', [0])
    plan = program_bridge(p)
    assert plan.steps[0].extract.name == 'entity'
    assert plan.steps[0].extract.kind == 'entity'
    assert plan.steps[1].depends_on == ['s1']
    assert plan.steps[1].query == '{s1.entity} phone'
    assert plan.steps[1].extract is None


@pytest.mark.parametrize('suffix', ['', ' ', '{entity} phone', '{bad} phone', '{x}', 'x' * 81])
def test_bridge_context_is_plain_bounded_text(suffix):
    with pytest.raises(ValueError):
        parse_proposal(json.dumps({'mode': 'entity_bridge', 'target_slot': 0,
            'first_query': 'order carrier', 'entity_description': 'carrier',
            'followup_prefix': '', 'followup_suffix': suffix}), 'unknown', [0])


def test_invalid_target_and_mode_rejected():
    for target in [-1, 1, True]:
        with pytest.raises(ValueError):
            parse_proposal(json.dumps({'mode': 'independent', 'target_slot': target, 'query': 'quota'}), 'independent', [0])
    with pytest.raises(ValueError):
        parse_proposal('{"mode":"entity_bridge"}', 'independent', [0])
    assert parse_proposal('{"mode":"unknown","reason":"no_grounded_query"}', 'unknown', [0]).mode == 'unknown'


def test_schema_is_grounded_in_the_actual_model_prompt():
    from types import SimpleNamespace
    from app.supplement_contract import build_supplement_contract
    from agent.supplement_proposal import make_proposal
    bodies = []
    def post(path, body, **kwargs):
        bodies.append(body)
        return {'message':{'content':'{"mode":"unknown","reason":"no_grounded_query"}'}}
    p = make_proposal(SimpleNamespace(_post=post), build_supplement_contract('A service policy?', []), [], [0])
    payload = json.loads(bodies[0]['messages'][1]['content'])
    assert payload['output_schema'] == bodies[0]['format']
    assert p.reason == 'no_grounded_query'
    assert 'independent' in bodies[0]['messages'][0]['content']
