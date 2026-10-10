from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from agent.supplement_repair import run_repair
from app.execution_budget import ExecutionBudget
from app.supplement_contract import build_supplement_contract


def row(cid='seed', text='Order A-1 carrier Acme. Phone unknown.'):
    return dict(chunk_id=cid, document_id=cid, version_id=cid+'-v1', source_sha256=cid+'-sha',
                title='Orders', text=text, parent_rank=1, lane_type='global', lane_sha256='global')


class Models:
    def __init__(self, proposal=None, entity='Acme'):
        self.proposal = proposal or {'mode': 'independent', 'target_slot': 0, 'query': 'A-1 carrier phone'}
        self.entity, self.bodies = entity, []
    def _post(self, path, body, **kwargs):
        self.bodies.append(body)
        return {'message': {'content': json.dumps(self.proposal) if len(self.bodies) == 1 else self.entity}}


def invoke(question='What is order A-1 carrier phone?', **kwargs):
    models = kwargs.pop('models', Models())
    seen = []
    def search(args):
        seen.append(args.query)
        return [row('new', 'Acme phone is 555.')]
    allocation = ExecutionBudget(SimpleNamespace(agent_task_timeout_seconds=180,
        agent_policy_max_calls=2, agent_judge_max_calls=0, agent_generation_max_calls=0, agent_judge_token_budget=0))
    result = run_repair(build_supplement_contract(question, []), kwargs.pop('seed', [row()]), [row()],
        models=models, search=kwargs.pop('search', search), budget=allocation,
        reauthorize=kwargs.pop('reauthorize', deepcopy),
        validate_scope=kwargs.pop('validate_scope', lambda original, query: True), **kwargs)
    return result, models, seen, allocation


def test_independent_dispatch_has_no_extraction():
    result, models, seen, allocation = invoke()
    assert result.trace['status'] == 'completed'
    assert len(models.bodies) == 1
    assert seen == ['What is order A-1 carrier phone?', 'A-1 carrier phone']
    assert allocation.state['calls']['search']['attempted'] == 2
    assert result.trace['mode'] == 'independent'


def test_bridge_has_program_structure_and_required_source():
    models = Models({'mode': 'entity_bridge', 'target_slot': 0, 'first_query': 'A-1 carrier',
        'entity_description': 'carrier', 'followup_prefix': '', 'followup_suffix': ' phone'})
    result, models, seen, _ = invoke(models=models)
    assert result.trace['status'] == 'completed'
    assert [b['options']['num_predict'] for b in models.bodies] == [300, 60]
    assert seen == ['phone', 'Acme phone']
    assert all('seed' in [r['chunk_id'] for r in a['evidence']] for a in result.arms.values())


def test_unknown_is_honest_skip():
    result, _, seen, _ = invoke(models=Models({'mode': 'unknown', 'reason': 'no_grounded_query'}))
    assert result.trace['status'] == 'route_unknown'
    assert not seen


def test_schema_rejection_has_fixed_reason_and_no_raw_values():
    result, _, seen, _ = invoke(models=Models({'mode': 'independent', 'target_slot': 0,
        'query': 'SECRET private', 'steps': 'SECRET'}))
    assert result.trace['rejections'][0]['codes'] == ['proposal_schema']
    assert 'SECRET' not in json.dumps(result.trace)
    assert not seen


def test_long_question_is_not_concatenated_or_truncated():
    q = 'What is order A-1 carrier phone? ' + 'background '*39
    assert len(q) < 500
    result, _, seen, _ = invoke(question=q)
    assert result.trace['status'] == 'completed'
    assert seen[0] == q and seen[1] == 'A-1 carrier phone'


def test_scope_change_rejected_before_dispatch():
    result, _, seen, _ = invoke(validate_scope=lambda original, query: query == original)
    assert result.trace['status'] == 'scope_mismatch'
    assert not seen


@pytest.mark.parametrize('call', [1, 2, 3, 4, 5, 6])
def test_stale_version_fails_closed(call):
    count = 0
    def auth(rows):
        nonlocal count
        count += 1
        return [dict(r, version_id='revoked') for r in rows] if count == call else deepcopy(rows)
    result, _, _, _ = invoke(reauthorize=auth)
    assert any(a['status'] == 'failed' for a in result.arms.values())


def test_failed_first_search_still_runs_second_with_no_retry():
    calls = []
    def search(args):
        calls.append(args.query)
        if len(calls) == 1:
            raise RuntimeError('SECRET')
        return [row('fresh')]
    result, _, _, allocation = invoke(search=search)
    assert result.trace['status'] == 'partial_failure'
    assert result.arms['control']['evidence'] is None
    assert result.arms['treatment']['status'] == 'selected'
    assert len(calls) == 2 and allocation.state['calls']['search']['failed'] == 1
    assert 'SECRET' not in json.dumps(result.trace)


def test_graph_does_not_accept_labels():
    with pytest.raises(TypeError):
        invoke(gold=['secret'])


def test_structurally_supported_literal_needs_no_model_call():
    result, model, seen, _ = invoke(question='Product A 的超时阈值是多少？',
        reauthorize=deepcopy, seed=[row(text='Product A 的超时阈值为 9 秒。')])
    assert result.trace['status'] == 'structurally_complete'
    assert not seen and not model.bodies


def test_unresolved_original_publication_scope_skips_before_model():
    from types import SimpleNamespace
    doc = SimpleNamespace(id='doc', active_version_id='v-doc',
                          metadata_json={'source':'Publisher P', 'published_at':'2026-01-03'})
    contract = build_supplement_contract('Publisher P article published February 8, 2026', [doc])
    model = Models()
    allocation = ExecutionBudget(SimpleNamespace(agent_task_timeout_seconds=180,
        agent_policy_max_calls=2, agent_judge_max_calls=0, agent_generation_max_calls=0, agent_judge_token_budget=0))
    result = run_repair(contract, [row()], [row()], models=model, reauthorize=deepcopy,
        search=lambda args: [], validate_scope=lambda original, query: True, budget=allocation)
    assert result.trace['status'] == 'original_scope_unresolved'
    assert not model.bodies


def test_interruption_is_propagated_for_runner_preservation():
    def interrupt(args):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        invoke(search=interrupt)


def test_absent_literal_bridge_is_rejection_instead_of_operational_failure():
    models = Models({'mode': 'entity_bridge', 'target_slot': 0, 'first_query': 'A-1 carrier',
        'entity_description': 'carrier', 'followup_prefix': '', 'followup_suffix': ' phone'}, entity='Imaginary')
    result, _, seen, _ = invoke(models=models)
    assert result.trace['status'] == 'binding_rejected'
    assert result.trace['rejections'][0]['codes'] == ['literal_binding_mismatch']
    assert not result.trace['errors'] and not seen


def test_bridge_after_character_700_reaches_model_and_binds_authorized_whole_bytes():
    models = Models({'mode': 'entity_bridge', 'target_slot': 0, 'first_query': 'A-1 carrier',
        'entity_description': 'carrier', 'followup_prefix': '', 'followup_suffix': ' phone'})
    text = 'Order A-1 carrier unknown. ' + 'Background. '*70 + 'Order A-1 carrier Acme.'
    result, models, seen, allocation = invoke(models=models, seed=[row(text=text)])
    assert result.trace['status'] == 'completed'
    assert text in models.bodies[1]['messages'][1]['content']
    assert seen == ['phone', 'Acme phone']
    assert allocation.state['calls']['policy']['attempted'] == 2


def test_bridge_oversize_payload_rejects_before_extraction_dispatch():
    models = Models({'mode': 'entity_bridge', 'target_slot': 0, 'first_query': 'A-1 carrier',
        'entity_description': 'carrier', 'followup_prefix': '', 'followup_suffix': ' phone'})
    result, models, seen, allocation = invoke(models=models,
        seed=[dict(row(), locator={'section': 'x'*11000})])
    assert result.trace['status'] == 'binding_rejected'
    assert result.trace['rejections'][0]['codes'] == ['payload_too_large']
    assert len(models.bodies) == allocation.state['calls']['policy']['attempted'] == 1
    assert not seen


def test_program_proposal_and_selector_reuse_graph_search_caps_without_model_http():
    from agent.supplement_proposal import FocusProposal
    from app.evidence_selection import select_slot_evidence
    proposals, selections = [], []
    def provider(models, contract, current, needed):
        proposals.append((contract.question, deepcopy(current), list(needed)))
        return FocusProposal(mode='independent', target_slot=0, query='A-1 carrier phone')
    def selector(contract, union, **kwargs):
        selections.append(deepcopy(kwargs))
        return select_slot_evidence(contract, union, **kwargs)
    result, models, seen, allocation = invoke(proposal_provider=provider, evidence_selector=selector)
    assert result.trace['status'] == 'completed'
    assert proposals[0][0] == 'What is order A-1 carrier phone?'
    assert proposals[0][2] == [0]
    assert len(selections) == 2
    assert all(k['limit'] == 8 and k['token_budget'] == 5000 and k['required_ids'] == [] for k in selections)
    assert selections[0]['focus_queries'] == {'slot_1': 'A-1 carrier phone'}
    assert seen == ['What is order A-1 carrier phone?', 'A-1 carrier phone']
    assert allocation.state['calls']['search']['attempted'] == 2
    assert allocation.state['limits']['search'] == allocation.state['limits']['policy'] == 2
    assert not models.bodies and allocation.state['calls'].get('policy') is None


@pytest.mark.parametrize('invalid', ['slot', 'type'])
def test_injected_program_proposal_rejects_invalid_slot_and_type_before_search(invalid):
    from agent.supplement_proposal import FocusProposal
    def provider(*args):
        if invalid == 'slot':
            return FocusProposal(mode='independent', target_slot=99, query='A-1 carrier phone')
        return SimpleNamespace(mode='independent', target_slot=0, query='A-1 carrier phone')
    result, models, seen, _ = invoke(proposal_provider=provider)
    assert result.trace['status'] == 'proposal_rejected'
    assert not models.bodies and not seen
