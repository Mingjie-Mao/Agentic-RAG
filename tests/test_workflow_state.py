import pytest

from agent.controller import _execute, create_task, run_task, task_payload
from agent.tools import KnowledgeTools, ToolResult
from agent.workflow_state import WorkflowState
from app.clients import Claim, GeneratedAnswer
from app.task_analysis import version_intent
from test_agent import agent_db


@pytest.fixture(autouse=True)
def legacy_workflow_configuration(monkeypatch):
    # These tests preserve the original workflow path for deployments opting out.
    # Shared contract routing/default behavior is exercised in its own test suite.
    from app.config import settings
    monkeypatch.setattr(settings(), "task_contract_enabled", False)
    monkeypatch.setattr(settings(), "adaptive_routing_enabled", False)


def test_subgoal_machine_keeps_candidates_separate_from_answer_coverage():
    state = WorkflowState.start(['RPO', 'RTO'])
    state.observe({'RPO':['c1'], 'RTO':[]})
    assert not state.evidence_ready
    assert [row['state'] for row in state.items] == ['evidence_candidate', 'evidence_missing']
    restored = WorkflowState.restore(state.snapshot(), ['RPO', 'RTO'])
    restored.check_answers(['RTO'], [{'text':'RPO为15分钟', 'evidence_ids':['c1']}], valid=True)
    assert [row['state'] for row in restored.items] == ['answer_covered', 'answer_missing']
    assert state.items[0]['state'] == 'evidence_candidate'
    restored.observe({}, blocked=True)
    restored.observe({})
    assert all(row['state'] == 'blocked' for row in restored.items)


@pytest.mark.parametrize('goal', [
    'Compare previous and current policy versions.',
    '列出2026年9月20日与2026年10月20日工单响应时限。',
])
def test_version_intent_supports_plain_history_requests(goal):
    assert version_intent(goal)
    assert not version_intent('Which newspaper reported the incident earlier?')


class CitedModels:
    def generate(self, question, evidence, **options):
        return GeneratedAnswer(answerable=True, claims=[
            Claim(text=f"{row['title']}：{row['text']}", evidence_ids=[row['id']], quotes=[row['text']])
            for row in evidence
        ]), {'answer_status':'answered'}


class VersionTools(KnowledgeTools):
    def __init__(self, db, user):
        super().__init__(db, user, models=object(), search=object())
        self.calls = []

    def call(self, name, arguments):
        self.calls.append(name)
        if name == 'search_documents':
            return ToolResult('ok', {'matches':[{'document_id':'doc-a', 'title':'恢复政策', 'snippet':'RPO为15分钟。'}]}, ['c2'], {})
        return super().call(name, arguments)


def test_workflow_discovers_unique_document_and_routes_history_without_input_id():
    db, user = agent_db()
    tools = VersionTools(db, user)
    task = create_task(db, user, '列出恢复政策历史版本的 RPO。', 'workflow', 6)
    run_task(db, user, task, models=CitedModels(), tools=tools)
    payload = task_payload(db, user, task)
    assert tools.calls == ['search_documents', 'get_document_version', 'compare_versions']
    assert {row['version_id'] for row in payload['result']['citations']} == {'v1','v2'}
    assert any(event['event_type'] == 'version_route' and event['payload']['status'] == 'routed' for event in payload['events'])
    assert task.input['_workflow_state']['items'][0]['state'] == 'answer_covered'


def test_tool_checkpoint_retains_version_plan_and_reauthorizes_replayed_refs():
    db, user = agent_db()
    tools = VersionTools(db, user)
    task = create_task(db, user, '恢复政策历史版本的 RPO。', 'workflow', 6)
    first = _execute(db, task, tools, 'get_document_version', {'document_id':'doc-a'}, reuse=True)
    replay = _execute(db, task, tools, 'get_document_version', {'document_id':'doc-a'}, reuse=True)
    assert replay.data['versions'] == first.data['versions']
    assert task.step_no == 1 and tools.calls == ['get_document_version']
    _execute(db, task, tools, 'compare_versions', {'document_id':'doc-a'}, reuse=True)
    replay = _execute(db, task, tools, 'compare_versions', {'document_id':'doc-a'}, reuse=True)
    assert set(replay.evidence_refs) == {'c1','c2'}
    from app.models import Document
    from fastapi import HTTPException
    db.get(Document, 'doc-a').deleted = True
    db.commit()
    with pytest.raises(HTTPException):
        _execute(db, task, tools, 'get_document_version', {'document_id':'doc-a'}, reuse=True)


def test_clients_cannot_inject_completed_subgoal_state():
    db, user = agent_db()
    task = create_task(db, user, 'RPO多少？', task_input={'_workflow_state':{'items':[{'state':'answer_covered'}]}})
    assert '_workflow_state' not in task.input


def test_current_version_fact_does_not_trigger_history_route():
    from app.task_analysis import historical_route_intent
    assert not historical_route_intent('当前 Python SDK 版本默认重试几次？')
    assert historical_route_intent('比较恢复政策之前与当前版本。')
    assert historical_route_intent('What did the previous policy require?')
    assert historical_route_intent('恢复政策在2026-08-20和2026-09-20的RPO。')
    assert version_intent('恢复政策在2026-08-20和2026-09-20的RPO。')
    from agent.planner import version_depth
    assert version_depth('Policy limits on 2026-08-20, 2026-09-20 and 2026-10-20') == 2


def test_missing_historical_chain_cannot_be_answered_with_a_current_chunk():
    db, user = agent_db()
    class NoHistoryTools(VersionTools):
        def call(self, name, arguments):
            if name == 'get_document_version':
                self.calls.append(name)
                return ToolResult('ok', {'versions':[{'version_id':'v2'}]}, [], {})
            return super().call(name, arguments)
    class MustNotGenerate:
        def generate(self, *args, **kwargs):
            raise AssertionError('Current evidence cannot establish historical facts')
    task = create_task(db, user, '列出恢复政策历史版本的 RPO。', 'workflow', 6)
    run_task(db, user, task, models=MustNotGenerate(), tools=NoHistoryTools(db,user))
    result = task_payload(db,user,task)['result']
    assert result['status'] == 'insufficient_evidence' and result['claims'] == []


def test_workflow_does_not_loop_after_candidate_and_answer_coverage():
    db, user = agent_db()
    class PlainTools(VersionTools):
        def call(self,name,args):
            self.calls.append(name)
            assert name in {'search_documents','retrieve_evidence'}
            return ToolResult('ok',{'matches':[{'title':'恢复政策','snippet':'RPO为15分钟。'}]},['c2'],{})
    task=create_task(db,user,'当前 RPO 是多少？','workflow',6)
    tools=PlainTools(db,user)
    run_task(db,user,task,models=CitedModels(),tools=tools)
    payload=task_payload(db,user,task)
    assert tools.calls == ['search_documents','retrieve_evidence']
    reasons=[event['payload']['reason'] for event in payload['events'] if event['event_type']=='workflow_stop']
    assert reasons == ['evidence_coverage_ready','answer_items_covered']


def test_exhausted_step_budget_blocks_new_calls_but_allows_authorized_replay():
    db,user=agent_db()
    tools=VersionTools(db,user)
    task=create_task(db,user,'恢复政策历史版本','workflow',1)
    _execute(db,task,tools,'get_document_version',{'document_id':'doc-a'},reuse=True)
    replay=_execute(db,task,tools,'get_document_version',{'document_id':'doc-a'},reuse=True)
    assert replay.data['versions']
    assert _execute(db,task,tools,'search_documents',{'query':'another query'}) is None
    assert tools.calls == ['get_document_version'] and task.step_no == 1


def test_cached_version_list_is_invalidated_when_active_version_changes():
    from fastapi import HTTPException
    from app.models import Document
    db,user=agent_db()
    tools=VersionTools(db,user)
    task=create_task(db,user,'恢复政策历史版本','workflow',3)
    _execute(db,task,tools,'get_document_version',{'document_id':'doc-a'},reuse=True)
    db.get(Document,'doc-a').active_version_id='v1'
    db.commit()
    with pytest.raises(HTTPException) as exc:
        _execute(db,task,tools,'get_document_version',{'document_id':'doc-a'},reuse=True)
    assert exc.value.status_code == 409


def test_explicit_three_timepoints_cannot_be_satisfied_by_two_versions():
    db, user = agent_db()
    class MustNotGenerate:
        def generate(self, *args, **kwargs):
            raise AssertionError('A missing explicitly requested version cannot disappear from the plan')
    task = create_task(db, user, '恢复政策在2026-08-20、2026-09-20、2026-10-20的 RPO。',
                       'workflow', 6, task_input={'document_id':'doc-a'})
    run_task(db, user, task, models=MustNotGenerate(), tools=VersionTools(db, user))
    payload = task_payload(db, user, task)
    assert payload['result']['status'] == 'insufficient_evidence'
    plans = [row['payload'] for row in payload['events'] if row['event_type'] == 'version_plan']
    assert plans[-1]['requested_pairs'] == 2 and plans[-1]['pairs'] == 1
