import json

from agent.controller import create_task, run_task, task_payload
from app.clients import Models, Claim, GeneratedAnswer
from app.config import settings
from app.models import Chunk, Document
from app.qa import validate_claims
from app.task_analysis import english_subquestions
from test_agent import agent_db
from test_source_facts import evidence


def test_focused_generation_corrects_exact_date_relation_without_extra_model_call(monkeypatch):
    monkeypatch.setattr(settings(), 'focused_generation_enabled', True)
    calls = []
    def backend(body):
        calls.append(body)
        return {'message': {'content': json.dumps({'status': 'answered', 'claims': [
            {'text': 'The opening dates differ because hours differ.', 'source_ids': ['E1:S1', 'E2:S1']}]
        })}}
    rows = [evidence('East store opens June 4, 2026 at 09:00.', 1),
            evidence('East store opens June 4, 2026 at 10:00.', 2)]
    result, usage = Models(backend).generate('Are the opening dates consistent despite different operating hours?', rows)
    assert len(calls) == 1 and '所问日期一致' in result.claims[0].text
    assert usage['deterministic_date_comparison']['value'] == 'yes'
    assert validate_claims(result, rows)[1] == 'answered'


def test_focused_generation_keeps_speaker_and_bounds_global_absence(monkeypatch):
    monkeypatch.setattr(settings(), 'focused_generation_enabled', True)
    def backend(_):
        return {'message': {'content': json.dumps({'status': 'answered', 'claims': [
            {'text': 'The report confirmed the merger helps buyers.', 'source_ids': ['E1:S1']},
            {'text': 'The whole report never mentions fines.', 'source_ids': ['E1:S1']}]
        })}}
    rows = [evidence('Lawyer Dana argued the merger helps buyers.')]
    result, usage = Models(backend).generate('What does Dana argue?', rows)
    assert len(usage['scope_repairs']) == 2
    assert all('Lawyer Dana argued' in claim.text for claim in result.claims)
    assert all('whole report never' not in claim.text for claim in result.claims)


def test_simple_route_skips_redundant_detail_but_still_validates_and_reauthorizes(monkeypatch):
    from agent.tools import ToolResult
    monkeypatch.setattr(settings(), 'adaptive_routing_enabled', True)
    db, user = agent_db()
    calls = []
    class Tools:
        def __init__(self):
            self.user = user
        def call(self, name, args):
            calls.append(name)
            return ToolResult('ok', {'matches': [{'title': '恢复政策', 'snippet': 'RPO 为 15 分钟。'}]}, ['c2'], {})
    class Model:
        def generate(self, question, evidence, **kwargs):
            return GeneratedAnswer(answerable=True, claims=[Claim(text='RPO 为 15 分钟。',
                evidence_ids=['E1'], quotes=['RPO 为 15 分钟。'])]), {'answer_status': 'answered'}
    task = create_task(db, user, 'RPO是多少？', 'workflow', 6, {'_execution_route': {'direct': False}})
    assert task.input['_execution_route']['direct']
    run_task(db, user, task, models=Model(), tools=Tools())
    assert calls == ['search_documents']
    assert task_payload(db, user, task)['result']['status'] == 'answered'
    db.get(Document, 'doc-a').owner_id = 'other'
    db.get(Document, 'doc-a').read_groups = ['finance']
    db.commit()
    assert task_payload(db, user, task)['result']['status'] == 'access_changed'


def test_window_neighbour_is_a_revocation_dependency_in_full_task(monkeypatch):
    from agent.tools import ToolResult
    monkeypatch.setattr(settings(), 'passage_window_enabled', True)
    db, user = agent_db()
    db.add(Chunk(id='c3', version_id='v2', ordinal=1, text='回滚阈值是 2%。', locator={}))
    db.commit()
    class Tools:
        def __init__(self):
            self.user = user
        def call(self, name, args):
            return ToolResult('ok', {'matches': [{'title': '恢复政策', 'snippet': 'RPO 为 15 分钟。'}]}, ['c2'], {})
    class Model:
        def generate(self, question, evidence, **kwargs):
            row = next(row for row in evidence if row['chunk_id'] == 'c3')
            return GeneratedAnswer(answerable=True, claims=[Claim(text='回滚阈值是 2%。',
                evidence_ids=[row['id']], quotes=[row['text']])]), {'answer_status': 'answered'}
    task = create_task(db, user, '回滚阈值是多少？')
    run_task(db, user, task, models=Model(), tools=Tools())
    assert 'c3' in task.evidence_chunk_ids
    assert task_payload(db, user, task)['result']['claims']
    db.get(Document, 'doc-a').deleted = True
    db.commit()
    revoked = json.dumps(task_payload(db, user, task), ensure_ascii=False)
    # Random task UUIDs may contain "c3"; reject the actual evidence handle.
    assert '2%' not in revoked and '"c3"' not in revoked


def test_english_independent_asks_split_without_breaking_shared_comparison():
    assert len(english_subquestions('What is the RPO and what is the RTO?')) == 2
    assert len(english_subquestions('Do A and B agree on the opening date?')) == 1


def test_historical_route_reads_requested_effective_version_not_latest_creation(monkeypatch):
    from agent.tools import KnowledgeTools
    from app.models import DocumentVersion
    monkeypatch.setattr(settings(), 'focused_generation_enabled', True)
    db, user = agent_db()
    db.get(Chunk, 'c1').text = 'Policy effective January 1, 2026 until March 1, 2026.\nPolicy limit is 30 minutes.'
    db.get(Chunk, 'c2').text = 'Policy effective March 1, 2026 until June 1, 2026.\nPolicy limit is 15 minutes.'
    db.add(DocumentVersion(id='v3', document_id='doc-a', filename='latest.md', media_type='text/markdown',
        content_hash='3' * 64, storage_key='latest', status='ready', pipeline={}, timings={}, parsed_blocks=[]))
    db.add(Chunk(id='c4', version_id='v3', ordinal=0,
        text='Policy effective June 1, 2026 through December 31, 2026.\nPolicy limit is 5 minutes.', locator={}))
    db.get(Document, 'doc-a').active_version_id = 'v3'
    db.commit()
    observed = []
    class Model:
        def generate(self, question, evidence, **kwargs):
            observed.extend(evidence)
            row = evidence[0]
            return GeneratedAnswer(answerable=True, claims=[Claim(
                text='Policy limit on February 1, 2026 was 30 minutes.',
                evidence_ids=[row['id'], row['id']], quotes=[
                    'Policy limit is 30 minutes.',
                    'Policy effective January 1, 2026 until March 1, 2026.'])]), {'answer_status': 'answered'}
    task = create_task(db, user, 'What was the policy limit on February 1, 2026?',
                       task_input={'document_id': 'doc-a'})
    tools = KnowledgeTools(db, user, models=object(), search=object())
    run_task(db, user, task, models=Model(), tools=tools)
    assert task.status == 'completed' and task.result['status'] == 'answered'
    assert {row['version_id'] for row in observed} == {'v1'}
    assert observed[0]['metadata']['requested_effective_dates'] == ['2026-02-01']


def test_public_trial_auto_cannot_bypass_dynamic_restriction(monkeypatch):
    from app.main import AgentTaskBody, start_agent_task
    from test_trial import trial_db, enable_trial
    db, user = trial_db()
    enable_trial(monkeypatch, limit=10)
    monkeypatch.setattr(settings(), 'adaptive_routing_enabled', True)
    payload = start_agent_task(AgentTaskBody(
        goal='If search finds a component then inspect its policy', mode='auto'), user, db)
    assert payload['mode'] == 'workflow'
    from app.models import AgentTask
    task = db.get(AgentTask, payload['id'])
    assert task.input['_execution_route']['reason'] == 'public_trial_fixed_workflow'


def test_skipping_conflict_judge_preserves_multi_document_conflict(monkeypatch):
    monkeypatch.setattr(settings(), 'focused_generation_enabled', True)
    def backend(_):
        return {'message': {'content': json.dumps({'status': 'conflict', 'claims': [
            {'text': 'The policies disagree: 15 versus 30 minutes.', 'source_ids': ['E1:S1', 'E2:S1']}]
        })}}
    result, usage = Models(backend).generate('What is the policy limit?',
        [evidence('Policy limit is 15 minutes.', 1), evidence('Policy limit is 30 minutes.', 2)])
    assert result.answerable and usage['answer_status'] == 'conflict'


def test_chinese_attribution_and_certainty_losses_preserve_source(monkeypatch):
    monkeypatch.setattr(settings(), 'focused_generation_enabled', True)
    def backend(_):
        return {'message': {'content': json.dumps({'status': 'answered', 'claims': [
            {'text': '报道认定交易合法。', 'source_ids': ['E1:S1']},
            {'text': '成本已经下降。', 'source_ids': ['E2:S1']}]
        }, ensure_ascii=False)}}
    rows = [evidence('报道转述：律师林主张交易合法。', 1),
            evidence('专家周预测成本可能下降。', 2)]
    result, usage = Models(backend).generate('有哪些观点？', rows, check_conflict=False)
    assert len(usage['scope_repairs']) == 2
    assert '律师林主张' in result.claims[0].text and '可能下降' in result.claims[1].text
