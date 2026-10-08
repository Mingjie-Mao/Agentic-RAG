"""Regressions for bound citations, temporal scope, branch gating and ID kinds."""
import pytest
from app.clients import Claim, GeneratedAnswer, agent_decision_schema
from app.qa import validate_claims
from app.answer_contract import quantitative_support_issues
from app.task_contract import bind_values, build_contract, contract_generate
from app.task_contract_runtime import prepare_contract, resolve_history_document
from app.temporal import complete_version_intervals, effective_interval, select_effective_versions
from agent.planner import compact_observation, evidence_gaps
from agent.tools import ToolResult


def evidence(text, cid='c', vid='v', document='doc', title='规则'):
    return dict(id=cid, chunk_id=cid, version_id=vid, document_id=document, title=title, text=text)


@pytest.mark.parametrize('claim,quotes,passes', [
    ('审计日志从3年变为5年。', ['审计日志保留3年。', '本版本自2026年8月1日起生效。'], False),
    ('审计日志从3年变为5年。', ['审计日志保留3年。', '审计日志保留5年。'], True),
    ('响应相差1小时40分钟。', ['首次响应20分钟。', '首次响应2小时。'], True),
    ('响应相差2小时。', ['首次响应20分钟。', '首次响应2小时。'], False),
    ('响应相差99分钟。', ['首次响应20分钟。', '首次响应2小时。'], False),
    ('响应相差一小时四十分钟。', ['首次响应20分钟。', '首次响应2小时。'], True),
    ('乙响应比甲长了1小时40分钟。', ['甲首次响应20分钟。', '乙首次响应2小时。'], True),
    ('乙响应比甲长2小时。', ['甲首次响应20分钟。', '乙首次响应2小时。'], False),
    ('甲响应比乙短一小时四十分钟。', ['甲首次响应20分钟。', '乙首次响应2小时。'], True),
    ('专业版配额比基础版多900次。', ['基础版300次，专业版1200次。'], True),
    ('专业版配额比基础版多1200次。', ['基础版300次，专业版1200次。'], False),
    ('响应比另一项长100分钟。', ['响应比另一项长100分钟。'], True),
    ('跨境订单比普通订单多1个工作日。', ['跨境订单在普通周期之外额外增加1个工作日。'], True),
    ('跨境订单比普通订单多2个工作日。', ['跨境订单在普通周期之外额外增加1个工作日。'], False),
    ('响应为100分钟。', ['首次响应20分钟。', '首次响应2小时。'], False),
    ('B is longer than A by 100 minutes.', ['A: 20 minutes.', 'B: 2 hours.'], True),
    ('2026年第四季度收货窗口为08:00至18:00。', ['收货窗口为08:00至18:00。'], True),
    ('2026 年 3 月配额为300次。', ['配额为300次。'], True),
    ('审计日志保留2026年。', ['审计日志保留2年。'], False),
    ('响应相差0分钟。', ['甲首次响应20分钟。', '乙首次响应20分钟。'], True),
    ('超时为90秒。', ['超时为1.5分钟。'], True),
    ('规则生效于2026年3月2日。', ['规则生效于2026年3月1日。'], False),
    ('2026年3月2日规则时限为20分钟。', ['Effective March 1, 2026 until April 1, 2026.', '时限20分钟。'], True),
])
def test_quantitative_support_is_bound_to_quotes(claim, quotes, passes):
    assert (not quantitative_support_issues(claim, quotes)) is passes


def test_an_uncited_correct_chunk_does_not_rescue_bad_citation():
    rows = [evidence('审计日志保留3年。', 'old'), evidence('审计日志保留5年。', 'new')]
    generated = GeneratedAnswer(answerable=True, claims=[Claim(text='审计日志保留5年。', evidence_ids=['old'], quotes=['审计日志保留3年。'])])
    assert validate_claims(generated, rows) == ([], 'verification_failed')


def test_document_identity_is_unique_relevant_source_not_first_rank():
    contract = build_contract('操作日志的历史保留期怎么变化？')
    rows = [evidence('重试为5次。', document='distractor'), evidence('操作日志保留80天。', document='logs')]
    assert resolve_history_document(contract, rows) == 'logs'
    rows.append(evidence('操作日志保留100天。', document='other', title='另一份规则'))
    assert resolve_history_document(contract, rows) is None


def history_rows():
    rows = []
    for i, (start, old, hotel, days) in enumerate([
        ('2024 年 4 月 1 日', None, 410, 40),
        ('2025 年 1 月 15 日', '2024 年 4 月 1 日', 570, 40),
        ('2025 年 7 月 1 日', '2025 年 1 月 15 日', 570, 25)]):
        text = f'本版本自 {start} 起生效' + (f'，取代 {old} 的版本。' if old else '。')
        text += f'\n一线城市住宿标准为每晚{hotel}元；材料须在结束后{days}天内提交。'
        rows.append(evidence(text, f'c{i}', f'v{i}', title='差旅规则'))
    return rows


def prepare_history(question):
    rows = history_rows()
    contract = build_contract(question)
    manifest = [dict(version_id=r['version_id'], filename=r['version_id'], status='ready',
                     created_at=str(3-i), is_active=i==2, effective_interval=effective_interval(r['text']))
                for i,r in enumerate(rows)]
    prepared, trace = prepare_contract(contract, [rows[-1], evidence('其他资料。', document='other')],
        search=lambda q: None, load=lambda refs,historical: [r for r in rows if r['chunk_id'] in refs],
        versions=lambda doc: ToolResult('ok', {'versions':manifest,'total_version_count':3},[],{}),
        compare=lambda doc,old,new: ToolResult('ok', {}, [r['chunk_id'] for r in rows if r['version_id'] in [old,new]],{}),
        open_version=None)
    return contract, prepared, trace


class NoModel:
    def generate(self, *args, **kwargs):
        raise AssertionError('verified numeric facts should not need a model call')


def test_history_month_and_second_attribute_survive_upload_order_and_reload():
    contract, rows, trace = prepare_history('2025 年 3 月出差住宿标准是多少？和现行版本相比，材料提交时限有什么变化？')
    assert contract.version_ids == ['v0','v1','v2']
    # Simulate framework checkpoint/resume: only source text is reloaded. Version
    # semantics must survive in the business contract, not ephemeral row metadata.
    reloaded = [{k:v for k,v in r.items() if k not in ['effective_interval','version_label']} for r in rows]
    answer, _ = contract_generate(NoModel(), contract.question, reloaded, contract=contract)
    assert answer.answerable
    text = ' '.join(c.text for c in answer.claims)
    assert '570 元' in text and '40 天' in text and '25 天' in text
    assert validate_claims(answer, reloaded)[1] == 'answered'
    assert trace['version_chain_complete']


def test_earliest_policy_uses_effective_date_not_upload_order():
    contract, rows, _ = prepare_history('差旅规则历史版本最早住宿标准是多少？')
    answer, _ = contract_generate(NoModel(), contract.question, rows, contract=contract)
    assert answer.answerable and '410 元' in answer.claims[0].text


def test_month_spanning_version_change_is_not_given_one_arbitrary_value():
    versions = [dict(version_id='a',effective_interval=dict(status='explicit',valid_from='2025-01-01',valid_to='2025-03-15')),
                dict(version_id='b',effective_interval=dict(status='explicit',valid_from='2025-03-15',valid_to=None))]
    contract, rows, _ = prepare_history('2025年3月历史住宿标准是多少？')
    contract.versions = versions
    contract.version_ids = ['a','b']
    for row,vid in zip(rows[:2],['a','b']):
        row['version_id']=vid
    assert not contract_generate(NoModel(), contract.question, rows[:2], contract=contract)[0].answerable


def test_unlinked_versions_do_not_invent_supersession():
    records = [dict(version_id='a',is_active=False,effective_interval=effective_interval('自2024年1月1日起生效。')),
               dict(version_id='b',is_active=True,effective_interval=effective_interval('自2025年1月1日起生效。'))]
    closed = complete_version_intervals(records, {'a':'自2024年1月1日起生效。','b':'自2025年1月1日起生效。'})
    assert closed[0]['effective_interval']['status'] == 'unknown'
    assert select_effective_versions('2024年3月1日',closed)['status'] == 'unresolved'


def test_subject_bound_active_branch_ignores_other_products_retries():
    rows = [evidence('Java SDK默认请求超时6秒。重试2次。', 'sdk', 's', title='Java SDK'),
            evidence('回调通知最多重试6次。', 'callback', 'b', title='回调通知'),
            evidence('重试3次。', 'router', 'r', title='路由服务')]
    q='先查Java SDK的默认请求超时；如果大于5秒，说明回调通知最多重试几次；否则说明幂等键请求头。'
    answer, usage = contract_generate(NoModel(), q, rows)
    assert answer.answerable and len(answer.claims)==2
    assert '6 次' in answer.claims[-1].text and '幂等' not in str(answer.claims)
    assert usage['task_contract']['slot_states']['false_1']=='inactive'


def test_difference_is_exact_when_hours_would_need_infinite_decimal():
    q='S2工单的首次响应时限比S1长多少？'
    rows=[evidence('| 等级 | 首次响应 |\n| S1 | 20分钟 |\n| S2 | 2小时 |')]
    answer,_=contract_generate(NoModel(),q,rows)
    assert '100 分钟' in answer.claims[-1].text
    assert validate_claims(answer,rows)[1]=='answered'


def test_policy_observation_and_schema_preserve_id_kinds():
    observation=compact_observation('search_documents', {'status':'ok'}, {'matches':[
        dict(document_id='doc',version_id='version',chunk_id='chunk',title='规则',snippet='超时为10秒。')]}, ['chunk'])
    schema=agent_decision_schema([observation])
    branches={r['properties']['action']['enum'][0]:r['properties']['arguments'] for r in schema['anyOf']}
    assert branches['get_document_version']['properties']['document_id']['enum']==['doc']
    assert branches['retrieve_evidence']['properties']['chunk_ids']['items']['enum']==['chunk']
    assert 'chunk' not in str(branches['get_document_version'])
    assert branches['final']['additionalProperties'] is False
    empty={r['properties']['action']['enum'][0] for r in agent_decision_schema([])['anyOf']}
    assert 'get_document_version' not in empty and 'retrieve_evidence' not in empty


def test_title_overlap_without_requested_value_is_not_evidence_ready():
    q='路由服务单个请求的超时阈值是多少？'
    assert evidence_gaps(q,[evidence('根因分析与改进措施。',title='路由服务超时')])
    assert not evidence_gaps(q,[evidence('路由服务单个请求超时为7秒。',title='路由服务')])


def test_general_source_comparisons_do_not_activate_numeric_or_version_contract():
    from app.task_analysis import historical_route_intent
    assert build_contract('华东仓和华南仓的收货窗口分别是什么？').intent=='ordinary'
    q='Does the article suggest innovation compared to previous years?'
    assert not historical_route_intent(q)
    assert build_contract(q).intent=='ordinary'
    assert historical_route_intent('Compare previous and current policy versions.')


def test_truncated_manifest_is_unknown_not_an_exception():
    contract=build_contract('操作日志的历史保留期怎么变化？')
    rows=[evidence('操作日志保留80天。')]
    prepared,trace=prepare_contract(contract,rows,search=None,load=lambda *a:rows,
        versions=lambda doc:ToolResult('ok',{'versions':[{'version_id':'v'}]},[],{}),
        compare=None,open_version=None)
    assert not contract.version_chain_complete and prepared==[]
    assert not contract_generate(NoModel(),contract.question,prepared,contract=contract)[0].answerable


def test_version_tools_are_unavailable_for_ordinary_current_question():
    obs=[dict(matches=[dict(document_id='d',chunk_id='c',version_id='v')])]
    schema=agent_decision_schema(obs,'查询默认重试次数')
    actions={b['properties']['action']['enum'][0] for b in schema['anyOf']}
    assert 'get_document_version' not in actions and 'compare_versions' not in actions
    assert 'retrieve_evidence' in actions


def test_quota_condition_preserves_entity_when_rate_unit_follows_it():
    q='先查银海基础版每分钟配额；如果低于800次，再给出专业版的配额；否则只回答基础版。'
    rows=[evidence('| 套餐 | 每分钟调用上限 |\n| 基础版 | 400次 |\n| 专业版 | 1600次 |',title='银海配额')]
    answer,usage=contract_generate(NoModel(),q,rows)
    assert answer.answerable and '400 次' in answer.claims[0].text and '1600 次' in answer.claims[1].text
    assert usage['task_contract']['condition']['result']=='true'


def test_branch_metric_is_bound_to_requested_action_not_other_service():
    q='先查生产数据库的RTO；如果超过30分钟，再给出发布灰度期间自动回滚的错误率阈值。'
    rows=[evidence('生产数据库的RTO为50分钟。','db','db',title='生产数据库'),
          evidence('## 自动回滚\n灰度期间错误率超过4%时，自动回滚到稳定版本。','release','release',title='发布规则'),
          evidence('错误率连续3分钟超过1.2%触发告警。','router','router',title='路由运维')]
    answer,_=contract_generate(NoModel(),q,rows)
    assert answer.answerable and '4 %' in answer.claims[-1].text
    assert '1.2' not in str(answer.claims)


def test_table_binding_uses_the_requested_column_and_retains_header():
    slot=build_contract('先查生产数据库的RPO，如果超过30分钟，再给出RTO。').slots[0]
    source=evidence('| 资源 | RPO | RTO |\n| 生产数据库 | 待确认 | 90分钟 |')
    assert bind_values(slot,[source])==[]
    source['text']='| 资源 | RPO | RTO |\n| 生产数据库 | 20分钟 | 90分钟 |'
    values=bind_values(slot,[source])
    assert len(values)==1 and values[0].value=='20'
    assert '| 资源 | RPO | RTO |' in values[0].quote
