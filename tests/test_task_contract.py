from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from agent.controller import create_task, run_task, task_payload
from agent.tools import KnowledgeTools, ToolResult
from app.config import settings
from app.models import Document, DocumentVersion
from app.qa import validate_claims
from app.task_contract import bind_values, build_contract, contract_generate, evaluate_condition
from app.task_contract_runtime import prepare_contract
from test_agent import agent_db


@pytest.mark.parametrize("raw,expected", [("十二万", 120000), ("十万", 100000), ("一亿二千万", 120000000), ("2.5万", 25000), ("50,000", 50000), ("八", 8)])
def test_scaled_and_grouped_numbers_preserve_magnitude(raw, expected):
    from app.task_contract import number
    assert number(raw) == expected


@pytest.mark.parametrize("raw", ["", "12,34", "50,00,0", "unknown"])
def test_unknown_and_malformed_numbers_remain_unbound(raw):
    from app.task_contract import number
    assert number(raw) is None


@pytest.mark.parametrize("value", ["七万", "70,000", "7 万"])
def test_import_limit_binding_excludes_alarm_shard_and_incident_counts(value):
    from app.task_contract import _slot
    slot = _slot("现在单次导入最多允许多少行？", "limit")
    assert slot.value_type == "number"
    evidence = [
        row("单次导入任务最多 " + value + " 行。", "limit"),
        row("单个任务超过上限通知值班。", "alarm"),
        row("每片最多7000行。", "shard"),
        row("此前导入了9万行，未分片。", "incident"),
        row("单次导入任务最多20分钟。", "wrong-unit"),
    ]
    assert [v.chunk_id for v in bind_values(slot, evidence)] == ["limit"]


def row(text, cid="c", vid="v", title="星河服务条款", **extra):
    return dict(id="E1", chunk_id=cid, version_id=vid, document_id="doc", title=title, text=text, locator={}, **extra)


class NoGeneration:
    def generate(self, *args, **kwargs):
        raise AssertionError("literal contracts do not require a model call")


def test_two_original_values_units_and_computed_difference_have_both_refs():
    q = "星河基础版和专业版的响应时限相差多少？"
    evidence = [row("基础版首次响应为90分钟。", "a"), row("专业版首次响应为1小时。", "b")]
    evidence[1]["id"] = "E2"
    generated, usage = contract_generate(NoGeneration(), q, evidence)
    assert generated.answerable
    assert "90 分钟" in generated.claims[0].text
    assert "1 小时" in generated.claims[1].text
    assert "30 分钟" in generated.claims[2].text
    assert generated.claims[2].evidence_ids == ["E1", "E2"]
    assert validate_claims(generated, evidence)[1] == "answered"
    assert len(usage["task_contract"]["bound_values"]) == 2


@pytest.mark.parametrize("text", ["基础版首次响应为90分钟。", "基础版首次响应为90分钟。专业版首次响应为2工作日。",
    "基础版首次响应为90分钟。专业版首次响应为30分钟或60分钟。"])
def test_incomplete_ambiguous_or_incompatible_comparisons_refuse(text):
    answer, _ = contract_generate(NoGeneration(), "星河基础版和专业版的响应时限相差多少？", [row(text)])
    assert not answer.answerable


@pytest.mark.parametrize("operator,result", [("超过", "false"), ("至少", "true"), ("低于", "false"), ("不超过", "true")])
def test_condition_exact_threshold_boundaries(operator, result):
    contract = build_contract(f"先查备份RPO，如果它{operator}30分钟，再给出回滚超时。")
    state = evaluate_condition(contract, [row("RPO为30分钟。")])
    assert state.result == result
    assert ("true_1" in state.active_slot_ids) == (result == "true")


@pytest.mark.parametrize("q,evidence", [
    ("先查RPO，如果它超过30分钟并且重试超过3次，再给出超时。", [row("RPO为40分钟。重试为5次。")]),
    ("先查RPO，如果它超过30分钟，再给出超时。", [row("RPO待确认。")]),
    ("先查RPO，如果它超过30分钟，再给出超时。", [row("RPO为2天。")]),
    ("先查RPO，如果它超过30分钟，再给出超时。", [row("RPO为20分钟或40分钟。")]),
])
def test_complex_missing_conflicting_or_wrong_units_stay_unknown(q, evidence):
    contract = build_contract(q)
    assert evaluate_condition(contract, evidence).result == "unknown"
    answer, usage = contract_generate(NoGeneration(), q, evidence)
    assert not answer.answerable and not answer.claims
    assert set(usage["task_contract"]["slot_states"].values()) == {"unknown"}


def test_date_predicate_uses_explicit_source_date_and_boundary():
    contract = build_contract("先查星河规则生效日期，如果它不晚于2026年5月1日，再给出重试次数。")
    assert contract.intent == "conditional"
    assert evaluate_condition(contract, [row("本规则生效日期为2026年5月1日。")]).result == "true"
    assert evaluate_condition(contract, [row("本规则生效日期为2026年5月2日。")]).result == "false"


def test_false_condition_does_not_search_or_generate_inactive_branch():
    contract = build_contract("先查RPO，如果它超过30分钟，再给出回滚超时。")
    queries = []
    def search(query):
        queries.append(query)
        return ToolResult("ok", {}, ["c"], {})
    evidence, trace = prepare_contract(contract, [], search=search, load=lambda *a: [row("RPO为20分钟。回滚超时为90秒。")],
        versions=None, compare=None, open_version=None)
    answer, _ = contract_generate(NoGeneration(), contract.question, evidence, contract=contract)
    assert queries == ["RPO"]
    assert trace["condition"]["result"] == "false"
    assert answer.answerable and "90" not in str(answer.claims)


def test_true_condition_includes_operand_and_active_numeric_branch():
    q = "先查RPO，如果它超过30分钟，再给出回滚超时。"
    answer, usage = contract_generate(NoGeneration(), q, [row("RPO为40分钟。回滚超时为90秒。")])
    assert answer.answerable and len(answer.claims) == 2
    assert usage["task_contract"]["condition"]["result"] == "true"


def test_wrong_subject_or_log_type_is_not_a_binding():
    slot = build_contract("星河基础版和专业版的响应时限相差多少？").slots[0]
    assert bind_values(slot, [row("专业版首次响应为30分钟。")]) == []
    slot = build_contract("最早版本的操作日志保留多久？").slots[0]
    assert bind_values(slot, [row("审计日志保留为3年。")]) == []


@pytest.mark.parametrize("unchanged", [True, False])
def test_history_compares_version_identity_instead_of_reporting_conflict(unchanged):
    contract = build_contract("各个历史版本的RPO有没有变化？")
    contract.version_ids, contract.version_chain_complete = ["old", "new"], True
    evidence = [row("RPO为30分钟。", "a", "old", version_label="旧版"), row(f"RPO为{30 if unchanged else 15}分钟。", "b", "new", version_label="新版")]
    evidence[1]["id"] = "E2"
    answer, usage = contract_generate(NoGeneration(), contract.question, evidence, contract=contract)
    assert answer.answerable and len(answer.claims) == 3
    assert ("保持不变" in answer.claims[-1].text) == unchanged
    assert "answer_status" not in usage
    contract.version_ids.append("missing")
    assert not contract_generate(NoGeneration(), contract.question, evidence, contract=contract)[0].answerable


def test_asof_date_requires_source_effective_intervals_with_exclusive_boundary():
    contract = build_contract("截至2026年5月1日历史版本的RPO是多少？")
    contract.version_ids, contract.version_chain_complete = ["old", "new"], True
    evidence = [row("RPO为30分钟。", "a", "old", effective_interval=dict(status="explicit", valid_from="2026-01-01", valid_to="2026-05-01", end_inclusive=False)),
                row("RPO为15分钟。", "b", "new", effective_interval=dict(status="explicit", valid_from="2026-05-01", valid_to=None))]
    evidence[1]["id"] = "E2"
    answer, _ = contract_generate(NoGeneration(), contract.question, evidence, contract=contract)
    assert answer.answerable and "15" in answer.claims[0].text
    evidence[1]["effective_interval"] = {}
    assert not contract_generate(NoGeneration(), contract.question, evidence, contract=contract)[0].answerable


@pytest.mark.parametrize("mode", ["workflow", "hybrid", "dynamic"])
def test_all_agent_modes_share_contract_route_and_reauthorize(monkeypatch, mode):
    monkeypatch.setattr(settings(), "task_contract_enabled", True)
    db, user = agent_db()
    class Tools:
        def call(self, name, arguments):
            assert name == "search_documents"
            return ToolResult("ok", {}, ["c2"], {})
    task = create_task(db, user, "先查RPO，如果它超过30分钟，再给出超时。", mode, 5,
                       task_input={"_task_contract": {"fake": True}})
    assert "_task_contract" not in task.input
    run_task(db, user, task, models=NoGeneration(), tools=Tools())
    assert task.result["status"] == "answered"
    assert task.result["usage"]["task_contract"]["condition"]["result"] == "false"
    doc = db.get(Document, "doc-a")
    doc.owner_id, doc.read_groups = "someone-else", []
    db.commit()
    assert task_payload(db, user, task)["result"]["status"] == "access_changed"


def test_history_route_collects_complete_chain_and_refuses_truncated_manifest(monkeypatch):
    monkeypatch.setattr(settings(), "task_contract_enabled", True)
    db, user = agent_db()
    db.get(DocumentVersion, "v1").created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    db.get(DocumentVersion, "v2").created_at = datetime(2025, 2, 1, tzinfo=timezone.utc)
    db.commit()
    task = create_task(db, user, "各个历史版本的RPO有没有变化？", "hybrid", 5, {"document_id": "doc-a"})
    run_task(db, user, task, models=NoGeneration(), tools=KnowledgeTools(db, user, models=NoGeneration(), search=object()))
    assert task.result["status"] == "answered"
    assert {c["version_id"] for c in task.result["citations"]} == {"v1", "v2"}
    assert "发生变化" in task.result["claims"][-1]["text"]
    task = create_task(db, user, "各个历史版本的RPO有没有变化？", "workflow", 1, {"document_id": "doc-a"})
    run_task(db, user, task, models=NoGeneration(), tools=KnowledgeTools(db, user, models=NoGeneration(), search=object()))
    assert task.result["status"] == "insufficient_evidence"


def test_permission_revoked_after_retrieval_cannot_generate(monkeypatch):
    monkeypatch.setattr(settings(), "task_contract_enabled", True)
    db, user = agent_db()
    class Tools:
        def call(self, *args):
            doc = db.get(Document, "doc-a")
            doc.owner_id, doc.read_groups = "someone-else", []
            db.commit()
            return ToolResult("ok", {}, ["c2"], {})
    task = create_task(db, user, "先查RPO，如果它超过30分钟，再给出超时。", "hybrid", 4)
    with pytest.raises(HTTPException):
        run_task(db, user, task, models=NoGeneration(), tools=Tools())
    assert task.status == "failed" and not task.result


def test_rag_path_uses_same_contract_and_final_acl_check(monkeypatch):
    from app import qa
    from types import SimpleNamespace
    from app.models import Answer
    monkeypatch.setattr(settings(), "task_contract_enabled", True)
    db, user = agent_db()
    calls = []
    initial = dict(row("RPO 为 15 分钟。", "c2", "v2"), document_id="doc-a")
    def retrieve(*args, **kwargs):
        calls.append(args[2])
        return SimpleNamespace(evidence=[initial], candidates=[{"chunk_id": "c2", "document_id": "doc-a", "title": "恢复政策"}],
            readable_documents=1, searchable_documents=1, blocked_reason=None, embed_ms=0, retrieval_ms=0, context_tokens=20)
    monkeypatch.setattr(qa, "retrieve_authorized", retrieve)
    monkeypatch.setattr("agent.tools.retrieve_authorized", retrieve)
    monkeypatch.setattr(qa, "Models", NoGeneration)
    result = qa.answer_question(db, user, "先查RPO，如果它超过30分钟，再给出超时。")
    assert result["status"] == "answered"
    assert result["usage"]["task_contract"]["condition"]["result"] == "false"
    assert all("超时" not in query for query in calls)
    assert result["citations"][0]["chunk_id"] == "c2"
    doc = db.get(Document, "doc-a")
    doc.owner_id, doc.read_groups = "someone-else", []
    db.commit()
    assert qa.visible_answer(db, user, db.get(Answer, result["id"]))["status"] == "access_changed"


def test_simple_first_search_miss_recovery_is_bounded_and_distinct():
    contract = build_contract("请查询星河基础版和专业版的响应时限相差多少？")
    calls = []
    def search(query):
        calls.append(query)
        return ToolResult("ok", {"matches": []}, [], {})
    prepare_contract(contract, [], search=search, load=lambda *a: [], versions=None, compare=None, open_version=None)
    # Two operand searches, at most one recovery for both; no retry for ACL errors.
    assert len(calls) <= 3
    assert len(set(calls)) == len(calls)


def test_negated_source_value_and_history_condition_are_not_positive_answers():
    contract = build_contract("先查RPO，如果它超过30分钟，再给出超时。")
    assert evaluate_condition(contract, [row("RPO不是40分钟。")]).result == "unknown"
    mixed = build_contract("先查最早历史版本的RPO，如果它超过30分钟，再给出超时。")
    assert mixed.intent == "conditional" and mixed.conditions[0].operator == "unknown"


@pytest.mark.parametrize("status,result", [("已批准", "true"), ("已拒绝", "false"), ("待确认", "unknown")])
def test_literal_enum_predicate(status, result):
    contract = build_contract("先查审批状态，如果它等于已批准，再给出超时。")
    assert evaluate_condition(contract, [row(f"审批状态为{status}。")]).result == result


def test_unsupported_conditional_syntax_is_unknown_instead_of_ordinary_generation():
    contract = build_contract("如果恢复状态满足正式验收标准再给出超时")
    assert contract.intent == "conditional"
    assert evaluate_condition(contract, [row("恢复状态为启用。")]).result == "unknown"


def test_same_sentence_implicit_subject_still_requires_local_attribute():
    q = "生产数据库的RPO和生产数据库的RTO相差多少？"
    answer, _ = contract_generate(NoGeneration(), q, [row("生产数据库恢复点目标RPO为15分钟，恢复时间目标RTO为60分钟。", title="数据库备份")])
    assert answer.answerable and "45 分钟" in answer.claims[-1].text


def test_first_response_attribute_is_not_first_version_scope():
    assert build_contract("各个历史版本的首次响应时限有没有变化？").history_kind == "changes"
    assert build_contract("最早版本的首次响应时限是多少？").history_kind == "first"


@pytest.mark.parametrize("question", [
    "Have video games changed in vibrancy compared to previous years?",
    "What was the first privacy change compared to older news reports?",
    "As of 2026, how has innovation changed compared to previous years?",
])
def test_qualitative_time_comparison_does_not_request_document_versions(question):
    assert build_contract(question).intent == "ordinary"


@pytest.mark.parametrize("question", [
    "What was the RPO in the first version?",
    "How did retention in previous policy versions change?",
])
def test_explicit_english_version_requests_keep_history_guards(question):
    assert build_contract(question).intent == "history"


@pytest.mark.parametrize("question", [
    "Does the semifinal path differ, with England defeating specific teams?",
    "Considering the Independent - Life and Style article, which couple separated?",
    "How has artificial intelligence changed, and what were the benefits?",
])
def test_if_inside_english_words_does_not_activate_a_condition(question):
    assert build_contract(question).intent == "ordinary"


def test_multi_attribute_history_never_returns_a_partial_answer_as_complete():
    contract = build_contract("各个历史版本的RPO和RTO有没有变化？")
    contract.version_chain_complete, contract.version_ids = True, ["v"]
    answer, usage = contract_generate(NoGeneration(), contract.question, [row("RPO为15分钟。RTO为60分钟。")], contract=contract)
    assert not answer.answerable and usage["task_contract"]["status"] == "history_scope_unknown"


def test_classified_table_subject_and_counter_unit_predicate():
    contract = build_contract("先查S1工单的首次响应时限，如果它超过30分钟，再给出超时。")
    assert evaluate_condition(contract, [row("| 等级 | 首次响应 |\n| S1 | 15分钟 |\n| S2 | 2小时 |")]).result == "false"
    contract = build_contract("先查门禁卡补办时间，如果它超过5个工作日，再给出有效期。")
    assert evaluate_condition(contract, [row("门禁卡补办为3个工作日。")]).result == "false"


@pytest.mark.parametrize("changed", [True, False])
def test_history_change_predicate_is_computed_only_after_complete_chain(changed):
    contract = build_contract("审计日志的历史版本保留期有没有变化？如果有，从多久变成了多久？")
    assert contract.intent == "history"
    contract.version_chain_complete, contract.version_ids = True, ["a", "b"]
    evidence = [row("审计日志保留3年。", "a", "a"), row(f"审计日志保留{5 if changed else 3}年。", "b", "b")]
    evidence[1]["id"] = "E2"
    answer, usage = contract_generate(NoGeneration(), contract.question, evidence, contract=contract)
    assert answer.answerable
    assert usage["task_contract"]["condition"]["result"] == ("true" if changed else "false")
    assert ("变为" in answer.claims[-1].text) == changed


class RecordingOrdinaryModel:
    def __init__(self):
        self.calls = []

    def generate(self, question, evidence, **options):
        from app.clients import Claim, GeneratedAnswer
        self.calls.append((question, evidence, options))
        return GeneratedAnswer(answerable=True, claims=[Claim(
            text=evidence[0]['text'], evidence_ids=[evidence[0]['id']],
            quotes=[evidence[0]['text']])]), {'prompt_tokens': 3}


@pytest.mark.parametrize('separator', ['？', '?', ';', '；'])
def test_ordinary_literal_policy_renders_every_explicit_numeric_ask(separator):
    q = f'日志服务的超时阈值是多少秒{separator}通知服务的超时阈值是多少秒？'
    evidence = [row('日志服务的超时阈值为9秒。', 'log'),
                row('通知服务的超时阈值为12秒。', 'notice')]
    evidence[1]['id'] = 'E2'
    answer, usage = contract_generate(NoGeneration(), q, evidence, ordinary_policy='literal_slots')
    assert answer.answerable and len(answer.claims) == 2
    assert [c.evidence_ids for c in answer.claims] == [['E1'], ['E2']]
    assert [c.quotes for c in answer.claims] == [[r['text'][:-1]] for r in evidence]
    assert all(s == 'supported' for s in usage['task_contract']['slot_states'].values())
    assert usage['task_contract']['status'] == 'ordinary_literal_complete'
    assert validate_claims(answer, evidence)[1] == 'answered'


@pytest.mark.parametrize('second', [None, '通知服务的超时阈值为12秒。通知服务的超时阈值为14秒。',
                                    '其他服务的超时阈值为12秒。'])
def test_ordinary_literal_policy_refuses_missing_conflicting_wrong_subject_slots(second):
    q = '日志服务的超时阈值是多少秒？通知服务的超时阈值是多少秒？'
    evidence = [row('日志服务的超时阈值为9秒。')]
    if second:
        evidence.append(row(second, 'other'))
    answer, usage = contract_generate(NoGeneration(), q, evidence, ordinary_policy='literal_slots')
    assert not answer.answerable and not answer.claims
    assert usage['task_contract']['status'] == 'ordinary_slot_incomplete'
    assert usage['task_contract']['slot_states']['slot_2'] == 'unknown'
    assert usage['task_contract']['slot_reasons']['slot_2']


@pytest.mark.parametrize('text', [
    '日志服务讨论通知服务的超时阈值为9秒。',
    '例如日志服务的超时阈值为9秒。', '日志服务的超时阈值为9秒（示例）。',
    '假设日志服务的超时阈值为9秒。', '日志服务的超时阈值暂定为9秒。',
    '日志服务的超时阈值不是9秒。', '日志服务的超时阈值未设置为9秒。',
    '日志服务的超时阈值为9秒，待确认。', '历史记录：日志服务的超时阈值为9秒。',
    '# 日志服务\n超时阈值为9秒（历史值）。',
])
def test_ordinary_literal_policy_rejects_unverified_predicates(text):
    answer, usage = contract_generate(NoGeneration(), '日志服务的超时阈值是多少秒？',
                                      [row(text)], ordinary_policy='literal_slots')
    assert not answer.answerable and not answer.claims
    assert usage['task_contract']['slot_states'] == {'slot_1': 'unknown'}


def test_ordinary_literal_policy_does_not_claim_undelimited_multi_subject_completion():
    q = '日志服务的超时阈值和通知服务的超时阈值是多少秒？'
    answer, usage = contract_generate(NoGeneration(), q, [row('日志服务的超时阈值为9秒。')],
                                      ordinary_policy='literal_slots')
    assert not answer.answerable and not answer.claims
    assert set(usage['task_contract']['slot_states'].values()) == {'unknown'}


@pytest.mark.parametrize('query,text,value', [
    ('日志服务的状态是什么？', '日志服务的状态为已启用。', '已启用'),
    ('日志服务的生效日期是什么？', '日志服务的生效日期为2026年5月1日。', '2026-05-01'),
])
def test_ordinary_literal_policy_accepts_complete_enum_and_date_predicates(query, text, value):
    answer, _ = contract_generate(NoGeneration(), query, [row(text)], ordinary_policy='literal_slots')
    assert answer.answerable and value in answer.claims[0].text
    assert answer.claims[0].quotes == [text[:-1]]


@pytest.mark.parametrize('query,text', [
    ('日志服务的状态是什么？', '日志服务的状态未启用。'),
    ('日志服务的状态是什么？', '日志服务的状态为已启用（待确认）。'),
    ('日志服务的生效日期是什么？', '日志服务的生效日期暂定为2026年5月1日。'),
    ('日志服务的生效日期是什么？', '日志服务的生效日期原为2026年5月1日。'),
    ('日志服务的生效日期是什么？', '# 日志服务\n生效日期为2026年5月1日（旧版）。'),
])
def test_ordinary_literal_policy_does_not_turn_qualified_date_enum_into_current_fact(query, text):
    answer, _ = contract_generate(NoGeneration(), query, [row(text)], ordinary_policy='literal_slots')
    assert not answer.answerable and not answer.claims


@pytest.mark.parametrize('quality', [False, True])
def test_ordinary_mixed_requests_use_one_complete_checklist_and_unknown_support(monkeypatch, quality):
    monkeypatch.setattr(settings(), 'answer_quality_enabled', quality)
    q = '日志服务的超时阈值是多少秒？通知服务由谁负责？日志服务由谁维护？'
    evidence = [row('日志服务的超时阈值为9秒。通知服务由值班负责。日志服务由运营维护。')]
    model = RecordingOrdinaryModel()
    recovery_calls = []
    def recover(models, question, rows, *, options):
        recovery_calls.append(question)
        return models.generate(question, rows, **options)
    monkeypatch.setattr('app.answer_quality.recover_answer', recover)
    answer, usage = contract_generate(model, q, evidence, ordinary_policy='literal_slots',
                                      options={'max_output_tokens': 321, 'check_conflict': False})
    assert answer.answerable and len(model.calls) == 1
    assert len(recovery_calls) == int(quality)
    assert model.calls[0][2] == {'max_output_tokens': 321, 'check_conflict': False,
        'acceptance_items_override': ['日志服务的超时阈值是多少秒', '通知服务由谁负责', '日志服务由谁维护']}
    assert set(usage['task_contract']['slot_states'].values()) == {'unknown'}
    assert usage['task_contract']['status'] == 'model_checklist_only'


def test_ordinary_literal_policy_never_truncates_explicit_asks_to_claim_limit():
    q = '？'.join(f'通知{i}服务的超时阈值是多少秒' for i in range(9)) + '？'
    evidence = [row(f'通知{i}服务的超时阈值为9秒。', str(i)) for i in range(9)]
    answer, usage = contract_generate(NoGeneration(), q, evidence, ordinary_policy='literal_slots')
    assert not answer.answerable and len(usage['task_contract']['slot_states']) == 9


def test_invalid_ordinary_policy_fails_before_any_generation():
    with pytest.raises(ValueError, match='ordinary_policy'):
        contract_generate(NoGeneration(), '日志服务由谁负责？', [], ordinary_policy='guess')


def test_legacy_ordinary_generation_keeps_existing_parser_and_options(monkeypatch):
    monkeypatch.setattr(settings(), 'answer_quality_enabled', False)
    q = '日志服务的超时阈值是多少秒？通知服务的超时阈值是多少秒？'
    evidence = [row('日志服务的超时阈值为9秒。')]
    default, explicit = RecordingOrdinaryModel(), RecordingOrdinaryModel()
    expected = contract_generate(default, q, evidence, options={'max_output_tokens': 321})
    actual = contract_generate(explicit, q, evidence, ordinary_policy='legacy', options={'max_output_tokens': 321})
    assert actual == expected and default.calls == explicit.calls
    assert len(build_contract(q).slots) == 1


def test_ordinary_policy_preserves_conditional_gates_and_earlier_history_versions():
    q = '先查RPO，如果它超过30分钟，再给出回滚超时。'
    evidence = [row('RPO为20分钟。回滚超时为90秒。')]
    assert contract_generate(NoGeneration(), q, evidence, ordinary_policy='literal_slots') == contract_generate(NoGeneration(), q, evidence)
    unknown = '如果恢复状态满足正式验收标准再给出超时'
    assert not contract_generate(NoGeneration(), unknown, evidence, ordinary_policy='literal_slots')[0].answerable
    history = build_contract('最早版本的操作日志保留多久？')
    history.version_chain_complete, history.version_ids = True, ['old', 'new']
    versions = [row('操作日志保留3年。', 'a', 'old'), row('操作日志保留5年。', 'b', 'new')]
    versions[1]['id'] = 'E2'
    assert contract_generate(NoGeneration(), history.question, versions, contract=history, ordinary_policy='literal_slots') == contract_generate(NoGeneration(), history.question, versions, contract=history)


@pytest.mark.parametrize('ambiguous', [
    '日志服务的超时阈值为10秒。日志服务的超时阈值为11秒。',
    '日志服务的超时阈值为10秒或11秒。',
    '日志服务的超时阈值未设置为10秒。',
])
def test_ordinary_literal_policy_does_not_hide_an_unbound_conflicting_version(ambiguous):
    evidence = [row('日志服务的超时阈值为9秒。', 'safe', 'one'),
                row(ambiguous, 'conflict', 'two')]
    answer, _ = contract_generate(NoGeneration(), '日志服务的超时阈值是多少秒？', evidence,
                                  ordinary_policy='literal_slots')
    assert not answer.answerable and not answer.claims


def test_ordinary_literal_policy_checks_requested_unit_dimension():
    answer, _ = contract_generate(NoGeneration(), '日志服务的超时阈值是多少秒？',
                                  [row('日志服务的超时阈值为9年。')], ordinary_policy='literal_slots')
    assert not answer.answerable


def test_ordinary_literal_policy_does_not_drop_safe_quotes_in_a_multi_fact_chunk():
    text = '日志服务的超时阈值为9秒。通知服务的超时阈值为12秒。'
    answer, _ = contract_generate(NoGeneration(),
        '日志服务的超时阈值是多少秒？通知服务的超时阈值是多少秒？', [row(text)],
        ordinary_policy='literal_slots')
    assert answer.answerable and len(answer.claims) == 2
    assert [c.quotes for c in answer.claims] == [['日志服务的超时阈值为9秒'], ['通知服务的超时阈值为12秒']]


@pytest.mark.parametrize('policy', ['', None, [], {}])
def test_invalid_ordinary_policy_types_fail_explicitly(policy):
    with pytest.raises(ValueError, match='ordinary_policy'):
        contract_generate(NoGeneration(), '日志服务的状态是什么？', [], ordinary_policy=policy)


def test_ordinary_text_checklist_keeps_more_than_four_explicit_requests(monkeypatch):
    monkeypatch.setattr(settings(), 'answer_quality_enabled', False)
    queries = [f'通知{i}服务由谁负责' for i in range(6)]
    model = RecordingOrdinaryModel()
    _, usage = contract_generate(model, '；'.join(queries), [row('通知服务由值班负责。')],
                                ordinary_policy='literal_slots')
    assert len(model.calls) == 1
    assert model.calls[0][2]['acceptance_items_override'] == queries
    assert len(usage['task_contract']['slot_states']) == 6
    assert set(usage['task_contract']['slot_states'].values()) == {'unknown'}


@pytest.mark.parametrize('separator', ['，', ',', '、', '并请给出'])
def test_ordinary_literal_policy_refuses_unrecognized_multiask_boundaries(separator):
    q = f'日志服务的超时阈值{separator}通知服务的超时阈值是多少秒？'
    evidence = [row('日志服务的超时阈值为9秒。', 'log'),
                row('通知服务的超时阈值为12秒。', 'notice')]
    answer, usage = contract_generate(NoGeneration(), q, evidence, ordinary_policy='literal_slots')
    assert not answer.answerable and not answer.claims
    assert usage['task_contract']['status'] == 'ordinary_slot_incomplete'
    assert set(usage['task_contract']['slot_states'].values()) == {'unknown'}


@pytest.mark.parametrize('qualifier', ['在2026年', '在生产环境下', '在新租户中', '发生故障后'])
def test_ordinary_literal_policy_refuses_unproven_question_constraints(qualifier):
    q = f'日志服务的超时阈值{qualifier}是多少秒？'
    evidence = [row('日志服务的超时阈值为9秒。', title='2026年生产环境新租户配置')]
    answer, usage = contract_generate(NoGeneration(), q, evidence, ordinary_policy='literal_slots')
    assert not answer.answerable and not answer.claims
    assert usage['task_contract']['status'] == 'ordinary_slot_incomplete'
    assert set(usage['task_contract']['slot_states'].values()) == {'unknown'}


@pytest.mark.parametrize('mixed', [False, True])
def test_ordinary_model_checklist_preserves_nine_explicit_requests(monkeypatch, mixed):
    monkeypatch.setattr(settings(), 'answer_quality_enabled', False)
    queries = [f'通知{i}服务由谁负责' for i in range(9)]
    if mixed:
        queries[0] = '日志服务的超时阈值是多少秒'
    model = RecordingOrdinaryModel()
    _, usage = contract_generate(model, '？'.join(queries) + '？',
        [row('日志服务的超时阈值为9秒。通知服务由值班负责。')], ordinary_policy='literal_slots')
    assert len(model.calls) == 1
    assert model.calls[0][2]['acceptance_items_override'] == queries
    assert len(usage['task_contract']['slot_states']) == 9
    assert set(usage['task_contract']['slot_states'].values()) == {'unknown'}
    assert usage['task_contract']['status'] == 'model_checklist_only'


@pytest.mark.parametrize('last_value', ['11秒', '11年'])
def test_ordinary_literal_policy_preserves_conflicting_versions_split_across_rows(last_value):
    evidence = [row('日志服务的超时阈值为9秒。', 'safe', 'v1'),
                row('日志服务的超时阈值为10秒。', 'conflict-a', 'v2'),
                row(f'日志服务的超时阈值为{last_value}。', 'conflict-b', 'v2')]
    answer, usage = contract_generate(NoGeneration(), '日志服务的超时阈值是多少秒？',
                                      evidence, ordinary_policy='literal_slots')
    assert not answer.answerable and not answer.claims
    assert usage['task_contract']['slot_states'] == {'slot_1': 'unknown'}
    assert usage['task_contract']['status'] == 'ordinary_slot_incomplete'


@pytest.mark.parametrize('unbound_text', [
    '超时阈值为10秒或11秒。', '超时阈值不是10秒。',
])
def test_ordinary_literal_policy_refuses_title_scoped_discarded_ambiguous_rows(unbound_text):
    evidence = [row('日志服务的超时阈值为9秒。', 'safe', 'v1'),
                row(unbound_text, 'unbound', 'v2', title='日志服务')]
    answer, usage = contract_generate(NoGeneration(), '日志服务的超时阈值是多少秒？',
                                      evidence, ordinary_policy='literal_slots')
    assert not answer.answerable and not answer.claims
    assert usage['task_contract']['slot_states'] == {'slot_1': 'unknown'}
    assert usage['task_contract']['status'] == 'ordinary_slot_incomplete'


@pytest.mark.parametrize('question', [
    '日志服务的超时阈值和通知服务的超时阈值是多少秒？',
    '日志服务的超时阈值是多少秒，并说明通知服务在生产环境的超时设置？',
    '日志服务的超时阈值是多少秒。2026年通知服务在测试环境的超时设置是什么？',
    '日志服务在生产环境的超时阈值是多少秒？',
])
def test_literal_fallback_preserves_whole_uncertain_request_in_one_checklist(monkeypatch, question):
    monkeypatch.setattr(settings(), 'answer_quality_enabled', False)
    evidence = [row('日志服务的超时阈值为9秒。')]
    model = RecordingOrdinaryModel()
    answer, usage = contract_generate(model, question, evidence, ordinary_policy='literal_fallback',
        options={'max_output_tokens': 321, 'acceptance_items_override': ['保留调用者的全部要求']})
    assert answer.answerable and len(model.calls) == 1
    actual_question, actual_evidence, options = model.calls[0]
    assert actual_question == question and actual_evidence == evidence
    assert question in options['acceptance_items_override']
    assert '保留调用者的全部要求' in options['acceptance_items_override']
    assert options['max_output_tokens'] == 321
    trace = usage['task_contract']
    assert trace['status'] == 'model_checklist_only'
    assert trace['literal_status'] == 'ordinary_slot_incomplete'
    assert set(trace['slot_states'].values()) == {'unknown'}
    assert set(trace['slot_reasons'].values()) == {'ordinary_scope_unresolved'}
    assert not trace['bound_values']


@pytest.mark.parametrize('question,text', [
    ('日志服务的超时阈值是多少秒？', '例如日志服务的超时阈值为9秒。'),
    ('日志服务的超时阈值是多少秒？', '日志服务的超时阈值为9秒，待确认。'),
    ('日志服务的超时阈值是多少秒？', '日志服务的超时阈值暂定为9秒。'),
    ('日志服务的状态是什么？', '日志服务的状态未启用。'),
    ('日志服务的生效日期是什么？', '日志服务的生效日期原为2026年5月1日。'),
])
def test_literal_fallback_retains_unknown_source_reasons(monkeypatch, question, text):
    monkeypatch.setattr(settings(), 'answer_quality_enabled', False)
    evidence = [row(text)]
    _, frozen = contract_generate(NoGeneration(), question, evidence, ordinary_policy='literal_slots')
    model = RecordingOrdinaryModel()
    answer, usage = contract_generate(model, question, evidence, ordinary_policy='literal_fallback')
    assert answer.answerable and len(model.calls) == 1
    assert usage['task_contract']['slot_states'] == frozen['task_contract']['slot_states']
    assert usage['task_contract']['slot_reasons'] == frozen['task_contract']['slot_reasons']
    assert not usage['task_contract']['bound_values']
    assert usage['task_contract']['status'] == 'model_checklist_only'


@pytest.mark.parametrize('second', [None, '通知服务的超时阈值为12秒或14秒。',
    '通知服务的超时阈值为12秒。通知服务的超时阈值为14秒。'])
def test_literal_fallback_delegates_all_slots_when_second_value_fails(monkeypatch, second):
    monkeypatch.setattr(settings(), 'answer_quality_enabled', False)
    question = '日志服务的超时阈值是多少秒？通知服务的超时阈值是多少秒？'
    evidence = [row('日志服务的超时阈值为9秒。')]
    if second:
        evidence.append(row(second, 'second'))
    _, frozen = contract_generate(NoGeneration(), question, evidence, ordinary_policy='literal_slots')
    model = RecordingOrdinaryModel()
    answer, usage = contract_generate(model, question, evidence, ordinary_policy='literal_fallback')
    assert answer.answerable and len(model.calls) == 1
    assert model.calls[0][2]['acceptance_items_override'] == [question,
        '日志服务的超时阈值是多少秒', '通知服务的超时阈值是多少秒']
    assert usage['task_contract']['slot_states'] == {'slot_1': 'unknown', 'slot_2': 'unknown'}
    assert usage['task_contract']['slot_reasons'] == frozen['task_contract']['slot_reasons']
    assert not usage['task_contract']['bound_values']


def test_literal_fallback_delegates_nine_literals_without_truncation(monkeypatch):
    monkeypatch.setattr(settings(), 'answer_quality_enabled', False)
    queries = [f'通知{i}服务的超时阈值是多少秒' for i in range(9)]
    question = '？'.join(queries) + '？'
    evidence = [row(f'通知{i}服务的超时阈值为9秒。', str(i)) for i in range(9)]
    model = RecordingOrdinaryModel()
    _, usage = contract_generate(model, question, evidence, ordinary_policy='literal_fallback')
    assert len(model.calls) == 1
    assert model.calls[0][2]['acceptance_items_override'] == [question, *queries]
    assert len(usage['task_contract']['slot_states']) == 9
    assert set(usage['task_contract']['slot_reasons'].values()) == {'ordinary_slot_budget'}
    assert set(usage['task_contract']['slot_states'].values()) == {'unknown'}


@pytest.mark.parametrize('question,text', [
    ('日志服务的超时阈值是多少秒？', '日志服务的超时阈值为9秒。'),
    ('日志服务的状态是什么？', '日志服务的状态为已启用。'),
    ('日志服务的生效日期是什么？', '日志服务的生效日期为2026年5月1日。'),
    ('日志服务的超时阈值是多少秒？通知服务的超时阈值是多少秒？',
     '日志服务的超时阈值为9秒。通知服务的超时阈值为12秒。'),
])
def test_literal_fallback_preserves_safe_literal_fast_path(question, text):
    evidence = [row(text)]
    assert contract_generate(NoGeneration(), question, evidence, ordinary_policy='literal_fallback') == (
        contract_generate(NoGeneration(), question, evidence, ordinary_policy='literal_slots'))


@pytest.mark.parametrize('question', [
    '通知服务由谁负责？',
    '日志服务的超时阈值是多少秒？通知服务由谁负责？',
])
def test_literal_fallback_text_and_mixed_asks_use_model_without_rule_proof(monkeypatch, question):
    monkeypatch.setattr(settings(), 'answer_quality_enabled', False)
    evidence = [row('日志服务的超时阈值为9秒。通知服务由值班负责。')]
    model = RecordingOrdinaryModel()
    _, usage = contract_generate(model, question, evidence, ordinary_policy='literal_fallback')
    assert len(model.calls) == 1
    assert question in model.calls[0][2]['acceptance_items_override']
    assert usage['task_contract']['status'] == 'model_checklist_only'
    assert set(usage['task_contract']['slot_states'].values()) == {'unknown'}
    assert not usage['task_contract']['bound_values']


@pytest.mark.parametrize('question', [
    '先查RPO，如果它超过30分钟，再给出回滚超时。',
    '如果RPO足够低，再给出回滚超时。',
    '星河基础版和专业版的响应时限相差多少？',
    '最早版本的RPO是多少分钟？',
])
def test_literal_fallback_preserves_nonordinary_contracts(question):
    evidence = [row('RPO为20分钟。回滚超时为90秒。基础版首次响应为90分钟。专业版首次响应为60分钟。')]
    assert contract_generate(NoGeneration(), question, evidence, ordinary_policy='literal_fallback') == (
        contract_generate(NoGeneration(), question, evidence))


@pytest.mark.parametrize('quality', [False, True])
def test_literal_fallback_model_refusal_does_not_start_another_attempt(monkeypatch, quality):
    from app.clients import GeneratedAnswer
    monkeypatch.setattr(settings(), 'answer_quality_enabled', quality)

    class RefusingModel(RecordingOrdinaryModel):
        def generate(self, question, evidence, **options):
            self.calls.append((question, evidence, options))
            return GeneratedAnswer(answerable=False, claims=[]), {'prompt_tokens': 7}

    model = RefusingModel()
    answer, usage = contract_generate(model, '日志服务的超时阈值是多少秒？',
        [row('日志服务的超时阈值暂定为9秒。')], ordinary_policy='literal_fallback')
    assert not answer.answerable and not answer.claims
    assert len(model.calls) == 1 and usage['prompt_tokens'] == 7
    assert usage['task_contract']['slot_states'] == {'slot_1': 'unknown'}
    if quality:
        assert usage['answer_recovery']['repair_attempts'] == 0


def test_literal_fallback_reuses_existing_single_repair_path(monkeypatch):
    from app.clients import Claim, GeneratedAnswer
    monkeypatch.setattr(settings(), 'answer_quality_enabled', True)
    evidence = [row('日志服务的超时阈值暂定为9秒。')]

    class RepairModel(RecordingOrdinaryModel):
        def generate(self, question, evidence, **options):
            self.calls.append((question, evidence, options))
            text = '日志服务的超时阈值为99秒。' if len(self.calls) == 1 else evidence[0]['text']
            return GeneratedAnswer(answerable=True, claims=[Claim(text=text,
                evidence_ids=['E1'], quotes=[evidence[0]['text']])]), {'prompt_tokens': 3}

    model = RepairModel()
    answer, usage = contract_generate(model, '日志服务的超时阈值是多少秒？', evidence,
        ordinary_policy='literal_fallback')
    assert answer.answerable and len(model.calls) == 2
    assert usage['answer_recovery']['repair_attempts'] == 1
    assert not usage['quality_validation_failed']
    assert usage['prompt_tokens'] == 6
    assert all(call[0] == '日志服务的超时阈值是多少秒？' for call in model.calls)
    assert model.calls[0][2]['acceptance_items_override'] == model.calls[1][2]['acceptance_items_override']
    assert usage['task_contract']['slot_states'] == {'slot_1': 'unknown'}


@pytest.mark.parametrize('quality', [False, True])
def test_literal_fallback_preserves_cumulative_model_budget(monkeypatch, quality):
    from app.execution_budget import BudgetExceeded, ExecutionBudget, bounded_model, use_budget
    monkeypatch.setattr(settings(), 'answer_quality_enabled', quality)
    monkeypatch.setattr(settings(), 'agent_generation_max_calls', 1)

    class BudgetedModel(RecordingOrdinaryModel):
        @bounded_model('generation')
        def generate(self, *args, **kwargs):
            return super().generate(*args, **kwargs)

    question = '日志服务的超时阈值是多少秒？'
    evidence = [row('日志服务的超时阈值暂定为9秒。')]
    budget = ExecutionBudget(settings())
    model = BudgetedModel()
    with use_budget(budget):
        contract_generate(model, question, evidence, ordinary_policy='literal_fallback')
    saved = budget.snapshot()
    resumed = ExecutionBudget(settings(), saved)
    with use_budget(resumed), pytest.raises(BudgetExceeded, match='generation_call_budget_exhausted'):
        contract_generate(model, question, evidence, ordinary_policy='literal_fallback')
    assert len(model.calls) == 1
    assert resumed.snapshot()['calls']['generation']['attempted'] == 1


@pytest.mark.parametrize('quality', [False, True])
def test_literal_fallback_propagates_model_failure_without_new_generation(monkeypatch, quality):
    from app.clients import DependencyError
    monkeypatch.setattr(settings(), 'answer_quality_enabled', quality)

    class FailingModel(RecordingOrdinaryModel):
        def generate(self, question, evidence, **options):
            self.calls.append((question, evidence, options))
            raise DependencyError('model unavailable', retryable=False)

    model = FailingModel()
    with pytest.raises(DependencyError, match='model unavailable'):
        contract_generate(model, '日志服务的超时阈值是多少秒？',
            [row('日志服务的超时阈值暂定为9秒。')], ordinary_policy='literal_fallback')
    assert len(model.calls) == 1


@pytest.mark.parametrize('mixed', [False, True])
def test_literal_fallback_delegates_unknown_optional_requested_slots(monkeypatch, mixed):
    from app.clients import GeneratedAnswer
    monkeypatch.setattr(settings(), 'answer_quality_enabled', False)
    question = ('日志服务的超时阈值是多少秒以及通知服务的超时阈值是多少秒？'
                if mixed else '通知服务的超时阈值是多少秒？')
    contract = build_contract(question)
    contract.slots[-1].required = False
    evidence = [row('日志服务的超时阈值为9秒。')] if mixed else []

    class RefusingModel(RecordingOrdinaryModel):
        def generate(self, question, evidence, **options):
            self.calls.append((question, evidence, options))
            return GeneratedAnswer(answerable=False, claims=[]), {'prompt_tokens': 7}

    frozen, frozen_usage = contract_generate(NoGeneration(), question, evidence, contract=contract,
        ordinary_policy='literal_slots')
    assert frozen.answerable and len(frozen.claims) == int(mixed)
    assert frozen_usage['task_contract']['status'] == 'ordinary_literal_complete'
    model = RefusingModel()
    answer, usage = contract_generate(model, question, evidence, contract=contract,
        ordinary_policy='literal_fallback')
    assert len(model.calls) == 1
    assert not answer.answerable and not answer.claims
    assert model.calls[0][2]['acceptance_items_override'] == [question,
        *(slot.query for slot in contract.slots)]
    assert usage['task_contract']['status'] == 'model_checklist_only'
    assert usage['task_contract']['literal_status'] == 'ordinary_slot_incomplete'
    assert set(usage['task_contract']['slot_states'].values()) == {'unknown'}
    assert usage['task_contract']['slot_reasons'] == frozen_usage['task_contract']['slot_reasons']
    assert not usage['task_contract']['bound_values']


def test_literal_fallback_verified_optional_slot_keeps_zero_call_fast_path():
    question = '通知服务的超时阈值是多少秒？'
    contract = build_contract(question)
    contract.slots[0].required = False
    evidence = [row('通知服务的超时阈值为12秒。')]
    answer, usage = contract_generate(NoGeneration(), question, evidence, contract=contract,
        ordinary_policy='literal_fallback')
    assert answer.answerable and len(answer.claims) == 1
    assert usage['task_contract']['status'] == 'ordinary_literal_complete'
    assert usage['task_contract']['slot_states'] == {'slot_1': 'supported'}
