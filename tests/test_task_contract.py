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
