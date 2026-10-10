import json
from datetime import date
from types import SimpleNamespace

import pytest

from agent.planned import (Plan, build_planned_graph, choose_version, extract_value, fill, literal_values, parse_day,
                           validate_plan)


def plan(*steps):
    return Plan.model_validate({"steps": list(steps)})


def test_plan_validation_rejects_unbound_placeholders_and_repairs_dropped_constraints():
    goal = "事故 QINGHE-INC-148 中出故障的那个服务，单个请求的超时阈值是多少秒？"
    good = plan({"id": "s1", "purpose": "找事故", "query": "事故复盘 故障服务",
                 "extract": {"name": "service", "kind": "entity", "description": "故障服务名"}},
                {"id": "s2", "purpose": "查阈值", "query": "{s1.service} 超时阈值", "depends_on": ["s1"]})
    assert validate_plan(goal, good) == []
    assert "QINGHE-INC-148" in good.steps[0].query  # the root query kept the user's identifier
    bad = plan({"id": "s1", "purpose": "x", "query": "{s2.service} 超时"},
               {"id": "s2", "purpose": "y", "query": "负责人", "foreach": True})
    issues = validate_plan(goal, bad)
    assert any("placeholder" in i for i in issues) and any("foreach" in i for i in issues)
    version = plan({"id": "s1", "purpose": "v", "kind": "version_as_of", "query": "门禁管理办法"})
    assert any("version_as_of" in i for i in validate_plan("门禁有效期？", version))


def test_fill_substitutes_and_bounds_fan_out():
    values = {"s1.service": ["白泽账务"], "s2.services": ["A", "B", "C", "D", "E"]}
    assert fill("{s1.service} 负责人", values) == ["白泽账务 负责人"]
    assert fill("{s2.services} 负责人", values) == ["A 负责人", "B 负责人", "C 负责人", "D 负责人"]
    assert fill("{s3.x} 负责人", values) == []


def test_bridge_value_must_occur_verbatim_and_the_program_finds_its_sentence():
    passages = [{"chunk_id": "c0", "title": "事故复盘", "text": "# 事故 INC-7 复盘"},
                {"chunk_id": "c1", "title": "事故复盘", "text": "## 概述\n2026-09-14，白泽账务出现定时任务漏跑。持续 82 分钟。"},
                {"chunk_id": "c2", "title": "规则", "text": "本规则适用于：白泽账务、青鸾网关、帝江排产。"}]
    assert literal_values("白泽账务。", passages, "entity") == (["白泽账务"], "c1", "2026-09-14，白泽账务出现定时任务漏跑。")
    assert literal_values("白泽账务、青鸾网关和帝江排产", passages, "list")[:2] == (["白泽账务", "青鸾网关", "帝江排产"], "c2")
    for invented in ("青鸾网关服务", "无", "", "白泽账务系统"):
        assert literal_values(invented, passages, "entity") == (None, None, None)


def test_version_as_of_selects_the_version_in_effect():
    versions = [{"version_id": v, "created_at": d} for v, d in
                (("v1", "2025-07-01T00:00:00+00:00"), ("v2", "2026-02-01T00:00:00+00:00"), ("v3", "2026-06-15T00:00:00+00:00"))]
    day = parse_day("自 2026-02-01 起修订")
    assert day == date(2026, 2, 1) and parse_day("2026 年 6 月 15 日") == date(2026, 6, 15)
    assert choose_version(versions, day, "before")["version_id"] == "v1"
    assert choose_version(versions, day, "at")["version_id"] == "v2"
    assert choose_version(versions, date(2025, 1, 1), "before") is None


class FakeModels:
    def __init__(self, replies):
        self.replies, self.payloads = list(replies), []

    def _post(self, path, body, base=None):
        self.payloads.append(body["messages"][1]["content"])
        reply = self.replies.pop(0)
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return {"message": {"content": content},
                "prompt_eval_count": 10, "eval_count": 5}


def extraction_step():
    return plan({'id': 's1', 'purpose': 'Find carrier', 'query': 'Order A-1 carrier',
                 'extract': {'name': 'carrier', 'kind': 'entity', 'description': 'Order A-1 carrier'}}).steps[0]


def test_complete_extraction_passes_late_literal_without_changing_legacy_default():
    text = 'Background. '*70 + 'Order A-1 carrier Acme.'
    passages = [{'chunk_id': 'late', 'title': 'Orders', 'text': text}]
    legacy = FakeModels(['Acme'])
    extract_value(legacy, 'Find carrier', extraction_step(), passages)
    assert 'Acme' not in legacy.payloads[0]
    complete = FakeModels(['Acme'])
    value = extract_value(complete, 'Find carrier', extraction_step(), passages, complete_passages=True)
    assert text in complete.payloads[0]
    assert literal_values(value, passages, 'entity')[:2] == (['Acme'], 'late')
    assert literal_values('Imaginary', passages, 'entity') == (None, None, None)
    assert literal_values('無', passages, 'entity') == (None, None, None)


def test_complete_extraction_rejects_serialized_payload_before_transport_and_accounts_attempt():
    from app.execution_budget import ExecutionBudget, use_budget
    models = FakeModels(['Acme'])
    allocation = ExecutionBudget(SimpleNamespace(agent_task_timeout_seconds=180,
        agent_policy_max_calls=2, agent_judge_max_calls=0, agent_generation_max_calls=0,
        agent_judge_token_budget=0))
    passages = [{'chunk_id': 'large', 'title': 'Orders', 'text': 'x'*11000}]
    with use_budget(allocation), pytest.raises(ValueError, match='payload'):
        extract_value(models, 'Find carrier', extraction_step(), passages, complete_passages=True)
    assert models.payloads == []
    assert allocation.state['calls']['policy']['attempted'] == 1
    assert allocation.state['calls']['policy']['failed'] == 1


def run_planner(replies, corpus, search):
    """corpus: chunk_id -> (document_id, text); search: query substring -> chunk ids."""
    events, calls = [], []
    task = SimpleNamespace(goal="事故 INC-7 中出故障的服务，超时阈值是多少秒？", step_no=0, max_steps=8, input={})

    def execute(db, task_, tools, name, arguments, reuse=False):
        task.step_no += 1
        calls.append((name, arguments))
        refs = next((ids for key, ids in search.items() if key in arguments.get("query", "")), [])
        return SimpleNamespace(status="ok", evidence_refs=refs, data={})

    def evidence(db, user, refs, allow_historical=False, limit=8):
        return [{"chunk_id": r, "document_id": corpus[r][0], "title": corpus[r][0], "text": corpus[r][1]} for r in refs][:limit]

    run_planner.task = task
    graph = build_planned_graph(None, None, task, FakeModels(replies), None, execute=execute,
                                event=lambda *a, **k: events.append((a[2], k.get("payload"))),
                                evidence_from_refs=evidence).compile()
    return graph.invoke({}), calls, events


def test_planner_uses_the_literal_bridge_value_for_the_next_hop():
    corpus = {"inc": ("inc-7", "2026-09-14，白泽账务出现定时任务漏跑。"), "svc": ("svc-11", "白泽账务单个请求超时阈值为 10 秒。")}
    replies = [
        {"steps": [{"id": "s1", "purpose": "找事故", "query": "事故 INC-7 复盘",
                    "extract": {"name": "service", "kind": "entity", "description": "故障服务"}},
                   {"id": "s2", "purpose": "查阈值", "query": "{s1.service} 运维手册 超时阈值", "depends_on": ["s1"]}]},
        "白泽账务",
    ]
    result, calls, events = run_planner(replies, corpus, {"INC-7": ["inc"], "白泽账务 运维手册": ["svc"]})
    assert [c[1]["query"] for c in calls if c[0] == "search_documents"] == ["事故 INC-7 复盘", "白泽账务 运维手册 超时阈值"]
    assert [c[1]["document_id"] for c in calls if c[0] == "open_document"] == ["inc-7", "svc-11"]
    assert result["refs"][0] == "svc" and "inc" in result["refs"]  # answer hop first, bridge kept
    assert result["planner"]["values"] == {"s1.service": ["白泽账务"]}
    recorded = run_planner.task.input["_planner_values"]
    assert recorded == [{"step": "s1", "what": "故障服务", "values": ["白泽账务"],
                         "quote": "2026-09-14，白泽账务出现定时任务漏跑。", "chunk_id": "inc"}]


def test_unverifiable_bridge_triggers_one_replan_then_stops():
    corpus = {"inc": ("inc-7", "2026-09-14，白泽账务出现定时任务漏跑。")}
    step = {"id": "s1", "purpose": "找事故", "query": "事故 INC-7 复盘",
            "extract": {"name": "service", "kind": "entity", "description": "故障服务"}}
    invented = "青鸾网关"
    replies = [{"steps": [step]}, invented, {"steps": [step]}, invented]
    result, calls, events = run_planner(replies, corpus, {"INC-7": ["inc"]})
    kinds = [kind for kind, _ in events]
    assert kinds.count("replan") == 1 and kinds.count("plan_extract_failed") == 2
    assert result["planner"]["values"] == {} and result["planner"]["replans"] == 1


def test_missing_extract_on_a_referenced_step_is_completed_from_its_query():
    p = plan({"id": "s1", "purpose": "找事故", "query": "事故 INC-7 中出故障的那个服务"},
             {"id": "s2", "purpose": "查阈值", "query": "{s1.name} 超时阈值", "depends_on": ["s1"]})
    assert validate_plan("事故 INC-7 中出故障的那个服务，超时阈值？", p) == []
    assert p.steps[0].extract.name == "name" and p.steps[0].extract.description == "事故 INC-7 中出故障的那个服务"


def test_reference_to_a_differently_named_single_extract_is_rebound():
    p = plan({"id": "s1", "purpose": "找事故", "query": "事故 INC-7 复盘",
              "extract": {"name": "service", "kind": "entity", "description": "故障服务"}},
             {"id": "s2", "purpose": "查阈值", "query": "{s1.faulty_service_name} 超时阈值", "depends_on": ["s1"]})
    assert validate_plan("事故 INC-7 中出故障的服务超时阈值？", p) == []
    assert p.steps[1].query == "{s1.service} 超时阈值"


def test_focus_document_prefers_distinctive_terms_and_skips_documents_already_read():
    from agent.planned import focus_document

    hits = [{"document_id": "svc-a", "title": "青鸾网关运维手册", "text": "单个请求超时阈值为 6 秒。"},
            {"document_id": "inc-7", "title": "事故 INC-7 复盘", "text": "白泽账务出现定时任务漏跑。"},
            {"document_id": "svc-b", "title": "白泽账务运维手册", "text": "单个请求超时阈值为 10 秒。"}]
    assert focus_document("白泽账务 单个请求超时阈值", hits, used=["inc-7"])[0] == "svc-b"
    glossary = [{"document_id": "form", "title": "跨仓调拨申请单管理规定", "text": "单笔金额超过 8000 元的须审批。"},
                {"document_id": "terms", "title": "内部常用俗称对照表", "text": "“外发单”：即外协加工单。审批使用正式名称。"}]
    assert focus_document("外发单 金额 审批", glossary)[0] == "terms"


def test_search_keyed_by_an_extracted_date_becomes_a_dated_version_step():
    def dated_plan():
        return plan({"id": "s1", "purpose": "查公告日期", "query": "行政公告 AN-16",
                     "extract": {"name": "day", "kind": "date", "description": "修订生效日期"}},
                    {"id": "s2", "purpose": "查有效期", "query": "门禁卡有效期 {s1.day}", "depends_on": ["s1"]})
    before = dated_plan()
    assert validate_plan("行政公告 AN-16 宣布的修订生效之前，门禁卡有效期是多少？", before) == []
    assert (before.steps[1].kind, before.steps[1].as_of, before.steps[1].when) == ("version_as_of", "{s1.day}", "before")
    after = dated_plan()
    validate_plan("行政公告 AN-16 宣布的修订生效之后、下一次修订之前，门禁卡有效期是多少？", after)
    assert after.steps[1].when == "at"


def test_per_item_question_gets_a_fan_out_step_for_the_asked_attribute():
    p = plan({"id": "s1", "purpose": "查规则覆盖的服务", "query": "变更冻结规则 R-8 覆盖的服务",
              "extract": {"name": "services", "kind": "entity", "description": "覆盖的服务"}})
    assert validate_plan("变更冻结规则 R-8 覆盖的服务分别由谁负责？", p) == []
    assert p.steps[0].extract.kind == "list"
    added = p.steps[1]
    assert added.foreach and added.depends_on == ["s1"] and added.query == "{s1.services} 由谁负责"
    single = plan({"id": "s1", "purpose": "查负责人", "query": "白泽账务 负责人",
                   "extract": {"name": "owner", "kind": "entity", "description": "负责人"}})
    validate_plan("白泽账务由谁负责？", single)
    assert len(single.steps) == 1


def test_value_is_not_taken_from_a_document_that_lacks_the_step_subject():
    from agent.planned import ungrounded_terms

    hits = [{"document_id": "terms", "title": "内部常用俗称对照表", "text": "“外发单”：即外协加工单。"},
            {"document_id": "form-a", "title": "跨仓调拨申请单管理规定", "text": "超过 8000 元的须审批。"},
            {"document_id": "form-b", "title": "报废处置单管理规定", "text": "超过 5000 元的须审批。"},
            {"document_id": "form-c", "title": "临时用工申请单管理规定", "text": "超过 50000 元的须审批。"}]
    assert ungrounded_terms("外发单 审批金额", hits, "form-a") == ["发单", "外发"]
    assert ungrounded_terms("外发单 审批金额", hits, "terms") == []


def test_ungrounded_nickname_is_resolved_from_the_document_that_defines_it():
    corpus = {"g": ("terms", "“外发单”：即外协加工单，因纸质版颜色得名。"),
              "f1": ("form-a", "跨仓调拨申请单：超过 8000 元的须由财务总监审批。"),
              "f2": ("form-b", "外协加工单：超过 5000 元的须由质量总监审批。"),
              "f3": ("form-c", "报废处置单：超过 3000 元的须由质量总监审批。"),
              "f4": ("form-d", "临时用工申请单：超过 50000 元的须由供应链总监审批。")}
    replies = [
        {"steps": [{"id": "s1", "purpose": "查门槛", "query": "外发单 审批金额",
                    "extract": {"name": "amount", "kind": "value", "description": "审批金额门槛"}}]},
        "无",          # the nickname's own document has no amount
        "外协加工单",   # resolution step: what "外发单" refers to
        "5000 元",      # rerun with the official name
    ]
    search = {"外发单 指的是": ["g", "f1"], "外发单": ["g", "f1", "f3", "f4"], "外协加工单": ["f2", "f1", "f3"]}
    result, calls, events = run_planner(replies, corpus, search)
    replans = [payload for kind, payload in events if kind == "replan"]
    assert [p.get("kind") for p in replans] == ["term_resolution"]  # no model replan was needed
    assert result["planner"]["values"].get("s0.resolved") == ["外协加工单"]
    assert result["planner"]["values"].get("s1.amount") == ["5000 元"]


def test_table_value_must_come_from_the_row_the_question_names():
    table = [{"chunk_id": "t", "title": "质量报表",
              "text": "## 退货率\n| 仓库 | 季度退货率 |\n| 东仓 | 4.6% |\n| 南仓 | 2.8% |"}]
    assert literal_values("4.6%", table, "value", query="南仓 第三季度 退货率") == (None, None, None)
    assert literal_values("2.8%", table, "value", query="南仓 第三季度 退货率") == (["2.8%"], "t", "| 南仓 | 2.8% |")


def test_claims_get_the_verified_bridge_quote_they_rely_on():
    from agent.planned import cite_bridges
    from app.clients import Claim, GeneratedAnswer

    evidence = [
        {"id": "E1", "chunk_id": "svc", "document_id": "svc-11", "title": "白泽账务运维手册", "text": "单个请求超时阈值为 10 秒。"},
        {"id": "E2", "chunk_id": "inc", "document_id": "inc-148", "title": "事故 QINGHE-INC-148 复盘",
         "text": "## 概述\n2026-09-14，白泽账务出现定时任务漏跑。"},
        {"id": "E3", "chunk_id": "svc2", "document_id": "svc-12", "title": "青鸾网关运维手册", "text": "单个请求超时阈值为 6 秒。"},
        {"id": "E4", "chunk_id": "inc-h", "document_id": "inc-148", "title": "事故 QINGHE-INC-148 复盘", "text": "# 事故 QINGHE-INC-148 复盘"},
        {"id": "E5", "chunk_id": "inc-r", "document_id": "inc-148", "title": "事故 QINGHE-INC-148 复盘", "text": "## 根因\n配置推送遗漏。"},
    ]  # the incident spans three chunks of one document
    verified = [{"step": "s1", "what": "故障服务", "values": ["白泽账务"], "quote": "2026-09-14，白泽账务出现定时任务漏跑。", "chunk_id": "inc"}]
    answer = GeneratedAnswer(answerable=True, claims=[
        Claim(text="事故 QINGHE-INC-148 中出故障的服务超时阈值是 10 秒。", evidence_ids=["E1"], quotes=["单个请求超时阈值为 10 秒。"]),
        Claim(text="青鸾网关的超时阈值是 6 秒。", evidence_ids=["E3"], quotes=["单个请求超时阈值为 6 秒。"]),
    ])
    answer, added = cite_bridges("事故 QINGHE-INC-148 中出故障的那个服务，超时阈值是多少秒？", answer, evidence, verified)
    assert answer.claims[0].evidence_ids == ["E1", "E2"] and added == [{"claim_index": 0, "chunk_id": "inc"}]
    assert answer.claims[1].evidence_ids == ["E3"]  # a claim not about the bridged subject is left alone
