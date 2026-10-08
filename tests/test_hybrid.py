import json
from types import SimpleNamespace

import pytest

from agent import hybrid
from agent.hybrid import ActionIntent, AgentState, analyze_goal, resolve_document, run_hybrid


class FakeDB:
    def refresh(self, _task):
        pass

    def commit(self):
        pass


def make_task(goal, task_input=None, max_steps=6):
    return SimpleNamespace(goal=goal, input=task_input or {}, step_no=0, max_steps=max_steps, status="running")


class Result(SimpleNamespace):
    pass


def ok(refs, matches=None):
    return Result(status="ok", evidence_refs=refs, data={"matches": matches or []})


CHUNKS = {
    "c1": {"chunk_id": "c1", "document_id": "d-log", "title": "日志制度", "text": "业务日志保留 90 天。"},
    "c2": {"chunk_id": "c2", "document_id": "d-x", "title": "上传指南", "text": "单个文件上限 20 MB。"},
    "c3": {"chunk_id": "c3", "document_id": "d-y", "title": "会议纪要", "text": "纪要须在会后 24 小时内归档。"},
    "c4": {"chunk_id": "c4", "document_id": "d-inc", "title": "事故 092 复盘",
           "text": "事故 092 的输入特征：单个超大文件阻塞了处理队列。"},
}


def evidence_from_refs(_db, _user, refs, allow_historical=False):
    return [dict(CHUNKS[ref], id=f"E{n}") for n, ref in enumerate(dict.fromkeys(refs), 1) if ref in CHUNKS]


def harness(task, responses, *, policy=None, versions=None):
    calls, events = [], []

    def execute(_db, t, _tools, name, arguments, *, reuse=False):
        calls.append((name, arguments))
        t.step_no += 1
        return responses.pop(0) if responses else ok([])

    def event(_db, _t, kind, *, payload=None, refs=None, tool_name=None):
        events.append((kind, payload or {}))

    walked = []

    def walk_versions(_db, t, _tools, document_id):
        walked.append(document_id)
        t.step_no += 2
        return versions or []

    models = SimpleNamespace(agent_policy_usage=None)
    intents = list(policy or [])

    def choose(_models, _state, _remaining):
        if not intents:
            raise AssertionError("the policy was asked although the controller should have decided")
        return intents.pop(0)

    return calls, events, walked, models, choose, execute, event, walk_versions


def run(task, responses, *, policy=None, versions=None, history=False, monkeypatch):
    calls, events, walked, models, choose, execute, event, walk = harness(
        task, responses, policy=policy, versions=versions
    )
    monkeypatch.setattr(hybrid, "choose_intent", choose)
    refs = run_hybrid(
        FakeDB(), SimpleNamespace(), task, models, None,
        execute=execute, event=event, evidence_from_refs=evidence_from_refs,
        walk_versions=walk, uses_history=lambda *_: history or bool(walked),
    )
    return refs, calls, events, walked


def test_the_policy_can_only_name_an_information_goal_never_an_id_or_argument():
    fields = set(ActionIntent.model_json_schema()["properties"])
    assert fields == {"kind", "subgoal_id", "query", "target", "purpose"}
    with pytest.raises(ValueError):
        ActionIntent.model_validate({"kind": "final", "document_id": "d-log"})
    with pytest.raises(ValueError):
        ActionIntent.model_validate({"kind": "compare_versions"})


def test_the_policy_view_holds_titles_and_coverage_but_no_ids():
    state = AgentState(
        goal="日志保留多少天？",
        subgoals=[{"id": "s1", "text": "日志保留多少天", "covered": False, "snippets": []}],
        version_required=False,
        known_documents={"d-secret-id": "日志制度"},
        evidence_refs=["c1"],
    )
    view = json.dumps(state.view(4), ensure_ascii=False)
    assert "日志制度" in view and "d-secret-id" not in view and "c1" not in view


def test_documents_are_resolved_from_held_ids_never_from_model_text():
    state = AgentState(goal="g", subgoals=[], version_required=True,
                       known_documents={"d1": "P1 支持政策", "d2": "发布手册"})
    assert resolve_document(make_task("g", {"document_id": "given"}), state, "anything") == "given"
    assert resolve_document(make_task("g"), state, "P1 支持政策") == "d1"
    assert resolve_document(make_task("g"), state, None) is None  # two candidates, no guess
    assert resolve_document(make_task("g"), state, "d2") is None  # an id typed by the model is not a title


def test_evidence_that_already_covers_the_goal_stops_without_asking_the_policy(monkeypatch):
    task = make_task("日志保留多少天？")
    refs, calls, events, _ = run(task, [ok(["c1"], [{"document_id": "d-log", "title": "日志制度"}])],
                                 monkeypatch=monkeypatch)
    assert refs == ["c1"]
    assert [name for name, _ in calls] == ["search_documents"]
    assert ("hybrid_stop", {"reason": "evidence_ready", "binding_failures": 0,
                            "basis": "lexical_coverage_proxy; not semantic correctness"}) in events


def test_an_empty_search_is_retried_once_by_protocol_not_by_the_policy(monkeypatch):
    task = make_task("客户鉴权突然失效，那个认证错误代码是什么意思？")
    _, calls, _, _ = run(
        task, [ok([]), ok([])],
        policy=[ActionIntent(kind="insufficient_evidence")], monkeypatch=monkeypatch,
    )
    searches = [args["query"] for name, args in calls if name == "search_documents"]
    assert len(searches) == 2 and searches[0] != searches[1]


def test_a_known_history_document_walks_its_versions_before_any_planning(monkeypatch):
    task = make_task("列出 v2 与 v3 的首次响应时限", {"document_id": "d-sla"})
    _, calls, _, walked = run(task, [], policy=[ActionIntent(kind="final")], monkeypatch=monkeypatch)
    assert walked == ["d-sla"] and calls == []


def test_the_policy_cannot_finish_a_history_question_before_the_history_is_read(monkeypatch):
    task = make_task("P1 支持政策历史上各版本的时限是多少？")
    first = ok(["c1"], [{"document_id": "d1", "title": "P1 支持政策"}])
    _, _, events, walked = run(
        task, [first], policy=[ActionIntent(kind="final"), ActionIntent(kind="final")],
        monkeypatch=monkeypatch,
    )
    assert walked == ["d1"]  # the controller read the history, then honoured the stop


def test_an_unbindable_intent_costs_no_tool_call_and_repeated_stalls_stop(monkeypatch):
    task = make_task("先查事故 092 的输入特征，再说明当前上传限制。")
    first = ok(["c1"], [{"document_id": "d-log", "title": "日志制度"},
                        {"document_id": "d-x", "title": "上传指南"}])
    _, calls, events, _ = run(
        task, [first],
        policy=[ActionIntent(kind="inspect_document"), ActionIntent(kind="inspect_document")],
        monkeypatch=monkeypatch,
    )
    assert [name for name, _ in calls] == ["search_documents"]
    assert [p["reason"] for kind, p in events if kind == "binding_failed"] == ["document_unresolved"] * 2
    assert any(kind == "hybrid_stop" and p["reason"] == "no_progress" for kind, p in events)


def test_repeating_an_already_checked_version_walk_stops_instead_of_looping(monkeypatch):
    task = make_task("差旅标准历史上最早的住宿标准是多少？", {"document_id": "d-travel"}, max_steps=7)
    again = ActionIntent(kind="inspect_versions", target="差旅标准")
    _, _, events, walked = run(task, [], policy=[again] * 2, history=True, monkeypatch=monkeypatch)
    assert walked == ["d-travel"]
    assert [p["reason"] for kind, p in events if kind == "binding_failed"] == ["already_checked"] * 2
    assert any(kind == "hybrid_stop" and p["reason"] == "no_progress" for kind, p in events)


def test_analysis_marks_history_from_the_task_input_or_the_goal():
    assert analyze_goal(make_task("日志保留多少天？")).version_required is False
    assert analyze_goal(make_task("日志保留多少天？", {"document_id": "d"})).version_required is True


def test_more_unrelated_chunks_are_not_progress(monkeypatch):
    task = make_task("日志保留多少天？", max_steps=8)
    _, calls, events, _ = run(
        task, [ok(["c2"]), ok(["c3"]), ok(["c3"])],
        policy=[ActionIntent(kind="follow_up_search", query="日志 保留期"),
                ActionIntent(kind="follow_up_search", query="日志 存档 天数")],
        monkeypatch=monkeypatch,
    )
    assert [name for name, _ in calls] == ["search_documents"] * 3
    assert any(kind == "hybrid_stop" and p["reason"] == "no_progress" for kind, p in events)


def test_a_follow_up_without_a_query_carries_only_relevant_first_hop_evidence(monkeypatch):
    task = make_task("先查事故 092 的输入特征，再说明当前上传限制。")
    first = ok(["c4", "c3"], [{"document_id": "d-inc", "title": "事故 092 复盘"},
                              {"document_id": "d-y", "title": "会议纪要"}])
    _, calls, _, _ = run(
        task, [first, ok(["c2"])],
        policy=[ActionIntent(kind="follow_up_search"), ActionIntent(kind="final")],
        monkeypatch=monkeypatch,
    )
    second = [args["query"] for name, args in calls if name == "search_documents"][1]
    assert "超大文件" in second and "纪要" not in second


def test_documents_that_cover_no_subgoal_are_dropped_from_the_final_evidence():
    state = AgentState(
        goal="g", version_required=False,
        subgoals=[{"id": "s1", "text": "a", "covered": True, "refs": ["c4"], "snippets": []},
                  {"id": "s2", "text": "b", "covered": False, "refs": [], "snippets": []}],
        evidence_refs=["c3", "c4", "c5"],
        ref_documents={"c3": "d-noise", "c4": "d-inc", "c5": "d-inc"},
    )
    assert hybrid.final_refs(state) == ["c4", "c5"]
    state.subgoals[0]["covered"] = False
    assert hybrid.final_refs(state) == ["c3", "c4", "c5"]  # nothing covered: keep all


def test_semantic_shadow_records_but_never_changes_behaviour(monkeypatch):
    from app import semantic_evidence
    from app.config import settings

    class FakeEvaluator(semantic_evidence.SemanticEvidenceEvaluator):
        def __init__(self):
            super().__init__(
                relevance_scorer=lambda q, texts: [5.0] * len(texts),
                judge=lambda payload: {"output": {"judgments": [
                    {"subgoal_id": s["id"], "status": "unsupported", "support_refs": [], "missing_slots": [],
                     "reason_code": "topic_only", "confidence": 0.9} for s in payload["subgoals"]]}},
            )

    monkeypatch.setattr(semantic_evidence, "SemanticEvidenceEvaluator", FakeEvaluator)

    def once(shadow):
        monkeypatch.setattr(settings(), "semantic_coverage_shadow", shadow)
        task = make_task("先查事故 092 的输入特征，再说明当前上传限制。")
        first = ok(["c4", "c3"], [{"document_id": "d-inc", "title": "事故 092 复盘", "chunk_id": "c4"},
                                  {"document_id": "d-y", "title": "会议纪要", "chunk_id": "c3"}])
        return run(task, [first, ok(["c2"])],
                   policy=[ActionIntent(kind="follow_up_search", query="上传限制"), ActionIntent(kind="final")],
                   monkeypatch=monkeypatch)

    off_refs, off_calls, off_events, _ = once(False)
    on_refs, on_calls, on_events, _ = once(True)
    assert on_refs == off_refs and on_calls == off_calls
    shadow = [p for kind, p in on_events if kind == "semantic_coverage_shadow"]
    assert shadow and all(p["mode"].startswith("shadow") for p in shadow)
    assert not any(kind == "semantic_coverage_shadow" for kind, _ in off_events)


def test_every_hybrid_event_type_fits_the_audit_column():
    import re
    from pathlib import Path

    from app.models import AgentEvent

    width = AgentEvent.__table__.c.event_type.type.length
    source = Path(hybrid.__file__).read_text()
    names = set(re.findall(r'event\(\s*(?:self\.)?db,\s*(?:self\.)?task,\s*"([a-z_]+)"', source))
    assert names and all(len(name) <= width for name in names), {n: len(n) for n in names}


def test_a_failing_shadow_never_breaks_the_task(monkeypatch):
    from app.config import settings

    class Exploding(hybrid.SemanticShadow):
        def __init__(self, db, *args):
            self.db, self.rolled_back = db, 0

        def _observe(self, **_kwargs):
            raise RuntimeError("shadow bug")

        def _final(self, *_args, **_kwargs):
            raise RuntimeError("shadow bug")

    class RollbackDB(FakeDB):
        def rollback(self):
            pass

    monkeypatch.setattr(hybrid, "SemanticShadow", Exploding)
    monkeypatch.setattr(settings(), "semantic_coverage_shadow", True)
    calls, events, walked, models, choose, execute, event, walk = harness(
        make_task("日志保留多少天？"), [ok(["c1"], [{"document_id": "d-log", "title": "日志制度"}])]
    )
    monkeypatch.setattr(hybrid, "choose_intent", choose)
    refs = run_hybrid(RollbackDB(), SimpleNamespace(), make_task("日志保留多少天？"), models, None,
                      execute=execute, event=event, evidence_from_refs=evidence_from_refs,
                      walk_versions=walk, uses_history=lambda *_: False)
    assert refs == ["c1"]


def test_semantic_control_stops_on_support_and_ranks_instead_of_deleting(monkeypatch):
    from app import semantic_evidence
    from app.config import settings

    scores = {"c1": 5.0, "c3": -9.0}

    class FakeEvaluator(semantic_evidence.SemanticEvidenceEvaluator):
        def __init__(self):
            super().__init__(
                relevance_scorer=lambda q, texts: [5.0 if "日志" in t else -9.0 for t in texts],
                judge=lambda payload: {"output": {"judgments": [
                    {"subgoal_id": s["id"], "status": "supported", "support_refs": ["E1"], "missing_slots": [],
                     "reason_code": "value_present", "confidence": 1.0} for s in payload["subgoals"]]}},
            )

    monkeypatch.setattr(semantic_evidence, "SemanticEvidenceEvaluator", FakeEvaluator)
    monkeypatch.setattr(settings(), "semantic_coverage_control", True)
    task = make_task("日志保留多少天？")
    first = ok(["c1", "c3"], [{"document_id": "d-log", "title": "日志制度", "chunk_id": "c1"},
                              {"document_id": "d-y", "title": "会议纪要", "chunk_id": "c3"}])
    refs, calls, events, _ = run(task, [first], monkeypatch=monkeypatch)  # no policy intents allowed
    assert refs == ["c1"]  # c3 is below the recall cut for every subgoal
    assert any(kind == "semantic_coverage_control" for kind, _ in events)
    assert scores  # documents the intent of the fixture


def test_numeric_limit_document_survives_hybrid_context_selection_without_extra_policy(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings(), "answer_quality_enabled", True)
    extra = {
        "incident": {"chunk_id":"incident","document_id":"incident-doc","title":"事故乙复盘","text":"事故乙的导入任务提交9万行，没有分片。"},
        "alarm": {"chunk_id":"alarm","document_id":"incident-doc","title":"导入改进","text":"新增导入行数告警，单个任务超过上限通知值班。"},
        "limit": {"chunk_id":"limit","document_id":"import-doc","title":"导入指南","text":"单次导入任务最多7万行。"},
    }
    for key, value in extra.items():
        monkeypatch.setitem(CHUNKS, key, value)
    task = make_task("事故乙的导入有什么特征？按这个特征对应的规则，现在单次导入最多允许多少行？")
    matches = [{"chunk_id":key,"document_id":value["document_id"],"title":value["title"],"snippet":value["text"]} for key,value in extra.items()]
    refs,calls,events,_ = run(task,[ok(list(extra),matches)],monkeypatch=monkeypatch)
    assert "limit" in refs and "incident" in refs
    assert len(calls)==1 and task.step_no==1
