from app.clients import Claim, DependencyError, GeneratedAnswer
from app.fact_integrity import missing_slots, required_slots
from app.qa import repair_exact_values


def evidence(text):
    return [{"id": "E1", "chunk_id": "c1", "document_id": "d1", "title": "Runbook", "text": text}]


def test_unique_labelled_event_id_and_numeric_threshold():
    rows = evidence("event_id: ABC-4826\n回滚阈值：错误率达到 2% 时执行回滚。")
    slots = required_slots("请给出 event_id 和回滚阈值", rows)
    assert {(row["field"], row["value"]) for row in slots} == {
        ("event_id", "ABC-4826"), ("rollback_threshold", "2%")
    }
    assert len(missing_slots(slots, [{"text": "编号是 XABC-48260，阈值是 12%。"}])) == 2
    assert missing_slots(slots, [{"text": "event_id 为 ABC-4826；回滚阈值 2.0%。"}]) == []


def test_ambiguous_values_are_not_forced_into_answer():
    rows = evidence("event_id: ABC-4826\n回滚阈值为 2% 或 3%。")
    rows += [{"id": "E2", "chunk_id": "c2", "text": "event_id: DEF-7777"}]
    assert required_slots("event_id 与回滚阈值？", rows) == []


def test_field_name_and_chinese_percent_are_checked_from_realistic_runbook_text():
    rows = evidence("按 event_id 建立去重表。\n回滚触发\n发布后错误率连续 5 分钟超过百分之二，应启动回滚。")
    slots = required_slots("给出客户端去重字段与回滚阈值", rows)
    assert {(slot["field"], slot["value"]) for slot in slots} == {
        ("event_id_field", "event_id"), ("rollback_threshold", "百分之二")
    }
    assert missing_slots(slots, [{"text": "按 event_id 去重，超过 2% 就回滚。"}]) == []


def test_rollback_target_is_taken_from_unique_cited_instruction():
    rows = evidence("回滚使用上一稳定版本镜像，保留当前事故日志。")
    slots = required_slots("若需回滚，请说明目标版本", rows)
    assert [(slot["field"], slot["value"]) for slot in slots] == [
        ("rollback_target", "上一稳定版本镜像")
    ]
    assert missing_slots(slots, [{"text": "应回滚至上一稳定版本镜像。"}]) == []
    assert len(missing_slots(slots, [{"text": "应回滚至当前版本。"}])) == 1
    conflicting = rows + [{"chunk_id": "c2", "text": "回滚使用另一个稳定版本镜像。"}]
    assert required_slots("若需回滚，请说明目标版本", conflicting) == []


def test_repair_adds_only_claim_with_the_missing_value_and_source():
    rows = evidence("event_id: ABC-4826\n回滚阈值：错误率达到 2% 时执行回滚。")

    class Stub:
        def generate(self, *_args, **_kwargs):
            return GeneratedAnswer(answerable=True, claims=[
                Claim(text="event_id 为 ABC-4826。", evidence_ids=["E1"], quotes=["event_id: ABC-4826"]),
                Claim(text="无关推论。", evidence_ids=["E1"], quotes=["event_id: ABC-4826"]),
            ]), {"prompt_tokens": 10, "completion_tokens": 5}

    claims, usage = repair_exact_values(
        Stub(), "event_id 是什么？", rows, [{"text": "事故已记录。", "evidence_ids": ["c1"]}], {}
    )
    assert len(claims) == 2
    assert claims[1]["text"] == "event_id 为 ABC-4826。"
    assert usage["exact_value_repair_claims"] == 1


def test_repair_failure_keeps_previous_cited_claims():
    class Broken:
        def generate(self, *_args, **_kwargs):
            raise DependencyError("offline", stage="generation")

    original = [{"text": "事故已记录。", "evidence_ids": ["c1"]}]
    claims, usage = repair_exact_values(
        Broken(), "event_id 是什么？", evidence("event_id: ABC-4826"), original, {}
    )
    assert claims == original
    assert usage["exact_value_repair_status"] == "unavailable"
