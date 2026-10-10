from agent.adaptive import escalation_reason


def row(chunk, document, title, text):
    return {"chunk_id": chunk, "document_id": document, "title": title, "text": text}


EVIDENCE = [
    row("g", "terms", "内部常用俗称对照表", "“外发单”：即外协加工单。"),
    row("f1", "form-a", "跨仓调拨申请单管理规定", "超过 8000 元的须由财务总监审批。"),
    row("f2", "form-b", "外协加工单管理规定", "超过 5000 元的须由质量总监审批。"),
    row("f3", "form-c", "报废处置单管理规定", "超过 5000 元的须由质量总监审批。"),
    row("f4", "form-d", "临时用工申请单管理规定", "超过 50000 元的须由供应链总监审批。"),
]


def answer(*chunks, status="answered"):
    return {"status": status, "citations": [{"chunk_id": c} for c in chunks]}


def test_abstention_escalates_and_a_grounded_answer_stands():
    goal = "外发单的金额超过多少元时需要更高级别审批？"
    assert escalation_reason(goal, answer(status="insufficient_evidence"), EVIDENCE)["reason"] == "first_pass_abstained"
    assert escalation_reason(goal, answer("f2", "g"), EVIDENCE) is None


def test_answer_that_never_cites_the_question_subject_escalates():
    reason = escalation_reason("外发单的金额超过多少元时需要更高级别审批？", answer("f2"), EVIDENCE)
    assert reason["reason"] == "answer_not_grounded_in_question_subject" and "外发" in reason["terms"]


def test_conflict_and_simple_lookups_are_not_escalated():
    assert escalation_reason("外发单审批", answer("g", status="conflict"), EVIDENCE) is None
    assert escalation_reason("外协加工单超过多少元由谁审批？", answer("f2"), EVIDENCE) is None
