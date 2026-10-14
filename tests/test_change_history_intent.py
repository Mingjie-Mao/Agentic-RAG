"""'变更' is a history signal only when it says that content changed."""

import pytest

from app.routing import execution_route
from app.task_analysis import historical_route_intent, version_intent
from app.task_contract import build_contract

BUSINESS_OBJECTS = [
    "变更冻结规则 ACME-R-800 覆盖的服务分别由谁负责？",
    "变更管理流程需要谁审批？",
    "日常变更需提前几个工作日申请？",
    "有几个变更冻结规则？",
    "哪些服务在变更窗口内不能发布？",
    "变更申请单由谁审批？",
]
CONTENT_CHANGES = [
    "报销制度有什么变更？",
    "审批时限变更了吗？",
    "变更前的门禁卡有效期是多少？",
    "最近一次变更是什么时候生效的？",
    "日志保留政策有没有变更？",
    "各版本的变更内容是什么？",
    "差旅标准发生过变更吗？",
    "这份制度做了哪些变更？",
    "住宿标准变更为多少？",
]


@pytest.mark.parametrize("question", BUSINESS_OBJECTS)
def test_change_as_business_object_is_not_history(question):
    assert not historical_route_intent(question)
    assert not version_intent(question)
    assert build_contract(question).intent != "history"
    assert execution_route(question)["reason"] != "explicit_version_workflow"


@pytest.mark.parametrize("question", CONTENT_CHANGES)
def test_change_of_content_is_still_history(question):
    assert historical_route_intent(question)
    assert version_intent(question)
    assert build_contract(question).intent == "history"
    assert execution_route(question)["reason"] == "explicit_version_workflow"
