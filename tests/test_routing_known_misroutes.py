"""Known `auto` misroutes from the 2026-10-12 review, pinned as strict xfails.

Each case states the desired route. They fail today by design; once routing is
fixed the strict xfail turns into a failure, so the marker must be removed
together with the fix. The router itself is intentionally unchanged here.
"""

import pytest

from app.routing import execution_route

KNOWN = pytest.mark.xfail(strict=True, reason="known auto misroute; see docs/problem-analysis-and-roadmap-20261012.md")


@KNOWN
@pytest.mark.parametrize("question", [
    "事故 ACME-INC-901 中出故障的那个服务，单个请求的超时阈值是多少秒？",
    "事故 ACME-INC-902 中出故障的服务，其负责人休假时由谁代班？代班人的办公分机是多少？",
])
def test_bridge_question_is_not_single_shot(question):
    # The asked attribute belongs to an entity that only the first hop reveals.
    assert not execution_route(question).get("direct")


@KNOWN
def test_plain_policy_conditional_does_not_need_an_agent():
    # A conditional sentence is not an observation dependency.
    assert execution_route("如果员工请假超过三天，需要提前几天申请？")["mode"] == "workflow"


def test_change_freeze_rule_is_not_a_version_history_request():
    # Fixed 2026-10-13: no longer a known misroute.
    # "变更" names the rule (change freeze), it does not ask for document history.
    assert execution_route("变更冻结规则 ACME-R-800 覆盖的服务分别由谁负责？")["reason"] != "explicit_version_workflow"


def test_explicit_history_request_still_uses_version_workflow():
    assert execution_route("报销制度的历史版本中审批时限有什么变化？")["reason"] == "explicit_version_workflow"
