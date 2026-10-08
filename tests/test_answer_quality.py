"""Business failures, bounded recovery and source-bound comparisons."""

import pytest

from app.clients import Claim, DependencyError, GeneratedAnswer
from app.config import settings
from app.execution_budget import BudgetExceeded, ExecutionBudget, use_budget
from app.answer_quality import recover_answer
from app.qa import claim_validation_issues, validate_claims


def row(text, key="a", **kwargs):
    return dict(
        id=key,
        chunk_id=key,
        document_id="doc-" + key,
        version_id="v-" + key,
        title="规则",
        text=text,
        **kwargs,
    )


class Drafts:
    def __init__(self, *drafts):
        self.drafts, self.calls = list(drafts), []

    def generate(self, q, evidence, **kwargs):
        self.calls.append(kwargs)
        answer = self.drafts.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer, {"prompt_tokens": 5, "completion_tokens": 3}


def answer(text, quote, key="a"):
    return GeneratedAnswer(
        answerable=True, claims=[Claim(text=text, evidence_ids=[key], quotes=[quote])]
    )


def test_local_repair_preserves_valid_claim_and_reports_actual_failure():
    evidence = [row("截止时间为7天。"), row("刷新间隔为12分钟。", "b")]
    valid = Claim(text="截止时间为7天。", evidence_ids=["a"], quotes=["截止时间为7天。"])
    draft = answer("刷新间隔为9分钟。", "刷新间隔为12分钟。", "b")
    draft.claims.insert(0, valid)
    models = Drafts(draft, answer("刷新间隔为12分钟。", "刷新间隔为12分钟。", "b"))
    result, usage = recover_answer(models, "截止时间与刷新间隔？", evidence)
    assert result.claims[0] == valid
    assert validate_claims(result, evidence)[1] == "answered"
    assert len(models.calls) == 2
    assert usage["prompt_tokens"] == 10
    assert usage["answer_recovery"]["first_issues"] == [
        {"claim_index": 1, "codes": ["unsupported_cited_quantity"]}
    ]
    assert not usage["answer_recovery"]["remaining_issues"]
    assert models.calls[1]["repair_context"]["retained_claims"] == [valid.text]


@pytest.mark.parametrize("value", ["5万", "五万", "50,000", "50000", "5 万"])
def test_row_quantities_normalize_without_equating_other_units(value):
    evidence = [row("单次最多5万行。")]
    assert not claim_validation_issues(answer(f"单次最多{value}行。", evidence[0]["text"]), evidence)
    assert claim_validation_issues(answer("单次最多50000次。", evidence[0]["text"]), evidence)


def test_unsupported_grouped_row_count_is_repaired_from_actual_limit():
    alarm = "新增行数告警：超过上限通知值班。"
    limit = "单次最多5万行。"
    evidence = [row(alarm), row(limit, "b")]
    models = Drafts(answer("上限为10,000行。", alarm), answer("上限为五万行。", limit, "b"))
    result, usage = recover_answer(models, "当前行数上限？", evidence)
    assert validate_claims(result, evidence)[1] == "answered"
    assert usage["answer_recovery"]["first_issues"][0]["codes"] == ["unsupported_cited_quantity"]
    assert usage["answer_recovery"]["repair_attempts"] == 1
    assert result.claims[0].evidence_ids == ["b"]


def test_claim_supported_only_by_furniture_still_goes_to_repair(monkeypatch):
    monkeypatch.setattr(settings(), "answer_quality_enabled", True)
    evidence = [row("## 改进措施\n新增告警。"), row("单次最多5万行。", "b")]
    models = Drafts(answer("单次最多5万行。", "改进措施"), answer("单次最多5万行。", "单次最多5万行。", "b"))
    result, usage = recover_answer(models, "上限？", evidence)
    assert "unrelated_heading_reference" in usage["answer_recovery"]["first_issues"][0]["codes"]
    assert usage["answer_recovery"]["repair_attempts"] == 1 and result.claims[0].evidence_ids == ["b"]


def test_shard_size_cannot_support_total_import_limit():
    evidence = [row("每片5000行。")]
    assert claim_validation_issues(answer("任务总上限为5万行。", evidence[0]["text"]), evidence)


def test_bare_unrelated_heading_does_not_support_numeric_fact(monkeypatch):
    monkeypatch.setattr(settings(), "answer_quality_enabled", True)
    evidence = [row("单次最多5万行。"), row("## 改进措施\n新增告警。", "b")]
    draft = GeneratedAnswer(answerable=True, claims=[Claim(
        text="单次最多5万行。", evidence_ids=["a", "b"], quotes=["单次最多5万行。", "改进措施"]
    )])
    assert claim_validation_issues(draft, evidence) == [{"claim_index": 0, "codes": ["unrelated_heading_reference"]}]


def test_embed_or_consent_boilerplate_cannot_support_a_claim(monkeypatch):
    monkeypatch.setattr(settings(), "answer_quality_enabled", True)
    embed = ("This Instagram post cannot be displayed in your browser. Please enable Javascript or try a "
             "different browser. View original content on Instagram The BBC is not responsible for the content "
             "of external sites. Skip instagram post by cadaea Allow Instagram content? We ask for your "
             "permission before anything is loaded, as they may be using cookies and other technologies.")
    content = "Sophie said the game finally let her play with one hand, which changed everything for her."
    evidence = [row(embed + " " + content), row("Copycat cookie paywalls quickly sprung up on news publishers.", "b")]
    claim = "Sophie says games became more accessible to her."
    bad = GeneratedAnswer(answerable=True, claims=[Claim(text=claim, evidence_ids=["a"], quotes=[embed])])
    assert claim_validation_issues(bad, evidence) == [{"claim_index": 0, "codes": ["boilerplate_reference"]}]
    # Reporting about cookies, and real content beside an embed placeholder, stay citable.
    good = GeneratedAnswer(answerable=True, claims=[
        Claim(text=claim, evidence_ids=["a"], quotes=[content]),
        Claim(text="News publishers copied cookie paywalls.", evidence_ids=["b"],
              quotes=["Copycat cookie paywalls quickly sprung up on news publishers."]),
    ])
    assert not claim_validation_issues(good, evidence)
    from app.qa import boilerplate_quote

    assert not any(boilerplate_quote(q) for q in (
        "Meta has been using cookies to track users across sites.",
        "Users must accept all cookies or pay a monthly fee.",
        "Meta's new cookie policy requires consent in the EU."))


def test_entity_heading_and_factual_numeric_headline_remain_citable(monkeypatch):
    monkeypatch.setattr(settings(), "answer_quality_enabled", True)
    evidence = [row("## 北区中心\n响应时间为7分钟。")]
    draft = GeneratedAnswer(answerable=True, claims=[Claim(
        text="北区中心响应时间为7分钟。", evidence_ids=["a", "a"], quotes=["北区中心", "响应时间为7分钟。"]
    )])
    assert not claim_validation_issues(draft, evidence)
    headline = "单次最多5万行。"
    assert not claim_validation_issues(answer(headline, headline), [row("## " + headline)])


@pytest.mark.parametrize("repair", [answer("截止为9天。", "截止为7天。"), DependencyError("offline")])
def test_failed_repair_never_returns_unsupported_answer_or_repeats_forever(repair):
    models = Drafts(answer("截止为9天。", "截止为7天。"), repair)
    result, usage = recover_answer(models, "截止时间？", [row("截止为7天。")])
    assert validate_claims(result, [row("截止为7天。")])[1] == "verification_failed"
    assert len(models.calls) == 2 and usage["answer_recovery"]["repair_attempts"] == 1


def test_valid_answer_and_genuine_abstention_do_not_trigger_repair():
    for draft in [answer("截止为7天。", "截止为7天。"), GeneratedAnswer(answerable=False, claims=[])]:
        models = Drafts(draft)
        result, usage = recover_answer(models, "截止时间？", [row("截止为7天。")])
        assert len(models.calls) == 1 and result == draft
        assert usage["answer_recovery"]["repair_attempts"] == 0


def test_malformed_citation_arity_produces_diagnostic_not_crash():
    draft = GeneratedAnswer(
        answerable=True,
        claims=[Claim(text="截止为7天。", evidence_ids=["missing", "a"], quotes=["截止为7天。"])],
    )
    assert set(claim_validation_issues(draft, [row("截止为7天。")])[0]["codes"]) == {
        "citation_arity",
        "unknown_evidence",
    }
    assert validate_claims(draft, [row("截止为7天。")])[1] == "verification_failed"


def test_budget_exhaustion_is_not_caught_as_optional_repair_failure():
    class BudgetModels(Drafts):
        def generate(self, *args, **kwargs):
            from app.execution_budget import current_budget

            with current_budget().call("generation"):
                return super().generate(*args, **kwargs)

    cfg = settings().model_copy(update={"agent_generation_max_calls": 1})
    budget = ExecutionBudget(cfg)
    models = BudgetModels(answer("截止为9天。", "截止为7天。"))
    with pytest.raises(BudgetExceeded), use_budget(budget):
        recover_answer(models, "截止？", [row("截止为7天。")])
    assert budget.snapshot()["calls"]["generation"]["attempted"] == 1


def test_window_comparison_binds_whole_period_and_both_entities_without_model():
    from app.evidence_scope import scoped_windows

    rows = [
        dict(
            row("工作日服务时段为09:00至17:00。", key="north"),
            title="北区中心服务时间",
            metadata={
                "document_scope": {
                    "period": ["2028-04-01", "2028-06-30"],
                    "clauses": ["本文件适用于2028年第二季度北区中心"],
                }
            },
        ),
        dict(
            row("工作日服务时段为10:00至16:00。", key="south"),
            title="南区中心服务时间",
            metadata={
                "document_scope": {
                    "period": ["2028-04-01", "2028-06-30"],
                    "clauses": ["本文件适用于2028年第二季度南区中心"],
                }
            },
        ),
        dict(
            row("工作日服务时段为06:00至22:00。", key="temp"),
            title="北区中心临时服务时间",
            metadata={
                "document_scope": {
                    "period": ["2028-04-01", "2028-04-30"],
                    "clauses": ["本通知适用于2028年4月北区中心"],
                }
            },
        ),
    ]
    a, trace = scoped_windows(
        "北区中心和南区中心2028年第二季度的工作日服务时段分别是什么？两者是否相同？", rows
    )
    assert a.answerable and trace["verdict"]["value"] == "no"
    assert [c.evidence_ids for c in a.claims] == [["north"], ["south"], ["north", "south"]]
    assert trace["rejected"] == [{"chunk_id": "temp", "reason": "period_not_covering_request"}]
    assert validate_claims(a, rows)[1] == "answered"


@pytest.mark.parametrize("other", ["missing", "overlap", "wrong_entity", "wrong_weekday"])
def test_window_unknown_or_conflicting_scope_never_becomes_complete_comparison(other):
    from app.evidence_scope import scoped_windows

    scope = {"document_scope": {"period": ["2028-04-01", "2028-06-30"], "clauses": []}}
    north = dict(row("工作日服务时段为09:00至17:00。"), title="北区中心服务时间", metadata=scope)
    south = dict(
        row("工作日服务时段为09:00至17:00。", key="b"), title="南区中心服务时间", metadata=scope
    )
    rows = [north, south]
    if other == "missing":
        south["metadata"] = {}
    if other == "overlap":
        rows.append(dict(south, chunk_id="c", id="c", text="工作日服务时段为10:00至16:00。"))
    if other == "wrong_entity":
        south["title"] = "另一区中心服务时间"
    if other == "wrong_weekday":
        south["text"] = "周末服务时段为09:00至17:00。"
    a, _ = scoped_windows(
        "北区中心和南区中心2028年第二季度的工作日服务时段分别是什么？两者是否相同？", rows
    )
    assert not a.answerable


@pytest.mark.parametrize("case", ["lookalike_only", "lookalike_extra", "cross_month", "operator_clause"])
def test_window_entity_is_whole_name_and_period_must_cover_request(case):
    from app.evidence_scope import scoped_windows

    quarter = {"document_scope": {"period": ["2028-04-01", "2028-06-30"], "clauses": []}}
    east = dict(row("工作日收货窗口为08:00至18:00。"), title="华东仓收货时间", metadata=quarter)
    south = dict(row("工作日收货窗口为09:00至17:00。", key="b"), title="华南仓收货时间", metadata=quarter)
    other = dict(row("工作日收货窗口为07:00至20:00。", key="c"), title="华东仓二号库收货时间",
                 metadata=quarter)
    rows = {"lookalike_only": [other, south], "lookalike_extra": [east, south, other],
            "cross_month": [dict(east, metadata={"document_scope": {
                "period": ["2028-05-01", "2028-07-31"], "clauses": []}}), south],
            "operator_clause": [east, dict(south, title="收货时间", metadata={"document_scope": {
                "period": ["2028-04-01", "2028-06-30"],
                "clauses": ["本文件适用于2028年第二季度的华南仓，由华东仓运营组代管"]}})]}[case]
    a, trace = scoped_windows("华东仓和华南仓2028年第二季度的工作日收货窗口分别是什么？两者是否相同？", rows)
    if case in {"lookalike_only", "cross_month"}:
        assert not a.answerable
    else:
        assert a.answerable and trace["verdict"]["value"] == "no"
        assert "07:00" not in " ".join(c.text for c in a.claims)
        assert a.claims[0].evidence_ids == ["a"] and a.claims[1].evidence_ids == ["b"]


def test_scope_never_uses_publication_year_as_applicability():
    from app.evidence_scope import source_scope

    assert source_scope("Published 2028年第二季度。适用于北区中心。")["status"] == "unknown"


def test_semantic_failure_is_repaired_locally_and_bad_second_attempt_is_rejected(monkeypatch):
    from app.clients import BoundAnswerAudit, BoundClaimCheck

    monkeypatch.setattr(settings(), "answer_semantic_audit_enabled", True)
    monkeypatch.setattr("app.answer_quality.require_audit_gate", lambda: None)

    class Audits(Drafts):
        def audit_bound_answer(self, q, a, e):
            self.checked = getattr(self, "checked", 0) + 1
            return BoundAnswerAudit(
                checks=[
                    BoundClaimCheck(
                        claim_index=1,
                        source_indices=[1],
                        support="unsupported",
                        reason="Subject changed",
                        source_excerpts=["乙已同意。"],
                    )
                ],
                question_complete=False,
                verdict="unclear",
                conclusion="",
                conclusion_claim_indices=[],
            ), {}

    m = Audits(answer("甲已同意。", "乙已同意。"), answer("甲已同意。", "乙已同意。"))
    result, usage = recover_answer(m, "谁已同意？", [row("乙已同意。")])
    assert len(m.calls) == 2 and m.checked == 2
    assert usage["quality_validation_failed"]
    assert usage["answer_recovery"]["remaining_issues"][0]["codes"] == ["bound_claim_unsupported"]


def test_verdict_comes_from_supported_claims_and_retains_bound_quotes(monkeypatch):
    from app.clients import BoundAnswerAudit, BoundClaimCheck
    from app.qa import answer_verdict

    monkeypatch.setattr(settings(), "answer_semantic_audit_enabled", True)
    monkeypatch.setattr("app.answer_quality.require_audit_gate", lambda: None)

    class Audits(Drafts):
        def audit_bound_answer(self, q, a, e):
            return BoundAnswerAudit(
                checks=[
                    BoundClaimCheck(
                        claim_index=1,
                        source_indices=[1],
                        support="supported",
                        reason="Explicit denial",
                        source_excerpts=["The report explicitly denies approval."],
                    )
                ],
                question_complete=True,
                verdict="no",
                conclusion="No, the report explicitly denies approval.",
                conclusion_claim_indices=[1],
            ), {}

    m = Audits(
        answer("The report explicitly denies approval.", "The report explicitly denies approval.")
    )
    q = "Did the report confirm approval?"
    ev = [row("The report explicitly denies approval.")]
    a, u = recover_answer(m, q, ev)
    claims, status = validate_claims(a, ev)
    assert not u["quality_validation_failed"]
    v, _ = answer_verdict(object(), q, claims, status, verified_verdict=u["verified_verdict"])
    assert v["value"] == "no" and v["claim_index"] == 2 and v["evidence_ids"] == ["a"]
    assert claims[-1]["quotes"] == claims[0]["quotes"]


def test_audit_unavailable_never_becomes_a_supported_answer(monkeypatch):
    monkeypatch.setattr(settings(), "answer_semantic_audit_enabled", True)
    monkeypatch.setattr("app.answer_quality.require_audit_gate", lambda: None)

    class Offline(Drafts):
        def audit_bound_answer(self, *args):
            raise DependencyError("offline", stage="answer_audit")

    m = Offline(answer("乙已同意。", "乙已同意。"))
    _, u = recover_answer(m, "谁已同意？", [row("乙已同意。")])
    assert len(m.calls) == 1 and u["quality_validation_failed"]


def test_only_transient_transport_failure_uses_official_framework_retry():
    models = Drafts(DependencyError("503", retryable=True), answer("截止为7天。", "截止为7天。"))
    _, u = recover_answer(models, "截止？", [row("截止为7天。")])
    assert len(models.calls) == 2 and u["answer_recovery"]["repair_attempts"] == 0
    models = Drafts(DependencyError("invalid JSON", stage="generation"))
    with pytest.raises(DependencyError):
        recover_answer(models, "截止？", [row("截止为7天。")])
    assert len(models.calls) == 1


def test_qualitative_role_request_is_not_misclassified_as_response_duration():
    from app.task_contract import _slot, bind_values

    assert _slot("紧急工单在首次响应时还必须通知哪个角色", "x").value_type == "text"
    s = _slot("网关服务单个请求的超时阈值是多少", "x")
    assert bind_values(s, [dict(row("单个请求的超时阈值为9秒。"), title="网关服务手册")])[0].value == "9"


def test_no_eligible_source_does_not_send_empty_enum_to_model(monkeypatch):
    monkeypatch.setattr(settings(), "answer_quality_enabled", True)
    m = Drafts()
    a, u = recover_answer(
        m, "作者是谁？", [row("This Instagram post cannot be displayed. End of instagram post")]
    )
    assert not a.answerable and not m.calls and u["reason"] == "no_eligible_source_spans"


def test_component_gate_requires_both_acceptance_and_rejection():
    from app.answer_quality import validation_summary

    gate = {
        "good_claim_acceptance_min": 0.95,
        "bad_claim_rejection_min": 0.95,
        "transport_or_format_failure_max": 0,
    }
    assert validation_summary(
        [
            {"label": "supported", "prediction": "supported"},
            {"label": "unsupported", "prediction": "unsupported"},
        ],
        gate,
    )["passed"]
    assert not validation_summary(
        [
            {"label": "supported", "prediction": "unsupported"},
            {"label": "unsupported", "prediction": "unsupported"},
        ],
        gate,
    )["passed"]
    assert not validation_summary(
        [
            {"label": "supported", "prediction": "supported"},
            {"label": "unsupported", "prediction": None},
        ],
        gate,
    )["passed"]


def test_missing_or_changed_component_gate_never_enables_judge(monkeypatch, tmp_path):
    import json
    from app.answer_quality import require_audit_gate

    monkeypatch.setattr(settings(), "answer_semantic_audit_gate_path", "")
    with pytest.raises(ValueError, match="validation gate required"):
        require_audit_gate()
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "protocol": "bound-answer-audit-gate-v1",
                "passed": True,
                "implementation_sha256": "changed",
            }
        )
    )
    monkeypatch.setattr(settings(), "answer_semantic_audit_gate_path", str(path))
    with pytest.raises(ValueError, match="mismatch"):
        require_audit_gate()


def test_audit_proof_cannot_borrow_another_claim_quote(monkeypatch):
    import json
    from app.clients import Models

    payload = {
        "checks": [
            {"claim_index": 1, "source_indices": [2], "reason": "Same topic", "support": "supported"},
            {"claim_index": 2, "source_indices": [1], "reason": "Exact quote", "support": "supported"},
        ],
        "question_complete": True,
        "verdict": "unclear",
        "conclusion": "",
        "conclusion_claim_indices": [],
    }
    m = Models(chat_backend=lambda body: {"message": {"content": json.dumps(payload)}})
    a = answer("甲已同意。", "甲正在等待回复。")
    a.claims.append(Claim(text="乙已同意。", evidence_ids=["b"], quotes=["乙已同意。"]))
    checked, _ = m.audit_bound_answer("谁已同意？", a, [row("甲正在等待回复。"), row("乙已同意。", "b")])
    assert checked.checks[0].support == "unsupported" and checked.checks[1].support == "supported"


@pytest.mark.parametrize("limit", ["scan", "context", "revoked"])
def test_scope_expansion_obeys_scan_context_and_current_authorization(monkeypatch, limit):
    from types import SimpleNamespace
    from app.evidence_scope import attach_scope, scoped_windows

    scope = SimpleNamespace(id="scope", text="本文件适用于2028年第二季度北区中心。", locator={})
    window = SimpleNamespace(id="win", text="工作日服务时段为09:00至17:00。", locator={})
    chunks = (
        [scope, window]
        if limit != "scan"
        else [scope, window]
        + [
            SimpleNamespace(id=f"x{i}", text="工作日服务时段为10:00至16:00。", locator={})
            for i in range(63)
        ]
    )

    class Db:
        def scalars(self, query):
            return chunks

    calls = []

    def authorize(db, user, cid, **kwargs):
        calls.append(cid)
        if limit == "revoked" and cid == "scope":
            raise PermissionError("revoked")

    monkeypatch.setattr("app.security.require_chunk", authorize)
    q = "北区中心和南区中心2028年第二季度的工作日服务时段分别是什么？两者是否相同？"
    source = dict(row(window.text, "win"), title="北区中心")
    original = [dict(source)]
    if limit == "context":
        monkeypatch.setattr(settings(), "context_token_budget", 1)
    if limit == "revoked":
        with pytest.raises(PermissionError):
            attach_scope(Db(), object(), q, original)
    else:
        enriched, trace = attach_scope(Db(), object(), q, original)
        assert len(enriched) == 1 and original == [source] and not trace["added_chunk_ids"]
        assert not scoped_windows(q, enriched)[0].answerable
    assert "win" in calls


def test_literal_renderer_removes_unproved_paraphrase_preserves_original_attribution():
    from app.answer_quality import bind_source_text

    q = "The lawyer alleged the transfer was for personal gain; the court has not confirmed it."
    draft = answer("The court confirmed personal gain.", q)
    a, trace = bind_source_text(draft, [row(q)])
    assert a.claims[0].text == q and a.claims[0].quotes == [q]
    assert trace["guarantee"].startswith("verbatim_provenance_only")


def test_literal_renderer_does_not_silently_drop_sources_or_accept_fabricated_quotes():
    from app.answer_quality import bind_source_text

    draft = answer("甲已同意。", "原文没有这句话。")
    a, trace = bind_source_text(draft, [row("甲仍在审核。")])
    assert a == draft and trace["status"] == "unrenderable_source_span"


def test_literal_renderer_keeps_condition_and_rejects_oversize_partial_claim():
    from app.answer_quality import bind_source_text

    q = "If Ari wants a promotion, division X is no longer an option."
    a, trace = bind_source_text(answer("Ari does not want a promotion.", q), [row(q)])
    assert a.claims[0].text == q
    source = "word " * 220
    _, trace = bind_source_text(answer("A short invented paraphrase.", source), [row(source)])
    assert trace["status"] == "unrenderable_source_span"


def test_source_selection_has_no_model_written_assertion(monkeypatch):
    import json
    from app.clients import Models

    monkeypatch.setattr(settings(), "answer_quality_enabled", True)
    monkeypatch.setattr(settings(), "answer_extractive_enabled", True)
    seen = []

    def backend(body):
        seen.append(body)
        return {"message": {"content": json.dumps({"status": "answered", "source_ids": ["a:S1"]})}}

    m = Models(chat_backend=backend)
    a, u = m.generate(
        "谁批准了录制？", [row("Dina explicitly refused recording.")], check_conflict=False
    )
    assert a.claims[0].text == "Dina explicitly refused recording."
    assert set(seen[0]["format"]["properties"]) == {"status", "source_ids"}
    assert "claims" not in seen[0]["format"]["properties"]


def test_source_plan_covers_each_question_and_allows_shared_proof(monkeypatch):
    import json
    from app.clients import Models

    monkeypatch.setattr(settings(), "answer_extractive_enabled", True)
    seen = []

    def backend(body):
        seen.append(body)
        return {
            "message": {
                "content": json.dumps(
                    {
                        "status": "answered",
                        "selections": [
                            {"item_index": 1, "source_ids": ["a:S1"]},
                            {"item_index": 2, "source_ids": ["a:S1"]},
                        ],
                    }
                )
            }
        }

    a, u = Models(chat_backend=backend).generate(
        "What are the two limits?",
        [row("Refresh takes 10 minutes; late arrivals add one day.")],
        acceptance_items_override=["Refresh interval", "Late arrival penalty"],
        check_conflict=False,
    )
    assert len(a.claims) == 1 and u["selection_missing_items"] == []
    assert seen[0]["format"]["properties"]["selections"]["minItems"] == 2


def test_partial_source_plan_gets_one_repair_instead_of_silent_success(monkeypatch):
    import json
    from app.clients import Models

    monkeypatch.setattr(settings(), "answer_extractive_enabled", True)
    monkeypatch.setattr(settings(), "answer_semantic_audit_enabled", False)
    payloads = [
        [{"item_index": 1, "source_ids": ["a:S1"]}, {"item_index": 2, "source_ids": []}],
        [{"item_index": 1, "source_ids": ["a:S1"]}, {"item_index": 2, "source_ids": ["b:S1"]}],
    ]
    seen = []

    def backend(body):
        seen.append(body)
        return {
            "message": {"content": json.dumps({"status": "answered", "selections": payloads.pop(0)})}
        }

    a, u = recover_answer(
        Models(chat_backend=backend),
        "What are the limits?",
        [
            row("Refresh takes 10 minutes."),
            row("Late arrivals add one day.", "b"),
        ],
        options={"acceptance_items_override": ["Refresh interval", "Late arrival penalty"]},
    )
    assert len(a.claims) == 2 and len(seen) == 2
    assert u["answer_recovery"]["first_issues"][0]["codes"] == ["selection_slots_missing"]
    assert not u["quality_validation_failed"]


def test_duplicate_source_plan_index_is_rejected(monkeypatch):
    import json
    from app.clients import Models

    monkeypatch.setattr(settings(), "answer_extractive_enabled", True)
    m = Models(
        chat_backend=lambda body: {
            "message": {
                "content": json.dumps(
                    {
                        "status": "answered",
                        "selections": [
                            {"item_index": 1, "source_ids": ["a:S1"]},
                            {"item_index": 1, "source_ids": ["a:S1"]},
                        ],
                    }
                )
            }
        }
    )
    with pytest.raises(DependencyError):
        m.generate(
            "Two limits?",
            [row("A rule.")],
            acceptance_items_override=["First", "Second"],
            check_conflict=False,
        )


def test_literal_composition_rejects_arbitrary_paraphrase_before_any_model_call():
    from app.clients import Models

    m = Models(chat_backend=lambda body: pytest.fail("must not call model"))
    with pytest.raises(DependencyError, match="自由改写"):
        m.compose_literal_answer(
            "Did Dina consent?",
            answer("Dina approved recording.", "Dina refused recording."),
            [row("Dina refused recording.")],
        )


def test_literal_identification_needs_no_semantic_model_call():
    from app.clients import Models

    m = Models(chat_backend=lambda body: pytest.fail("no judgment required"))
    q = "Avery is the project coordinator."
    checked, u = m.compose_literal_answer("Who is the coordinator?", answer(q, q), [row(q)])
    assert checked.checks[0].support == "supported" and u["model_calls"] == 0


@pytest.mark.parametrize("literal_only", [False, True])
def test_judgment_composes_all_authorized_evidence_once_without_weak_prefilter(monkeypatch, literal_only):
    from app.clients import BoundAnswerAudit, BoundClaimCheck

    monkeypatch.setattr(settings(), "answer_extractive_enabled", not literal_only)
    monkeypatch.setattr(settings(), "answer_semantic_audit_enabled", not literal_only)
    monkeypatch.setattr(settings(), "answer_literal_judgment_enabled", literal_only)
    monkeypatch.setattr("app.answer_quality.require_audit_gate", lambda: None)
    evidence = [row("The participant approved filming.")]

    class Composer:
        calls = 0

        def generate(self, *args, **kwargs):
            pytest.fail("judgment must not lose counterevidence in a prior selection")

        def compose_literal_answer(self, question, generated, context):
            self.calls += 1
            assert context == evidence and not generated.claims
            c = Claim(text=context[0]["text"], quotes=[context[0]["text"]], evidence_ids=["a"])
            return BoundAnswerAudit(
                checks=[
                    BoundClaimCheck(
                        claim_index=1, source_indices=[1], support="supported", reason="literal"
                    )
                ],
                question_complete=True,
                verdict="no",
                conclusion="No. The participant approved filming.",
                conclusion_claim_indices=[1],
                selected_claims=[c],
            ), {"prompt_tokens": 15, "completion_tokens": 20}

    m = Composer()
    a, u = recover_answer(m, "Did the participant not consent to filming?", evidence)
    assert m.calls == 1 and len(a.claims) == 2
    assert u["verified_verdict"]["value"] == "no"
    assert u["answer_recovery"]["repair_attempts"] == 0


def test_literal_judgment_does_not_replace_ordinary_multi_slot_generation(monkeypatch):
    monkeypatch.setattr(settings(), "answer_literal_judgment_enabled", True)
    monkeypatch.setattr(settings(), "answer_semantic_audit_enabled", False)
    monkeypatch.setattr(settings(), "answer_extractive_enabled", False)
    monkeypatch.setattr("app.answer_quality.require_audit_gate", lambda: pytest.fail("ordinary QA requires no relationship model"))
    evidence = [row("重试7次。截止5天。")]
    models = Drafts(answer("重试7次；截止5天。", evidence[0]["text"]))
    generated, usage = recover_answer(models, "重试次数和截止天数是多少？", evidence)
    assert len(models.calls) == 1 and len(generated.claims) == 1
    assert usage["answer_recovery"]["repair_attempts"] == 0


def test_literal_judgment_cannot_borrow_arbitrary_claim_validation_protocol(monkeypatch, tmp_path):
    import json
    from app.answer_quality import require_audit_gate

    monkeypatch.setattr(settings(), "answer_literal_judgment_enabled", True)
    monkeypatch.setattr(settings(), "answer_semantic_audit_enabled", False)
    path = tmp_path / "wrong-gate.json"
    path.write_text(json.dumps({"protocol": "bound-answer-audit-gate-v1", "passed": True}))
    monkeypatch.setattr(settings(), "answer_semantic_audit_gate_path", str(path))
    with pytest.raises(ValueError, match="protocol mismatch"):
        require_audit_gate()


def test_composition_cannot_cite_a_source_outside_authorized_context():
    import json
    from app.clients import Models

    m = Models(
        chat_backend=lambda body: {
            "message": {
                "content": json.dumps(
                    {
                        "propositions": [
                            {
                                "item_index": 1,
                                "source_ids": ["unauthorized:S1"],
                                "truth": "supported",
                            }
                        ],
                    }
                )
            }
        }
    )
    with pytest.raises(DependencyError, match="返回格式无效"):
        m.compose_literal_answer(
            "Did Dina consent?",
            GeneratedAnswer(answerable=True, claims=[]),
            [row("Dina approved recording.")],
        )


def test_empty_answer_cannot_be_an_answered_task():
    a = GeneratedAnswer(answerable=True, claims=[])
    assert validate_claims(a, [row("无答案依据。")])[1] == "verification_failed"


def test_window_qualifier_is_local_and_clock_padding_does_not_change_equality():
    from app.evidence_scope import scoped_windows

    scope = {"document_scope": {"period": ["2028-04-01", "2028-06-30"], "clauses": []}}
    rows = [
        dict(
            row("周末服务时段为06:00至22:00。\n工作日服务时段为9:00至17:00。"),
            title="北区中心",
            metadata=scope,
        ),
        dict(row("工作日服务时段为09:00至17:00。", "b"), title="南区中心", metadata=scope),
    ]
    a, t = scoped_windows("北区中心和南区中心2028年第二季度的工作日服务时段是否相同？", rows)
    assert a.answerable and t["verdict"]["value"] == "yes" and "06:00" not in a.claims[0].text
    assert validate_claims(a, rows)[1] == "answered"
    rows[0]["text"] = "工作日服务时段为09:00至17:00或10:00至18:00。"
    assert not scoped_windows("北区中心和南区中心2028年第二季度的工作日服务时段是否相同？", rows)[
        0
    ].answerable


def test_source_scope_excludes_another_publication_without_losing_named_sources():
    from app.answer_contract import source_scoped_evidence

    rows = [
        row("Limited personal privacy.", "a", metadata={"source": "Daily Record"}),
        row("Private experiences kept private.", "b", metadata={"source": "Weekly Ledger"}),
        row("Unrelated public event.", "c", metadata={"source": "Other Wire"}),
    ]
    selected, trace = source_scoped_evidence(
        "Does the Daily Record report compare to the Weekly Ledger article?", rows
    )
    assert [r["id"] for r in selected] == ["a", "b"]
    assert trace["input_chunks"] == 3 and trace["scoped_chunks"] == 2


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Did Ari actually lack interest in joining a contender?", "unclear"),
        ("Does the report suggest Ari lacked interest in joining a contender?", "no"),
        ("Did the report never mention interest?", "unclear"),
    ],
)
def test_unsupported_source_characterization_does_not_invent_actual_world_negation(question, expected):
    import json
    from app.clients import Models

    m = Models(
        chat_backend=lambda body: {
            "message": {
                "content": json.dumps(
                    {
                        "propositions": [
                            {
                                "item_index": 1,
                                "source_ids": ["a:S1"],
                                "truth": "unsupported_characterization",
                            }
                        ]
                    }
                )
            }
        }
    )
    checked, _ = m.compose_literal_answer(
        question,
        GeneratedAnswer(answerable=True, claims=[]),
        [row("If Ari wants to join a contender, North is no longer an option.")],
    )
    assert checked.verdict == expected
    assert checked.selected_claims[0].text.startswith("If Ari")


def test_source_scope_preserves_two_requested_dates_from_same_publication():
    from app.answer_contract import source_scoped_evidence

    rows = [
        row("First report.", "a", metadata={"source": "TechCrunch", "published_at": "2031-10-31"}),
        row("Second report.", "b", metadata={"source": "TechCrunch", "published_at": "2031-12-15"}),
        row("Other report.", "c", metadata={"source": "Other Wire"}),
    ]
    selected, trace = source_scoped_evidence(
        "Did TechCrunch report this on October 31, 2031, and then on December 15, 2031?", rows
    )
    assert [r["id"] for r in selected] == ["a", "b"]
    assert len(trace["groups"]) == 1
    assert len(trace["groups"][0]["key"]) == 2


@pytest.mark.parametrize("missing_side", [False, True])
def test_comparison_requires_both_publications_and_original_attribute_proofs(missing_side):
    import json
    from app.clients import Models

    def backend(body):
        prop = body["format"]["$defs"]["BoundSourceProposition"]
        assert "comparison_sources" in prop["required"]
        return {
            "message": {
                "content": json.dumps(
                    {
                        "propositions": [
                            {
                                "item_index": 1,
                                "source_ids": [],
                                "truth": "supported",
                                "comparison_sources": [
                                    {"group_index": 1, "source_ids": ["a:S1"]},
                                    {"group_index": 2, "source_ids": [] if missing_side else ["b:S1"]},
                                ],
                            }
                        ]
                    }
                )
            }
        }

    rows = [
        row("The city received less funding.", "a", metadata={"source": "Daily Record"}),
        row("The other city received more funding.", "b", metadata={"source": "Weekly Ledger"}),
    ]
    audit, _ = Models(chat_backend=backend).compose_literal_answer(
        "Does the Daily Record report imply less funding compared to the Weekly Ledger article?",
        GeneratedAnswer(answerable=True, claims=[]),
        rows,
    )
    assert audit.verdict == ("unclear" if missing_side else "yes")
    assert audit.question_complete is not missing_side
    assert [c.evidence_ids for c in audit.selected_claims] == (
        [["a"]] if missing_side else [["a"], ["b"]]
    )


def test_comparison_rejects_evidence_laundered_into_another_publication():
    import json
    from app.clients import Models

    model = Models(
        chat_backend=lambda body: {
            "message": {
                "content": json.dumps(
                    {
                        "propositions": [
                            {
                                "item_index": 1,
                                "source_ids": [],
                                "truth": "contradicted",
                                "comparison_sources": [
                                    {"group_index": 1, "source_ids": ["a:S1"]},
                                    {"group_index": 2, "source_ids": ["a:S1"]},
                                ],
                            }
                        ]
                    }
                )
            }
        }
    )
    rows = [
        row("One value.", "a", metadata={"source": "Daily Record"}),
        row("Other value.", "b", metadata={"source": "Weekly Ledger"}),
    ]
    with pytest.raises(DependencyError, match="组合返回格式无效"):
        model.compose_literal_answer(
            "Are Daily Record and Weekly Ledger consistent?",
            GeneratedAnswer(answerable=True, claims=[]),
            rows,
        )


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Does the report suggest Ari's actual move was motivated by disinterest?", "no"),
        ("Was Ari's actual move motivated by disinterest?", "unclear"),
        ("Does the report describe what happens if Ari wants to join a contender?", "yes"),
    ],
)
def test_model_cannot_promote_explicit_conditional_only_quote_to_actual_fact(question, expected):
    import json
    from app.clients import Models

    model = Models(
        chat_backend=lambda body: {
            "message": {
                "content": json.dumps(
                    {"propositions": [{"item_index": 1, "source_ids": ["a:S1"], "truth": "supported"}]}
                )
            }
        }
    )
    checked, _ = model.compose_literal_answer(
        question,
        GeneratedAnswer(answerable=True, claims=[]),
        [row("If Ari wants to join a contender, North is no longer an option.")],
    )
    assert checked.verdict == expected
    assert len(checked.selected_claims) == 1


def test_factual_headline_is_still_an_exact_citable_source(monkeypatch):
    from app.clients import evidence_spans

    monkeypatch.setattr(settings(), "answer_quality_enabled", True)
    sources, _ = evidence_spans([row("# Publisher files antitrust suit against the search provider")])
    assert sources["a:S1"]["quote"] == "Publisher files antitrust suit against the search provider"
    assert sources["a:S1"]["quote"] in "# Publisher files antitrust suit against the search provider"


def test_comparison_focus_keeps_original_authorized_rows_and_both_source_groups(monkeypatch):
    from app.answer_contract import comparison_relevant_evidence

    rows = [
        row("A private fact.", "a"),
        row("Unrelated public commentary.", "b"),
        row("Another private fact.", "c"),
        row("Creative work.", "d"),
    ]
    rows[1]["document_id"] = rows[0]["document_id"]
    rows[3]["document_id"] = rows[2]["document_id"]
    scope = {
        "groups": [
            {"key": ["doc-a"], "mention": "Report A", "clause": "privacy in A"},
            {"key": ["doc-c"], "mention": "Report B", "clause": "privacy in B"},
        ]
    }
    queries = []

    def rank(query, candidates, **kwargs):
        queries.append(query)
        # Ranking cannot inject altered text into evidence consumed by the model.
        return [
            {**r, "text": "untrusted replacement", "rerank_score": 10 - i}
            for i, r in enumerate(candidates)
        ]

    monkeypatch.setattr("app.rerank.rerank", rank)
    chosen, trace = comparison_relevant_evidence(
        "Are Report A and Report B consistent about privacy?", rows, scope
    )
    assert chosen == [rows[0], rows[2]]
    assert all("privacy" in query for query in queries)
    assert trace["ranking_only_not_entailment"] is True
    assert trace["candidate_chunk_ids"] == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    "question,keys",
    [
        ("Are the reports consistent about their color?", ["doc-a"]),
        ("Are totals consistent about privacy?", ["doc-a"]),
        ("Are the reports consistent about privacy?", ["doc-a", "doc-b"]),
    ],
)
def test_comparison_focus_never_prunes_unknown_dimensions_aggregates_or_multiple_articles(
    monkeypatch, question, keys
):
    from app.answer_contract import comparison_relevant_evidence

    def should_not_rank(*args, **kwargs):
        pytest.fail("This question needs the full context")

    monkeypatch.setattr("app.rerank.rerank", should_not_rank)
    rows = [row("One fact.", "a"), row("Other fact.", "b")]
    scope = {"groups": [{"key": keys, "mention": "First"}, {"key": ["doc-b"], "mention": "Second"}]}
    chosen, trace = comparison_relevant_evidence(question, rows, scope)
    assert chosen is rows and trace["status"] == "full_context"


def test_comparison_focus_rejects_reranker_source_id_outside_authorized_group(monkeypatch):
    from app.answer_contract import comparison_relevant_evidence

    rows = [row("A private fact.", "a"), row("Another paragraph.", "b"), row("A fact in C.", "c")]
    rows[1]["document_id"] = rows[0]["document_id"]
    scope = {"groups": [{"key": ["doc-a"], "mention": "A"}, {"key": ["doc-c"], "mention": "C"}]}
    monkeypatch.setattr("app.rerank.rerank", lambda *a, **k: [{"id": "unauthorized", "rerank_score": 5}])
    with pytest.raises(DependencyError, match="比较重排未返回原始来源"):
        comparison_relevant_evidence("Are A and C consistent about privacy?", rows, scope)


def test_reported_pronoun_keeps_adjacent_original_context_without_resolving_identity(monkeypatch):
    from app.clients import citation_context, evidence_spans
    from app.config import Settings
    monkeypatch.setattr('app.clients.settings', lambda: Settings(answer_quality_enabled=True))
    quote = '"We welcome the change," she said.'
    prior = 'Meanwhile Dana, the organizer, spoke to the paper.'
    text = 'Background about another topic.\n\n' + prior + '\n\n' + quote
    assert citation_context(quote, text) == prior + '\n\n' + quote
    item = row(text)
    spans, _ = evidence_spans([item])
    assert any(s['quote'] == prior + '\n\n' + quote for s in spans.values())
    assert all(s['quote'] in text for s in spans.values())
    assert citation_context(quote, quote) == quote
    assert citation_context(quote, '# Speech\n\n' + quote) == quote
    assert citation_context(quote, '| Name |\n\n' + quote) == quote
    assert citation_context(quote, text + '\n\n' + quote) == quote
    assert citation_context(quote, 'x' * 1000 + '\n\n' + quote) == quote


def test_reported_speech_context_never_crosses_embedded_heading_or_table():
    from app.clients import citation_context
    quote = '"Welcome," she said.'
    assert citation_context(quote, 'Old context\n# New section\n\n' + quote) == quote
    assert citation_context(quote, 'Old context\n| Heading |\n\n' + quote) == quote


def test_publication_mismatch_repair_names_the_evidence_of_that_publication(monkeypatch):
    monkeypatch.setattr(settings(), "answer_quality_enabled", True)
    evidence = [
        dict(row("Britney lived a restricted life.", "g"), metadata={"source": "The Guardian"}),
        dict(row("They don't know about it, she told NPR.", "b"), metadata={"source": "BBC News - Entertainment & Arts"}),
        dict(row("There is chaos outside the restaurant.", "i"), metadata={"source": "The Independent - Life and Style"}),
    ]
    claim = "The BBC News - Entertainment & Arts article implies Swift keeps some events private."
    models = Drafts(answer(claim, "There is chaos outside the restaurant.", "i"),
                    answer(claim, "They don't know about it, she told NPR.", "b"))
    result, usage = recover_answer(models, "Does the BBC article imply privacy?", evidence)
    reason = models.calls[1]["repair_context"]["failed_claims"][0]["reason"]
    assert "b" in reason.split("：")[-1] and "i" not in reason.split("：")[-1]
    assert usage["answer_recovery"]["repair_attempts"] == 1 and result.claims[0].evidence_ids == ["b"]
