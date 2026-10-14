"""Synthetic transports exercise the real audit API; these are not model accuracy labels."""

import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from app import clients
from app.config import Settings


def source(cid="chunk-a", text="Ari approved filming.", **changes):
    row = dict(
        chunk_id=cid,
        document_id="doc-" + cid,
        version_id="v1",
        source_sha256=hashlib.sha256(("full-source-" + cid).encode()).hexdigest(),
        title="Synthetic notice",
        text=text,
        locator={"paragraph": 1},
        metadata={"source": "Synthetic Gazette", "published_at": "2026-01-10"},
    )
    return dict(row, **changes)


def answer(rows=None, claims=None, verdict=None):
    rows = rows or [source()]
    return dict(
        status="answered",
        claims=claims
        or [dict(text=rows[0]["text"], evidence_ids=[rows[0]["chunk_id"]], quotes=[rows[0]["text"]])],
        citations=copy.deepcopy(rows),
        verdict=verdict,
    )


def audit_reply(n=1, **changes):
    result = dict(
        checks=[
            dict(claim_index=i, source_indices=[1], reason="Literal support", support="supported")
            for i in range(1, n + 1)
        ],
        question_complete=True,
        verdict="unclear",
        conclusion="",
        conclusion_claim_indices=[],
    )
    return dict(result, **changes)


def response(result=None, **counts):
    return dict(message={"content": json.dumps(result or audit_reply())}, **counts)


def run(
    payload=None,
    evidence=None,
    *,
    backend=None,
    cfg=None,
    question="Who approved filming?",
    reauthorize=None,
    models=None,
):
    assert importlib.util.find_spec("scripts.run_bound_citation_dev") is not None, (
        "bound audit helper missing"
    )
    module = importlib.import_module("scripts.run_bound_citation_dev")
    evidence = evidence or [source()]
    return module.audit_answer(
        question,
        payload or answer(evidence),
        evidence,
        models
        or clients.Models(
            chat_backend=backend or (lambda body: response(prompt_eval_count=50, eval_count=20))
        ),
        cfg or Settings(_env_file=None),
        reauthorize=reauthorize or (lambda rows: copy.deepcopy(rows)),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_id", "fabricated"),
        ("document_id", "other"),
        ("version_id", "old"),
        ("source_sha256", "chunk-hash"),
        ("title", "altered"),
        ("text", "altered"),
        ("locator", {"paragraph": 2}),
        ("metadata", {"source": "Other"}),
    ],
)
def test_original_source_fields_must_match_before_dispatch(field, value):
    payload = answer()
    payload["citations"][0][field] = value
    result = run(payload, backend=lambda body: pytest.fail("malformed binding reached model"))
    assert result["status"] == "mechanical_block"
    assert result["semantic_pass"] is None
    assert result["cost"]["dispatches"] == 0
    assert result["cost"]["provider_prompt_tokens"] == result["cost"]["provider_completion_tokens"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_field",
        "missing_citation",
        "fake_claim_id",
        "fake_quote",
        "arity",
        "empty_quote",
        "duplicate_citation",
        "duplicate_claim",
        "extra_citation",
    ],
)
def test_malformed_claim_bindings_never_dispatch(mutation):
    payload = answer()
    if mutation == "missing_field":
        del payload["citations"][0]["source_sha256"]
    elif mutation == "missing_citation":
        payload["citations"] = []
    elif mutation == "fake_claim_id":
        payload["claims"][0]["evidence_ids"] = ["E1"]
    elif mutation == "fake_quote":
        payload["claims"][0]["quotes"] = ["Ari rejected filming."]
    elif mutation == "arity":
        payload["claims"][0]["quotes"].append("Ari approved filming.")
    elif mutation == "empty_quote":
        payload["claims"][0]["quotes"] = [""]
    elif mutation == "duplicate_citation":
        payload["citations"] *= 2
    elif mutation == "duplicate_claim":
        payload["claims"] *= 2
    else:
        payload["citations"].append(source("unused"))
    result = run(payload, backend=lambda body: pytest.fail("bad binding reached model"))
    assert result["status"] == "mechanical_block"
    assert result["cost"]["dispatches"] == 0


def test_whole_source_identity_is_not_chunk_hash_and_claim_sources_are_local():
    rows = [
        source("a", "Ari requested filming. Unquoted private context."),
        source("b", "Bo approved filming."),
    ]
    payload = answer(
        rows,
        claims=[
            dict(text="Ari requested filming.", evidence_ids=["a"], quotes=["Ari requested filming."]),
            dict(text="Bo approved filming.", evidence_ids=["b"], quotes=["Bo approved filming."]),
        ],
    )
    original = copy.deepcopy(payload)
    bodies = []

    def backend(body):
        bodies.append(body)
        return response(audit_reply(2), prompt_eval_count=80, eval_count=40)

    result = run(payload, rows, backend=backend)
    assert result["status"] == "observed" and result["semantic_pass"] is True
    assert payload == original
    assert (
        result["original_payload_sha256"]
        == hashlib.sha256(
            json.dumps(original, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert len(bodies) == 1
    sent = json.loads(bodies[0]["messages"][1]["content"])
    assert [c["bound_sources"][0]["quote"] for c in sent["claims"]] == [
        "Ari requested filming.",
        "Bo approved filming.",
    ]
    assert "Unquoted private context" not in json.dumps(bodies)
    assert bodies[0]["options"]["num_ctx"] == 8192
    assert bodies[0]["options"]["num_predict"] == 260
    budget = result["cost"]["business_budget"]
    assert budget["calls"]["judge"]["attempted"] == budget["calls"]["judge"]["succeeded"] == 1
    assert all(budget["limits"][role] == 0 for role in ("generation", "policy", "search", "embedding"))


@pytest.mark.parametrize(
    "status", ["insufficient_evidence", "verification_failed", "dependency_unavailable", "refused"]
)
def test_refusals_and_unavailable_answers_are_not_passes(status):
    payload = answer()
    payload["status"] = status
    result = run(payload, backend=lambda body: pytest.fail("not applicable"))
    assert result["status"] == "not_applicable" and result["semantic_pass"] is None
    assert result["cost"]["dispatches"] == 0


def test_empty_answer_is_not_a_pass():
    payload = answer()
    payload["claims"] = []
    assert run(payload)["status"] == "not_applicable"


def test_authorization_is_required_before_audit():
    module = importlib.import_module("scripts.run_bound_citation_dev")
    result = module.audit_answer(
        "Who approved?",
        answer(),
        [source()],
        clients.Models(chat_backend=lambda body: pytest.fail("unauthorized")),
        Settings(_env_file=None),
        reauthorize=None,
    )
    assert result["status"] == "authorization_failure" and result["cost"]["dispatches"] == 0


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_reauthorization_source_change_or_revocation_is_unknown_with_cost(phase):
    calls = []

    def reauth(rows):
        calls.append(rows)
        if phase == "pre" or len(calls) == 2:
            return []
        return rows

    result = run(reauthorize=reauth)
    assert result["status"] == "authorization_failure" and result["semantic_pass"] is None
    assert result["cost"]["dispatches"] == (0 if phase == "pre" else 1)
    if phase == "post":
        assert result["cost"]["provider_prompt_tokens"] == 50


def test_post_reauthorization_compares_unquoted_original_source_fields():
    calls = []

    def reauth(rows):
        calls.append(1)
        if len(calls) == 2:
            rows[0]["locator"]["paragraph"] = 9
        return rows

    payload = answer()
    original = copy.deepcopy(payload)
    result = run(payload, reauthorize=reauth)
    assert result["status"] == "authorization_failure"
    assert result["cost"]["dispatches"] == 1 and payload == original


def test_fake_source_index_cannot_borrow_other_claim_or_infer_conclusion():
    rows = [source("a", "Ari requested filming."), source("b", "Bo approved filming.")]
    claims = [
        dict(text="Ari approved filming.", evidence_ids=["a"], quotes=[rows[0]["text"]]),
        dict(text="Bo approved filming.", evidence_ids=["b"], quotes=[rows[1]["text"]]),
    ]
    reply = audit_reply(2, verdict="yes", conclusion="Both approved.", conclusion_claim_indices=[1, 2])
    reply["checks"][0]["source_indices"] = [2]
    result = run(
        answer(rows, claims),
        rows,
        question="Did both approve filming?",
        backend=lambda body: response(reply),
    )
    assert result["assertions_supported"] is False and result["semantic_pass"] is False
    assert result["audit"]["checks"][0]["source_excerpts"] == []
    assert result["model_judgment_relation"]["supported_by_checked_claims"] is False


@pytest.mark.parametrize("fault", ["duplicate", "incomplete", "bad_conclusion"])
def test_official_schema_checks_fail_and_keep_physical_cost(fault):
    reply = audit_reply(2 if fault == "duplicate" else 1)
    if fault == "duplicate":
        reply["checks"][1]["claim_index"] = 1
    elif fault == "incomplete":
        reply["checks"] = []
    else:
        reply.update(verdict="yes", conclusion="Approved", conclusion_claim_indices=[9])
    result = run(
        question="Did Ari approve?",
        backend=lambda body: response(reply, prompt_eval_count=60, eval_count=25),
    )
    assert result["status"] == "schema_failure" and result["semantic_pass"] is None
    assert result["cost"]["schema_failures"] == 1 and result["cost"]["transport_failures"] == 0
    assert result["cost"]["provider_prompt_tokens"] == 60
    assert result["cost"]["business_budget"]["calls"]["judge"]["failed"] == 1
    assert result["cost"]["business_budget"]["judge_tokens_reserved"] == 8372


def test_supported_but_question_incomplete_is_separate():
    result = run(backend=lambda body: response(audit_reply(question_complete=False)))
    assert result["assertions_supported"] is True
    assert result["question_complete"] is False and result["semantic_pass"] is False


def test_original_verdict_value_mismatch_is_shadow_and_preserved():
    payload = answer(verdict={"value": "no", "reason": "Frozen answer"})
    original = copy.deepcopy(payload)
    reply = audit_reply(verdict="yes", conclusion="Ari approved.", conclusion_claim_indices=[1])
    result = run(payload, question="Did Ari approve filming?", backend=lambda body: response(reply))
    assert result["original_verdict_mismatch"] is True
    assert result["semantic_pass"] is False
    assert result["model_judgment_relation"]["value"] == "yes"
    assert result["assertions_supported"] is True and result["question_complete"] is True
    assert payload == original and result["original_verdict"] == original["verdict"]


@pytest.mark.parametrize("counts", [{}, {"prompt_eval_count": 50}, {"eval_count": 20}])
def test_missing_provider_counts_are_null_and_conservative_bookkeeping_is_explicit(counts):
    result = run(backend=lambda body: response(**counts))
    assert result["status"] == "observed" and result["semantic_pass"] is True
    assert result["cost"]["provider_prompt_tokens"] == counts.get("prompt_eval_count")
    assert result["cost"]["provider_completion_tokens"] == counts.get("eval_count")
    assert result["cost"]["token_accounting"] == "conservative_estimate"
    budget = result["cost"]["business_budget"]
    assert budget["calls"]["judge"]["succeeded"] == 1 and budget["calls"]["judge"]["failed"] == 0
    assert budget["judge_tokens_reserved"] >= 180
    assert budget["model_call_records"][0]["token_accounting"] == "conservative_estimate"


def test_transport_failure_retains_dispatch_and_null_cost():
    def backend(body):
        raise httpx.ReadTimeout("synthetic timeout")

    result = run(backend=backend)
    assert result["status"] == "transport_failure" and result["semantic_pass"] is None
    assert result["cost"]["dispatches"] == result["cost"]["transport_failures"] == 1
    assert result["cost"]["provider_prompt_tokens"] is None
    assert result["cost"]["business_budget"]["calls"]["judge"]["failed"] == 1


def test_failed_transport_usage_is_retained():
    def backend(body):
        raise clients.DependencyError(
            "synthetic failure", stage="judge", usage={"prompt_tokens": 75, "completion_tokens": 4}
        )

    result = run(backend=backend)
    assert result["cost"]["provider_prompt_tokens"] == 75
    assert result["cost"]["provider_completion_tokens"] == 4


def test_whole_serialized_schema_request_preflight_is_estimate_and_blocks(monkeypatch):
    module = importlib.import_module("scripts.run_bound_citation_dev")
    observed = []

    def count(text):
        observed.append(json.loads(text))
        return 8100

    monkeypatch.setattr(module, "token_count", count)
    result = run(backend=lambda body: pytest.fail("oversized request"))
    assert result["status"] == "preflight_block" and result["cost"]["dispatches"] == 0
    assert observed[0]["format"]["$defs"]["BoundClaimCheck"]
    assert len(observed[0]["messages"]) == 2 and observed[0]["options"]["num_predict"] == 180
    assert result["cost"]["input_token_estimate"] == 8100
    assert result["cost"]["token_counter"] == "cl100k_base_estimate_not_provider_exact"


@pytest.mark.parametrize(
    "counts",
    [
        {"prompt_eval_count": 8190, "eval_count": 20},
        {"prompt_eval_count": 50, "eval_count": 181},
        {"prompt_eval_count": 8193},
    ],
)
def test_detected_provider_ceiling_violation_is_unknown_with_actual_cost(counts):
    result = run(backend=lambda body: response(**counts))
    assert result["status"] == "ceiling_violation" and result["semantic_pass"] is None
    assert result["cost"]["dispatches"] == 1
    assert result["cost"]["provider_prompt_tokens"] == counts.get("prompt_eval_count")


def test_budget_reservation_blocks_without_dispatch():
    result = run(
        cfg=Settings(_env_file=None, agent_judge_token_budget=100),
        backend=lambda body: pytest.fail("budget should block"),
    )
    assert result["status"] == "budget_exhausted" and result["cost"]["dispatches"] == 0


def test_elapsed_deadline_keeps_actual_cost_and_unknown_outcome(monkeypatch):
    module = importlib.import_module("scripts.run_bound_citation_dev")
    real = module.ExecutionBudget
    now = [0]

    def budget(cfg, **kwargs):
        return real(cfg, clock=lambda: now[0], **kwargs)

    monkeypatch.setattr(module, "ExecutionBudget", budget)

    def backend(body):
        now[0] = 181
        return response(prompt_eval_count=100, eval_count=10)

    result = run(backend=backend)
    assert result["status"] == "budget_exhausted" and result["semantic_pass"] is None
    assert result["cost"]["provider_prompt_tokens"] == 100
    assert result["cost"]["business_budget"]["calls"]["judge"]["failed"] == 1


def test_real_http_transport_is_bounded_and_counted_once(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(
            200, request=httpx.Request("POST", url), json=response(prompt_eval_count=80, eval_count=25)
        )

    monkeypatch.setattr(clients.httpx, "post", post)
    cfg = Settings(
        _env_file=None,
        semantic_judge_url="http://synthetic.invalid",
        semantic_judge_model="synthetic-model",
        model_timeout_seconds=999,
    )
    result = run(cfg=cfg, models=clients.Models())
    assert result["status"] == "observed" and len(calls) == 1
    assert calls[0][0] == "http://synthetic.invalid/api/chat"
    assert calls[0][1]["json"]["model"] == "synthetic-model"
    assert 0 < calls[0][1]["timeout"] <= 180
    assert calls[0][1]["trust_env"] is False


def test_authored_contrast_fixture_is_complete_and_not_human_gold():
    path = Path("fixtures/source_contract/evidence-first-20261011-v1.json")
    assert path.exists(), "authored contrast fixture missing"
    fixture = json.loads(path.read_text())
    assert fixture["label_author"] == "assistant" and fixture["human_gold"] is False
    assert fixture["purpose"] == "synthetic_component_contrasts_not_dev_accuracy"
    required = {
        "wrong_subject",
        "wrong_value",
        "wrong_unit",
        "background",
        "hypothetical_intent",
        "requesting_consent",
        "accusation_gain",
        "source_attribution",
        "publication_event_date",
        "omitted_second_slot",
        "joint_comparison",
        "cross_claim_borrowing",
    }
    assert {pair["contrast"] for pair in fixture["pairs"]} == required
    for pair in fixture["pairs"]:
        assert pair["faithful"]["expected"]["assertions_supported"] is True
        for case in (pair["faithful"], pair["contrast_case"]):
            assert case["expected"]["stage"] in {"semantic_observation", "mechanical_or_semantic"}
            assert case["payload"]["citations"] == case["evidence"]


@pytest.mark.parametrize("duplicate", ["source", "conclusion"])
def test_duplicate_proof_indices_are_schema_failure_with_cost(duplicate):
    reply = audit_reply(verdict="yes", conclusion="Ari approved filming.", conclusion_claim_indices=[1])
    if duplicate == "source":
        reply["checks"][0]["source_indices"] = [1, 1]
    else:
        reply["conclusion_claim_indices"] = [1, 1]
    result = run(
        question="Did Ari approve filming?",
        backend=lambda body: response(reply, prompt_eval_count=60, eval_count=25),
    )
    assert result["status"] == "schema_failure"
    assert result["cost"]["schema_failures"] == 1 and result["cost"]["provider_completion_tokens"] == 25
    assert result["cost"]["business_budget"]["calls"]["judge"]["failed"] == 1


def test_nonjudgment_cannot_receive_fabricated_judgment_relation():
    reply = audit_reply(verdict="yes", conclusion="Ari approved.", conclusion_claim_indices=[1])
    result = run(backend=lambda body: response(reply))
    assert result["status"] == "schema_failure" and result["semantic_pass"] is None


def test_raw_serialized_payload_retains_original_bytes_digest():
    raw = ("  " + json.dumps(answer(), indent=3) + "\n").encode()
    result = run(raw)
    assert result["status"] == "observed"
    assert result["original_payload_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["payload_digest_encoding"] == "original_bytes"


def test_original_quote_whitespace_cannot_be_fabricated():
    payload = answer()
    payload["claims"][0]["quotes"] = ["Ari  approved filming."]
    result = run(payload, backend=lambda body: pytest.fail("only original bytes may be quoted"))
    assert result["status"] == "mechanical_block"


def test_optional_serialized_metadata_omission_uses_trusted_metadata():
    payload = answer()
    del payload["citations"][0]["metadata"]
    bodies = []

    def backend(body):
        bodies.append(body)
        return response()

    assert run(payload, backend=backend)["status"] == "observed"
    sent = json.loads(bodies[0]["messages"][1]["content"])
    assert sent["claims"][0]["bound_sources"][0]["source"] == "Synthetic Gazette"
    assert sent["claims"][0]["bound_sources"][0]["published_at"] == "2026-01-10"


def test_joint_bound_source_omission_is_unsupported_without_borrowing():
    rows = [source("a", "Ari approved filming."), source("b", "Bo approved filming.")]
    claims = [
        dict(
            text="Ari and Bo approved filming.",
            evidence_ids=["a", "b"],
            quotes=[r["text"] for r in rows],
        )
    ]
    reply = audit_reply()
    result = run(answer(rows, claims), rows, backend=lambda body: response(reply))
    assert result["status"] == "observed" and result["assertions_supported"] is False
    assert result["audit"]["checks"][0]["reason"] == "unverified_bound_source"


def test_jointly_supported_comparison_uses_both_original_quotes():
    rows = [
        source("a", "A hall has capacity of 9 units."),
        source("b", "B hall has capacity of 5 units."),
    ]
    claims = [
        dict(
            text="A hall has capacity of 9 units and B hall has capacity of 5 units; A has greater capacity.",
            evidence_ids=["a", "b"],
            quotes=[r["text"] for r in rows],
        )
    ]
    reply = audit_reply(
        verdict="yes", conclusion="A has greater capacity.", conclusion_claim_indices=[1]
    )
    reply["checks"][0]["source_indices"] = [1, 2]
    result = run(
        answer(rows, claims, verdict={"value": "yes"}),
        rows,
        question="Does A hall have greater capacity than B hall?",
        backend=lambda body: response(reply),
    )
    assert result["status"] == "observed" and result["semantic_pass"] is True
    assert result["audit"]["checks"][0]["source_excerpts"] == [r["text"] for r in rows]


def test_numeric_negative_is_mechanical_block_not_semantic_detection():
    rows = [source("numeric", "The batch limit is 9 rows.")]
    payload = answer(
        rows,
        [dict(text="The batch limit is 12 rows.", evidence_ids=["numeric"], quotes=[rows[0]["text"]])],
    )
    result = run(payload, rows, backend=lambda body: pytest.fail("numeric validator already blocks"))
    assert result["status"] == "mechanical_block"
    assert result["semantic_pass"] is None and result["cost"]["dispatches"] == 0


def test_schema_failure_with_unknown_usage_is_failed_but_not_zero_cost():
    result = run(backend=lambda body: {"message": {"content": "malformed JSON"}})
    assert result["status"] == "schema_failure"
    assert result["cost"]["provider_prompt_tokens"] is None
    assert result["cost"]["provider_completion_tokens"] is None
    budget = result["cost"]["business_budget"]
    assert budget["calls"]["judge"]["failed"] == 1 and budget["judge_tokens_reserved"] == 8372
    assert budget["model_call_records"][0]["token_accounting"] == "conservative_estimate"


def test_helper_does_not_activate_semantic_defaults_or_production_gate():
    from app.answer_quality import require_audit_gate
    from app.config import settings

    before = settings().model_dump()
    gate = require_audit_gate
    result = run()
    assert result["mode"] == "shadow"
    assert settings().model_dump() == before
    assert require_audit_gate is gate


def test_keyboard_interruption_keeps_dispatch_reservation_and_unknown_outcome():
    def backend(body):
        raise KeyboardInterrupt("synthetic interruption")

    try:
        result = run(backend=backend)
    except KeyboardInterrupt:
        pytest.fail("interrupted physical dispatch was not returned as an observation")
    assert result["status"] == "interrupted" and result["semantic_pass"] is None
    assert result["cost"]["dispatches"] == 1
    assert result["cost"]["provider_prompt_tokens"] is None
    assert result["cost"]["business_budget"]["calls"]["judge"]["failed"] == 1
    assert result["cost"]["business_budget"]["judge_tokens_reserved"] == 8372


def test_cancelled_scope_check_is_interruption_not_schema_or_transport_failure():
    from app.execution_budget import TaskCancelled

    def reauth(rows):
        raise TaskCancelled("synthetic cancellation")

    result = run(reauthorize=reauth)
    assert result["status"] == "interrupted"
    assert (
        result["cost"]["dispatches"]
        == result["cost"]["schema_failures"]
        == result["cost"]["transport_failures"]
        == 0
    )


def test_meter_discloses_model_and_shadow_independence():
    cfg = Settings(_env_file=None, semantic_judge_model="synthetic-audit-model")
    result = run(cfg=cfg)
    assert result["cost"]["model"] == cfg.semantic_judge_model
    assert result["independent"] is False
    assert result["scope"] == "each_claim_bound_quotes_only"


def test_judgment_with_unresolved_relation_cannot_pass_supported_claims():
    result = run(
        answer(verdict={"value": "unclear"}),
        question="Did Ari approve filming?",
        backend=lambda body: response(audit_reply()),
    )
    assert result["status"] == "observed" and result["assertions_supported"] is True
    assert result["question_complete"] is True and result["semantic_pass"] is False


def test_judgment_without_original_verdict_cannot_pass_a_new_model_conclusion():
    reply = audit_reply(verdict="yes", conclusion="Ari approved.", conclusion_claim_indices=[1])
    result = run(question="Did Ari approve filming?", backend=lambda body: response(reply))
    assert result["assertions_supported"] is True and result["question_complete"] is True
    assert result["original_verdict"] is None and result["semantic_pass"] is False


@pytest.mark.parametrize(
    "field,original,replacement",
    [
        ("locator", {"paragraph": 1}, {"paragraph": True}),
        ("locator", {"paragraph": 1}, {"paragraph": 1.0}),
        ("metadata", {"active": True}, {"active": 1}),
        ("metadata", {"active": 1}, {"active": 1.0}),
    ],
)
def test_type_drift_in_original_citation_fields_blocks_before_dispatch(field, original, replacement):
    rows = [source(**{field: original})]
    payload = answer(rows)
    payload["citations"][0][field] = replacement
    result = run(payload, rows, backend=lambda body: pytest.fail("type drift reached semantic model"))
    assert result["status"] == "mechanical_block" and result["semantic_pass"] is None
    assert result["cost"]["dispatches"] == 0


@pytest.mark.parametrize("phase", ["pre", "post"])
@pytest.mark.parametrize(
    "field,original,replacement",
    [
        ("locator", {"paragraph": 1}, {"paragraph": True}),
        ("locator", {"paragraph": 1}, {"paragraph": 1.0}),
        ("metadata", {"active": True}, {"active": 1}),
    ],
)
def test_reauthorization_type_drift_is_unknown_and_preserves_actual_cost(
    phase, field, original, replacement
):
    rows = [source(**{field: original})]
    payload = answer(rows)
    original_payload = copy.deepcopy(payload)
    checked = []

    def reauthorize(current):
        checked.append(1)
        if phase == "pre" or len(checked) == 2:
            current[0][field] = replacement
        return current

    result = run(payload, rows, reauthorize=reauthorize)
    assert result["status"] == "authorization_failure" and result["semantic_pass"] is None
    assert result["cost"]["dispatches"] == (0 if phase == "pre" else 1)
    assert payload == original_payload
    if phase == "post":
        assert result["cost"]["provider_prompt_tokens"] == 50
        assert result["cost"]["provider_completion_tokens"] == 20
        assert result["cost"]["business_budget"]["calls"]["judge"]["failed"] == 1


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), float("-inf"), (1,), {1: "invalid-key"}]
)
def test_non_json_or_nonfinite_source_fields_are_rejected_before_dispatch(invalid):
    rows = [source(locator={"paragraph": 1, "annotation": invalid})]
    payload = answer(rows)
    result = run(payload, rows, backend=lambda body: pytest.fail("non-JSON source field reached model"))
    assert result["status"] == "mechanical_block" and result["semantic_pass"] is None
    assert result["cost"]["dispatches"] == 0


@pytest.mark.parametrize("phase", ["pre", "post"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), (1,), {1: "invalid-key"}])
def test_non_json_reauthorization_records_are_unknown_with_retained_cost(phase, invalid):
    checked = []

    def reauthorize(current):
        checked.append(1)
        if phase == "pre" or len(checked) == 2:
            current[0]["metadata"]["annotation"] = invalid
        return current

    result = run(reauthorize=reauthorize)
    assert result["status"] == "authorization_failure" and result["semantic_pass"] is None
    assert result["cost"]["dispatches"] == (0 if phase == "pre" else 1)
    if phase == "post":
        assert result["cost"]["provider_prompt_tokens"] == 50


def test_record_key_order_is_irrelevant_but_evidence_order_is_preserved():
    def reauthorize(current):
        return [{key: row[key] for key in reversed(row)} for row in current]

    assert run(reauthorize=reauthorize)["status"] == "observed"
    rows = [source("a"), source("b")]
    payload = answer(
        rows,
        claims=[
            dict(text=row["text"], evidence_ids=[row["chunk_id"]], quotes=[row["text"]]) for row in rows
        ],
    )
    result = run(
        payload,
        rows,
        reauthorize=lambda current: list(reversed(current)),
        backend=lambda body: pytest.fail("evidence order changed"),
    )
    assert result["status"] == "authorization_failure" and result["cost"]["dispatches"] == 0


# Registered runner inputs are entirely synthetic and use the standard review packet.
def test_runner_freezes_complete_input_identity_and_all_positions():
    runner = importlib.import_module("scripts.run_bound_citation_dev")
    assert callable(getattr(runner, "registration_for", None)), "registered shadow runner missing"


def test_runner_has_narrow_register_execute_cli():
    runner = importlib.import_module("scripts.run_bound_citation_dev")
    assert callable(getattr(runner, "main", None)), "register/execute CLI missing"


def test_fixture_truthfully_discloses_synthetic_caller_owned_identities():
    fixture = json.loads(Path("fixtures/source_contract/evidence-first-20261011-v1.json").read_bytes())
    assert fixture["source_identity"] == "synthetic_caller_owned_identity_not_verified_full_source_bytes"


@pytest.fixture
def shadow_tree(tmp_path, monkeypatch):
    from types import SimpleNamespace

    r = importlib.import_module("scripts.run_bound_citation_dev")
    spec = importlib.util.spec_from_file_location(
        "synthetic_review_helpers",
        Path(".runtime/evidence-first-20261011/test_generation_review_helpers_v2.py"),
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    m = helpers.m
    original_synthetic = helpers.synthetic

    def synthetic():
        raw, suite, _ = original_synthetic()
        raw["registration"]["code"] = copy.deepcopy(fingerprints)
        for task in suite["tasks"]:
            task["user"] = "lt-eng"
        for arm in m.ARMS:
            for row in raw["results"][arm]:
                row["answer_payload"] = answer()
                row["retrieved_evidence"] = [dict(source(), rank=1, parent_rank=1)]
                # Preserve the exact original ordered trusted evidence as citations.
                row["answer_payload"]["citations"] = copy.deepcopy(row["retrieved_evidence"])
                row["status"] = "answered"
        return raw, suite, m.answer_review_packet(raw, suite)

    monkeypatch.setattr(helpers, "synthetic", synthetic)
    # Simulate the historical Stage3 205-source schema only inside this fixture.
    # Real source fingerprints and production lineage validation remain unchanged.
    # Derive the 205 historical paths from the frozen archive instead of naming each
    # later script, so adding an unrelated new source does not break this fixture.
    archive = Path(".runtime/evidence-first-20261011/frozen-code-v4")
    historical = {str(p.relative_to(archive)) for folder in ("app", "agent", "scripts")
                  for p in (archive / folder).rglob("*.py")} | {"pyproject.toml", "AGENTS.md"}
    runtime_fingerprints = {k: v for k, v in m.code_fingerprints().items()
                            if k in historical or k == "scripts/run_bound_citation_dev.py"}
    fingerprints = copy.deepcopy(runtime_fingerprints)
    fingerprints.pop("scripts/run_bound_citation_dev.py")
    monkeypatch.setattr(m, "code_fingerprints", lambda: fingerprints)
    monkeypatch.setattr(r, "code_fingerprints", lambda: runtime_fingerprints)
    # The reused helper validates imported bytes against the synthetic 205-source parent.
    paths = helpers.synthetic_tree(tmp_path)
    parent = {
        "identity": {"corpus": {"documents": []}, "source_index": {"index": "synthetic", "chunks": []}}
    }
    parent["identity"]["corpus_sha256"] = m.digest(parent["identity"]["corpus"])
    parent["identity"]["source_index_sha256"] = m.digest(parent["identity"]["source_index"])
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    raw = m.read_json(paths["run"] / "raw.json")
    raw["corpus_sha256"] = parent["identity"]["corpus_sha256"]
    raw["registration"]["input_bytes_sha256"] = [hashlib.sha256(parent_path.read_bytes()).hexdigest()]
    for name, obj in [
        ("raw.json", raw),
        ("review-packet.json", m.answer_review_packet(raw, m.read_json(paths["run"] / "suite.json"))),
        ("registration.json", raw["registration"]),
    ]:
        (paths["run"] / name).write_text(json.dumps(obj))
    paths["generation_registration"].write_text(json.dumps(raw["registration"]))
    public = m.read_json(paths["source_artifact"])
    public.update(identity=raw["registration"], private_raw_sha256=m.digest(raw))
    paths["source_artifact"].write_text(json.dumps(public))
    m.prepare(paths)
    for number in (1, 2, 3):
        part = m.read_json(paths["out"] / f"answer-part-{number}.json")
        reviews = [
            dict(
                id=i["id"],
                fact_verdicts={"f1": "supported"},
                claim_verdicts=["supported"] * len(i["answer"]["claims"]),
                citation_verdicts=["supported"] * len(i["answer"]["citations"]),
                reason="PRIVATE_REASON",
                **dict.fromkeys(m.CHECKS, True),
            )
            for i in part["items"]
        ]
        m.write_new(paths["out"] / f"answer-part-{number}-reviews.json", {"reviews": reviews})
    m.score(paths)
    out = paths["out"]
    packet = m.read_json(paths["run"] / "review-packet.json")
    args = SimpleNamespace(
        run=str(paths["run"]),
        review_root=str(out),
        generation_registration=str(paths["generation_registration"]),
        generation_public=str(paths["source_artifact"]),
        review_registration=str(paths["registration"]),
        review_public=str(paths["public"]),
        parent=str(parent_path),
        review_attestation=str(tmp_path / "attestation.json"),
        fixture=str(Path("fixtures/source_contract/evidence-first-20261011-v1.json").resolve()),
        registration=str(tmp_path / "shadow-registration.json"),
        private_root=str(tmp_path / "private"),
        output=str(tmp_path / "shadow-public.json"),
        deduplicate=True,
    )
    cfg = m.settings().model_copy(deep=True)
    monkeypatch.setattr(r, "settings", lambda: cfg, raising=False)
    identity = {
        "policy": {
            "name": cfg.semantic_judge_model,
            "digest": "a" * 64,
            "endpoint_sha256": "b" * 64,
            "runtime": {"version": "1"},
            "status": "active",
        },
        "libraries": {"tiktoken": "synthetic"},
    }
    monkeypatch.setattr(
        r, "active_model_identity", lambda config: copy.deepcopy(identity), raising=False
    )
    monkeypatch.setattr(r, "private_root", lambda path: Path(path), raising=False)
    monkeypatch.setattr(r, "ROOT", Path.cwd(), raising=False)
    monkeypatch.setattr(
        r,
        "active_scope",
        lambda db: ({"lt-eng": "registered-user"}, [], parent["identity"]["corpus"]),
        raising=False,
    )
    monkeypatch.setattr(
        r, "index_ledger", lambda search, records: parent["identity"]["source_index"], raising=False
    )
    import app.db

    statements = []

    class DB:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement):
            statements.append(str(statement))

    monkeypatch.setattr(app.db, "SessionLocal", DB)
    monkeypatch.setattr(clients, "Search", lambda **kw: SimpleNamespace(index=kw["index"]))
    calls, authorized = [], []

    def rebind(db, user, rows, corpus):
        authorized.append(user)
        return copy.deepcopy(rows)

    monkeypatch.setattr(r, "rebind_rows", rebind, raising=False)

    def backend(body):
        calls.append(copy.deepcopy(body))
        return response(prompt_eval_count=50, eval_count=20)

    monkeypatch.setattr(r, "Models", lambda: clients.Models(chat_backend=backend))
    # Keep approved component's official API lookup intact while replacing runner construction only.
    monkeypatch.setattr(r, "make_models", lambda: clients.Models(chat_backend=backend), raising=False)
    monkeypatch.setattr(r, "Models", clients.Models)
    if callable(getattr(r, "prepare_review_attestation", None)):
        r.prepare_review_attestation(args)
    return SimpleNamespace(
        r=r,
        args=args,
        paths=paths,
        calls=calls,
        authorized=authorized,
        statements=statements,
        cfg=cfg,
        identity=identity,
        root=tmp_path,
        m=m,
        raw=raw,
        packet=packet,
        parent=parent,
    )


def freeze_shadow(f):
    reg = f.r.registration_for(f.args)
    Path(f.args.registration).write_text(json.dumps(reg))
    return reg


def test_shadow_registration_maps_all_positions_without_reference_grade_access(shadow_tree, monkeypatch):
    f = shadow_tree
    original = json.loads

    def loads(value, *a, **kw):
        assert "PRIVATE_REASON" not in (value.decode() if isinstance(value, bytes) else value)
        return original(value, *a, **kw)

    monkeypatch.setattr(f.r.json, "loads", loads)
    reg = freeze_shadow(f)
    assert reg["dev_positions"] == 94 and reg["authored_positions"] == 24
    assert len(reg["positions"]) == 118
    assert reg["primary_tasks"] == 40 and reg["pending_tasks"] == 7
    assert len(reg["upstream_code"]) == 205
    assert len({p["audit_input_sha256"] for p in reg["positions"] if p["kind"] == "dev"}) == 1
    assert not f.calls


def test_shadow_complete_replay_deduplicates_and_compares_only_after_runtime(shadow_tree, monkeypatch):
    f = shadow_tree
    _reg = freeze_shadow(f)
    original = f.r.compare_references

    def compare(*a, **kw):
        assert len(f.calls) > 1 and len(a[0]) == 118
        return original(*a, **kw)

    monkeypatch.setattr(f.r, "compare_references", compare)
    result = f.r.execute(f.args, {"run_id": "synthetic-complete"})
    assert result["valid"] and result["status"] == "complete"
    assert len(result["positions"]) == 118
    dev = [p for p in result["positions"] if p["kind"] == "dev"]
    assert sum(p["shared_physical_cost"] for p in dev) == 93
    assert result["comparison"]["primary_positions"] == 80
    assert result["comparison"]["pending_positions"] == 14
    assert f.statements == ["SET TRANSACTION READ ONLY"]
    assert set(f.authorized) == {"registered-user"}
    public = Path(f.args.output).read_text()
    assert all(s not in public for s in ("PRIVATE_", "Ari approved", "Literal support"))
    assert all("required_facts" not in json.dumps(body) for body in f.calls)
    raw = json.loads((Path(f.args.private_root) / "synthetic-complete/raw.json").read_bytes())
    assert all(
        p["payload"] == f.raw["results"][p["arm"]][p["task_index"]]["answer_payload"]
        for p in raw["positions"]
        if p["kind"] == "dev"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "raw",
        "packet",
        "payload",
        "evidence",
        "review_public",
        "pending",
        "parent",
        "mapping",
        "code",
        "settings",
        "model",
    ],
)
def test_shadow_rejects_drift_before_any_model_and_preserves_denominator(
    shadow_tree, mutation, monkeypatch
):
    f = shadow_tree
    freeze_shadow(f)
    if mutation in ("settings", "model", "code"):
        if mutation == "settings":
            f.cfg.semantic_judge_model = "changed"
        elif mutation == "model":
            f.identity["policy"]["digest"] = "c" * 64
        else:
            monkeypatch.setattr(f.r, "code_fingerprints", lambda: {})
    else:
        path = {
            "raw": Path(f.args.run) / "raw.json",
            "packet": Path(f.args.run) / "review-packet.json",
            "payload": Path(f.args.run) / "raw.json",
            "evidence": Path(f.args.run) / "raw.json",
            "review_public": Path(f.args.review_public),
            "pending": Path(f.args.review_registration),
            "parent": Path(f.args.parent),
            "mapping": Path(f.args.review_root) / "mapping.json",
        }[mutation]
        path.write_bytes(path.read_bytes() + b"\n")
    result = f.r.execute(f.args, {"run_id": "bad-" + mutation})
    assert not result["valid"] and not f.calls
    assert len(result["positions"]) == 118
    assert all(p["semantic_pass"] is None for p in result["positions"])


def test_shadow_unknown_reference_support_is_not_false():
    r = importlib.import_module("scripts.run_bound_citation_dev")
    assert r.reference_support(["supported", "unclear"]) is None
    assert r.reference_support(["unsupported", "unclear"]) is False
    assert r.reference_support([]) is None


def test_shadow_public_cost_projection_rejects_untrusted_scalar_text():
    r = importlib.import_module("scripts.run_bound_citation_dev")
    cost = {
        "dispatches": 1,
        "provider_prompt_tokens": "PRIVATE_TEXT",
        "business_budget": {"text": "PRIVATE_TEXT"},
    }
    projected = r.public_cost(cost)
    assert projected["provider_prompt_tokens"] is None
    assert "PRIVATE_TEXT" not in json.dumps(projected)


def test_shadow_source_revocation_after_dispatch_remains_unknown_and_keeps_cost(
    shadow_tree, monkeypatch
):
    f = shadow_tree
    freeze_shadow(f)
    count = 0

    def rebind(db, user, rows, corpus):
        nonlocal count
        count += 1
        if count == 2:
            raise PermissionError("PRIVATE_ERROR")
        return copy.deepcopy(rows)

    monkeypatch.setattr(f.r, "rebind_rows", rebind)
    result = f.r.execute(f.args, {"run_id": "revoked"})
    dev = [p for p in result["positions"] if p["kind"] == "dev"]
    assert all(p["status"] == "authorization_failure" and p["semantic_pass"] is None for p in dev)
    assert result["physical"][0]["cost"]["dispatches"] == 1
    assert "PRIVATE_ERROR" not in Path(f.args.output).read_text()


def test_shadow_interruption_keeps_cost_and_all_unobserved_positions(shadow_tree, monkeypatch):
    f = shadow_tree
    freeze_shadow(f)

    def backend(body):
        raise KeyboardInterrupt()

    monkeypatch.setattr(f.r, "make_models", lambda: clients.Models(chat_backend=backend))
    result = f.r.execute(f.args, {"run_id": "interrupted"})
    assert result["status"] == "interrupted" and not result["valid"]
    assert len(result["positions"]) == 118 and result["comparison"] is None
    assert result["positions"][0]["status"] == "interrupted"
    assert sum(p["status"] == "not_observed" for p in result["positions"]) == 117
    assert result["physical"][0]["cost"]["dispatches"] == 1
    assert result["physical"][0]["cost"]["provider_prompt_tokens"] is None
    assert (Path(f.args.private_root) / "interrupted/raw.json").exists()


def test_shadow_unknown_provider_usage_is_null_without_fabricated_failure(shadow_tree, monkeypatch):
    f = shadow_tree
    freeze_shadow(f)
    monkeypatch.setattr(f.r, "make_models", lambda: clients.Models(chat_backend=lambda body: response()))
    result = f.r.execute(f.args, {"run_id": "unknown-usage"})
    assert result["valid"]
    assert result["summary"]["physical"]["provider_prompt_tokens"] is None
    assert result["positions"][0]["status"] == "observed"
    raw = json.loads((Path(f.args.private_root) / "unknown-usage/raw.json").read_bytes())
    record = raw["physical"][0]["observation"]["cost"]
    assert record["conservative_reserved_tokens"] > 0
    assert record["business_budget"]["calls"]["judge"]["prompt_tokens"] == 8192


def test_shadow_outputs_are_exclusive(shadow_tree):
    f = shadow_tree
    freeze_shadow(f)
    first = f.r.execute(f.args, {"run_id": "exclusive"})
    original = Path(f.args.output).read_bytes()
    with pytest.raises(FileExistsError):
        f.r.execute(f.args, {"run_id": "exclusive"})
    assert Path(f.args.output).read_bytes() == original
    assert first["valid"]


def test_shadow_identity_drift_stops_comparison_before_grade_loading(shadow_tree, monkeypatch):
    f = shadow_tree
    freeze_shadow(f)
    _original = f.r.make_models
    count = 0

    def backend(body):
        nonlocal count
        count += 1
        if count == 1:
            f.identity["policy"]["digest"] = "c" * 64
        return response(prompt_eval_count=50, eval_count=20)

    monkeypatch.setattr(f.r, "make_models", lambda: clients.Models(chat_backend=backend))
    comparisons = []
    monkeypatch.setattr(f.r, "compare_references", lambda *a: comparisons.append(True))
    result = f.r.execute(f.args, {"run_id": "model-drift"})
    assert not result["valid"] and result["status"] == "invalid_drift"
    assert result["comparison"] is None and len(result["physical"]) > 0
    assert not comparisons, "drifted runtime exposed grades"


def test_shadow_untrusted_registration_strings_never_reach_public(shadow_tree):
    f = shadow_tree
    reg = freeze_shadow(f)
    reg["positions"][0]["payload_sha256"] = "PRIVATE_SECRET"
    Path(f.args.registration).write_text(json.dumps(reg))
    result = f.r.execute(f.args, {"run_id": "malformed-registration"})
    assert not result["valid"] and not f.calls
    assert "PRIVATE_SECRET" not in Path(f.args.output).read_text()


def test_shadow_registered_full_corpus_and_index_identity(shadow_tree):
    f = shadow_tree
    reg = freeze_shadow(f)
    assert reg["corpus_sha256"] == f.parent["identity"]["corpus_sha256"]
    assert reg["index_sha256"] == f.parent["identity"]["source_index_sha256"]


def test_shadow_direct_script_help_requires_no_import_environment():
    import subprocess

    result = subprocess.run(
        [".venv/bin/python", "scripts/run_bound_citation_dev.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "register" in result.stdout and "execute" in result.stdout


def test_shadow_authored_errors_use_only_observed_semantic_denominators(shadow_tree):
    f = shadow_tree
    freeze_shadow(f)
    result = f.r.execute(f.args, {"run_id": "authored-metrics"})
    metrics = result["comparison"]["authored"]
    assert metrics["registered"] == 24
    assert metrics["false_accepts"] > 0
    assert metrics["false_rejects"] == sum(
        row.get("predicted_pass") is False and row.get("expected_pass") is True
        for row in result["comparison"]["rows"]
        if row["kind"] == "authored"
    )
    assert metrics["mechanical_blocked"] > 0
    assert metrics["semantic_false_accept_denominator"] + metrics["negative_unknown"] == 12
    assert metrics["mechanical_blocked"] <= metrics["negative_unknown"]


def test_shadow_preflight_incomplete_upstream_never_registers(shadow_tree):
    f = shadow_tree
    raw = copy.deepcopy(f.raw)
    raw["results"]["literal_fallback"].pop()
    (Path(f.args.run) / "raw.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="complete|Complete"):
        f.r.registration_for(f.args)
    assert not Path(f.args.registration).exists() and not f.calls


def test_shadow_register_discloses_tokenizer_and_unverified_provider_framing(shadow_tree):
    f = shadow_tree
    reg = freeze_shadow(f)
    assert len(reg["tokenizer_library_sha256"]) == 64
    assert reg["configuration"]["provider_framing_and_truncation_verified"] is False


def test_shadow_reviewed_byte_tamper_aborts_before_inference(shadow_tree):
    f = shadow_tree
    freeze_shadow(f)
    path = Path(f.args.review_root) / "reviewed.json"
    path.write_bytes(path.read_bytes() + b"\n")
    result = f.r.execute(f.args, {"run_id": "reference-byte-drift"})
    assert not result["valid"] and not f.calls
    assert result["comparison"] is None and len(result["positions"]) == 118


def test_shadow_operational_mechanical_and_support_outcomes_remain_separate(shadow_tree):
    f = shadow_tree
    freeze_shadow(f)
    result = f.r.execute(f.args, {"run_id": "separate-denominators"})
    summary = result["summary"]["authored"]
    assert summary["registered"] == 24
    assert summary["operational_observed"] == summary["observed"]
    assert summary["semantic_reject"] + summary["semantic_accept"] == summary["observed"]
    assert summary["mechanical_blocked"] == summary["statuses"]["mechanical_block"]
    assert summary["unknown"] + summary["observed"] == summary["registered"]


def test_shadow_permission_failure_invalidates_run_even_without_corpus_ledger_change(
    shadow_tree, monkeypatch
):
    f = shadow_tree
    freeze_shadow(f)
    monkeypatch.setattr(
        f.r, "rebind_rows", lambda *a: (_ for _ in ()).throw(PermissionError("PRIVATE_REASON"))
    )
    result = f.r.execute(f.args, {"run_id": "permission-invalid"})
    assert not result["valid"]
    assert result["drift"]["authorization"] is True
    assert result["comparison"] is None


def test_shadow_no_dedup_executes_each_valid_original_position(shadow_tree):
    f = shadow_tree
    f.args.deduplicate = False
    freeze_shadow(f)
    result = f.r.execute(f.args, {"run_id": "no-sharing"})
    assert result["valid"]
    assert sum(p["kind"] == "dev" and p["status"] == "observed" for p in result["positions"]) == 94
    assert len([x for x in result["physical"] if x["cost"]["dispatches"] == 1]) >= 94
    assert not any(p["shared_physical_cost"] for p in result["positions"])


def test_shadow_attestation_rejects_strict_score_public_reference_mismatch(shadow_tree):
    f = shadow_tree
    Path(f.args.review_attestation).unlink()
    path = Path(f.args.review_root) / "scored-primary.json"
    scored = json.loads(path.read_bytes())
    tid = next(t for t in scored["details"]["shared_contract"] if t not in f.m.PENDING_IDS)
    original = scored["details"]["shared_contract"][tid]["answer"]["strict_task_success"]
    scored["details"]["shared_contract"][tid]["answer"]["strict_task_success"] = not original
    path.write_text(json.dumps(scored))
    with pytest.raises(ValueError, match="scored_public_reference_changed"):
        f.r.prepare_review_attestation(f.args)
    assert not Path(f.args.review_attestation).exists() and not f.calls


def test_shadow_relation_projection_is_enum_and_bool_only():
    r = importlib.import_module("scripts.run_bound_citation_dev")
    row = dict(
        id="a" * 64,
        kind="dev",
        arm="shared_contract",
        task_id="t",
        task_index=0,
        pending=False,
        payload_sha256="b" * 64,
        evidence_sha256="c" * 64,
        audit_input_sha256="d" * 64,
        observation={
            "status": "observed",
            "model_judgment_relation": {
                "value": "yes",
                "conclusion": "PRIVATE_CONCLUSION",
                "claim_indices": [1],
                "supported_by_checked_claims": True,
            },
        },
    )
    projected = r._public_position(row)
    assert (
        projected["judgment_relation"] == "yes"
        and projected["relation_supported_by_checked_claims"] is True
    )
    assert "PRIVATE_CONCLUSION" not in json.dumps(projected)
    row["observation"]["model_judgment_relation"]["value"] = "PRIVATE_RELATION"
    assert r._public_position(row)["judgment_relation"] is None


def test_shadow_cached_position_still_reauthorizes_exact_task_user(shadow_tree, monkeypatch):
    f = shadow_tree
    freeze_shadow(f)
    count = 0

    def rebind(db, user, rows, corpus):
        nonlocal count
        count += 1
        if count == 3:
            raise PermissionError("revoked before next cached position")
        return copy.deepcopy(rows)

    monkeypatch.setattr(f.r, "rebind_rows", rebind)
    result = f.r.execute(f.args, {"run_id": "cached-revocation"})
    assert not result["valid"] and result["comparison"] is None
    assert result["positions"][1]["status"] == "authorization_failure"
    assert result["positions"][1]["semantic_pass"] is None
    assert result["physical"][0]["cost"]["dispatches"] == 1


def test_shadow_drift_during_comparison_keeps_private_observations_with_null_public_agreement(
    shadow_tree, monkeypatch
):
    f = shadow_tree
    freeze_shadow(f)
    original = f.r.compare_references

    def compare(*args):
        result = original(*args)
        f.identity["policy"]["digest"] = "c" * 64
        return result

    monkeypatch.setattr(f.r, "compare_references", compare)
    result = f.r.execute(f.args, {"run_id": "comparison-drift"})
    assert not result["valid"] and result["comparison"] is None
    raw = json.loads((Path(f.args.private_root) / "comparison-drift/raw.json").read_bytes())
    assert raw["comparison"] is not None and len(raw["physical"]) > 0


def test_shadow_registration_never_parses_any_grade_bearing_artifact(shadow_tree, monkeypatch):
    f = shadow_tree
    assert Path(f.args.review_attestation).is_file(), "metadata attestation missing"
    forbidden = {
        Path(f.args.review_public).resolve(),
        Path(f.args.review_root) / "reviewed.json",
        Path(f.args.review_root) / "scored-primary.json",
        Path(f.args.review_root) / "canonical-evidence.json",
        f.paths["v1_artifact"],
    }
    forbidden.update((Path(f.args.review_root) / f"answer-part-{i}-reviews.json") for i in (1, 2, 3))
    forbidden.update((f.paths["v1_out"] / f"evidence-part-{i}-reviews.json") for i in (1, 2))
    original = f.r._read

    def read(path):
        assert Path(path).resolve() not in forbidden, "grade-bearing file parsed before runtime"
        return original(path)

    monkeypatch.setattr(f.r, "_read", read)
    freeze_shadow(f)
    assert not f.calls


@pytest.mark.parametrize(
    "mutation",
    ["delete", "empty", "extra", "substitute", "anchor", "attestation-delete", "attestation-bytes"],
)
def test_shadow_complete_upstream_bindings_and_attestation_required(shadow_tree, mutation):
    f = shadow_tree
    reg = f.m.read_json(f.args.review_registration)
    if mutation in ("delete", "empty", "extra", "substitute"):
        if mutation == "delete":
            reg["frozen_files"].pop("v1_review_evidence-part-1-reviews.json")
        elif mutation == "empty":
            reg["frozen_files"] = {}
        elif mutation == "extra":
            reg["frozen_files"]["extra"] = copy.deepcopy(reg["frozen_files"]["raw"])
        else:
            reg["frozen_files"]["raw"] = copy.deepcopy(reg["frozen_files"]["suite"])
        Path(f.args.review_registration).write_text(json.dumps(reg))
        public = f.m.read_json(f.args.review_public)
        public["review_registration_sha256"] = f.m.digest(reg)
        Path(f.args.review_public).write_text(json.dumps(public))
    elif mutation == "anchor":
        (Path(f.args.review_root) / "registration-sha256.json").write_text("{}")
    elif mutation == "attestation-delete":
        Path(f.args.review_attestation).unlink()
    else:
        Path(f.args.review_attestation).write_bytes(Path(f.args.review_attestation).read_bytes() + b"\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        f.r.registration_for(f.args)
    assert not f.calls


def test_shadow_public_physical_audit_id_never_exports_nested_private_text():
    r = importlib.import_module("scripts.run_bound_citation_dev")
    row = dict(
        id="a" * 64,
        kind="dev",
        arm="shared_contract",
        task_id="t",
        task_index=0,
        pending=False,
        payload_sha256="b" * 64,
        evidence_sha256="c" * 64,
        audit_input_sha256="d" * 64,
        physical_audit_id={"private": "PRIVATE_SOURCE"},
        observation={},
    )
    projected = r._public_position(row)
    assert projected["physical_audit_id"] is None
    assert "PRIVATE_SOURCE" not in json.dumps(projected)


def test_shadow_metadata_attestation_contains_no_grades_or_reasons_and_is_exclusive(shadow_tree):
    f = shadow_tree
    att = f.m.read_json(f.args.review_attestation)
    assert att["version"] == "bound-citation-review-metadata-v1"
    assert len(att["frozen_files"]) == 235
    assert not any(
        s in json.dumps(att)
        for s in ("PRIVATE_", "fact_verdicts", "strict_task_success", "correctness", "threshold")
    )
    assert not f.calls
    with pytest.raises(FileExistsError):
        f.r.prepare_review_attestation(f.args)


def test_shadow_execute_defers_all_reference_json_until_all_118_positions(shadow_tree, monkeypatch):
    f = shadow_tree
    freeze_shadow(f)
    gradepaths = {p.resolve() for p in f.r._reference_paths(f.args).values()}
    gradepaths.add(f.paths["v1_artifact"].resolve())
    gradepaths.update((f.paths["v1_out"] / f"evidence-part-{i}-reviews.json").resolve() for i in (1, 2))
    original = f.r._read
    compared = []
    original_compare = f.r.compare_references

    def compare(positions, *a):
        assert len(positions) == 118 and all("observation" in p for p in positions)
        compared.append(True)
        return original_compare(positions, *a)

    def read(path):
        if Path(path).resolve() in gradepaths:
            assert compared, "grade parsed during registration/runtime"
        return original(path)

    monkeypatch.setattr(f.r, "_read", read)
    monkeypatch.setattr(f.r, "compare_references", compare)
    result = f.r.execute(f.args, {"run_id": "grade-blind-complete"})
    assert result["valid"] and compared


def test_shadow_attestation_preparation_rejects_reference_grade_substitution(shadow_tree):
    f = shadow_tree
    Path(f.args.review_attestation).unlink()
    reviewed = f.m.read_json(Path(f.args.review_root) / "reviewed.json")
    reviewed["reviews"][0]["claim_verdicts"] = ["unsupported"]
    (Path(f.args.review_root) / "reviewed.json").write_text(json.dumps(reviewed))
    public = f.m.read_json(f.args.review_public)
    public["reviewed_sha256"] = f.m.digest(reviewed)
    Path(f.args.review_public).write_text(json.dumps(public))
    with pytest.raises(ValueError, match="expanded_fresh_review_changed"):
        f.r.prepare_review_attestation(f.args)
    assert not Path(f.args.review_attestation).exists()


def test_shadow_nested_physical_id_tampering_never_leaks_failed_artifact(shadow_tree):
    f = shadow_tree
    reg = freeze_shadow(f)
    reg["positions"][0]["physical_audit_id"] = {"raw": "PRIVATE_SECRET_SOURCE"}
    Path(f.args.registration).write_text(json.dumps(reg))
    result = f.r.execute(f.args, {"run_id": "nested-id-preflight-failure"})
    assert not result["valid"] and not f.calls
    assert "PRIVATE_SECRET_SOURCE" not in Path(f.args.output).read_text()


@pytest.mark.parametrize(
    "field,value",
    [("version", "wrong"), ("canonical_evidence_tasks", 46), ("settings_sha256", "0" * 64)],
)
def test_shadow_attestation_rejects_incomplete_review_registration_metadata(shadow_tree, field, value):
    f = shadow_tree
    Path(f.args.review_attestation).unlink()
    reg = f.m.read_json(f.args.review_registration)
    reg[field] = value
    Path(f.args.review_registration).write_text(json.dumps(reg))
    public = f.m.read_json(f.args.review_public)
    public["review_registration_sha256"] = f.m.digest(reg)
    Path(f.args.review_public).write_text(json.dumps(public))
    anchor = {
        "registration_path": str(Path(f.args.review_registration).resolve()),
        "registration_bytes_sha256": f.m.byte_sha(f.args.review_registration),
    }
    (Path(f.args.review_root) / "registration-sha256.json").write_text(json.dumps(anchor))
    with pytest.raises(ValueError):
        f.r.prepare_review_attestation(f.args)
    assert not Path(f.args.review_attestation).exists()


def test_shadow_attestation_rejects_reference_bytes_changed_during_preparation(shadow_tree, monkeypatch):
    f = shadow_tree
    Path(f.args.review_attestation).unlink()
    original = f.r._read

    def read(path):
        value = original(path)
        if Path(path).name == "scored-primary.json":
            target = Path(f.args.review_root) / "reviewed.json"
            target.write_bytes(target.read_bytes() + b"\n")
        return value

    monkeypatch.setattr(f.r, "_read", read)
    with pytest.raises(ValueError):
        f.r.prepare_review_attestation(f.args)
    assert not Path(f.args.review_attestation).exists()


def test_shadow_frozen_path_and_anchor_accept_only_same_file_symlink_alias(shadow_tree):
    f = shadow_tree
    alias = f.root / "legacy-project"
    alias.symlink_to(f.root, target_is_directory=True)
    reg = f.m.read_json(f.args.review_registration)
    original = reg["frozen_files"]["raw"]["path"]
    reg["frozen_files"]["raw"]["path"] = str(alias / Path(original).relative_to(f.root))
    anchor_path = Path(f.args.review_root) / "registration-sha256.json"
    anchor = f.m.read_json(anchor_path)
    anchor["registration_path"] = str(alias / Path(f.args.review_registration).relative_to(f.root))
    anchor_path.write_text(json.dumps(anchor))
    _, preserved = f.r._validate_review_bindings(
        f.args, f.r._review_helper(), f.raw["registration"], reg
    )
    assert preserved == anchor
    assert reg["frozen_files"]["raw"]["path"] != original


@pytest.mark.parametrize(
    "mutation", ["same_hash_different_path", "different_content", "missing_file", "extra_record_field"]
)
def test_shadow_frozen_alias_compatibility_does_not_relax_paths_or_hashes(shadow_tree, mutation):
    f = shadow_tree
    reg = f.m.read_json(f.args.review_registration)
    record = reg["frozen_files"]["raw"]
    if mutation == "same_hash_different_path":
        duplicate = f.root / "duplicate-raw.json"
        duplicate.write_bytes(Path(record["path"]).read_bytes())
        record["path"] = str(duplicate)
    elif mutation == "different_content":
        record["bytes_sha256"] = "0" * 64
    elif mutation == "missing_file":
        record["path"] = str(f.root / "missing.json")
    else:
        record["private"] = "PRIVATE_NESTED"
    with pytest.raises((ValueError, FileNotFoundError)):
        f.r._validate_review_bindings(f.args, f.r._review_helper(), f.raw["registration"], reg)


def test_shadow_anchor_same_hash_at_wrong_path_is_rejected(shadow_tree):
    f = shadow_tree
    path = Path(f.args.review_root) / "registration-sha256.json"
    anchor = f.m.read_json(path)
    duplicate = f.root / "duplicate-registration.json"
    duplicate.write_bytes(Path(f.args.review_registration).read_bytes())
    anchor["registration_path"] = str(duplicate)
    path.write_text(json.dumps(anchor))
    with pytest.raises(ValueError):
        f.r._validate_review_bindings(
            f.args,
            f.r._review_helper(),
            f.raw["registration"],
            f.m.read_json(f.args.review_registration),
        )


@pytest.mark.parametrize(
    "value", ["PRIVATE_COMPLETENESS", 1, 0, None, {"private": "PRIVATE_COMPLETENESS"}, [True]]
)
def test_shadow_authored_expected_requires_exact_boolean_types_before_registration(shadow_tree, value):
    f = shadow_tree
    fixture = f.m.read_json(f.args.fixture)
    fixture["pairs"][0]["faithful"]["expected"]["assertions_supported"] = copy.deepcopy(value)
    path = f.root / "malformed-authored.json"
    path.write_text(json.dumps(fixture))
    f.args.fixture = str(path)
    with pytest.raises(ValueError, match="authored"):
        f.r.registration_for(f.args)
    assert not f.calls and not Path(f.args.registration).exists()


@pytest.mark.parametrize(
    "mutation",
    ["extra_expected_field", "missing_expected_field", "unknown_stage", "extra_case", "duplicate_pair"],
)
def test_shadow_authored_expected_schema_is_complete_and_exact(shadow_tree, mutation):
    f = shadow_tree
    fixture = f.m.read_json(f.args.fixture)
    case = fixture["pairs"][0]["faithful"]
    if mutation == "extra_expected_field":
        case["expected"]["private"] = {"reason": "PRIVATE_REASON"}
    elif mutation == "missing_expected_field":
        del case["expected"]["question_complete"]
    elif mutation == "unknown_stage":
        case["expected"]["stage"] = "PRIVATE_STAGE"
    elif mutation == "extra_case":
        fixture["pairs"][0]["extra_case"] = {"expected": "PRIVATE_REASON"}
    else:
        fixture["pairs"][1] = copy.deepcopy(fixture["pairs"][0])
    path = f.root / "malformed-schema.json"
    path.write_text(json.dumps(fixture))
    f.args.fixture = str(path)
    with pytest.raises(ValueError, match="authored"):
        f.r.registration_for(f.args)
    assert not f.calls


def test_shadow_malformed_authored_fixture_failed_artifact_remains_private(shadow_tree):
    f = shadow_tree
    freeze_shadow(f)
    fixture = f.m.read_json(f.args.fixture)
    fixture["pairs"][0]["faithful"]["expected"].update(
        assertions_supported="PRIVATE_SUPPORT", question_complete={"reason": "PRIVATE_COMPLETENESS"}
    )
    path = f.root / "malformed-private-fixture.json"
    path.write_text(json.dumps(fixture))
    f.args.fixture = str(path)
    result = f.r.execute(f.args, {"run_id": "malformed-authored-private"})
    assert not result["valid"] and not f.calls and result["comparison"] is None
    assert "PRIVATE_" not in Path(f.args.output).read_text()


def test_shadow_comparison_expected_pass_has_final_boolean_null_whitelist(shadow_tree):
    f = shadow_tree
    inputs = f.r._load_inputs(f.args)
    positions = f.r._positions(inputs)
    positions[-1]["expected"].update(
        assertions_supported="PRIVATE_SUPPORT", question_complete="PRIVATE_COMPLETENESS"
    )
    result = f.r.compare_references(positions, inputs, f.args)
    assert result["rows"][-1]["expected_pass"] is None
    assert all(type(row.get("expected_pass")) in (bool, type(None)) for row in result["rows"])
    assert "PRIVATE_" not in json.dumps(result)


@pytest.mark.parametrize(
    "exception,status",
    [
        ("cancelled", "interrupted"),
        ("budget", "budget_exhausted"),
        ("keyboard", "interrupted"),
        ("permission", "authorization_failure"),
    ],
)
def test_shadow_cached_authorization_stop_never_publishes_reused_pass(
    shadow_tree, monkeypatch, exception, status
):
    from app.execution_budget import TaskCancelled, BudgetExceeded

    f = shadow_tree
    freeze_shadow(f)
    calls = 0
    errors = {
        "cancelled": TaskCancelled("PRIVATE_CANCEL"),
        "budget": BudgetExceeded("PRIVATE_BUDGET"),
        "keyboard": KeyboardInterrupt("PRIVATE_INTERRUPT"),
        "permission": PermissionError("PRIVATE_PERMISSION"),
    }

    def rebind(db, user, rows, corpus):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise errors[exception]
        return copy.deepcopy(rows)

    monkeypatch.setattr(f.r, "rebind_rows", rebind)
    result = f.r.execute(f.args, {"run_id": "cached-stop-" + exception})
    assert not result["valid"] and result["comparison"] is None
    assert (
        result["positions"][0]["status"] == "observed"
        and result["positions"][0]["semantic_pass"] is True
    )
    second = result["positions"][1]
    assert second["status"] == status and second["shared_physical_cost"] is True
    assert all(second[field] is None for field in f.r._BOOLEAN_FIELDS)
    assert len(f.calls) == 1 and len(result["physical"]) == 1
    assert result["physical"][0]["cost"]["dispatches"] == 1
    assert all(
        row["status"] == "not_observed" and row["semantic_pass"] is None
        for row in result["positions"][2:]
    )
    raw = f.m.read_json(Path(f.args.private_root) / ("cached-stop-" + exception) / "raw.json")
    assert raw["positions"][1]["payload"] == f.raw["results"]["literal_fallback"][0]["answer_payload"]
    assert raw["positions"][1]["observation"]["cost"] == raw["physical"][0]["observation"]["cost"]
    assert "PRIVATE_" not in Path(f.args.output).read_text()


def test_shadow_synthetic_205_fixture_does_not_hide_real_new_source_files(shadow_tree, monkeypatch):
    from scripts.run_retrieval_quality_dev import code_fingerprints as real_fingerprints

    f = shadow_tree
    actual = real_fingerprints()
    added = {"scripts/run_policy_ablation_dev.py", "scripts/run_policy_ablation_replay_dev.py"}
    assert added <= set(actual)
    assert len(f.raw["registration"]["code"]) == 205
    monkeypatch.setattr(f.r, "code_fingerprints", lambda: actual)
    registration = f.m.read_json(f.paths["registration"])
    with pytest.raises(ValueError, match="unregistered_new_or_missing_source"):
        f.r._upstream_sources(f.raw["registration"], registration)


@pytest.mark.parametrize("mutation", ["rogue", "mutated"])
def test_shadow_synthetic_fixture_still_rejects_unregistered_or_changed_code(
    shadow_tree, monkeypatch, mutation
):
    f = shadow_tree
    actual = copy.deepcopy(f.r.code_fingerprints())
    if mutation == "rogue":
        actual["scripts/rogue_synthetic_source.py"] = "a" * 64
    else:
        actual[next(iter(f.raw["registration"]["code"]))] = "0" * 64
    monkeypatch.setattr(f.r, "code_fingerprints", lambda: actual)
    with pytest.raises(ValueError):
        f.r._upstream_sources(f.raw["registration"], f.m.read_json(f.paths["registration"]))
