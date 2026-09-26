import json

import pytest

from app.clients import Models
from app.config import Settings
from app.qa import answer_verdict
from app.verdict import StructuredVerdict, constrained_schema, question_slots, resolve_verdict

QUESTION = "Did Alpha approve the change and did Beta reject the change?"
CLAIMS = [
    {"text": "Alpha approved the change.", "evidence_ids": ["c1"]},
    {"text": "Beta rejected the change.", "evidence_ids": ["c2"]},
]


def decision(truths=("supported", "supported"), **updates):
    value = {
        "operator": "all", "question_complete": True, "comparison_dimension": "",
        "propositions": [
            {"question_span": span, "truth": truth, "claim_indices": [index] if truth != "unknown" else []}
            for index, (span, truth) in enumerate(zip(
                ["Did Alpha approve the change", "did Beta reject the change"], truths, strict=True
            ), 1)
        ],
    }
    value.update(updates)
    return StructuredVerdict.model_validate(value)


@pytest.mark.parametrize("truths,operator,expected", [
    (("supported", "supported"), "all", "yes"),
    (("supported", "contradicted"), "all", "no"),
    (("supported", "unknown"), "all", "unclear"),
    (("contradicted", "unknown"), "all", "no"),
    (("supported", "unknown"), "any", "yes"),
    (("contradicted", "contradicted"), "any", "no"),
    (("contradicted", "unknown"), "any", "unclear"),
])
def test_logic_and_multiclaim_citations(truths, operator, expected):
    result = resolve_verdict(decision(truths, operator=operator), QUESTION, CLAIMS)
    assert result["value"] == expected
    if truths == ("supported", "supported"):
        assert result["claim_indices"] == [1, 2]
        assert result["evidence_ids"] == ["c1", "c2"]
    if expected == "unclear":
        assert result["evidence_ids"] == []


@pytest.mark.parametrize("mutation,issue", [
    (lambda d: setattr(d, "question_complete", False), "incomplete_question"),
    (lambda d: setattr(d.propositions[0], "claim_indices", [8]), "invalid_claim_reference"),
    (lambda d: setattr(d.propositions[0], "claim_indices", []), "missing_claim_reference"),
    (lambda d: setattr(d.propositions[0], "question_span", "invented"), "question_span_not_found"),
    (lambda d: setattr(d.propositions[1], "question_span", d.propositions[0].question_span), "duplicate_proposition"),
    (lambda d: setattr(d, "operator", "atomic"), "invalid_operator_arity"),
])
def test_invalid_decisions_cannot_become_confident(mutation, issue):
    d = decision()
    mutation(d)
    result = resolve_verdict(d, QUESTION, CLAIMS)
    assert result["value"] == "unclear"
    assert issue in result["validation_issues"]
    assert result["evidence_ids"] == []


def test_comparison_dimension_is_tied_to_question():
    d = decision(operator="comparison", propositions=[{
        "question_span": "Are opening dates consistent", "truth": "supported", "claim_indices": [1, 2]
    }], comparison_dimension="opening hours")
    question = "Are opening dates consistent across both reports?"
    assert resolve_verdict(d, question, CLAIMS)["value"] == "unclear"
    d.comparison_dimension = "opening dates"
    assert resolve_verdict(d, question, CLAIMS)["value"] == "yes"


def test_structured_protocol_uses_one_bounded_call_and_qa_exports_all_refs(monkeypatch):
    monkeypatch.setattr("app.clients.settings", lambda: Settings(verdict_protocol="structured"))
    calls = []
    models = Models(chat_backend=lambda body: calls.append(body) or {
        "message": {"content": decision().model_dump_json()}, "prompt_eval_count": 20,
    })
    result, usage = answer_verdict(models, QUESTION, CLAIMS, "answered")
    assert result["value"] == "yes" and result["evidence_ids"] == ["c1", "c2"]
    assert usage["prompt_tokens"] == 20
    assert len(calls) == 1 and calls[0]["options"]["num_predict"] == 500
    inputs = json.loads(calls[0]["messages"][1]["content"])
    assert inputs["claims"] == [{"index": 1, "text": CLAIMS[0]["text"]}, {"index": 2, "text": CLAIMS[1]["text"]}]


def test_constrained_slots_are_given_by_schema_and_all_must_be_covered():
    slots = question_slots(QUESTION)
    assert slots == ["Did Alpha approve the change", "did Beta reject the change"]
    schema = constrained_schema(QUESTION)
    assert schema["$defs"]["PropositionVerdict"]["properties"]["question_span"]["enum"] == slots
    assert schema["properties"]["propositions"]["maxItems"] == 2
    d = decision()
    assert resolve_verdict(d, QUESTION, CLAIMS, constrained=True)["value"] == "yes"
    d.propositions[0].question_span = "Alpha approve the change"
    result = resolve_verdict(d, QUESTION, CLAIMS, constrained=True)
    assert result["value"] == "unclear" and "question_slots_not_covered" in result["validation_issues"]
    single = constrained_schema("Are opening dates consistent?")
    assert single["properties"]["propositions"]["maxItems"] == 1


def test_comparison_cannot_invent_an_extra_property_in_schema():
    question = "Are the launch dates consistent in the two reports?"
    schema = constrained_schema(question)
    assert schema["$defs"]["PropositionVerdict"]["properties"]["question_span"]["enum"] == [question.rstrip("?")]
    assert schema["properties"]["propositions"]["maxItems"] == 1


def test_external_oracle_manifest_is_fresh_balanced_and_pinned():
    from pathlib import Path
    import hashlib

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "fixtures/verdict/external-oracle-r4-8.json").read_text())
    batch = root / "fixtures/multihop/subset-r4.json"
    assert manifest["batch_sha256"] == hashlib.sha256(batch.read_bytes()).hexdigest()
    selected = {row["query_sha256"] for row in manifest["items"]}
    assert len(selected) == 8
    assert {row["question_type"] for row in manifest["items"]} == {"comparison_query", "temporal_query"}
    for name in ["subset.json", "subset-r2.json", "subset-r3.json"]:
        used = {row["query_sha256"] for row in json.loads((root / "fixtures/multihop" / name).read_text())["items"]}
        assert selected.isdisjoint(used)


@pytest.mark.parametrize("records", [
    [{"id": "one", "protocol": "legacy", "expected": "yes"}],
    [{"id": "one", "protocol": "legacy", "expected": "yes"}, {"id": "one", "protocol": "legacy", "expected": "yes"}],
    [{"id": "one", "protocol": "legacy", "expected": "yes"}, {"id": "one", "protocol": "structured", "expected": "no"}],
])
def test_experiment_summary_rejects_partial_duplicate_or_mismatched_pairs(records, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from summarize_verdict_experiments import summarize

    with pytest.raises(ValueError):
        summarize({"records": records})
