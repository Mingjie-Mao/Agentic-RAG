import json

import pytest

from app.clients import Models, DependencyError
from app.config import Settings
from app.qa import answer_verdict, merge_verdict_usage
from app.verdict import (
    StructuredVerdict, constrained_schema, question_slots, resolve_verdict,
    IndexedStructuredVerdict, indexed_schema, expand_indexed_verdict,
    SourceBoundIndexedVerdict, source_bound_schema, expand_source_bound_verdict,
)

QUESTION = "Did Alpha approve the change and did Beta reject the change?"
CLAIMS = [
    {"text": "Alpha approved the change.", "evidence_ids": ["c1"]},
    {"text": "Beta rejected the change.", "evidence_ids": ["c2"]},
]


def decision(truths=("supported", "supported"), **updates):
    value = {
        "operator": "all",
        "question_complete": True,
        "comparison_dimension": "",
        "propositions": [
            {
                "question_span": span,
                "truth": truth,
                "claim_indices": [index] if truth != "unknown" else [],
            }
            for index, (span, truth) in enumerate(
                zip(["Did Alpha approve the change", "did Beta reject the change"], truths, strict=True),
                1,
            )
        ],
    }
    value.update(updates)
    return StructuredVerdict.model_validate(value)


@pytest.mark.parametrize(
    "truths,operator,expected",
    [
        (("supported", "supported"), "all", "yes"),
        (("supported", "contradicted"), "all", "no"),
        (("supported", "unknown"), "all", "unclear"),
        (("contradicted", "unknown"), "all", "no"),
        (("supported", "unknown"), "any", "yes"),
        (("contradicted", "contradicted"), "any", "no"),
        (("contradicted", "unknown"), "any", "unclear"),
    ],
)
def test_logic_and_multiclaim_citations(truths, operator, expected):
    result = resolve_verdict(decision(truths, operator=operator), QUESTION, CLAIMS)
    assert result["value"] == expected
    if truths == ("supported", "supported"):
        assert result["claim_indices"] == [1, 2]
        assert result["evidence_ids"] == ["c1", "c2"]
    if expected == "unclear":
        assert result["evidence_ids"] == []


@pytest.mark.parametrize(
    "mutation,issue",
    [
        (lambda d: setattr(d, "question_complete", False), "incomplete_question"),
        (lambda d: setattr(d.propositions[0], "claim_indices", [8]), "invalid_claim_reference"),
        (lambda d: setattr(d.propositions[0], "claim_indices", []), "missing_claim_reference"),
        (lambda d: setattr(d.propositions[0], "question_span", "invented"), "question_span_not_found"),
        (
            lambda d: setattr(d.propositions[1], "question_span", d.propositions[0].question_span),
            "duplicate_proposition",
        ),
        (lambda d: setattr(d, "operator", "atomic"), "invalid_operator_arity"),
    ],
)
def test_invalid_decisions_cannot_become_confident(mutation, issue):
    d = decision()
    mutation(d)
    result = resolve_verdict(d, QUESTION, CLAIMS)
    assert result["value"] == "unclear"
    assert issue in result["validation_issues"]
    assert result["evidence_ids"] == []


def test_comparison_dimension_is_tied_to_question():
    d = decision(
        operator="comparison",
        propositions=[
            {
                "question_span": "Are opening dates consistent",
                "truth": "supported",
                "claim_indices": [1, 2],
            }
        ],
        comparison_dimension="opening hours",
    )
    question = "Are opening dates consistent across both reports?"
    assert resolve_verdict(d, question, CLAIMS)["value"] == "unclear"
    d.comparison_dimension = "opening dates"
    assert resolve_verdict(d, question, CLAIMS)["value"] == "yes"


def test_structured_protocol_uses_one_bounded_call_and_qa_exports_all_refs(monkeypatch):
    monkeypatch.setattr("app.clients.settings", lambda: Settings(verdict_protocol="structured"))
    calls = []
    models = Models(
        chat_backend=lambda body: calls.append(body)
        or {
            "message": {"content": json.dumps({
                "operator": "all", "question_complete": True, "comparison_dimension": "",
                "propositions": [
                    {"slot_index": i, "truth": "supported", "claim_indices": [i]}
                    for i in (1, 2)
                ],
            })},
            "prompt_eval_count": 20,
        }
    )
    result, usage = answer_verdict(models, QUESTION, CLAIMS, "answered")
    assert result["value"] == "yes" and result["evidence_ids"] == ["c1", "c2"]
    assert usage["prompt_tokens"] == 20
    assert len(calls) == 1 and calls[0]["options"]["num_predict"] == 500
    inputs = json.loads(calls[0]["messages"][1]["content"])
    assert inputs["claims"] == [
        {"index": 1, "text": CLAIMS[0]["text"]},
        {"index": 2, "text": CLAIMS[1]["text"]},
    ]


def test_quoted_question_never_enters_transport_schema_and_exact_spans_survive():
    question = 'Does Alpha call the product "new", while Beta calls it "accessible"?'
    schema = indexed_schema(question, 2)
    assert "new" not in json.dumps(schema) and "accessible" not in json.dumps(schema)
    wire = IndexedStructuredVerdict.model_validate({
        "operator": "all", "question_complete": True, "comparison_dimension": "",
        "propositions": [
            {"slot_index": 2, "truth": "supported", "claim_indices": [2]},
            {"slot_index": 1, "truth": "supported", "claim_indices": [1]},
        ],
    })
    expanded = expand_indexed_verdict(wire, question)
    assert expanded.propositions[1].question_span == question_slots(question)[0]
    assert resolve_verdict(expanded, question, CLAIMS, constrained=True)["value"] == "yes"
    wire.propositions[1].slot_index = 2
    with pytest.raises(ValueError, match="question_slots_not_covered"):
        expand_indexed_verdict(wire, question)


def test_verdict_failure_is_diagnostic_and_unknown_cost_is_not_zero(monkeypatch):
    class Unavailable:
        def decide_verdict(self, *args):
            raise DependencyError("unavailable", stage="verdict_input")
    value, usage = answer_verdict(Unavailable(), QUESTION, CLAIMS, "answered")
    assert value is None
    merged = merge_verdict_usage({}, usage)
    assert merged["verdict_status"] == "unavailable"
    assert merged["verdict_failure_stage"] == "verdict_input"
    assert "verdict_prompt_tokens" not in merged


def test_verdict_input_is_never_sliced_into_incomplete_json(monkeypatch):
    monkeypatch.setattr("app.clients.settings", lambda: Settings(verdict_protocol="structured"))
    def forbidden(body):
        raise AssertionError("Oversized input must fail before the model call")
    with pytest.raises(DependencyError, match="上下文限制"):
        Models(chat_backend=forbidden).decide_verdict(QUESTION, [{"text": "a" * 12001}])


def test_invalid_wire_keeps_actual_consumed_usage(monkeypatch):
    monkeypatch.setattr("app.clients.settings", lambda: Settings(verdict_protocol="structured"))
    models = Models(chat_backend=lambda body: {
        "message": {"content": '{"operator":"all"}'},
        "prompt_eval_count": 70, "eval_count": 4,
    })
    value, usage = answer_verdict(models, QUESTION, CLAIMS, "answered")
    assert value is None
    merged = merge_verdict_usage({}, usage)
    assert merged["verdict_prompt_tokens"] == 70
    assert merged["verdict_completion_tokens"] == 4
    assert merged["verdict_failure_stage"] == "verdict"


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
    assert schema["$defs"]["PropositionVerdict"]["properties"]["question_span"]["enum"] == [
        question.rstrip("?")
    ]
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
        used = {
            row["query_sha256"]
            for row in json.loads((root / "fixtures/multihop" / name).read_text())["items"]
        }
        assert selected.isdisjoint(used)


@pytest.mark.parametrize(
    "records",
    [
        [{"id": "one", "protocol": "legacy", "expected": "yes"}],
        [
            {"id": "one", "protocol": "legacy", "expected": "yes"},
            {"id": "one", "protocol": "legacy", "expected": "yes"},
        ],
        [
            {"id": "one", "protocol": "legacy", "expected": "yes"},
            {"id": "one", "protocol": "structured", "expected": "no"},
        ],
    ],
)
def test_experiment_summary_rejects_partial_duplicate_or_mismatched_pairs(records, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from summarize_verdict_experiments import summarize

    with pytest.raises(ValueError):
        summarize({"records": records})


def test_while_source_clauses_keep_negation_and_are_not_partial_judgments():
    q = "Does Report A suggest a lack of interest, while Report B implies an expected reward?"
    assert question_slots(q) == [
        "Does Report A suggest a lack of interest",
        "Report B implies an expected reward",
    ]
    assert all(part in q for part in question_slots(q))


def test_judgment_reads_only_server_copied_labels_for_cited_chunks(monkeypatch):
    monkeypatch.setattr('app.qa.settings', lambda: Settings(verdict_protocol='structured', answer_quality_enabled=True))
    class Judge:
        def decide_verdict(self, question, claims):
            self.seen = claims
            return decision(), {}
    judge = Judge()
    evidence = [dict(id='E1', chunk_id='c1', document_id='a', title='Change notice', metadata={'source':'Paper A'}, text='Full unquoted text'),
                dict(id='E2', chunk_id='c2', document_id='b', title='Reply notice', metadata={'source':'Paper B'}, text='Full unquoted text'),
                dict(id='E3', chunk_id='other', document_id='c', title='Unrelated', metadata={'source':'Paper C'}, text='Unrelated text')]
    value, _ = answer_verdict(judge, QUESTION, CLAIMS, 'answered', evidence=evidence)
    assert value['value'] == 'yes'
    assert judge.seen[0]['sources'] == [{'publication':'Paper A','title':'Change notice'}]
    assert judge.seen[1]['sources'] == [{'publication':'Paper B','title':'Reply notice'}]
    assert 'Full unquoted text' not in json.dumps(judge.seen)
    assert 'Paper C' not in json.dumps(judge.seen)
    assert all('sources' not in c for c in CLAIMS)


SOURCE_QUESTION = "Is control over privacy consistent across Paper Alpha and Paper Beta?"
PROOF_GROUPS = [
    {"key": "g1", "source": "Paper Alpha", "claim_indices": [1]},
    {"key": "g2", "source": "Paper Beta", "claim_indices": [2]},
]


def source_decision(*, truth="supported", proofs=None):
    return SourceBoundIndexedVerdict.model_validate({
        "operator": "comparison", "question_complete": True,
        "comparison_dimension": "target",
        "propositions": [{"slot_index": 1, "truth": truth,
                          "source_proofs": proofs or {"g1": [1], "g2": [2]}}],
    })


def test_source_bound_schema_and_expansion_require_each_selected_source():
    schema = source_bound_schema(PROOF_GROUPS)
    proofs = schema["$defs"]["DecisiveSourceBoundPropositionVerdict"]["properties"]["source_proofs"]
    assert proofs["required"] == ["g1", "g2"]
    assert proofs["properties"]["g2"]["items"]["enum"] == [2]
    assert proofs["properties"]["g2"]["minItems"] == 1
    assert schema["$defs"]["UnknownSourceBoundPropositionVerdict"]["properties"]["source_proofs"]["properties"]["g2"]["minItems"] == 0
    assert "Paper Alpha" not in json.dumps(schema)
    expanded = expand_source_bound_verdict(source_decision(), SOURCE_QUESTION, PROOF_GROUPS)
    result = resolve_verdict(expanded, SOURCE_QUESTION, CLAIMS, constrained=True)
    assert result["value"] == "yes" and result["claim_indices"] == [1, 2]


@pytest.mark.parametrize("truth,value", [("contradicted", "no"), ("unknown", "unclear")])
def test_source_constraints_never_force_positive_truth(truth, value):
    decision = source_decision(truth=truth)
    result = resolve_verdict(
        expand_source_bound_verdict(decision, SOURCE_QUESTION, PROOF_GROUPS),
        SOURCE_QUESTION, CLAIMS, constrained=True,
    )
    assert result["value"] == value


@pytest.mark.parametrize("proofs,issue", [
    ({"g1": [1]}, "source_proof_groups_not_covered"),
    ({"g1": [1], "g2": [1]}, "invalid_source_proof_reference"),
    ({"g1": [1], "g2": [2], "g3": [2]}, "source_proof_groups_not_covered"),
])
def test_source_binding_rechecks_model_output_beyond_transport(proofs, issue):
    with pytest.raises(ValueError, match=issue):
        expand_source_bound_verdict(source_decision(proofs=proofs), SOURCE_QUESTION, PROOF_GROUPS)


def test_empty_group_has_unknown_path_and_does_not_rescue_one_sided_yes():
    groups = [PROOF_GROUPS[0], {**PROOF_GROUPS[1], "claim_indices": []}]
    schema = source_bound_schema(groups)
    missing = schema["$defs"]["UnknownSourceBoundPropositionVerdict"]["properties"]["source_proofs"]["properties"]["g2"]
    assert missing["maxItems"] == 0 and "enum" not in missing["items"]
    assert "DecisiveSourceBoundPropositionVerdict" not in schema["$defs"]
    for truth in ("supported", "unknown"):
        d = source_decision(truth=truth, proofs={"g1": [1], "g2": []})
        expanded = expand_source_bound_verdict(d, SOURCE_QUESTION, groups)
        assert resolve_verdict(expanded, SOURCE_QUESTION, CLAIMS)["value"] == "unclear"


def test_single_claim_can_bind_both_sources_without_including_unselected_claims():
    groups = [{**g, "claim_indices": [1, 2]} for g in PROOF_GROUPS]
    d = source_decision(proofs={"g1": [1], "g2": [1]})
    expanded = expand_source_bound_verdict(d, SOURCE_QUESTION, groups)
    assert expanded.propositions[0].claim_indices == [1]


def test_qa_builds_bound_choices_from_cited_sources_and_attributed_alternatives(monkeypatch):
    from app.answer_contract import verdict_proof_groups
    cfg = Settings(verdict_protocol="structured", answer_quality_enabled=True)
    monkeypatch.setattr("app.qa.settings", lambda: cfg)
    monkeypatch.setattr("app.clients.settings", lambda: cfg)
    evidence = [
        dict(id="E1", chunk_id="c1", document_id="a", title="Alpha interview",
             metadata={"source": "Paper Alpha"}, text="Alpha privacy statement"),
        dict(id="E2", chunk_id="c2", document_id="b", title="Beta interview",
             metadata={"source": "Paper Beta"}, text="Beta privacy statement"),
        dict(id="E3", chunk_id="c3", document_id="c", title="Third party",
             metadata={"source": "Paper Gamma"}, text="Paper Beta reports control over privacy."),
    ]
    claims = [{**CLAIMS[0], "quotes": ["Alpha privacy statement"]},
              {**CLAIMS[1], "quotes": ["Beta privacy statement"]},
              {"text": "Paper Beta reports control over privacy.", "evidence_ids": ["c3"],
               "quotes": ["Paper Beta reports control over privacy."]}]
    groups = verdict_proof_groups(SOURCE_QUESTION, evidence, claims)
    assert groups[0]["claim_indices"] == [1] and groups[1]["claim_indices"] == [2, 3]
    assert verdict_proof_groups("Did Paper Alpha agree and did Paper Beta reject?", evidence, claims) == []
    assert verdict_proof_groups("Do both Paper Alpha and Paper Beta support privacy?", evidence, claims) == []
    calls = []
    model = Models(chat_backend=lambda body: calls.append(body) or {
        "message": {"content": source_decision(proofs={"g1": [1], "g2": [3]}).model_dump_json()},
    })
    result, _ = answer_verdict(model, SOURCE_QUESTION, claims, "answered", evidence=evidence)
    assert result["value"] == "yes" and result["claim_indices"] == [1, 3]
    inputs = json.loads(calls[0]["messages"][1]["content"])
    assert inputs["source_proof_groups"] == groups
    assert len(calls) == 1
