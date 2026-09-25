"""Failure signals must remain review prompts, not fabricated root-cause labels."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "attribute_multihop_failures", ROOT / "scripts/attribute_multihop_failures.py"
)
assert SPEC and SPEC.loader
ATTRIBUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ATTRIBUTION)


def test_document_presence_does_not_claim_that_the_answer_fact_was_found():
    record = {"gold_documents": 2, "gold_documents_retrieved": 2, "verdict": "yes"}
    signals = ATTRIBUTION.signals_for(
        record,
        ["unsupported"],
        {"completeness": "partial", "verdict_reading": "no"},
    )
    assert "gold_document_present_fact_unverified" in signals
    assert "reviewed_claim_not_fully_supported" in signals
    assert "reviewed_answer_incomplete" in signals
    assert "system_verdict_disagrees_with_reviewed_reading" in signals
    assert "gold_document_missing" not in signals


def test_missing_document_is_distinct_from_semantic_unknown():
    record = {"gold_documents": 2, "gold_documents_retrieved": 1}
    assert ATTRIBUTION.signals_for(record, [], None) == [
        "gold_document_missing",
        "no_reviewed_semantic_label",
    ]


def test_frozen_gold_is_checked_before_building_a_report(tmp_path):
    claims = tmp_path / "gold-v3-claims.csv"
    answers = tmp_path / "gold-v3-answers.csv"
    claims.write_text("claim\n")
    answers.write_text("answer\n")
    manifest = tmp_path / "gold-v3-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sha256": {
                    claims.name: ATTRIBUTION.sha256(claims),
                    answers.name: ATTRIBUTION.sha256(answers),
                }
            }
        )
    )
    ATTRIBUTION.verify_gold(manifest, claims, answers)
    answers.write_text("changed\n")
    with pytest.raises(ValueError, match="frozen semantic labels changed"):
        ATTRIBUTION.verify_gold(manifest, claims, answers)
