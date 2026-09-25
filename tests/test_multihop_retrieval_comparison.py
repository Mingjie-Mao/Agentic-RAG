"""A retrieval ablation must compare the same questions and denominators."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_multihop_retrieval", ROOT / "scripts/compare_multihop_retrieval.py"
)
assert SPEC and SPEC.loader
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def row(key, facts, *, batch="b1"):
    return {
        "batch": batch,
        "id": key,
        "question_type": "comparison_query",
        "gold_documents": 2,
        "gold_documents_found": facts,
        "gold_facts": 2,
        "gold_facts_delivered": facts,
        "retrieval_ms": 100,
    }


def test_paired_comparison_rejects_question_or_gold_mismatch():
    with pytest.raises(ValueError, match="different question IDs"):
        COMPARE.align([row("a", 1)], [row("b", 1)])
    with pytest.raises(ValueError, match="different gold data"):
        COMPARE.align([row("a", 1)], [{**row("a", 1), "gold_facts": 3}])
    with pytest.raises(ValueError, match="duplicate question"):
        COMPARE.align([row("a", 1), row("a", 1)], [row("a", 1)])


def test_paired_delta_is_calculated_from_fact_counts():
    pairs = COMPARE.align([row("a", 0), row("b", 1)], [row("a", 2), row("b", 1)])
    result = COMPARE.summarize(pairs, samples=100, seed=0)
    assert result["gold_fact_recall"]["before"] == 0.25
    assert result["gold_fact_recall"]["after"] == 0.75
    assert result["gold_fact_recall"]["delta"] == 0.5
    assert result["gold_fact_recall"]["delta_ci95"][0] >= 0


def test_comparison_rejects_different_index_state(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    base = {"label": "index-v1", "batches": ["b1"], "metric": "fact",
            "document_quota": 2, "clause_queries": "off", "results": [row("a", 1)]}
    left.write_text(json.dumps(base))
    right.write_text(json.dumps({**base, "label": "index-v2"}))
    with pytest.raises(ValueError, match="differ beyond routing: label"):
        COMPARE.compare(left, right, samples=10)
