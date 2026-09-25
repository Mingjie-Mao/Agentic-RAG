import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "summarize_multihop_r3", Path(__file__).resolve().parents[1] / "scripts/summarize_multihop_r3.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_comparison_uses_identical_question_denominator():
    rag = [{"id": "a", "answer_correct": True}, {"id": "b", "answer_correct": False}]
    workflow = [{"id": "b", "answer_correct": True}, {"id": "a", "answer_correct": False}]
    result = MODULE.paired(rag, workflow, samples=100)
    assert result["questions"] == 2
    assert result["workflow_only_correct"] == 1
    assert result["rag_only_correct"] == 1
    assert result["accuracy_delta_workflow_minus_rag"] == 0


def test_yes_no_majority_baseline_and_null_false_answers():
    rows = [
        {"id": "a", "question_type": "comparison_query", "answer_correct": True,
         "status": "answered", "latency_ms": 1, "prompt_tokens": 10,
         "gold_documents": 1, "gold_documents_retrieved": 1, "all_gold_retrieved": True},
        {"id": "b", "question_type": "comparison_query", "answer_correct": False,
         "status": "insufficient_evidence", "latency_ms": 2, "prompt_tokens": 20,
         "gold_documents": 1, "gold_documents_retrieved": 0, "all_gold_retrieved": False},
        {"id": "c", "question_type": "null_query", "answer_correct": False,
         "status": "answered", "latency_ms": 3, "prompt_tokens": 30},
    ]
    result = MODULE.summarize(rows, {"a": "Yes", "b": "Yes", "c": "Insufficient information."})
    assert result["yes_no_majority_baseline"] == 1
    assert result["yes_no_accuracy"] == .5
    assert result["literal_yes_no_accuracy"] == .5
    assert result["literal_yes_no_majority_baseline"] == 1
    assert result["null_false_answer_rate"] == 1
    assert result["gold_document_recall_in_answer_context"] == .5


def test_true_false_gap_is_diagnostic_and_does_not_rewrite_frozen_score():
    rows = [
        {"id": "a", "question_type": "comparison_query", "answer_correct": False,
         "status": "answered", "verdict": "yes", "latency_ms": 1, "prompt_tokens": 10},
        {"id": "b", "question_type": "comparison_query", "answer_correct": False,
         "status": "answered", "verdict": "yes", "latency_ms": 2, "prompt_tokens": 10},
    ]
    result = MODULE.summarize(rows, {"a": "True", "b": "False"})
    assert result["answer_correct"] == 0
    assert result["true_false_gold_questions"] == 2
    assert result["true_false_verdict_correct_posthoc"] == 1
    assert result["true_false_correct_hidden_by_v2"] == 1
