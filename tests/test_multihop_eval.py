"""External-evaluation scoring and the sampling that feeds human annotation."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Scripts import their shared helpers the way they do when run directly, where the
# script's own directory is on the path.
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = load_script("run_multihop_eval")
SAMPLE = load_script("build_semantic_annotation_set")


def test_null_question_is_answered_correctly_only_by_refusing():
    assert EVAL.answer_matches("null_query", "Insufficient information.", "insufficient_evidence", "")
    assert not EVAL.answer_matches("null_query", "Insufficient information.", "answered", "anything")


def test_yes_no_scoring_reads_the_verdict_field_and_falls_back_to_words():
    prose = "the two reports do not agree on the cause"
    # v1 hunts for a word in prose: this answer is invisible to it.
    assert not EVAL.answer_matches("comparison_query", "No", "answered", prose, None, "v1")
    # v2 reads the explicit field.
    assert EVAL.answer_matches("comparison_query", "No", "answered", prose, {"value": "no"}, "v2")
    assert not EVAL.answer_matches("comparison_query", "No", "answered", prose, {"value": "yes"}, "v2")
    # Declining to judge is not an answer.
    assert not EVAL.answer_matches(
        "comparison_query", "No", "answered", prose, {"value": "unclear"}, "v2"
    )
    # With no verdict at all, v2 is v1 rather than free credit.
    assert EVAL.answer_matches(
        "comparison_query", "No", "answered", "no, they differ", None, "v2"
    )
    assert not EVAL.answer_matches("comparison_query", "No", "answered", prose, None, "v2")


def test_entity_answers_are_scored_by_containment_in_either_rule():
    for rule in ("v1", "v2"):
        assert EVAL.answer_matches(
            "inference_query", "Sam Bankman-Fried", "answered", "the trial of sam bankman-fried", None, rule
        )
        assert not EVAL.answer_matches(
            "inference_query", "Sam Bankman-Fried", "insufficient_evidence", "", None, rule
        )


def test_summary_reports_answerability_calibration_alongside_accuracy():
    rows = [
        {"question_type": "inference_query", "answer_correct": True, "status": "answered",
         "latency_ms": 10, "prompt_tokens": 1, "gold_documents": 1, "gold_documents_retrieved": 1,
         "gold_documents_cited": 1, "all_gold_retrieved": True},
        {"question_type": "inference_query", "answer_correct": False, "status": "insufficient_evidence",
         "latency_ms": 10, "prompt_tokens": 1, "gold_documents": 1, "gold_documents_retrieved": 0,
         "gold_documents_cited": 0, "all_gold_retrieved": False},
        {"question_type": "null_query", "answer_correct": True, "status": "insufficient_evidence",
         "latency_ms": 10, "prompt_tokens": 1, "gold_documents": 0, "gold_documents_retrieved": 0,
         "gold_documents_cited": 0, "all_gold_retrieved": False},
        {"question_type": "null_query", "answer_correct": False, "status": "answered",
         "latency_ms": 10, "prompt_tokens": 1, "gold_documents": 0, "gold_documents_retrieved": 0,
         "gold_documents_cited": 0, "all_gold_retrieved": False},
    ]
    summary = EVAL.summarize(rows)
    assert summary["answerable_questions"] == 2
    assert summary["answerable_accuracy"] == 0.5
    assert summary["refusal_rate_on_answerable"] == 0.5
    # Coverage and abstention have to be readable together: one null was answered.
    assert summary["null_refusal_recall"] == 0.5
    assert summary["false_answer_rate_on_null"] == 0.5


def test_annotation_sample_is_stratified_interleaved_and_reproducible():
    pool = [
        {"item_id": f"B{batch}-{kind}-{index}", "batch": batch, "question_type": kind, "claims": [1]}
        for batch in (1, 2)
        for kind in ("comparison_query", "inference_query")
        for index in range(10)
    ]
    chosen = SAMPLE.sample(pool, 3)
    assert len(chosen) == 12
    assert SAMPLE.sample(pool, 3) == chosen  # content addressed, not random
    # The first row of every stratum comes before the second row of any stratum, so a
    # partially labelled file is still balanced.
    assert len({(row["batch"], row["question_type"]) for row in chosen[:4]}) == 4
    assert SAMPLE.arm_for("B1-MH-abc") == SAMPLE.arm_for("B1-MH-abc")
    assert {SAMPLE.arm_for(f"item-{index}") for index in range(20)} == {"rag", "workflow"}


MERGE = load_script("merge_semantic_labels")


def _write(path, header, rows):
    import csv

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_two_label_sets_produce_only_the_contested_items_for_a_person(tmp_path):
    header = ["item_id", "claim_index", "entailment_MODEL", "relevance_MODEL", "note"]
    first = _write(
        tmp_path / "a.csv",
        header,
        [
            ["i1", "1", "supported", "answers", ""],
            ["i2", "1", "partial", "answers", ""],
            ["i3", "1", "unsupported", "related", ""],
        ],
    )
    second = _write(
        tmp_path / "b.csv",
        ["item_id", "claim_index", "entailment_gpt", "relevance_gpt", "note"],
        [
            ["i1", "1", "supported", "answers", ""],
            ["i2", "1", "supported", "answers", ""],  # disagrees on entailment only
            ["i3", "1", "unsupported", "answers", ""],  # disagrees on relevance only
        ],
    )
    summary, contested, agreed, audited = MERGE.compare("claims", first, second, 0.5)
    assert summary["items_compared"] == 3
    assert summary["contested_items"] == 2 and summary["agreed_items"] == 1
    assert summary["by_label"]["entailment"]["agreement_rate"] == round(2 / 3, 3)
    assert summary["by_label"]["relevance"]["agreement_rate"] == round(2 / 3, 3)
    # A person sees the disagreements plus a fixed slice of the agreements, because
    # two models agreeing can still be two models sharing one blind spot.
    assert summary["human_judgements_required"] == 2 + len(audited)
    assert audited == {("i1", "1")}
    # The audit slice is content addressed, so it cannot be redrawn until it is small.
    assert MERGE.audit_sample([("i1", "1")], 0.5) == {("i1", "1")}


def test_label_columns_are_found_whatever_the_suffix():
    columns = MERGE.label_columns(
        ["item_id", "entailment(supported|partial|unsupported)", "relevance_gpt", "note"]
    )
    assert columns["entailment"] == "entailment(supported|partial|unsupported)"
    assert columns["relevance"] == "relevance_gpt"
    assert "completeness" not in columns


def test_unsupported_agreements_are_audited_in_full(tmp_path):
    header = ["item_id", "claim_index", "entailment_MODEL", "relevance_MODEL", "note"]
    rows = [[f"i{n}", "1", "supported", "answers", ""] for n in range(8)]
    rows += [[f"u{n}", "1", "unsupported", "answers", ""] for n in range(3)]
    first = _write(tmp_path / "a.csv", header, rows)
    second = _write(tmp_path / "b.csv", header, rows)
    summary, _, _, audited = MERGE.compare(
        "claims", first, second, 0.2, "entailment", "unsupported", 1.0
    )
    assert summary["agreed_items"] == 11
    # Every agreed `unsupported` is checked, because a person ratifying "both models
    # said this claim has no support" is the one judgement P2 depends on.
    assert summary["audit_critical"] == 3
    assert {key for key in audited if key[0].startswith("u")} == {("u0", "1"), ("u1", "1"), ("u2", "1")}
    # The rest keeps the ordinary sampling rate.
    assert len([key for key in audited if key[0].startswith("i")]) == 2
