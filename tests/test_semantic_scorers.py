"""P2 scorer calibration: weighted metrics, dev-only thresholds, the selection rule,
the freeze check, and the inverse-probability weights of the v3 gold."""

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = load_script("evaluate_semantic_scorers")
SCORERS = load_script("semantic_scorers")
GOLD = load_script("build_semantic_gold_v3")


def row(gold, pred, w=1.0, gold3=None, item="i"):
    return {"item": item, "w": w, "gold": gold, "pred": pred, "gold3": gold3 or ("supported" if gold else "unsupported")}


def test_weights_move_balanced_accuracy_and_pass_rate():
    rows = [
        row(True, True),
        row(False, True, w=3.0),  # a sampled row stands for three claims
        row(False, False),
    ]
    metrics = EVAL.binary_metrics(rows)
    assert metrics["recall_positive"] == 1.0
    assert metrics["recall_negative"] == pytest.approx(0.25)
    assert metrics["pass_rate_unsupported"] == pytest.approx(0.75)
    assert EVAL.binary_metrics([{**r, "w": 1.0} for r in rows])["pass_rate_unsupported"] == pytest.approx(0.5)


def test_partial_counts_as_not_supported_but_is_reported_separately():
    rows = [row(True, True), row(False, True, gold3="partial"), row(False, False, gold3="unsupported")]
    metrics = EVAL.binary_metrics(rows)
    assert metrics["pass_rate_partial"] == 1.0
    assert metrics["pass_rate_unsupported"] == 0.0


def test_threshold_ties_resolve_to_the_stricter_value():
    rows = [
        {"item": "a", "w": 1.0, "gold": True, "score": 0.9},
        {"item": "b", "w": 1.0, "gold": False, "score": 0.2},
    ]
    # 0.9 and any value above 0.2 separate perfectly; the stricter one wins.
    assert EVAL.best_threshold(rows) == 0.9


def test_auc_is_weighted_and_counts_ties_as_half():
    rows = [
        {"w": 1.0, "gold": True, "score": 0.5},
        {"w": 1.0, "gold": False, "score": 0.5},
        {"w": 1.0, "gold": False, "score": 0.1},
    ]
    assert EVAL.auc(rows) == pytest.approx(0.75)


def test_selection_drops_random_scorers_and_picks_the_cheapest_eligible():
    def entry(ba, ci, pass_rate, ms):
        return {"test": {"balanced_accuracy": ba, "pass_rate_unsupported": pass_rate},
                "test_ci95": {"balanced_accuracy": ci}, "median_ms_per_claim": ms}

    report = {
        "rule": entry(0.55, [0.45, 0.65], 0.5, 0.1),  # CI touches 0.5: eliminated
        "embedding": entry(0.70, [0.60, 0.81], 0.25, 300),
        "cross_encoder": entry(0.80, [0.70, 0.90], 0.20, 150),
        "llm_judge": entry(0.95, [0.90, 0.99], 0.00, 8000),  # reported, never the baseline
    }
    chosen = EVAL.select(report)
    assert chosen["baseline"] == "cross_encoder"
    assert chosen["eliminated_as_random"] == ["rule"]
    assert chosen["reported_not_ranked"] == ["llm_judge"]
    # Embedding is eligible but slower than the baseline, so the baseline itself is chosen.
    assert chosen["selected"] == "cross_encoder"


def test_selection_prefers_a_cheaper_scorer_that_is_not_credibly_worse():
    def entry(ba, ci, pass_rate, ms):
        return {"test": {"balanced_accuracy": ba, "pass_rate_unsupported": pass_rate},
                "test_ci95": {"balanced_accuracy": ci}, "median_ms_per_claim": ms}

    report = {
        "rule": entry(0.74, [0.64, 0.84], 0.25, 0.1),
        "embedding": entry(0.70, [0.60, 0.81], 0.45, 300),  # passes too many unsupported
        "cross_encoder": entry(0.78, [0.68, 0.88], 0.20, 150),
    }
    chosen = EVAL.select(report)
    assert chosen["eligible"] == ["rule", "cross_encoder"]
    assert chosen["selected"] == "rule"


def test_selection_reports_no_winner_when_everything_is_random():
    report = {"rule": {"test": {"balanced_accuracy": 0.5, "pass_rate_unsupported": 0.5},
                       "test_ci95": {"balanced_accuracy": [0.4, 0.6]}, "median_ms_per_claim": 0.1}}
    assert EVAL.select(report)["selected"] is None


def test_evaluation_refuses_edited_or_unfrozen_gold(tmp_path):
    with pytest.raises(SystemExit):
        EVAL.verify_freeze(tmp_path)
    (tmp_path / "gold.csv").write_text("a\n")
    import hashlib

    (tmp_path / "gold-v3-manifest.json").write_text(json.dumps({
        "frozen_at": "x", "sha256": {"gold.csv": hashlib.sha256(b"a\n").hexdigest()}}))
    EVAL.verify_freeze(tmp_path)
    (tmp_path / "gold.csv").write_text("b\n")
    with pytest.raises(SystemExit):
        EVAL.verify_freeze(tmp_path)


def test_split_keeps_every_claim_of_a_question_together():
    assert EVAL.split_of("B1-MH-0c18610f435a") == EVAL.split_of("B1-MH-0c18610f435a")
    shares = [EVAL.split_of(f"B1-MH-{n:012x}") for n in range(400)]
    assert 0.3 < shares.count("dev") / 400 < 0.5


def test_rule_scorer_counts_content_words_and_cjk_bigrams():
    score = SCORERS.rule_scorer()
    assert score("Google paid $26.3 billion in 2021", "Google paid 26.3 billion") == 1.0
    assert score("unrelated text", "Google paid 26.3 billion") == 0.0
    assert SCORERS.content_terms("情报失误") == {"情报", "报失", "失误"}


def test_windows_cover_long_blocks_with_overlap():
    evidence = "a" * 3000 + "\n---\n" + "short"
    parts = SCORERS.windows(evidence)
    assert parts[-1] == "short"
    assert all(len(part) <= SCORERS.WINDOW for part in parts)
    assert sum(len(p) for p in parts[:-1]) >= 3000


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_v3_gold_weights_only_the_sampled_stratum(tmp_path):
    review = tmp_path / "round2"
    review.mkdir()
    claims = [("R1-a", "1", "q", "c one", "e"), ("R2-a", "1", "q", "c two", "e"),
              ("R2-b", "1", "q", "c three", "e"), ("R2-c", "1", "q", "c four", "e"),
              ("R2-d", "1", "q", "reports are consistent", "e"), ("R2-e", "1", "q", "c six", "e")]
    write_csv(tmp_path / "claims-to-label.csv", ["item_id", "claim_index", "question", "claim", "cited_evidence"], claims)
    write_csv(tmp_path / "gold-claims.csv", ["item_id", "claim_index", "entailment", "relevance", "provenance", "basis"],
              [("R1-a", "1", "supported", "answers", "gpt_final_review_accept", "")])
    labels = {"R2-a": ("supported", "supported"), "R2-b": ("supported", "supported"),
              "R2-c": ("unsupported", "unsupported"), "R2-d": ("supported", "supported"),
              "R2-e": ("partial", "supported")}
    write_csv(tmp_path / "machine-prepass-claims-rest.csv", ["item_id", "claim_index", "entailment_MODEL"],
              [(k, "1", a) for k, (a, _) in labels.items()])
    write_csv(tmp_path / "gpt-prepass-claims-rest.csv", ["item_id", "claim_index", "entailment(x)"],
              [(k, "1", b) for k, (_, b) in labels.items()])
    write_csv(review / "final-review-entailment.csv", ["item_id", "claim_index", "reason", "final_entailment(x)"],
              [("R2-a", "1", "audit_sample_20pct", "supported"),
               ("R2-c", "1", "audit_unsupported_agreement", "unsupported"),
               ("R2-d", "1", "v3_rule12_recheck", "unsupported"),
               ("R2-e", "1", "disagreement", "partial")])
    write_csv(review / "final-review-relevance.csv", ["item_id", "claim_index", "final_relevance(x)"],
              [(c[0], "1", "answers") for c in claims])
    rows, weights = GOLD.build_claims(tmp_path, review)
    by_id = {r["item_id"]: r for r in rows}
    # Stratum = agreed, not unsupported, not relational: R2-a and R2-b; one of two sampled.
    assert weights == {"stratum": 2, "sampled": 1, "weight": 2.0}
    assert by_id["R2-a"]["weight"] == 2.0
    assert by_id["R2-b"]["weight"] == 0.0 and by_id["R2-b"]["entailment_provenance"] == "model_consensus"
    assert by_id["R2-c"]["weight"] == 1.0
    assert by_id["R2-d"]["entailment"] == "unsupported"
    assert by_id["R2-d"]["entailment_provenance"] == "gpt_final_review_v3_override"
    assert by_id["R2-e"]["entailment_provenance"] == "gpt_final_review_v3_disagreement"
    assert by_id["R1-a"]["entailment_provenance"] == "v2_gpt_final_review_accept"


def test_v3_gold_rejects_an_unfilled_review(tmp_path):
    review = tmp_path / "round2"
    review.mkdir()
    write_csv(tmp_path / "claims-to-label.csv", ["item_id", "claim_index", "question", "claim", "cited_evidence"],
              [("R2-a", "1", "q", "c", "e")])
    write_csv(tmp_path / "gold-claims.csv", ["item_id", "claim_index", "entailment", "relevance", "provenance", "basis"], [])
    write_csv(tmp_path / "machine-prepass-claims-rest.csv", ["item_id", "claim_index", "entailment_MODEL"], [("R2-a", "1", "partial")])
    write_csv(tmp_path / "gpt-prepass-claims-rest.csv", ["item_id", "claim_index", "entailment(x)"], [("R2-a", "1", "supported")])
    write_csv(review / "final-review-entailment.csv", ["item_id", "claim_index", "reason", "final_entailment(x)"],
              [("R2-a", "1", "disagreement", "")])
    write_csv(review / "final-review-relevance.csv", ["item_id", "claim_index", "final_relevance(x)"], [("R2-a", "1", "answers")])
    with pytest.raises(SystemExit):
        GOLD.build_claims(tmp_path, review)
