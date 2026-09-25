"""Audit the pre-registered B3 answer slice without changing its scorer or questions."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "fixtures/multihop/subset-r3-eval32.json"
RUN = ROOT / "artifacts/multihop-external-r3-eval32.json"
OUT = ROOT / "artifacts/multihop-r3-eval32-analysis.json"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def percentile(values, fraction):
    ordered = sorted(values)
    return round(ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)], 1)


def summarize(rows, gold):
    answerable = [row for row in rows if row["question_type"] != "null_query"]
    null = [row for row in rows if row["question_type"] == "null_query"]
    yes_no = [row for row in rows if gold[row["id"]].lower() in {"yes", "no", "true", "false"}]
    labels = [gold[row["id"]].lower() in {"yes", "true"} for row in yes_no]
    majority = max(sum(labels), len(labels) - sum(labels)) if labels else 0
    literal_yes_no = [row for row in yes_no if gold[row["id"]].lower() in {"yes", "no"}]
    literal_labels = [gold[row["id"]].lower() == "yes" for row in literal_yes_no]
    literal_majority = max(sum(literal_labels), len(literal_labels) - sum(literal_labels)) if literal_labels else 0
    # Frozen v2 only normalizes literal Yes/No. Keep the pre-registered score
    # intact, and show its True/False blind spot as a post-hoc diagnostic.
    true_false = [row for row in yes_no if gold[row["id"]].lower() in {"true", "false"}]
    verdict_scored = [row for row in true_false if row.get("verdict") in {"yes", "no"}]
    verdict_correct = sum(
        (row["verdict"] == "yes") == (gold[row["id"]].lower() == "true")
        for row in verdict_scored
    )
    gold_documents = sum(row.get("gold_documents", 0) for row in answerable)
    return {
        "questions": len(rows),
        "answer_correct": sum(row["answer_correct"] for row in rows),
        "answer_correct_rate": rate(sum(row["answer_correct"] for row in rows), len(rows)),
        "answerable_accuracy": rate(sum(row["answer_correct"] for row in answerable), len(answerable)),
        "false_refusal_rate_answerable": rate(sum(row["status"] == "insufficient_evidence" for row in answerable), len(answerable)),
        "null_false_answer_rate": rate(sum(row["status"] != "insufficient_evidence" for row in null), len(null)),
        "gold_document_recall_in_answer_context": rate(
            sum(row.get("gold_documents_retrieved", 0) for row in answerable), gold_documents
        ),
        "gold_document_citation_recall": rate(
            sum(row.get("gold_documents_cited", 0) for row in answerable), gold_documents
        ),
        "all_gold_documents_retrieved_rate": rate(
            sum(bool(row.get("all_gold_retrieved")) for row in answerable), len(answerable)
        ),
        "coverage_repair_questions": sum(
            "coverage_repair" in row.get("planner_events", []) for row in rows
        ),
        "query_recovery_questions": sum(
            "query_recovery" in row.get("planner_events", []) for row in rows
        ),
        "yes_no_questions": len(yes_no),
        "yes_no_correct": sum(row["answer_correct"] for row in yes_no),
        "yes_no_accuracy": rate(sum(row["answer_correct"] for row in yes_no), len(yes_no)),
        "yes_no_majority_baseline": rate(majority, len(yes_no)),
        "literal_yes_no_questions_v2_comparable": len(literal_yes_no),
        "literal_yes_no_accuracy": rate(sum(row["answer_correct"] for row in literal_yes_no), len(literal_yes_no)),
        "literal_yes_no_majority_baseline": rate(literal_majority, len(literal_yes_no)),
        "true_false_gold_questions": len(true_false),
        "true_false_with_explicit_verdict": len(verdict_scored),
        "true_false_verdict_correct_posthoc": verdict_correct,
        "true_false_correct_hidden_by_v2": sum(
            (row["verdict"] == "yes") == (gold[row["id"]].lower() == "true")
            and not row["answer_correct"] for row in verdict_scored
        ),
        "p50_latency_ms": round(statistics.median(row["latency_ms"] for row in rows), 1),
        "p95_latency_ms": percentile([row["latency_ms"] for row in rows], .95),
        "mean_prompt_tokens": round(statistics.mean(row["prompt_tokens"] for row in rows), 1),
    }


def paired(rows_a, rows_b, *, samples=2000):
    a, b = {row["id"]: row for row in rows_a}, {row["id"]: row for row in rows_b}
    if a.keys() != b.keys():
        raise ValueError("paired arms do not have identical question IDs")
    ids = sorted(a)
    deltas = [int(b[key]["answer_correct"]) - int(a[key]["answer_correct"]) for key in ids]
    rng = random.Random(0)
    boot = sorted(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
                  for _ in range(samples))
    return {
        "questions": len(ids),
        "workflow_only_correct": sum(value == 1 for value in deltas),
        "rag_only_correct": sum(value == -1 for value in deltas),
        "accuracy_delta_workflow_minus_rag": rate(sum(deltas), len(deltas)),
        "delta_ci95_question_bootstrap": [round(boot[int(.025 * samples)], 4), round(boot[int(.975 * samples)], 4)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=SUBSET)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    subset = json.loads(args.subset.read_text())
    run = json.loads(args.run.read_text())
    if run["subset_sha256"] != hashlib.sha256(args.subset.read_bytes()).hexdigest():
        raise ValueError("run used a different frozen subset")
    if run["scoring_rule"] != "v2" or run["mode"] != "frozen_subset":
        raise ValueError("unexpected scoring protocol")
    dataset = {digest(row["query"]): row for row in json.loads((ROOT / ".runtime/multihop/MultiHopRAG.json").read_bytes())}
    gold = {}
    for item in subset["items"]:
        raw = dataset[item["query_sha256"]]
        if digest(raw["answer"]) != item["answer_sha256"]:
            raise ValueError(f"gold hash mismatch: {item['id']}")
        gold[item["id"]] = raw["answer"].strip()
    for arm in ("rag", "workflow"):
        if {row["id"] for row in run["results"][arm]} != set(gold):
            raise ValueError(f"incomplete {arm} arm")
    report = {
        "scope": "pre-registered 32-question independent B3 slice; literal v2 scorer, not human semantic accuracy",
        "subset_sha256": run["subset_sha256"],
        "run_sha256": hashlib.sha256(args.run.read_bytes()).hexdigest(),
        "arms": {arm: summarize(run["results"][arm], gold) for arm in ("rag", "workflow")},
        "paired": paired(run["results"]["rag"], run["results"]["workflow"]),
        "by_type": {
            kind: {
                "arms": {
                    arm: summarize(
                        [row for row in run["results"][arm] if row["question_type"] == kind], gold
                    ) for arm in ("rag", "workflow")
                },
                "paired": paired(
                    [row for row in run["results"]["rag"] if row["question_type"] == kind],
                    [row for row in run["results"]["workflow"] if row["question_type"] == kind],
                ),
            }
            for kind in sorted({row["question_type"] for row in run["results"]["rag"]})
        },
        "diagnostic_note": "True/False verdict counts are post-hoc scorer diagnostics; the frozen v2 answer_correct score is unchanged.",
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
