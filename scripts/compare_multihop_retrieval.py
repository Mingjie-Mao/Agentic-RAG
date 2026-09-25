"""Paired, retrieval-only comparison on aligned frozen batches.

The two inputs must contain exactly the same question IDs and gold denominators.
Bootstrap resamples whole questions, preserving the pairing and all facts from a
question together. This cannot establish an end-to-end answer improvement.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/multihop-retrieval-v2"


def keyed(rows):
    result = {}
    for row in rows:
        key = (row["batch"], row["id"])
        if key in result:
            raise ValueError(f"duplicate question: {key}")
        result[key] = row
    return result


def align(before, after):
    left, right = keyed(before), keyed(after)
    if left.keys() != right.keys():
        raise ValueError("retrieval runs have different question IDs")
    for key in left:
        for field in ("question_type", "gold_documents", "gold_facts"):
            if left[key][field] != right[key][field]:
                raise ValueError(f"retrieval runs have different gold data for {key}: {field}")
    return [(left[key], right[key]) for key in sorted(left)]


def recall(pairs, field, denominator, side):
    total = sum(pair[side][denominator] for pair in pairs)
    return sum(pair[side][field] for pair in pairs) / total if total else None


def summarize(pairs, samples=2000, seed=0):
    if not pairs:
        return {"questions": 0}
    rng = random.Random(seed)
    metrics = {}
    for name, numerator, denominator in (
        ("gold_document_recall", "gold_documents_found", "gold_documents"),
        ("gold_fact_recall", "gold_facts_delivered", "gold_facts"),
    ):
        before = recall(pairs, numerator, denominator, 0)
        after = recall(pairs, numerator, denominator, 1)
        deltas = []
        if before is not None:
            for _ in range(samples):
                chosen = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                a = recall(chosen, numerator, denominator, 0)
                b = recall(chosen, numerator, denominator, 1)
                if a is not None and b is not None:
                    deltas.append(b - a)
        deltas.sort()
        metrics[name] = {
            "before": round(before, 4) if before is not None else None,
            "after": round(after, 4) if after is not None else None,
            "delta": round(after - before, 4) if before is not None else None,
            "delta_ci95": [
                round(deltas[int(0.025 * len(deltas))], 4),
                round(deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))], 4),
            ] if deltas else None,
        }
    return {
        "questions": len(pairs),
        **metrics,
        "median_retrieval_ms": {
            "before": round(statistics.median(row[0]["retrieval_ms"] for row in pairs), 1),
            "after": round(statistics.median(row[1]["retrieval_ms"] for row in pairs), 1),
        },
    }


def compare(before_path, after_path, samples=2000):
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    for field in ("label", "batches", "metric", "document_quota", "clause_queries", "passage_rerank"):
        if before.get(field) != after.get(field):
            raise ValueError(f"retrieval runs differ beyond routing: {field}")
    pairs = align(before["results"], after["results"])
    types = sorted({row[0]["question_type"] for row in pairs})
    scope = (
        "one-time frozen B3 retrieval validation; no post-result tuning"
        if before.get("batches") == ["b3"] else
        "retrieval only; B1/B2 MultiHop-RAG batches are development data"
    )
    return {
        "scope": scope,
        "metric": before["metric"],
        "bootstrap": {"resample": "question", "samples": samples, "seed": 0},
        "inputs": {
            "before": {"file": str(before_path), "sha256": hashlib.sha256(before_path.read_bytes()).hexdigest()},
            "after": {"file": str(after_path), "sha256": hashlib.sha256(after_path.read_bytes()).hexdigest()},
        },
        "all": summarize(pairs, samples),
        "by_type": {
            kind: summarize([pair for pair in pairs if pair[0]["question_type"] == kind], samples)
            for kind in types
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=RUNS / "index-v2-routing-off-q2.json")
    parser.add_argument("--after", type=Path, default=RUNS / "index-v2-routing-on-q2.json")
    parser.add_argument("--out", type=Path, default=RUNS / "paired-source-routing.json")
    args = parser.parse_args()
    report = compare(args.before, args.after)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["all"], ensure_ascii=False, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
