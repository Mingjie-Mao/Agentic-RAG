"""Decide whether post-training is justified by real unresolved policy failures."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "artifacts/agent-failure-corpus.json"
DEFAULT_OUT = ROOT / "artifacts/agent-training-gate.json"
MIN_FAILURES = 50
MIN_TYPES = 3


def assess(rows):
    qualified = [
        row for row in rows
        if row.get("failure_layer") == "policy"
        and row.get("adjudicated") is True
        and row.get("resolved") is False
        and row.get("replay")
    ]
    # Several runs of the same task and failure type are one failure, not several.
    unique = {(row.get("task_id"), row.get("failure_type")): row for row in qualified}
    kinds = sorted({row.get("failure_type") for row in unique.values()})
    ready = len(unique) >= MIN_FAILURES and len(kinds) >= MIN_TYPES
    return {
        "decision": "consider_training_experiment" if ready else "do_not_train",
        "historical_failures_in_corpus": len(rows),
        "marked_resolved_in_corpus": sum(row.get("resolved") is True for row in rows),
        "independently_adjudicated": sum(row.get("adjudicated") is True for row in rows),
        "qualified_unresolved_unique_policy_failures": len(unique),
        "qualified_failure_types": kinds,
        "minimum_unique_failures": MIN_FAILURES,
        "minimum_failure_types": MIN_TYPES,
        "requires_frozen_unseen_evaluation": True,
        "note": "A positive gate permits a cost-capped comparison; it does not select SFT, distillation, preference optimization or RL.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    report = assess(corpus["failures"])
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
