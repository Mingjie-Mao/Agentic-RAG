"""Derive corrected metrics without rewriting the frozen v1 artifact or its scores."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_runtime import atomic_json, summary, task_safety_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/unseen-benchmark-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/unseen-benchmark-v1-p0-audit.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("derived report exists; use a new output path")
    original = args.input.read_bytes()
    result = json.loads(original)
    suite = json.loads(Path("fixtures/unseen/tasks.json").read_text())
    spec = json.loads(Path("fixtures/unseen/documents.json").read_text())
    labels = task_safety_labels(suite, spec, Path.cwd())
    atomic_json(args.output, {
        "source": str(args.input), "source_sha256": hashlib.sha256(original).hexdigest(),
        "scores_changed": False, "summary": {arm: summary(rows, labels)
                                               for arm, rows in result["results"].items()},
        "safety_classification": {key: t.get("forbidden_kind", "other_forbidden_output")
                                  for key, t in labels.items() if t.get("forbidden")},
        "classification_method": "conditions by category; ACL by forbidden values unique to "
                                 "corpus documents denied to the specified user",
    })
    assert args.input.read_bytes() == original


if __name__ == "__main__":
    main()
