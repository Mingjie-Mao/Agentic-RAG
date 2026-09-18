"""Build a reviewable failure corpus from real Agent benchmark runs."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "artifacts/b1-agent-benchmark.json",
    "artifacts/b1-agent-conflict-gate.json",
    "artifacts/b2-agent-benchmark-rag-workflow.json",
    "artifacts/b2-agent-regression-gate.json",
    "artifacts/b2-agent-final-gate.json",
    "artifacts/b2-agent-dynamic-hard.json",
    "artifacts/memory-ab-v1.json",
]
RESOLUTION_SOURCES = [
    "artifacts/b2-agent-benchmark-final.json",
    "artifacts/b2-agent-dynamic-hard.json",
    "artifacts/b2-agent-dynamic-a22-final.json",
    "artifacts/memory-ab-a26-foreign-tenant.json",
]


def taxonomy(row):
    if row.get("execution_error"):
        return "execution_failure"
    if any(row.get("forbidden_present", {}).values()):
        return "security_leak"
    if row.get("expected_status") == "conflict" and row.get("status") != "conflict":
        return "conflict_miss"
    if row.get("status") == "insufficient_evidence" and row.get("expected_status") == "answered":
        return "false_refusal"
    if row.get("status") == "answered" and row.get("expected_status") == "insufficient_evidence":
        return "false_answer"
    if any(not value for value in row.get("facts", {}).values()):
        return "incomplete_answer"
    if not row.get("status_ok", True):
        return "status_error"
    return "other"


def signature(row, arm):
    return (
        row.get("task_id"),
        arm,
        row.get("status"),
        row.get("expected_status"),
        tuple(key for key, value in row.get("facts", {}).items() if not value),
        tuple(key for key, value in row.get("forbidden_present", {}).items() if value),
        row.get("execution_error"),
    )


def rows(path):
    payload = json.loads((ROOT / path).read_text())
    for arm, items in payload.get("results", {}).items():
        if not isinstance(items, list):
            continue
        for row in items:
            yield arm, row


def main():
    tasks = {
        row["id"]: row for row in json.loads((ROOT / "fixtures/agent/tasks.json").read_text())["tasks"]
    }
    successes = {}
    for source in RESOLUTION_SOURCES:
        for arm, row in rows(source):
            if row.get("correct"):
                successes[(row.get("task_id"), arm)] = source

    failures = {}
    for source in SOURCES:
        for arm, row in rows(source):
            if row.get("correct") is not False and not row.get("execution_error"):
                continue
            key = signature(row, arm)
            if key not in failures:
                missing = [key for key, value in row.get("facts", {}).items() if not value]
                leaked = [key for key, value in row.get("forbidden_present", {}).items() if value]
                task = tasks.get(row.get("task_id"), {})
                failures[key] = {
                    "task_id": row.get("task_id"),
                    "arm": arm,
                    "category": task.get("category"),
                    "goal": task.get("goal"),
                    "failure_type": taxonomy(row),
                    "observed_status": row.get("status"),
                    "expected_status": row.get("expected_status"),
                    "missing_facts": missing,
                    "leaked_forbidden": leaked,
                    "execution_error": row.get("execution_error"),
                    "source_files": [],
                    "occurrences": 0,
                }
            failures[key]["source_files"].append(source)
            failures[key]["occurrences"] += 1

    output_rows = []
    for item in failures.values():
        resolved_in = successes.get((item["task_id"], item["arm"]))
        item["resolved"] = resolved_in is not None
        item["resolved_in"] = resolved_in
        output_rows.append(item)
    output_rows.sort(key=lambda item: (item["task_id"], item["arm"], item["failure_type"]))
    unresolved = [item for item in output_rows if not item["resolved"]]
    categories = Counter(item["failure_type"] for item in output_rows)
    payload = {
        "corpus": "agent-observed-failures-v1",
        "scope": "actual local benchmark runs over the synthetic Agent task suite",
        "summary": {
            "unique_failures": len(output_rows),
            "unresolved": len(unresolved),
            "resolved": len(output_rows) - len(unresolved),
            "by_type": dict(sorted(categories.items())),
        },
        "training_gate": {
            "decision": "do_not_train",
            "reason": "too few unresolved failures; observed failures were fixed by deterministic system changes",
            "minimum_unique_adjudicated_failures": 50,
            "minimum_failure_types": 3,
            "requires_frozen_unseen_evaluation": True,
        },
        "failures": output_rows,
    }
    destination = ROOT / "artifacts/agent-failure-corpus.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["summary"] | payload["training_gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
