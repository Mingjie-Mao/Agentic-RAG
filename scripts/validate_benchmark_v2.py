"""Read-only check of the PROPOSED Benchmark v2 manifest against repository suites.

Adapted from the external v2 draft validator. Exact question overlap across
suites fails unless it is registered in ``known_exact_question_overlap``; a
registration that no longer matches the data also fails. Only IDs and counts
are printed, never question text. Nothing is executed or modified.

Usage: .venv/bin/python scripts/validate_benchmark_v2.py --repo-root .
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/enterprise_rag/v2/manifest.proposed.json"
LEAKAGE_KEYS = ("gold_answer", "gold_documents", "required_facts", "expected_behavior")


def validate_manifest(data):
    issues = []
    names = [m.get("name") for m in data.get("methods") or []]
    if len(names) != 4 or len(set(names)) != 4:
        issues.append("Expected exactly four distinct primary arms A/B/C/D")
    for prefix in ("A_", "B_", "C_", "D_"):
        if sum(isinstance(n, str) and n.startswith(prefix) for n in names) != 1:
            issues.append(f"Expected exactly one {prefix} arm")
    if data.get("status") != "design_not_frozen_not_executed":
        issues.append("The proposed manifest must not claim tested/frozen status")
    ids = [s.get("id") for s in data.get("suites") or []]
    if len(ids) != len(set(ids)):
        issues.append("Duplicate suite IDs")
    for suite in data.get("suites") or []:
        if suite.get("expected_tasks", 0) < 1:
            issues.append(f"Invalid expected_tasks: {suite.get('id')}")
        if not suite.get("path"):
            issues.append(f"Missing suite path: {suite.get('id')}")
    if data.get("primary_test_metric") != "dynamic_test.StrictTaskSuccessRate":
        issues.append("Unexpected v2 primary metric")
    if data.get("original_v1_main_metric_unchanged") != "core.StrictTaskSuccessRate":
        issues.append("Do not change v1 headline metric")
    prohibited = set(data.get("must_not_enter_method_input", []))
    issues += [f"Potential answer leakage: exclude {k} from method input" for k in LEAKAGE_KEYS if k not in prohibited]
    return issues


def load_suite(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, list) else obj["tasks"]


def normalized(question):
    return "".join(str(question).split()).casefold()


def validate_repository(data, root):
    issues, info = [], []
    seen_ids, seen_questions, overlaps = {}, {}, {}
    for suite in data["suites"]:
        label = suite["id"]
        if suite.get("status") == "proposed_not_created":
            info.append(f"{label}: proposed only, not checked")
            continue
        path = root / suite["path"]
        if not path.is_file():
            issues.append(f"{label}: suite not found: {suite['path']}")
            continue
        tasks = load_suite(path)
        if len(tasks) != suite["expected_tasks"]:
            issues.append(f"{label}: expected {suite['expected_tasks']} tasks, found {len(tasks)}")
        families = Counter()
        for item in tasks:
            tid, question = str(item.get("id", "")), item.get("question") or item.get("goal") or ""
            if not tid or not question:
                issues.append(f"{label}: task missing id/question")
                continue
            if tid in seen_ids:
                issues.append(f"Duplicate task ID {tid!r} in {label} and {seen_ids[tid]}")
            seen_ids[tid] = label
            key = normalized(question)
            other = seen_questions.get(key)
            if other is not None and other[0] != label:
                overlaps.setdefault(label, set()).add(tid)
            seen_questions.setdefault(key, (label, tid))
            families[item.get("dynamic_type") or item.get("category") or "unknown"] += 1
        info.append(f"{label}: {len(tasks)} tasks; categories={dict(families)}")
    known = data.get("known_exact_question_overlap") or {}
    registered = {suite: set(ids) for suite, ids in known.items() if isinstance(ids, list)}
    for suite in sorted(set(overlaps) | set(registered)):
        found, listed = overlaps.get(suite, set()), registered.get(suite, set())
        if found - listed:
            issues.append(f"{suite}: unregistered exact question overlap {sorted(found - listed)}")
        if listed - found:
            issues.append(f"{suite}: registered overlap no longer present {sorted(listed - found)}")
        if found and found == listed:
            info.append(f"{suite}: {len(found)} registered exact-overlap tasks (report separately)")
    return issues, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    problems = validate_manifest(manifest)
    more, info = validate_repository(manifest, args.repo_root)
    problems += more
    for line in info:
        print(line)
    if problems:
        print("FAILED:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("OK: proposed v2 manifest is consistent. Baselines are not implemented and nothing has run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
