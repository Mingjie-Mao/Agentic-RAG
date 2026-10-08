"""Summarize a dynamic-benchmark run by dynamic type and by the measured necessity label.

Strict success comes only from a reviewed report; without one it stays null. Latency
uses the standard nearest rank and keeps every row, including failures.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def nearest_rank(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--necessity", type=Path, required=True)
    p.add_argument("--report", type=Path, help="reviewed report; strict success is null without it")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    raw = json.loads(args.raw.read_text())
    tasks = {t["id"]: t for t in json.loads(args.suite.read_text())["tasks"]}
    need = {r["task_id"]: r["needs_dynamic"] for r in json.loads(args.necessity.read_text())["rows"]}
    details = json.loads(args.report.read_text())["details"] if args.report else None
    out = {}
    for method, rows in raw["results"].items():
        def success(r):
            return details[method][r["task_id"]]["answer"]["strict_task_success"] if details else None

        def group(selected):
            ok = [success(r) for r in selected]
            lat = [r["latency_ms"] for r in selected if r.get("latency_ms") is not None]
            return {"n": len(selected),
                    "strict_success": sum(bool(x) for x in ok) if details else None,
                    "mean_steps": round(statistics.mean(r["steps"] for r in selected), 2) if selected else None,
                    "p50_s": round(statistics.median(lat) / 1000, 1) if lat else None,
                    "p95_s": round(nearest_rank(lat, 0.95) / 1000, 1) if lat else None,
                    "execution_failures": sum(r["status"] == "execution_failed" for r in selected)}

        by_type = {}
        for r in rows:
            by_type.setdefault(tasks[r["task_id"]]["dynamic_type"], []).append(r)
        controls = [r for r in rows if tasks[r["task_id"]]["dynamic_type"] == "control"]
        out[method] = {
            "all": group(rows),
            "needs_dynamic": group([r for r in rows if need.get(r["task_id"])]),
            "one_shot_sufficient": group([r for r in rows if need.get(r["task_id"]) is False]),
            "by_type": {k: group(v) for k, v in sorted(by_type.items())},
            "control_over_planning": sum(bool(r.get("over_planned")) for r in controls),
        }
    args.output.write_text(json.dumps({"raw": str(args.raw), "reviewed": bool(details), "methods": out},
                                      ensure_ascii=False, indent=2) + "\n")
    for method, m in out.items():
        print(method, "all", m["all"]["strict_success"], "/", m["all"]["n"], "| needs_dynamic",
              m["needs_dynamic"]["strict_success"], "/", m["needs_dynamic"]["n"], "| steps", m["all"]["mean_steps"],
              "| p50/p95", m["all"]["p50_s"], m["all"]["p95_s"], "| over-planning", m["control_over_planning"])


if __name__ == "__main__":
    main()
