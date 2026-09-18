"""Paired long-term-memory ablation on the frozen Agent development suite.

The primary metric is enterprise-answer correctness. Memory contains only user
preferences and must therefore be non-inferior on facts and access control. The
two arms use the same workflow policy and model settings; task order is
counterbalanced to reduce warm-cache bias. Progress is written after every arm.
"""

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import statistics
import time

from agent.controller import create_task, run_task, task_payload
from app.db import SessionLocal
from app.models import AgentTask, User


ROOT = Path(__file__).resolve().parents[1]


def normalized(value):
    text = "".join(str(value).lower().split())
    chinese_digits = {"零": "0", "一": "1", "二": "2", "三": "3", "四": "4", "五": "5"}
    for chinese, digit in chinese_digits.items():
        text = text.replace(f"百分之{chinese}", f"{digit}%")
    return text


def score(task, payload, latency_ms):
    claims = payload.get("claims", [])
    text = normalized(" ".join(claim.get("text", "") for claim in claims))
    facts = {fact: normalized(fact) in text for fact in task["facts"]}
    forbidden = {fact: normalized(fact) in text for fact in task["forbidden"]}
    status_ok = payload.get("status") == task["expected_status"]
    usage = payload.get("usage", {})
    return {
        "status": payload.get("status"),
        "expected_status": task["expected_status"],
        "status_ok": status_ok,
        "facts": facts,
        "forbidden_present": forbidden,
        "correct": status_ok and all(facts.values()) and not any(forbidden.values()),
        "claim_count": len(claims),
        "citation_count": len(payload.get("citations", [])),
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
    }


def run_arm(db, user, task, use_memory):
    started = time.monotonic()
    row = create_task(
        db,
        user,
        task["goal"],
        "workflow",
        4,
        {
            "use_memory": use_memory,
            "benchmark_id": task["id"],
            "experiment": "memory-ab-v1",
        },
    )
    execution_error = None
    try:
        run_task(db, user, row)
    except Exception as exc:  # A dependency or protocol failure is an observed outcome.
        execution_error = f"{type(exc).__name__}: {exc}"
        db.expire_all()
    view = task_payload(db, user, row)
    payload = view["result"] or {
        "status": "execution_failed",
        "claims": [],
        "citations": [],
    }
    result = score(task, payload, (time.monotonic() - started) * 1000)
    result.update(
        {
            "task_id": task["id"],
            "category": task.get("category", "uncategorized"),
            "run_id": row.id,
            "steps": view["step_no"],
        }
    )
    if execution_error:
        result["execution_error"] = execution_error
    return result


def arm_summary(rows):
    if not rows:
        return {}
    return {
        "tasks": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": round(sum(row["correct"] for row in rows) / len(rows), 4),
        "status_accuracy": round(sum(row["status_ok"] for row in rows) / len(rows), 4),
        "forbidden_leaks": sum(any(row["forbidden_present"].values()) for row in rows),
        "median_latency_ms": round(statistics.median(row["latency_ms"] for row in rows), 1),
        "mean_prompt_tokens": round(statistics.mean(row["prompt_tokens"] for row in rows), 1),
        "mean_completion_tokens": round(
            statistics.mean(row["completion_tokens"] for row in rows), 1
        ),
    }


def exact_mcnemar_p(b, c):
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(0, min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_summary(results):
    off = {row["task_id"]: row for row in results["memory_off"]}
    on = {row["task_id"]: row for row in results["memory_on"]}
    paired = sorted(set(off) & set(on))
    off_only = sum(off[key]["correct"] and not on[key]["correct"] for key in paired)
    on_only = sum(on[key]["correct"] and not off[key]["correct"] for key in paired)
    return {
        "pairs": len(paired),
        "both_correct": sum(off[key]["correct"] and on[key]["correct"] for key in paired),
        "both_wrong": sum(not off[key]["correct"] and not on[key]["correct"] for key in paired),
        "memory_off_only_correct": off_only,
        "memory_on_only_correct": on_only,
        "accuracy_delta": round(
            (on_only - off_only) / len(paired), 4
        ) if paired else None,
        "mcnemar_exact_p": round(exact_mcnemar_p(off_only, on_only), 6),
        "changed_tasks": [
            key for key in paired if off[key]["correct"] != on[key]["correct"]
        ],
    }


def rescore_stored_runs(db, task_by_id, results):
    """Recompute metrics from persisted raw results after scorer changes."""
    for rows in results.values():
        for index, row in enumerate(rows):
            task = task_by_id[row["task_id"]]
            stored = db.get(AgentTask, row["run_id"])
            if stored is None or not stored.result:
                continue
            rescored = score(task, stored.result, row["latency_ms"])
            rows[index] = row | rescored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="fixtures/agent/tasks.json")
    parser.add_argument("--out", default="artifacts/memory-ab-v1.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="+", help="Run only these task ids")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    task_path = ROOT / args.tasks
    raw = task_path.read_bytes()
    suite = json.loads(raw)
    tasks = suite["tasks"]
    if args.ids:
        wanted = set(args.ids)
        tasks = [task for task in tasks if task["id"] in wanted]
        missing = wanted - {task["id"] for task in tasks}
        if missing:
            parser.error(f"Unknown task ids: {', '.join(sorted(missing))}")
    tasks = tasks[: args.limit]
    task_by_id = {task["id"]: task for task in tasks}
    out = ROOT / args.out
    results = {"memory_off": [], "memory_on": []}
    if out.exists() and not args.fresh:
        previous = json.loads(out.read_text())
        if previous.get("task_sha256") != hashlib.sha256(raw).hexdigest():
            raise SystemExit("existing output belongs to a different task suite; use --fresh")
        results = previous["results"]

    def payload():
        return {
            "experiment": "memory-ab-v1",
            "design": {
                "suite": suite["version"],
                "primary_metric": "paired exact task correctness",
                "memory_contract": "preferences only; never enterprise evidence",
                "order": "odd tasks memory_off first; even tasks memory_on first",
                "model_controls": "temperature=0, seed=42, workflow policy, max_steps=4",
            },
            "task_sha256": hashlib.sha256(raw).hexdigest(),
            "results": results,
            "summary": {arm: arm_summary(rows) for arm, rows in results.items()},
            "paired": paired_summary(results),
        }

    completed = {
        arm: {row["task_id"] for row in rows} for arm, rows in results.items()
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        for index, task in enumerate(tasks):
            user = db.get(User, task["user"])
            if user is None:
                raise RuntimeError(f"missing seeded user {task['user']}")
            order = ["memory_off", "memory_on"] if index % 2 == 0 else ["memory_on", "memory_off"]
            for arm in order:
                if task["id"] in completed[arm]:
                    continue
                print(f"{task['id']} {arm} ...", flush=True)
                results[arm].append(run_arm(db, user, task, arm == "memory_on"))
                completed[arm].add(task["id"])
                out.write_text(json.dumps(payload(), ensure_ascii=False, indent=2) + "\n")
        rescore_stored_runs(db, task_by_id, results)
    final = payload()
    out.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": final["summary"], "paired": final["paired"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
