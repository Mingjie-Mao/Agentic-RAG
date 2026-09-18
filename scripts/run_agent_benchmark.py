"""Run a paired, frozen benchmark across RAG and both Agent policies."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from agent.controller import create_task, run_task, task_payload
from app.db import SessionLocal
from app.models import User
from app.qa import answer_question


ROOT = Path(__file__).resolve().parents[1]


def normalized(value):
    text = "".join(str(value).lower().split())
    chinese_digits = {"零": "0", "一": "1", "二": "2", "三": "3", "四": "4", "五": "5"}
    for chinese, digit in chinese_digits.items():
        text = text.replace(f"百分之{chinese}", f"{digit}%")
    return text


def score(task, payload, latency_ms):
    text = normalized(
        " ".join(claim.get("text", "") for claim in payload.get("claims", []))
    )
    facts = {fact: normalized(fact) in text for fact in task["facts"]}
    forbidden = {fact: normalized(fact) in text for fact in task["forbidden"]}
    status_ok = payload.get("status") == task["expected_status"]
    return {
        "category": task.get("category", "uncategorized"),
        "status": payload.get("status"),
        "expected_status": task["expected_status"],
        "status_ok": status_ok,
        "facts": facts,
        "forbidden_present": forbidden,
        "correct": status_ok and all(facts.values()) and not any(forbidden.values()),
        "citation_count": len(payload.get("citations", [])),
        "latency_ms": round(latency_ms, 1),
        "usage": payload.get("usage", {}),
    }


def run_arm(db, user, task, arm):
    started = time.monotonic()
    if arm == "rag":
        payload = answer_question(db, user, task["goal"])
        steps = 1
        task_id = None
    else:
        row = create_task(
            db,
            user,
            task["goal"],
            arm,
            4 if arm == "workflow" else 6,
            {"use_memory": False, "benchmark_id": task["id"]},
        )
        execution_error = None
        try:
            run_task(db, user, row)
        except Exception as exc:  # A policy/protocol failure is a benchmark result.
            execution_error = f"{type(exc).__name__}: {exc}"
            db.expire_all()
        task_view = task_payload(db, user, row)
        payload = task_view["result"] or {
            "status": "execution_failed",
            "claims": [],
            "citations": [],
        }
        steps = task_view["step_no"]
        task_id = row.id
    result = score(task, payload, (time.monotonic() - started) * 1000)
    result.update({"task_id": task["id"], "run_id": task_id, "steps": steps})
    if arm != "rag" and execution_error:
        result["execution_error"] = execution_error
    return result


def summarize(rows):
    return {
        "tasks": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": round(sum(row["correct"] for row in rows) / len(rows), 3),
        "median_latency_ms": round(statistics.median(row["latency_ms"] for row in rows), 1),
        "total_latency_ms": round(sum(row["latency_ms"] for row in rows), 1),
        "mean_steps": round(statistics.mean(row["steps"] for row in rows), 2),
        "forbidden_leaks": sum(any(row["forbidden_present"].values()) for row in rows),
    }


def summarize_categories(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)
    return {category: summarize(items) for category, items in sorted(grouped.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="fixtures/agent/tasks.json")
    parser.add_argument("--arms", nargs="+", choices=["rag", "workflow", "dynamic"], default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="+", help="Run only these task ids")
    parser.add_argument("--out", default="artifacts/b1-agent-benchmark.json")
    args = parser.parse_args()
    arms = args.arms or ["rag", "workflow", "dynamic"]
    task_path = ROOT / args.tasks
    raw = task_path.read_bytes()
    suite = json.loads(raw)
    tasks = suite["tasks"]
    if args.ids:
        wanted = set(args.ids)
        tasks = [task for task in tasks if task["id"] in wanted]
        missing = wanted - {task["id"] for task in tasks}
        if missing:
            raise SystemExit(f"unknown task ids: {', '.join(sorted(missing))}")
    tasks = tasks[: args.limit]
    results = {arm: [] for arm in arms}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    def output_payload():
        return {
            "benchmark": suite["version"],
            "task_sha256": hashlib.sha256(raw).hexdigest(),
            "arms": arms,
            "results": results,
            "summary": {arm: summarize(rows) for arm, rows in results.items() if rows},
            "by_category": {
                arm: summarize_categories(rows) for arm, rows in results.items() if rows
            },
            "completed_tasks": min((len(rows) for rows in results.values()), default=0),
            "requested_tasks": len(tasks),
            "note": "开发基准，不是独立留出集；本地 Ollama 的 api_cost 为 0。",
        }

    with SessionLocal() as db:
        for task in tasks:
            user = db.get(User, task["user"])
            if user is None:
                raise RuntimeError(f"missing seeded user {task['user']}")
            for arm in arms:
                print(f"{task['id']} {arm} ...", flush=True)
                results[arm].append(run_arm(db, user, task, arm))
            out.write_text(json.dumps(output_payload(), ensure_ascii=False, indent=2) + "\n")
    output = output_payload()
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
