"""Paired ablation: does handing the model its own subgoal checklist help or hurt?

The Agent splits a compound goal into subgoals to decide what is still unanswered.
The tempting next step is to pass that list to the first generation as
`acceptance_items`. This measures that step instead of assuming it. Both arms use the
same workflow policy, the same tools and the same model settings; the only difference
is whether the first generation receives the splitter's list.

Arm A (`baseline`) is what ships: the first pass sees the question exactly as the
single-turn path does. Arm B (`checklist`) injects the subgoals. Facts are scored with
the Hard Benchmark's own matchers so the two are directly comparable.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import time

from agent.controller import create_task, run_task, task_payload
from agent.planner import subgoals
from app.clients import Models
from app.db import SessionLocal
from app.models import User


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = ["H01", "H02", "H04", "H05", "H13"]


def benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "run_agent_hard_benchmark", ROOT / "scripts/run_agent_hard_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChecklistModels(Models):
    """Inject the splitter's subgoals into the first generation of one task."""

    def __init__(self, goal):
        self.goal = goal

    def generate(self, question, evidence, memory_context=None, **options):
        if question == self.goal and "acceptance_items_override" not in options:
            options["acceptance_items_override"] = subgoals(self.goal)
        return super().generate(question, evidence, memory_context, **options)


def run_arm(db, user, task, arm, scorer):
    started = time.monotonic()
    models = ChecklistModels(task["goal"]) if arm == "checklist" else Models()
    row = create_task(db, user, task["goal"], "workflow", task.get("max_steps", 8), task.get("task_input", {}))
    run_task(db, user, row, models=models)
    payload = task_payload(db, user, row)["result"] or {}
    text = scorer.normalized(" ".join(claim.get("text", "") for claim in payload.get("claims", [])))
    facts = {
        (row_.get("id") if isinstance(row_, dict) else str(row_)): scorer.fact_matches(row_, text)
        for row_ in (task.get("fact_matchers") or task.get("facts", []))
    }
    return {
        "task_id": task["id"],
        "arm": arm,
        "run_id": row.id,
        "subgoals": subgoals(task["goal"]),
        "status": payload.get("status"),
        "facts": facts,
        "facts_covered": sum(facts.values()),
        "facts_total": len(facts),
        "all_facts": all(facts.values()),
        "claims": [claim.get("text") for claim in payload.get("claims", [])],
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--out", default="artifacts/checklist-ab.json")
    args = parser.parse_args()
    scorer = benchmark_module()
    suite = json.loads((ROOT / "fixtures/agent/hard_tasks.json").read_text())
    tasks = [row for row in suite["tasks"] if row["id"] in set(args.ids)]
    results = []
    with SessionLocal() as db:
        for task in tasks:
            user = db.get(User, task["user"])
            # Counterbalanced within a task so a warm model cache cannot favour one arm.
            order = ["baseline", "checklist"] if len(results) % 4 == 0 else ["checklist", "baseline"]
            for arm in order:
                print(f"{task['id']} {arm} ...", flush=True)
                results.append(run_arm(db, user, task, arm, scorer))
    by_arm = {
        arm: {
            "tasks": len([row for row in results if row["arm"] == arm]),
            "all_facts": sum(row["all_facts"] for row in results if row["arm"] == arm),
            "facts_covered": sum(row["facts_covered"] for row in results if row["arm"] == arm),
            "facts_total": sum(row["facts_total"] for row in results if row["arm"] == arm),
        }
        for arm in ("baseline", "checklist")
    }
    output = {
        "benchmark": suite["version"],
        "arms": by_arm,
        "results": results,
        "note": (
            "同一组题、同一 workflow、同一模型设置；唯一差别是首轮生成是否收到子目标清单。"
            "样本量小，只用于判断该改动是否安全，不作为泛化成绩。"
        ),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(by_arm, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
