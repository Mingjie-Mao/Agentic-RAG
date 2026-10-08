"""Finish the one-shot unseen run after an infrastructure hang, without rerunning it.

The first run (scripts/run_unseen_benchmark.py) completed 133 of 140 arm-task runs and
then blocked for hours on a lost Ollama request (hybrid-semantic U29); it writes its
output only at the end, so nothing was saved. Nothing was changed between that run and
this recovery: the freeze is checked again here.

Completed runs are rebuilt from what the run stored in the database (RAG answers,
Agent task results, tool executions, events) and scored with the same scorer. Every
rebuilt score is checked against the success line the original run printed; any
mismatch aborts. The unseen tasks have no scenario events, so the stored rows are all
the scorer needs. Latency is the one field not stored: it is taken from database
timestamps and marked as such.

The "hang" was not infrastructure: on U29 the frozen Hybrid loop never terminates (the
policy repeats an inspect_versions intent the binder rejects as already checked; the
rejection uses no step and is counted as progress). The original run made 1,636 judge
calls on it and a second attempt looped the same way. That run is scored as what it is,
an execution failure (non-termination). The remaining hybrid-semantic tasks run with the
frozen code under a wall-clock guard (TASK_LIMIT_S, over 3x the slowest completed run);
a task that hits it is likewise an execution failure.
"""

import json
import re
import signal
import sys
from pathlib import Path

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
LOG = ROOT / ".runtime/unseen-run.log"
TASK_LIMIT_S = 600


class NonTermination(Exception):
    pass


def _alarm(_signum, _frame):
    raise NonTermination(f"task exceeded {TASK_LIMIT_S}s")

import run_unseen_benchmark as frozen  # noqa: E402


def main():
    from app.config import settings
    from app.db import SessionLocal
    from app.models import AgentTask, Answer, User
    from agent.controller import task_payload
    from run_agent_hard_benchmark import _tool_trace, run_arm, score_task

    suite = json.loads(frozen.TASKS.read_text())
    if frozen.OUT.exists():
        raise SystemExit(f"{frozen.OUT.relative_to(ROOT)} exists")
    freeze = json.loads(frozen.FREEZE.read_text())
    changed = [p for p, d in frozen.inputs().items() if freeze["inputs"].get(p) != d]
    if changed or frozen.configuration() != freeze["configuration"]:
        raise SystemExit(f"inputs differ from the freeze: {changed or 'configuration'}")

    printed = {}
    for line in LOG.read_text().split("\n"):
        if m := re.match(r"\[(\w+)\] (U\d+) success=(True|False)$", line):
            printed[(m[1], m[2])] = m[3] == "True"
    tasks = suite["tasks"]
    by_id = {t["id"]: t for t in tasks}
    results = {arm: [] for arm in frozen.ARMS}
    with SessionLocal() as db:
        answers = db.scalars(select(Answer).where(Answer.tenant_id == "lanting").order_by(Answer.created_at)).all()
        runs = db.scalars(select(AgentTask).where(AgentTask.tenant_id == "lanting").order_by(AgentTask.created_at)).all()
        workflow = [r for r in runs if r.mode == "workflow"]
        hybrid = [r for r in runs if r.mode == "hybrid"]
        if len(answers) != len(tasks) or len(workflow) != len(tasks):
            raise SystemExit(f"unexpected stored rows: {len(answers)} answers, {len(workflow)} workflow runs")
        stored = {"workflow": workflow, "hybrid": hybrid[:len(tasks)], "hybrid_semantic": hybrid[len(tasks):]}
        completed = sum(1 for k in printed if k[0] == "hybrid_semantic")
        hung = stored["hybrid_semantic"][completed:completed + 1]
        retried = [r.id for r in stored["hybrid_semantic"][completed + 1:]]

        previous = None
        for task, answer in zip(tasks, answers):
            if answer.question != task["goal"]:
                raise SystemExit(f"rag {task['id']}: stored question differs")
            payload = {"id": answer.id, "question": answer.question, **answer.payload}
            latency = (answer.created_at - previous).total_seconds() * 1000 if previous else None
            previous = answer.created_at
            row = score_task(task, payload, [], 1, latency or 0, observable_payload=payload, scenario_events={})
            row.update({"task_id": task["id"], "run_id": None, "latency_source": "db_timestamps"})
            if latency is None:
                row["latency_ms"], row["latency_source"] = None, "unrecoverable"
            results["rag"].append(row)

        for arm in ("workflow", "hybrid", "hybrid_semantic"):
            for run in stored[arm]:
                if run.id in retried:
                    continue
                task = by_id[run.input["benchmark_id"]]
                user = db.get(User, task["user"])
                observable = task_payload(db, user, run)
                payload = observable["result"] or {"status": "execution_failed", "claims": [], "citations": []}
                latency = (run.updated_at - run.created_at).total_seconds() * 1000
                row = score_task(task, payload, _tool_trace(db, run.id), observable["step_no"], latency,
                                 observable_payload=observable, scenario_events={})
                row.update({"task_id": task["id"], "run_id": run.id, "latency_source": "db_timestamps"})
                if arm == "hybrid_semantic":
                    row["judge_usage"] = frozen.judge_usage(db, run.id)
                results[arm].append(row)

        loop = results["hybrid_semantic"].pop() if hung else None
        mismatched = [(arm, r["task_id"]) for arm, rows in results.items() for r in rows
                      if printed.get((arm, r["task_id"])) != r["task_success"]]
        if mismatched or sum(len(rows) for rows in results.values()) != len(printed):
            raise SystemExit(f"rebuilt scores differ from the original run: {mismatched}")
        print(f"rebuilt {len(printed)} runs; every score matches the original run's output", flush=True)

        if loop is not None:
            task = by_id[loop["task_id"]]
            checks = frozen.judge_usage(db, hung[0].id).get("calls", 0)
            failed = {"status": "execution_failed", "claims": [], "citations": []}
            row = score_task(task, failed, loop["tool_trace"], loop["steps"], 0, scenario_events={},
                             execution_error_kind="unexpected")
            row.update({"task_id": task["id"], "run_id": hung[0].id, "latency_source": "unrecoverable",
                        "execution_error": f"NonTermination: policy/binder loop, {checks} judge calls before stop",
                        "judge_usage": frozen.judge_usage(db, hung[0].id)})
            results["hybrid_semantic"].append(row)
            print(f"[hybrid_semantic] {task['id']} success=False (non-termination, {checks} judge calls)", flush=True)

        done = {r["task_id"] for r in results["hybrid_semantic"]}
        signal.signal(signal.SIGALRM, _alarm)
        settings().semantic_coverage_control = True
        for task in tasks:
            if task["id"] in done:
                continue
            signal.alarm(TASK_LIMIT_S)
            try:
                row = run_arm(db, db.get(User, task["user"]), task, "hybrid")
            finally:
                signal.alarm(0)
            row["judge_usage"] = frozen.judge_usage(db, row["run_id"])
            row["latency_source"] = "measured"
            results["hybrid_semantic"].append(row)
            print(f"[hybrid_semantic] {task['id']} success={row.get('task_success')}", flush=True)
        settings().semantic_coverage_control = False

    categories = sorted({t["category"] for t in tasks})
    output = {
        "benchmark": suite["version"], "freeze": str(frozen.FREEZE.relative_to(ROOT)), "arms": frozen.ARMS,
        "recovery": {
            "reason": "the frozen Hybrid loop did not terminate on hybrid_semantic U29; the runner saves only at the end",
            "rebuilt_from_db": len(printed), "rerun": [r["task_id"] for r in results["hybrid_semantic"]
                                                      if r.get("latency_source") == "measured"],
            "non_terminating_run": hung[0].id if hung else None, "discarded_retry_runs": retried,
            "task_limit_s": TASK_LIMIT_S,
            "latency": "rebuilt runs: database timestamps (agent: task created->updated; rag: gap between "
                       "consecutive answers, first one unrecoverable and recorded as 0)",
        },
        "summary": {arm: frozen.summarize(rows) for arm, rows in results.items()},
        "by_category": {arm: {c: frozen.summarize([r for r in rows if r["category"] == c]) for c in categories}
                        for arm, rows in results.items()},
        "results": results,
    }
    frozen.OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
