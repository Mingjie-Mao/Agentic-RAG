"""Replay shadow on committed development runs and prove stored primary state is unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_runtime import atomic_json  # noqa: E402
from research_review import digest  # noqa: E402


def main():
    from sqlalchemy import select
    from app.db import SessionLocal
    from app.models import AgentTask, AgentEvent, ToolExecution
    from app.slot_shadow import observe_completed

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/agent-hard-hybrid-qwen25-7b-shadow-20261001.json")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("results immutable")
    source = json.loads(args.source.read_text())
    ids = [r["run_id"] for r in source["results"]["hybrid"] if r.get("run_id")][:args.limit]
    results = []
    def primary(db, tid):
        task = db.get(AgentTask, tid, populate_existing=True)
        tools = db.scalars(select(ToolExecution).where(ToolExecution.task_id == tid).order_by(ToolExecution.id)).all()
        events = db.scalars(select(AgentEvent).where(AgentEvent.task_id == tid).order_by(AgentEvent.sequence)).all()
        return {"status": task.status, "result": task.result, "input": task.input,
                "tools": [{"id": t.id, "name": t.tool_name, "arguments": t.arguments, "result": t.result} for t in tools],
                "events": [{"id": e.id, "type": e.event_type, "refs": e.evidence_chunk_ids, "payload": e.payload} for e in events]}
    for tid in ids:
        with SessionLocal() as db:
            task = db.get(AgentTask, tid)
            if not task or task.status != "completed" or task.tenant_id != "xingqiao":
                raise ValueError("only completed, already-used Chinese Hard development runs")
            before = digest(primary(db, tid))
            observation = observe_completed("agent", tid, task.user_id)
            db.expire_all()
            unchanged = digest(primary(db, tid)) == before
            if not unchanged:
                raise RuntimeError("shadow mutated primary output/budget/tools/evidence")
            results.append({"run_id": tid, "primary_sha256": before, "primary_unchanged": unchanged,
                            "shadow": observation})
    atomic_json(args.output, {"version": "slot-shadow-v2-development-smoke", "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                              "tasks": len(results), "isolation_passed": bool(results) and all(r["primary_unchanged"] for r in results),
                              "quality_claim": None, "results": results})
    print(json.dumps({"tasks": len(results), "isolation_passed": True}, indent=2))


if __name__ == "__main__":
    main()
