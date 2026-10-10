"""Dev47 input-only gate for lossless JSON compression; never score Core/outputs.

Capture the first generation request after real authorized retrieval/preparation,
without executing the generator. Raw third-party text stays in ignored .runtime.
"""
import hashlib
import json
from pathlib import Path
import statistics
from unittest.mock import patch
from uuid import uuid4

import tiktoken

from app.clients import Models
from app.db import SessionLocal
from app.models import User
from app.qa import answer_question
from scripts.benchmark_runtime import atomic_json

ROOT = Path(__file__).resolve().parents[1]


class Captured(BaseException):
    pass


def main():
    suite_path = ROOT / ".runtime/benchmark-package/v1/dev/tasks.json"
    suite = json.loads(suite_path.read_text())
    if suite["split"] != "dev" or len(suite["tasks"]) != 47:
        raise ValueError("Only the complete Dev47 suite is allowed")
    run_id = str(uuid4())
    directory = ROOT / ".runtime/demo-performance" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    encoder = tiktoken.get_encoding("cl100k_base")
    rows, requests = [], []
    registration = {"kind": "dev_generation_input_probe_not_quality_benchmark", "split": "dev",
                    "task_ids": [t["id"] for t in suite["tasks"]],
                    "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                    "minimum_mean_token_reduction": 0.15,
                    "token_counter": "cl100k_base_estimate_not_Qwen_actual",
                    "policy": "No removal of source text, scope, conditions or instructions; no Core"}
    atomic_json(directory / "registration.json", registration)
    def capture(_, body):
        requests.append(json.loads(json.dumps(body)))
        raise Captured()
    try:
        with SessionLocal() as db, patch.object(Models, "_chat", capture):
            for task in suite["tasks"]:
                before = len(requests)
                user = db.get(User, task["user"])
                if user is None:
                    raise ValueError("Dev identity missing: " + task["user"])
                try:
                    answer_question(db, user, task["goal"], benchmark_run_token=run_id)
                except Captured:
                    pass
                row = {"task_id": task["id"], "generation_requested": len(requests) > before}
                if len(requests) > before:
                    body = requests[-1]
                    original = body["messages"][-1]["content"]
                    compact = json.dumps(json.loads(original), ensure_ascii=False, separators=(",", ":"))
                    row.update(original_tokens_estimate=len(encoder.encode(original)),
                               compact_tokens_estimate=len(encoder.encode(compact)),
                               lossless=json.loads(original) == json.loads(compact))
                rows.append(row)
                atomic_json(directory / "requests.json", requests)
                atomic_json(directory / "rows.json", rows)
                if len(rows) % 10 == 0:
                    print(f"Dev inputs captured {len(rows)}/47; no generation calls", flush=True)
        compared = [r for r in rows if r["generation_requested"]]
        reductions = [1 - r["compact_tokens_estimate"] / r["original_tokens_estimate"] for r in compared]
        reduction = statistics.mean(reductions) if reductions else None
        report = {**registration, "completed_tasks": len(rows), "generation_calls": 0,
                  "compared_inputs": len(compared), "mean_estimated_token_reduction": reduction,
                  "all_json_values_preserved": all(r["lossless"] for r in compared),
                  "latency_ms": None, "quality_score": None,
                  "decision": "eligible_for_paired_generation_test" if reduction is not None and reduction >= 0.15
                  else "reject_before_generation_insufficient_input_reduction",
                  "rows": rows, "local_raw_capture": str(directory.relative_to(ROOT))}
        output = ROOT / "artifacts" / f"demo-prompt-dev-{run_id}.json"
        with output.open("x") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(json.dumps({"report": str(output), "decision": report["decision"],
                          "mean_estimated_token_reduction": reduction}), flush=True)
    except BaseException as exc:
        atomic_json(directory / "failure.json", {"type": type(exc).__name__, "completed_tasks": len(rows)})
        raise


if __name__ == "__main__":
    main()
