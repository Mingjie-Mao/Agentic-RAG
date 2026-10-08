"""Split measured task latency into model/tool stages from a benchmark checkpoint.

Only recorded telemetry is summed. Rows or calls without server timings are counted
as unknown instead of 0 ms, and every completed row (including failures) stays in P95.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def nearest_rank(values, q):
    # Standard nearest-rank percentile (ceil(q*n)), as used by the earlier old35 comparison.
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else None


def stages(row):
    usage = row.get("usage") or {}
    calls = (usage.get("execution_budget") or {}).get("model_call_records")
    out = {"total_ms": row.get("latency_ms"), "tool_ms": usage.get("tool_wall_ms")}
    if calls is None:
        # Plain RAG has no Agent budget; its generation timings are on usage itself.
        calls = [{"role": "generation", "wall_ms": usage.get("generation_wall_ms") or usage.get("model_duration_ms"),
                  **{k: usage.get(k) for k in ("prompt_tokens", "prompt_eval_ms", "completion_eval_ms")},
                  "load_ms": usage.get("model_load_ms"), "server_ms": usage.get("model_duration_ms")}]
    for role in ("policy", "generation", "judge"):
        mine = [c for c in calls if c.get("role") == role]
        out[f"{role}_calls"] = len(mine)
        out[f"{role}_wall_ms"] = sum(c.get("wall_ms") or 0 for c in mine)
        for key in ("prompt_eval_ms", "completion_eval_ms", "load_ms"):
            known = [c[key] for c in mine if c.get(key) is not None]
            out[f"{role}_{key}"] = sum(known) if len(known) == len(mine) and mine else None
        out[f"{role}_prompt_tokens"] = sum(c.get("prompt_tokens") or 0 for c in mine)
    if out["total_ms"] is not None:
        known = sum(out[f"{r}_wall_ms"] for r in ("policy", "generation", "judge")) + (out["tool_ms"] or 0)
        out["other_ms"] = round(out["total_ms"] - known, 1)
    return out


def summarize(rows):
    measured = [r for r in rows if r.get("latency_ms") is not None]
    latencies = [r["latency_ms"] for r in measured]
    p95 = nearest_rank(latencies, 0.95)
    tail = [stages(r) | {"key": r["key"]} for r in measured if r["latency_ms"] >= p95] if p95 else []
    return {"tasks": len(rows), "latency_observed": len(measured),
            "p50_ms": round(statistics.median(latencies), 1) if latencies else None,
            "p95_ms": p95, "tail_rows": sorted(tail, key=lambda r: -r["total_ms"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--task-prefix", default="", help="Restrict to task IDs with this prefix, e.g. U for old35")
    args = p.parse_args()
    completed = {k: v for k, v in json.loads(args.checkpoint.read_text())["completed"].items()
                 if k.split(":", 1)[1].startswith(args.task_prefix)}
    by_method = {}
    for key, row in completed.items():
        by_method.setdefault(key.split(":")[0], []).append(dict(row, key=key))
    every = [stages(r) for rows in by_method.values() for r in rows if r.get("latency_ms") is not None]
    gen = [s for s in every if s["generation_prompt_eval_ms"] and s["generation_prompt_tokens"]]
    report = {"source": str(args.checkpoint), "methods": {m: summarize(rows) for m, rows in sorted(by_method.items())},
              "generation_prefill_tokens_per_s": round(sum(s["generation_prompt_tokens"] for s in gen)
                                                        / (sum(s["generation_prompt_eval_ms"] for s in gen) / 1000), 1)
              if gen else None,
              "policy_calls_without_server_timing": sum(s["policy_calls"] for s in every if s["policy_prompt_eval_ms"] is None)}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    for method, s in report["methods"].items():
        print(method, "p50", s["p50_ms"], "p95", s["p95_ms"], "tail", [
            (t["key"], round(t["total_ms"] / 1000, 1), "gen", round(t["generation_wall_ms"] / 1000, 1),
             "prefill", t["generation_prompt_eval_ms"] and round(t["generation_prompt_eval_ms"] / 1000, 1),
             "policy", round(t["policy_wall_ms"] / 1000, 1)) for t in s["tail_rows"]])
    print("prefill tok/s", report["generation_prefill_tokens_per_s"],
          "policy calls lacking server timing", report["policy_calls_without_server_timing"])


if __name__ == "__main__":
    main()
