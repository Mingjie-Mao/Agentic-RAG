"""Where the lexical proxies and the semantic evaluator disagree, and what happened next.

Reads one Hard-set replay run with `RAG_SEMANTIC_COVERAGE_SHADOW=true`. The run's
behaviour was decided by the lexical proxies; the semantic judgements were only
recorded. For each task this lines up, at the point where the Agent stopped:

  lexical said ready / not ready   vs   semantic said ready / not ready
  how many evidence chunks the lexical filter dropped, and how the relevance scorer
  rated them
  the task's outcome (success, answer, over-planning, early stop)
  the judge's cost

This is a regression-set diagnosis (Hard 30 is development data). It is not used to
choose thresholds, and it is not evidence of generalisation.
"""

import argparse
import collections
import json
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models import AgentEvent, AgentTask

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="artifacts/agent-hard-hybrid-qwen25-7b-shadow-20261001.json")
    parser.add_argument("--out", default="artifacts/semantic-shadow-analysis-20261001.json")
    args = parser.parse_args()
    run = json.loads((ROOT / args.run).read_text())
    rows = [r for r in run["results"]["hybrid"] if not r.get("skipped")]
    tasks, totals = [], collections.Counter()
    with SessionLocal() as db:
        for r in rows:
            events = db.scalars(select(AgentEvent).where(AgentEvent.task_id == r["run_id"]).order_by(AgentEvent.sequence)).all()
            checks = [e.payload for e in events if e.event_type == "semantic_coverage_shadow"]
            dropped = [e.payload for e in events if e.event_type == "semantic_filter_shadow"]
            stop = next((e.payload.get("reason") for e in events if e.event_type == "hybrid_stop"), None)
            last = checks[-1] if checks else None
            usage = collections.Counter()
            for c in checks:
                usage.update({k: v for k, v in c["judge_usage"].items() if isinstance(v, (int, float))})
            final_lexical = last["lexical_ready"] if last else None
            final_semantic = last["semantic_ready"] if last else None
            if last is None:
                pattern = "no_check"
            elif final_lexical == final_semantic:
                pattern = "agree_ready" if final_lexical else "agree_not_ready"
            else:
                pattern = "lexical_ready_semantic_not" if final_lexical else "semantic_ready_lexical_not"
            stored = db.get(AgentTask, r["run_id"])
            claim_scores = (((stored.result or {}) if stored else {}).get("shadow_scores") or {})
            task = {
                "task_id": r["task_id"], "category": r["category"], "controlled": r["controlled"],
                "task_success": r["task_success"], "answer_correct": r["answer_correct"],
                "over_planned": r["over_planned"], "early_stop": r["early_stop"], "stop_reason": stop,
                "checks": len(checks), "pattern_at_stop": pattern,
                "subgoal_disagreements_at_stop": len(last["disagreements"]) if last else 0,
                "semantic_report_at_stop": last["report"] if last else None,
                "lexically_dropped": sum(d["lexically_dropped"] for d in dropped),
                "dropped_semantic_labels": dict(sum((collections.Counter(d["semantic_labels_of_dropped"]) for d in dropped),
                                                    collections.Counter())),
                "judge_calls": usage["calls"], "judge_ms": round(usage["wall_ms"], 1),
                "judge_prompt_tokens": usage["prompt_tokens"],
                "claim_support": claim_scores.get("semantic"),
            }
            tasks.append(task)
            totals[pattern] += 1

    def outcome(pattern):
        group = [t for t in tasks if t["pattern_at_stop"] == pattern]
        return {"tasks": len(group), "success": sum(t["task_success"] for t in group),
                "ids": [t["task_id"] for t in group]}

    dropped_relevant = [t["task_id"] for t in tasks if t["dropped_semantic_labels"].get("relevant")]
    output = {
        "run": args.run,
        "note": "Hard 30 is a development regression set; diagnosis only, no threshold is chosen here.",
        "pattern_at_stop": {p: outcome(p) for p in sorted(totals)},
        "evidence_filter": {
            "tasks_with_dropped_chunks": sum(1 for t in tasks if t["lexically_dropped"]),
            "chunks_dropped": sum(t["lexically_dropped"] for t in tasks),
            "dropped_by_semantic_label": dict(sum((collections.Counter(t["dropped_semantic_labels"]) for t in tasks),
                                                  collections.Counter())),
            "tasks_where_a_highly_relevant_chunk_was_dropped": dropped_relevant,
        },
        "judge_cost": {
            "tasks": len(tasks), "calls": sum(t["judge_calls"] for t in tasks),
            "calls_per_task": round(sum(t["judge_calls"] for t in tasks) / len(tasks), 2),
            "seconds_per_task": round(sum(t["judge_ms"] for t in tasks) / len(tasks) / 1000, 1),
            "prompt_tokens_per_task": round(sum(t["judge_prompt_tokens"] for t in tasks) / len(tasks), 1),
        },
        "tasks": tasks,
    }
    (ROOT / args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "tasks"}, ensure_ascii=False, indent=2))
    for t in tasks:
        print(t["task_id"], t["category"][:12].ljust(12), "ok" if t["task_success"] else "FAIL",
              t["stop_reason"], t["pattern_at_stop"], t["semantic_report_at_stop"], "dropped", t["dropped_semantic_labels"])


if __name__ == "__main__":
    main()
