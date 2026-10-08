"""Paired prefill/decode probe: the same real Dev contexts on two Ollama servers.

Server A is the configured runtime; server B differs only by OLLAMA_FLASH_ATTENTION=1.
Calls alternate A/B and B/A so host-load drift hits both arms. This measures model
throughput and output drift only; it is not a task-success or end-to-end P95 result.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import httpx


def prompts(review_packet, limit):
    seen, out = set(), []
    for item in json.loads(Path(review_packet).read_text())["items"]:
        task = item["task"]
        evidence = item.get("retrieved_evidence") or []
        if task["id"] in seen or not evidence:
            continue
        context = "\n\n".join(f"[E{i}] {row['title']}\n{row['text']}" for i, row in enumerate(evidence, 1))
        out.append((task["id"], f"Evidence:\n{context}\n\nQuestion: {task['goal']}\nAnswer with cited claims."))
        seen.add(task["id"])
    # Longest contexts first: they dominate the measured tail.
    return sorted(out, key=lambda p: -len(p[1]))[:limit]


def call(base, model, prompt):
    started = time.monotonic()
    r = httpx.post(f"{base}/api/generate", timeout=600, json={
        "model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 200}}).json()
    return {"wall_ms": round((time.monotonic() - started) * 1000, 1), "prompt_tokens": r["prompt_eval_count"],
            "prefill_ms": round(r["prompt_eval_duration"] / 1e6, 1), "decode_tokens": r["eval_count"],
            "decode_ms": round(r["eval_duration"] / 1e6, 1), "load_ms": round(r.get("load_duration", 0) / 1e6, 1),
            "response": r["response"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packet", required=True)
    p.add_argument("--baseline", default="http://127.0.0.1:11436")
    p.add_argument("--candidate", default="http://127.0.0.1:11438")
    p.add_argument("--model", default="qwen2.5:7b-instruct")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for base in (args.baseline, args.candidate):  # warm both; not measured
        call(base, args.model, "warm up")
    rows = []
    for index, (task_id, prompt) in enumerate(prompts(args.packet, args.limit)):
        order = [("baseline", args.baseline), ("candidate", args.candidate)]
        if index % 2:
            order.reverse()
        result = {name: call(base, args.model, prompt) for name, base in order}
        result["task_id"], result["order"] = task_id, [name for name, _ in order]
        result["same_output"] = result["baseline"]["response"] == result["candidate"]["response"]
        rows.append(result)
        print(task_id, {n: (result[n]["prompt_tokens"], result[n]["prefill_ms"], result[n]["decode_ms"])
                        for n in ("baseline", "candidate")}, "same", result["same_output"], flush=True)

    def rate(arm, tokens, ms):
        return round(sum(r[arm][tokens] for r in rows) / (sum(r[arm][ms] for r in rows) / 1000), 1)

    summary = {arm: {"prefill_tok_s": rate(arm, "prompt_tokens", "prefill_ms"),
                     "decode_tok_s": rate(arm, "decode_tokens", "decode_ms"),
                     "median_wall_ms": statistics.median(r[arm]["wall_ms"] for r in rows)}
               for arm in ("baseline", "candidate")}
    report = {"scope": __doc__.strip(), "baseline": args.baseline, "candidate": args.candidate, "model": args.model,
              "prompts": len(rows), "identical_outputs": sum(r["same_output"] for r in rows),
              "summary": summary, "rows": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2), "identical", report["identical_outputs"], "/", len(rows))


if __name__ == "__main__":
    main()
