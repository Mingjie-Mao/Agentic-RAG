"""P3 same-snapshot paired cost/quality checks, and gates used by P4/runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_runtime import atomic_json  # noqa: E402
from research_review import digest  # noqa: E402


def code_inputs(root=ROOT):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ("app", "agent", "scripts") for p in (root / name).rglob("*.py")}


def require_gate(path, version):
    data = json.loads(path.read_text())
    if data.get("version") != version or data.get("passed") is not True:
        raise ValueError(f"{version} not passed: semantic arm remains disabled")
    if not data.get("evidence_files") or not data.get("code_inputs") or not data.get("model_signature"):
        raise ValueError("gate missing bound review, code or model evidence")
    for name, sha in data["evidence_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError("gate evidence changed")
    if data["code_inputs"] != code_inputs():
        raise ValueError("gate code differs; do not apply an old gate to new code")
    if version == "semantic-slot-v2-gate":
        if data.get("independent_human_labels", 0) < 660 or not {"s3", "mh"} <= set(data.get("corpora", {})):
            raise ValueError("independent bilingual component review incomplete")
        if not all(all(c["checks"].values()) for c in data["corpora"].values()):
            raise ValueError("component corpus gate failed")
    return data


def cost_gate(lexical, semantic):
    lexical, semantic = [[r for r in rows if not r.get("skipped")] for rows in (lexical, semantic)]
    if not lexical or {r["task_id"] for r in lexical} != {r["task_id"] for r in semantic}:
        raise ValueError("paired task sets differ")
    def costs(rows):
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        budget = [r.get("usage", {}).get("execution_budget", {}).get("calls", {}) for r in rows]
        complete = len(lat) == len(rows) and all("calls" in r.get("usage", {}).get("execution_budget", {}) for r in rows)
        tokens = sum(v.get("prompt_tokens", 0)+v.get("completion_tokens", 0) for b in budget for v in b.values())
        return {"complete": complete, "p50": statistics.median(lat) if lat else None,
                "p95": sorted(lat)[min(len(lat)-1, math.ceil(len(lat)*.95)-1)] if lat else None,
                "tokens": tokens, "judge_attempts": sum(b.get("judge", {}).get("attempted", 0) for b in budget)}
    a, b = costs(lexical), costs(semantic)
    ratios = {key: b[key]/a[key] if a[key] and b[key] is not None else
              1.0 if key == "tokens" and a[key] == b[key] == 0 else None for key in ("p50", "p95", "tokens")}
    checks = {"complete_telemetry": a["complete"] and b["complete"],
              "judge_mean": b["judge_attempts"]/len(semantic) <= .5,
              **{key: ratios[key] is not None and ratios[key] <= limit for key, limit in (("p50", 1.25), ("p95", 1.30), ("tokens", 1.30))},
              "acl_zero": not any(any(r.get("forbidden_present", {}).values()) for r in lexical+semantic),
              "termination_zero": not any(r.get("status") == "execution_failed" and r.get("execution_error_kind") != "authorization_change" for r in lexical+semantic)}
    return {"passed": all(checks.values()), "checks": checks, "ratios": ratios,
            "lexical": a, "semantic": b, "judge_mean": b["judge_attempts"]/len(semantic)}


def disagreement_packet(paths):
    from research_review import packet
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        for slot in data.get("slots", []):
            lexical = data.get("lexical_covered", {}).get(slot["id"], False)
            judgment = next((j for j in data.get("judgments", []) if j["subgoal_id"] == slot["id"]), None)
            semantic = slot["id"] in data.get("report", {}).get("complete", [])
            if lexical != semantic:
                rows.append({"id": digest([data["run_id"], slot["id"]]), "kind": "disagreement",
                             "source_sha256": digest(data), "run_id": data["run_id"], "slot": slot,
                             "lexical_covered": lexical, "judgment": judgment,
                             "judge_context": data.get("judge_context", {}),
                             "raw_judgments": data.get("raw_judgments", []),
                             "categories": ["lexical_early_stop", "semantic_false_rejection", "splitter_error",
                                            "judge_context_omission", "citation_mismatch", "missing_evidence", "unclear"]})
    return packet(rows, "independent shadow disagreement review; categories are hypotheses")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("disagreements", "cost"))
    p.add_argument("--shadow-dir", type=Path, default=ROOT / ".runtime/slot-shadow")
    p.add_argument("--results", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("report exists; use a fresh output")
    if args.command == "disagreements":
        result = disagreement_packet(sorted(args.shadow_dir.rglob("*.json")))
    else:
        raw = json.loads(args.results.read_text())
        result = cost_gate(raw["results"]["hybrid"], raw["results"]["hybrid_semantic"])
        # Cost alone never passes P3. Independent disagreement and paired answer
        # review plus bound P2 gate are required to create an activation gate.
        result.update(version="semantic-paired-v2-cost", activation=False,
                      reason="cost_only_not_a_P3_quality_gate", raw_sha256=digest(raw))
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
