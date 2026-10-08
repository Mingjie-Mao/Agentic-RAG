"""Finalize P3/P4 only from bound raw results and independent semantic reviews."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_runtime import atomic_json  # noqa: E402
from paired_semantic_replay import code_inputs, cost_gate, require_gate  # noqa: E402
from research_review import digest, validated_labels  # noqa: E402
from unseen_v2_scoring import quality_gate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("p3", "p4"))
    for field in ("results", "suite", "answer_review", "output"):
        p.add_argument("--" + field.replace("_", "-"), type=Path, required=True)
    p.add_argument("--p2-gate", type=Path)
    p.add_argument("--disagreement-source", type=Path)
    p.add_argument("--disagreement-review", type=Path)
    p.add_argument("--candidate", default="hybrid")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("gate exists; reports are immutable")
    raw, suite, review = [json.loads(path.read_text()) for path in (args.results, args.suite, args.answer_review)]
    candidate = "hybrid_semantic" if args.phase == "p3" else args.candidate
    baseline = "hybrid" if args.phase == "p3" else "workflow"
    quality = quality_gate(raw, suite, review, candidate=candidate, baseline=baseline)
    frozen = json.loads((ROOT / raw["freeze"]).read_text())
    frozen_code = {name: sha for name, sha in frozen["inputs"].items() if name.endswith(".py")}
    if frozen_code != code_inputs() or raw["configuration"] != frozen["configuration"]:
        raise ValueError("raw result snapshot/configuration differs from gate code")
    if not any(r.get("controlled") for rows in raw["results"].values() for r in rows):
        raise ValueError("controlled revocation/ACL safety evidence required")
    cost = cost_gate(raw["results"][baseline], raw["results"][candidate])
    files = [args.results, args.suite, args.answer_review]
    if args.phase == "p3":
        if not all((args.p2_gate, args.disagreement_source, args.disagreement_review)):
            raise ValueError("P3 requires passed P2 gate and independent disagreement review")
        p2 = require_gate(args.p2_gate, "semantic-slot-v2-gate")
        disagreements, reviewed = [json.loads(p.read_text()) for p in (args.disagreement_source, args.disagreement_review)]
        labels = validated_labels(disagreements, reviewed)
        if any(label == "unclear" for label in labels.values()):
            raise ValueError("unresolved shadow disagreements cannot activate control")
        files += [args.p2_gate, args.disagreement_source, args.disagreement_review]
        # P3 requires non-regression, P4 additionally needs a positive +5pp gain.
        quality["checks"]["quality"] = all(r["difference"] >= 0 and r["ci"][0] is not None and r["ci"][0] >= 0
                                               for r in quality["reports"].values())
        model_signature = p2["model_signature"]
    else:
        model_signature = raw["configuration"]["models"]
    result = {"version": "semantic-paired-v2-gate" if args.phase == "p3" else "unseen-v2-final-gate",
              "passed": all(quality["checks"].values()) and cost["passed"], "quality": quality, "cost": cost,
              "code_inputs": code_inputs(), "model_signature": model_signature,
              "evidence_files": {path.resolve().relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
              "decision": "enable_candidate" if all(quality["checks"].values()) and cost["passed"] else "do_not_enable",
              "raw_sha256": digest(raw)}
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
