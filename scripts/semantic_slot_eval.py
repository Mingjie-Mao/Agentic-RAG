"""Human-calibrated v2 component evaluation: dev-only selection, frozen test metrics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_runtime import atomic_json  # noqa: E402
from research_review import binary, component_input_key, digest, macro_f1, validated_labels, validate_component_partitions  # noqa: E402


BASELINES = ("lexical", "bge_only", "rule_b", "slot_v2")


def distinct_inputs(source, labels):
    """Repeated runs of identical claims are review rows, not new samples."""
    validate_component_partitions(source["items"])
    unique = {}
    for row in source["items"]:
        key = component_input_key(row)
        previous = unique.get(key)
        if previous:
            if (previous["family_id"], previous["split"]) != (row["family_id"], row["split"]):
                raise ValueError("identical component inputs cross source families/splits")
            if labels[previous["id"]] != labels[row["id"]]:
                raise ValueError("duplicate input labels disagree; independent adjudication required")
        else:
            unique[key] = row
    return list(unique.values())


def require_relevance_scores(rows, scores):
    for row in rows:
        value = scores.get(row["id"], {}).get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("complete finite relevance scores required; interrupted attempts are not scores")


def select_thresholds(source, labels, scores):
    result = {}
    items = distinct_inputs(source, labels)
    require_relevance_scores([r for r in items if r["kind"] == "relevance" and r["split"] == "dev"], scores)
    for corpus in sorted({r["corpus"] for r in source["items"]}):
        rows = [r for r in items if r["kind"] == "relevance" and r["split"] == "dev"
                and r["corpus"] == corpus and labels[r["id"]] != "unclear"]
        if not rows or not any(labels[r["id"]] == "relevant" for r in rows):
            raise ValueError(f"no independently labelled relevance dev positives for {corpus}")
        thresholds = sorted({scores[r["id"]]["score"] for r in rows})
        valid = [t for t in thresholds if binary([(labels[r["id"]] == "relevant", scores[r["id"]]["score"] >= t)
                                                  for r in rows])["recall"] >= .95]
        keep = max(valid)
        high = [t for t in thresholds if t >= keep and (binary([(labels[r["id"]] == "relevant", scores[r["id"]]["score"] >= t)
                                                  for r in rows])["precision"] or 0) >= .95]
        if not high:
            raise ValueError(f"no dev high-precision relevance rule for {corpus}")
        result[corpus] = {"keep": keep, "high": min(high), "dev_n": len(rows)}
    return {"version": "semantic-slot-v2", "source_sha256": source["source_sha256"],
            "selection_split": "dev", "thresholds": result,
            "coverage_rule": "all_actual_quoted_refs_high; exact_quotes; scoped_conflicts; no_confidence",
            "claim_control": False}


def gain_ci(rows, scores, labels, *, trials=2000):
    blocks = {}
    for r in rows:
        blocks.setdefault(r["family_id"], []).append(r)
    if len(blocks) < 2:
        return [None, None]
    rng, keys, values = random.Random(42), list(blocks), []
    for _ in range(trials):
        sample = [r for k in rng.choices(keys, k=len(keys)) for r in blocks[k]]
        values.append(macro_f1([(labels[r["id"]], scores[r["id"]]["slot_v2"]) for r in sample])
                      - macro_f1([(labels[r["id"]], scores[r["id"]]["lexical"]) for r in sample]))
    values.sort()
    return [values[int(trials*.025)], values[int(trials*.975)]]


def evaluate(source, labels, predictions, calibration):
    if predictions.get("source_sha256") != source["source_sha256"] or predictions.get("calibration_sha256") != digest(calibration):
        raise ValueError("predictions must bind exact reviewed inputs and frozen dev calibration")
    scores = predictions["rows"]
    if set(scores) != {r["id"] for r in source["items"]}:
        raise ValueError("incomplete or extra predictions")
    items = distinct_inputs(source, labels)
    require_relevance_scores([r for r in items if r["kind"] == "relevance"], scores)
    corpora, reasons = {}, []
    for corpus in sorted({r["corpus"] for r in source["items"]}):
        test = [r for r in items if r["corpus"] == corpus and r["split"] == "test"]
        rel = [r for r in test if r["kind"] == "relevance" and labels[r["id"]] != "unclear"]
        cut = calibration["thresholds"][corpus]["keep"]
        relevance = binary([(labels[r["id"]] == "relevant", scores[r["id"]]["score"] >= cut) for r in rel])
        cov = [r for r in test if r["kind"] == "coverage" and labels[r["id"]] != "unclear"]
        baselines = {}
        for arm in BASELINES:
            rows = [(labels[r["id"]], scores[r["id"]][arm]) for r in cov]
            baselines[arm] = {"supported": binary([(g == "supported", p == "supported") for g, p in rows]),
                              "macro_f1": macro_f1(rows),
                              "by_label": {label: binary([(g == label, p == label) for g, p in rows])
                                           for label in ("supported", "partial", "unsupported", "contradicted")},
                              "unknown": sum(p == "unknown" for _, p in rows)}
        gain = baselines["slot_v2"]["macro_f1"] - baselines["lexical"]["macro_f1"]
        ci = gain_ci(cov, scores, labels)
        support = baselines["slot_v2"]["supported"]
        counts = Counter(r["kind"] for r in items if r["corpus"] == corpus)
        # Point thresholds alone do not establish the proposed high precision gate.
        checks = {"relevance_recall": (relevance["recall_ci"][0] or 0) >= .95,
                  "coverage_precision": (support["precision_ci"][0] or 0) >= .95,
                  "coverage_recall": (support["recall_ci"][0] or 0) >= .80,
                  "macro_f1_gain": gain >= .10 and ci[0] is not None and ci[0] > 0,
                  "partial_test_examples": any(labels[r["id"]] == "partial" for r in cov),
                  "conflict_test_examples": any(labels[r["id"]] == "contradicted" for r in cov)}
        if not all(checks.values()):
            reasons.extend(f"{corpus}:{key}" for key, value in checks.items() if not value)
        claims = [r for r in test if r["kind"] == "claim" and labels[r["id"]] != "unclear"]
        corpora[corpus] = {"available_counts": dict(counts), "test_n": len(test), "relevance": relevance,
                           "coverage": baselines, "macro_f1_gain": gain, "macro_f1_gain_ci": ci,
                           "claim": {"n": len(claims), "four_class_macro_f1": macro_f1([
                               (labels[r["id"]], scores[r["id"]]["slot_v2"]) for r in claims]),
                               "runtime_interception": False}, "checks": checks,
                           "unclear": dict(Counter(r["kind"] for r in test if labels[r["id"]] == "unclear"))}
    sizes = Counter(r["kind"] for r in items)
    if any(sizes[k] < n for k, n in (("relevance", 300), ("coverage", 240), ("claim", 120))):
        reasons.append("minimum_sample_size")
    return {"version": "semantic-slot-v2-gate", "passed": not reasons, "reasons": reasons,
            "source_sha256": source["source_sha256"], "calibration_sha256": digest(calibration),
            "prediction_sha256": digest(predictions), "independent_human_labels": len(labels),
            "distinct_input_counts": dict(sizes), "duplicate_review_rows": len(source["items"])-len(items),
            "corpora": corpora, "claim_control": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("calibrate", "evaluate"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reviewed", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("output exists; calibration and test reports are immutable")
    source, reviewed, predictions = [json.loads(path.read_text()) for path in (args.source, args.reviewed, args.predictions)]
    labels = validated_labels(source, reviewed)
    if predictions.get("source_sha256") != source["source_sha256"]:
        raise ValueError("score inputs differ")
    if args.command == "calibrate":
        result = select_thresholds(source, labels, predictions["rows"])
        result["review_sha256"] = digest(reviewed)
    else:
        if not args.calibration:
            raise ValueError("frozen dev calibration required")
        calibration = json.loads(args.calibration.read_text())
        if calibration["review_sha256"] != digest(reviewed):
            raise ValueError("labels changed after calibration")
        result = evaluate(source, labels, predictions, calibration)
        from paired_semantic_replay import code_inputs
        import hashlib
        if predictions.get("code_inputs") != code_inputs() or not predictions.get("model_signature"):
            raise ValueError("prediction code/runtime/model signature missing or stale")
        result.update(code_inputs=code_inputs(), model_signature=predictions["model_signature"],
                      evidence_files={path.resolve().relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (args.source, args.reviewed, args.predictions, args.calibration)})
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
