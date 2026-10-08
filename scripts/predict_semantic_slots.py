"""Generate auditable component predictions. Unreviewed inputs stay development only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_runtime import atomic_json, CheckpointStore, checkpoint_lock  # noqa: E402
from research_review import digest  # noqa: E402


def predict(source, *, calibration=None, limit=None, relevance_only=False, checkpoint=None, resume=False):
    from agent.planner import evidence_coverage
    from app.config import settings
    from app.execution_budget import ExecutionBudget, use_budget
    from app.semantic_evidence import SlotEvidenceEvaluator, SemanticEvidenceEvaluator, Thresholds
    from paired_semantic_replay import code_inputs
    from run_unseen_benchmark import configuration

    if source["source_sha256"] != digest(source["items"]):
        raise ValueError("frozen review input changed")
    cfg = settings()
    frozen = configuration()
    signature = frozen["models"]
    output = {"version": "semantic-slot-v2-predictions", "source_sha256": source["source_sha256"],
              "calibration_sha256": digest(calibration) if calibration else None,
              "scope": "calibrated" if calibration else "development_only_uncalibrated",
              "model_signature": signature, "code_inputs": code_inputs(), "rows": {}}
    selected = [r for r in source["items"] if not relevance_only or r["kind"] == "relevance"]
    if limit is not None:
        selected = selected[:limit]
    output["mode"] = "relevance_only" if relevance_only else "all_components"
    store = None
    if checkpoint:
        identity = {k: v for k, v in output.items() if k != "rows"}
        identity.update(configuration_sha256=digest(frozen), selected_ids=[r["id"] for r in selected])
        store = CheckpointStore(checkpoint, identity, resume=resume)
        output["rows"] = {r["task_id"]: {k: v for k, v in r.items() if k != "task_id"}
                          for r in store.data["completed"].values()}
    elif resume:
        raise ValueError("resume requires a checkpoint")
    evidence_manifest = {p["chunk_id"]: digest(p) for r in source["items"] for p in r["evidence"]}
    # Chunk metadata can differ by row; bind the entire row-local evidence content.
    reused = {}
    for index, row in enumerate(selected):
        passages, item = row["evidence"], row["slot"]
        reuse_key = digest([row["kind"], row["question"], row["corpus"], item, passages])
        if row["id"] in output["rows"]:
            previous = output["rows"][row["id"]]
            if not previous.get("error") and previous.get("slot_v2") != "unknown":
                reused.setdefault(reuse_key, previous.get("reused_from", row["id"]))
            continue
        pending_key = "components:" + row["id"]
        if store and pending_key in store.data["pending"]:
            saved = store.data["pending"][pending_key].get("execution_budget")
            result = {arm: "unknown" for arm in ("lexical", "bge_only", "rule_b", "slot_v2")}
            result.update(error="interrupted_prediction", execution_budget=ExecutionBudget(cfg, saved).snapshot())
            store.finish("components", row["id"], {"task_id": row["id"], **result})
            output["rows"][row["id"]] = result
            continue
        if store:
            store.begin("components", row["id"], digest([source["source_sha256"], row["id"]]))
        if reuse_key in reused:
            original_id = reused[reuse_key]
            # Frozen offline inputs must match byte-for-byte, including slot ID
            # and cited content. Copy verdicts, but never duplicate model costs.
            result = {**output["rows"][original_id], "reused_from": original_id,
                      "judge_usage": {}, "legacy_judge_usage": {},
                      "execution_budget": {"calls": {}}}
            if store:
                store.finish("components", row["id"], {"task_id": row["id"], **result})
            output["rows"][row["id"]] = result
            continue
        row_manifest = {p["chunk_id"]: digest(p) for p in passages}
        def authorize(p):
            return row_manifest.get(p["chunk_id"]) == digest(p)
        threshold = Thresholds()
        if calibration:
            cut = calibration["thresholds"][row["corpus"]]
            threshold = Thresholds(keep=cut["keep"], high=cut["high"], version="semantic-slot-v2")
        evaluator = SlotEvidenceEvaluator(authorize=authorize, thresholds=threshold, model_fingerprint=digest(signature))
        old = SemanticEvidenceEvaluator()  # Frozen v1 rule B keeps its original cuts.
        def persist(snapshot):
            if store:
                store.data["pending"][pending_key]["execution_budget"] = snapshot
                atomic_json(store.path, store.data)
        budget = ExecutionBudget(cfg.model_copy(update={"agent_task_timeout_seconds": 180, "agent_judge_max_calls": 2,
                                                        "agent_judge_token_budget": 48000}), persist=persist)
        result = {}
        try:
            with use_budget(budget):
                relevance = evaluator.relevance([item], passages)
                if row["kind"] == "relevance":
                    result = {"score": relevance[0].score, "lexical": bool(evidence_coverage([item["text"]], passages, row["question"]).get(item["text"]))}
                else:
                    judgments = evaluator.coverage(row["question"], [item], passages, relevance)
                    old_relevance = [r.model_copy(update={"label": "relevant" if r.score >= old.thresholds.high
                        else "uncertain" if r.score >= old.thresholds.keep else "irrelevant"}) for r in relevance]
                    legacy = old.coverage(row["question"], [item], passages, old_relevance)
                    lexical = bool(evidence_coverage([item["text"]], passages, row["question"]).get(item["text"]))
                    result = {"lexical": "supported" if lexical else "unsupported",
                              "bge_only": "supported" if any(r.score >= threshold.high for r in relevance) else "unsupported",
                              "rule_b": legacy[0].status, "slot_v2": judgments[0].status if judgments[0].evaluation_validity == "evaluated" else "unknown",
                              "judgment": judgments[0].model_dump(), "judge_usage": evaluator.usage.as_dict(),
                              "legacy_judge_usage": old.usage.as_dict()}
        except Exception as exc:
            if row["kind"] == "relevance":
                raise  # Unavailable ranking scores cannot be invented for calibration.
            result = {"lexical": "unknown", "bge_only": "unknown", "rule_b": "unknown", "slot_v2": "unknown", "error": type(exc).__name__}
        result["execution_budget"] = budget.snapshot()
        if store:
            store.finish("components", row["id"], {"task_id": row["id"], **result})
        output["rows"][row["id"]] = result
        if not result.get("error") and result.get("slot_v2") != "unknown":
            reused[reuse_key] = row["id"]
        print(f'{index+1} {row["kind"]} {row["id"]}: {result.get("slot_v2", result.get("score"))}', flush=True)
    output["complete"] = len(output["rows"]) == len(source["items"])
    output["selected_complete"] = len(output["rows"]) == len(selected)
    output["reused_rows"] = sum("reused_from" in r for r in output["rows"].values())
    output["evidence_manifest_sha256"] = digest(evidence_manifest)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--relevance-only", action="store_true", help="Score ranking inputs without any judge calls for dev calibration")
    p.add_argument("--resume", action="store_true", help="Resume the same frozen checkpoint; interrupted attempts stay unknown")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("predictions immutable; use a fresh output")
    source = json.loads(args.source.read_text())
    if source["source_sha256"] != digest(source["items"]):
        raise ValueError("frozen review input changed")
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    if calibration and (args.limit or args.relevance_only):
        raise ValueError("calibrated test requires the full frozen input")
    checkpoint = args.output.with_suffix(".checkpoint.json")
    with checkpoint_lock(checkpoint):
        atomic_json(args.output, predict(source, calibration=calibration, limit=args.limit,
                                        relevance_only=args.relevance_only, checkpoint=checkpoint, resume=args.resume))
