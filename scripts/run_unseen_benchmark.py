"""One-shot comparison on the unseen benchmark: RAG, fixed workflow, Hybrid (lexical),
Hybrid (semantic coverage control).

The tasks and corpus in fixtures/unseen/ were written after the Hybrid architecture,
the splitter fix and the semantic evaluator (thresholds and coverage rule) were fixed,
and before any arm ran on them. `--freeze` records the hashes of the tasks, the corpus,
every code file the four arms execute, and the model configuration. A run refuses to
start if any of them changed, and refuses to overwrite an existing result: this set is
run once. Scoring reuses the Hard benchmark's scorer unchanged.

Arms share everything except what is being compared: the generator is qwen2.5-7b in all
four, the Hybrid policy is qwen2.5-7b in both Hybrid arms, and only the hybrid-semantic
arm calls the coverage judge (its calls are recorded in their own events).
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import os
import subprocess
import sys
from functools import lru_cache
from uuid import uuid4

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = ROOT / "fixtures/unseen/tasks.json"
SPEC = ROOT / "fixtures/unseen/documents.json"
FREEZE = ROOT / "fixtures/unseen/freeze.json"
OUT = ROOT / "artifacts/unseen-benchmark-v1.json"
CODE = [
    "agent/hybrid.py",
    "agent/controller.py",
    "agent/planner.py",
    "agent/tools.py",
    "agent/workflow_state.py",
    "app/semantic_evidence.py",
    "app/evidence_selection.py",
    "app/rerank.py",
    "app/retrieval.py",
    "app/qa.py",
    "app/clients.py",
    "app/task_analysis.py",
    "app/config.py",
    "app/security.py",
    "scripts/run_agent_hard_benchmark.py",
    "scripts/run_unseen_benchmark.py",
]
ARMS = ["rag", "workflow", "dynamic", "hybrid", "hybrid_semantic"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=64)
def fingerprint_file(name, size, mtime):
    with Path(name).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inputs():
    spec = json.loads(SPEC.read_text())
    files = {str(TASKS.relative_to(ROOT)): sha(TASKS), str(SPEC.relative_to(ROOT)): sha(SPEC)}
    for doc in spec["documents"]:
        for version in doc["versions"]:
            files[version["path"]] = sha(ROOT / version["path"])
    # Freeze transitive local Python dependencies, not only the hand-picked entrypoints.
    for directory in ("app", "agent", "scripts"):
        for path in (ROOT / directory).rglob("*.py"):
            files[str(path.relative_to(ROOT))] = sha(path)
    for path in [ROOT / "pyproject.toml", *ROOT.glob("requirements*.lock")]:
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    suite = json.loads(TASKS.read_text())
    for key in (
        "review_packet",
        "reviewed_packet",
        "p2_gate",
        "p3_gate",
        "calibration",
        "package_manifest",
    ):
        if suite.get(key):
            if (
                key == "reviewed_packet"
                and suite.get("evaluation_protocol") == "enterprise-benchmark-v1"
                and suite.get("split") == "dev"
                and not (ROOT / suite[key]).exists()
            ):
                continue
            files[suite[key]] = sha(ROOT / suite[key])
    from app.config import settings

    if settings().answer_semantic_audit_enabled or settings().answer_literal_judgment_enabled:
        gate_path = Path(settings().answer_semantic_audit_gate_path)
        for path in [
            gate_path,
            *[
                Path(value)
                for key, value in json.loads(gate_path.read_text()).items()
                if key in {"cases_path", "result_path"}
            ],
        ]:
            path = path if path.is_absolute() else ROOT / path
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    return files


def configuration():
    from app.config import settings
    from app.semantic_evidence import Thresholds

    cfg = settings()
    import httpx
    import platform

    model_records = {}
    model_roles = [
        ("generation", cfg.ollama_url, cfg.chat_model),
        ("embedding", cfg.ollama_url, cfg.embed_model),
        ("policy", cfg.agent_policy_url or cfg.ollama_url, cfg.agent_policy_model),
        ("judge", cfg.semantic_judge_url or cfg.ollama_url, cfg.semantic_judge_model),
    ]
    if cfg.answer_literal_judgment_enabled or (cfg.answer_semantic_audit_enabled and cfg.answer_extractive_enabled):
        model_roles.append(
            ("answer_composition", cfg.answer_composition_url, cfg.answer_composition_model)
        )
    for role, endpoint, model in model_roles:
        response = httpx.get(endpoint + "/api/tags", timeout=10, trust_env=False)
        response.raise_for_status()
        installed = next((row for row in response.json()["models"] if row["name"] == model), None)
        if not installed or not installed.get("digest"):
            raise ValueError(f"model must be installed and fingerprinted before freeze: {role}")
        runtime = httpx.get(endpoint + "/api/version", timeout=10, trust_env=False)
        runtime.raise_for_status()
        model_records[role] = {
            "model": model,
            "digest": installed["digest"],
            "runtime": runtime.json()["version"],
        }
    # The Python BGE cross-encoder is separate from Ollama's embedding model.
    # Bind its actual weights/tokenizer, rather than only its mutable Hub name.
    hf_home = Path(os.environ.get("HF_HOME", ROOT / ".runtime/huggingface")).resolve()
    repo = hf_home / "hub" / ("models--" + cfg.rerank_model.replace("/", "--"))
    revision = (repo / "refs/main").read_text().strip()
    snapshot = repo / "snapshots" / revision
    encoder_files = {}
    for path in sorted(snapshot.iterdir()):
        if path.is_file() and path.suffix in {".json", ".safetensors", ".model", ".txt"}:
            stat = path.stat()
            encoder_files[path.name] = fingerprint_file(str(path), stat.st_size, stat.st_mtime_ns)
    if not any(name.endswith(".safetensors") for name in encoder_files):
        raise ValueError("cross-encoder weights missing from fingerprint")
    model_records["cross_encoder"] = {
        "model": cfg.rerank_model,
        "revision": revision,
        "files": encoder_files,
        "max_tokens": cfg.rerank_max_tokens,
    }
    from importlib.metadata import version

    # Full behavioral configuration is frozen. Credentials never enter artifacts.
    effective = cfg.model_dump(
        mode="json", exclude={"database_url", "demo_password", "memory_tokens_json"}
    )
    effective["storage_dir"] = str(cfg.storage_dir.resolve())
    return {
        "python": platform.python_version(),
        "models": model_records,
        "libraries": {
            name: version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "httpx",
                "pydantic",
                "sqlalchemy",
                "langgraph",
                "langgraph-checkpoint-postgres",
                "langgraph-checkpoint-sqlite",
            )
        },
        "effective_configuration": effective,
        "chat_model": cfg.chat_model,
        "agent_policy_model": cfg.agent_policy_model,
        "agent_policy_url": cfg.agent_policy_url,
        "semantic_judge_model": cfg.semantic_judge_model,
        "semantic_judge_url": cfg.semantic_judge_url,
        "retrieval_mode": cfg.retrieval_mode,
        "thresholds": Thresholds().__dict__,
        "coverage_rule": "rule B (dev-selected, semantic-evidence-v1)",
        "flags": {
            key: getattr(cfg, key)
            for key in (
                "task_contract_enabled",
                "answer_contract_enabled",
                "source_facts_enabled",
                "focused_generation_enabled",
                "passage_window_enabled",
                "claim_consistency_enabled",
                "source_routing",
                "rerank_mode",
                "context_token_budget",
                "min_similarity",
                "agent_task_timeout_seconds",
                "agent_policy_max_calls",
                "agent_judge_max_calls",
                "agent_judge_token_budget",
                "agent_generation_max_calls",
            )
        },
    }


def validate(suite, spec):
    known = {doc["document_id"] for doc in spec["documents"]}
    errors, ids = [], [t["id"] for t in suite["tasks"]]
    if len(set(ids)) != len(ids):
        errors.append("duplicate task ids")
    for task in suite["tasks"]:
        for field in (
            "user",
            "goal",
            "expected_status",
            "facts",
            "forbidden",
            "scenario_events",
            "category",
        ):
            if field not in task:
                errors.append(f"{task['id']}: missing {field}")
        docs = set(task.get("expected_evidence_documents", []))
        transition = task.get("conditional_transition") or {}
        docs |= {v for v in (transition.get("observe_document"), transition.get("then_document")) if v}
        if docs - known:
            errors.append(f"{task['id']}: unknown documents {sorted(docs - known)}")
        for matcher in task.get("fact_matchers", []):
            if not matcher.get("id") or not (matcher.get("aliases") or matcher.get("patterns")):
                errors.append(f"{task['id']}: invalid matcher")
    if errors:
        raise SystemExit("\n".join(errors))


def judge_usage(db, run_id):
    from app.models import AgentEvent

    totals = Counter()
    for event in db.scalars(
        select(AgentEvent).where(
            AgentEvent.task_id == run_id, AgentEvent.event_type == "semantic_coverage_control"
        )
    ):
        totals.update(
            {
                k: v
                for k, v in event.payload.get("judge_usage", {}).items()
                if isinstance(v, (int, float))
            }
        )
    from app.models import AgentTask

    task = db.get(AgentTask, run_id)
    recorded = task.input.get("_execution_budget", {}).get("calls", {}).get("judge", {}) if task else {}
    if recorded:
        return dict(totals) | recorded | {"calls": recorded["attempted"]}
    return dict(totals)


def summarize(rows):
    scored = [r for r in rows if not r.get("skipped")]
    if not scored:
        return {"tasks": 0}
    latencies = [
        r["latency_ms"]
        for r in scored
        if r.get("latency_ms") is not None and r.get("latency_source") != "unrecoverable"
    ]
    tokens = [
        (r.get("usage") or {}).get("total_prompt_tokens")
        or (r.get("usage") or {}).get("prompt_tokens")
        or 0
        for r in scored
    ]
    stopping = [r for r in scored if r["over_planned"] is not None]
    return {
        "tasks": len(scored),
        "task_success": sum(r["task_success"] for r in scored),
        "answer_correct": sum(r["answer_correct"] for r in scored),
        "over_planning": f"{sum(bool(r['over_planned']) for r in stopping)}/{len(stopping)}",
        "early_stop": sum(bool(r["early_stop"]) for r in scored),
        "forbidden_output_violations": sum(any(r["forbidden_present"].values()) for r in scored),
        "execution_failures": sum(r["status"] == "execution_failed" for r in scored),
        "mean_steps": round(statistics.mean(r["steps"] for r in scored), 2),
        "unknown_latency": len(scored) - len(latencies),
        "p50_latency_s": round(statistics.median(latencies) / 1000, 1) if latencies else None,
        "p95_latency_s": round(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)] / 1000, 1)
        if latencies
        else None,
        "mean_prompt_tokens": round(statistics.mean(tokens), 1),
        "judge_calls": sum((r.get("judge_usage") or {}).get("calls", 0) for r in scored),
        "judge_seconds_per_task": round(
            sum((r.get("judge_usage") or {}).get("wall_ms", 0) for r in scored) / len(scored) / 1000, 1
        ),
    }


def watch_host(out):
    """Mid-run host check: two consecutive probes below baseline stop the run cleanly.

    The checkpoint keeps every finished row; resume from the frozen snapshot later.
    """
    import time

    from app.config import settings
    from benchmark_runtime import atomic_json, host_throughput_probe

    path = out.with_suffix(".host-probe.json")
    for attempt in range(2):
        probe = host_throughput_probe(settings().ollama_url, settings().chat_model) | {"phase": "during"}
        probes = json.loads(path.read_text()) if path.exists() else []
        atomic_json(path, probes + [probe])
        if probe["meets_baseline"]:
            return
        if attempt == 0:
            time.sleep(60)
    raise SystemExit("host degraded during the run; finished rows are checkpointed. "
                     "Resume with scripts/resume_frozen_snapshot.py once the host meets the baseline.")


def main():
    global TASKS, SPEC, FREEZE, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--suite-dir", type=Path, default=TASKS.parent)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-dev-freeze",
        action="store_true",
        help="Refresh only a package Dev snapshot before a new experiment",
    )
    parser.add_argument(
        "--allow-slow-host",
        action="store_true",
        help="Run although the host throughput probe is below baseline; the probe is still recorded",
    )
    parser.add_argument("--snapshot-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    suite_dir = args.suite_dir.resolve()
    TASKS, SPEC, FREEZE = (suite_dir / name for name in ("tasks.json", "documents.json", "freeze.json"))
    OUT = args.output.resolve()
    suite, spec = json.loads(TASKS.read_text()), json.loads(SPEC.read_text())
    validate(suite, spec)
    is_v2 = suite.get("evaluation_protocol") == "unseen-v2-semantic-review"
    is_paired = suite.get("evaluation_protocol") == "semantic-paired-v2-development"
    is_package = suite.get("evaluation_protocol") == "enterprise-benchmark-v1"
    uses_slots = is_v2 or is_paired or is_package
    if args.refresh_dev_freeze and not (args.freeze and is_package and suite.get("split") == "dev"):
        raise SystemExit("freeze refresh is allowed only for package Dev, never Core/Security/External")
    if uses_slots:
        from app.config import settings

        allowed_flags = {
            "task_contract_enabled",
            "answer_quality_enabled",
            "answer_comparison_focus_enabled",
            "answer_extractive_enabled",
            "answer_semantic_audit_enabled",
            "answer_literal_judgment_enabled",
            "answer_contract_enabled",
            "focused_generation_enabled",
            "claim_consistency_enabled",
            "semantic_coverage_shadow",
            "semantic_slot_shadow_enabled",
            "semantic_coverage_control",
            "semantic_slot_control_enabled",
        }
        for key, value in suite.get("shared_configuration", {}).items():
            package_values = {
                "retrieval_mode": {"bm25", "dense", "hybrid"},
                "rewrite_mode": {"off", "rule"},
            }
            package_values["verdict_protocol"] = {"legacy", "structured"}
            package_values["answer_composition_reasoning"] = {"off", "low", "medium", "high"}
            if is_package and key in package_values:
                valid = value in package_values[key]
            elif is_package and key == "passage_rerank":
                valid = isinstance(value, bool)
            elif is_package and key == "answer_composition_max_tokens":
                valid = type(value) is int and 200 <= value <= 2400
            elif is_package and key in {
                "semantic_judge_model",
                "semantic_judge_url",
                "answer_composition_model",
                "answer_composition_url",
                "answer_semantic_audit_gate_path",
            }:
                valid = isinstance(value, str) and bool(value.strip())
                if key == "answer_semantic_audit_gate_path":
                    valid = (
                        valid
                        and not Path(value).is_absolute()
                        and (ROOT / value).resolve().is_relative_to(ROOT)
                    )
            else:
                valid = key in allowed_flags and isinstance(value, bool)
            if not valid:
                raise ValueError("invalid shared evaluator/contract configuration")
            setattr(settings(), key, value)
    arms = suite.get("arms", ARMS)
    if not arms or len(set(arms)) != len(arms) or (not is_package and set(arms) - set(ARMS)):
        raise SystemExit("invalid arm list")
    if uses_slots:
        from app.config import settings

        if "hybrid_semantic" in arms:
            settings().semantic_slot_gate_path = suite["p2_gate"]
            settings().semantic_slot_calibration_path = suite["calibration"]
    if is_paired:
        from paired_semantic_replay import require_gate

        require_gate(ROOT / suite["p2_gate"], "semantic-slot-v2-gate")
    if is_v2:
        from unseen_v2_scoring import reviewed_suite

        reviewed_suite(suite, ROOT)
        if "hybrid_semantic" in arms:
            from paired_semantic_replay import require_gate

            require_gate(ROOT / suite["p2_gate"], "semantic-slot-v2-gate")
            require_gate(ROOT / suite["p3_gate"], "semantic-paired-v2-gate")
    if is_package:
        from scripts.benchmark_package_scoring import require_task_review
        from scripts.benchmark_package_runtime import method_matrix

        matrix = method_matrix(suite)
        if suite["split"] != "dev":
            require_task_review(suite, spec, ROOT)
    if args.freeze:
        if FREEZE.exists() and not args.refresh_dev_freeze:
            raise SystemExit("already frozen; the freeze is not rewritten")
        extra = {}
        if is_package:
            from scripts.benchmark_package_runtime import indexed_corpus

            extra["indexed_corpus"] = indexed_corpus(spec)
        FREEZE.write_text(
            json.dumps(
                {
                    "inputs": inputs(),
                    "configuration": configuration(),
                    "tasks": len(suite["tasks"]),
                    "categories": suite["categories"],
                    **extra,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(f"frozen {len(suite['tasks'])} tasks -> {FREEZE.relative_to(ROOT)}")
        return
    if not FREEZE.exists():
        raise SystemExit("freeze first (--freeze)")
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: this benchmark is run once")
    frozen = json.loads(FREEZE.read_text())
    current_inputs = inputs()
    changed = [
        path
        for path in current_inputs.keys() | frozen["inputs"].keys()
        if current_inputs.get(path) != frozen["inputs"].get(path)
    ]
    if changed or configuration() != frozen["configuration"]:
        raise SystemExit(f"inputs differ from the freeze: {changed or 'configuration'}")
    if is_package:
        from scripts.benchmark_package_runtime import indexed_corpus

        if indexed_corpus(spec) != frozen["indexed_corpus"]:
            raise SystemExit("indexed corpus/ACL/users changed since freeze")
        if args.snapshot_worker and suite["split"] != "dev":
            from scripts.benchmark_package_runtime import validate_test_worker

            validate_test_worker(ROOT, FREEZE, OUT)

    from benchmark_runtime import (
        CheckpointStore,
        atomic_json,
        immutable_snapshot,
        summary,
        task_safety_labels,
        checkpoint_lock,
    )

    if not args.snapshot_worker:
        from app.config import settings

        # A different --output must not secretly rerun a consumed test matrix.
        # This ledger lives beside the original suite, outside the code snapshot.
        if is_package and suite["split"] != "dev":
            from scripts.benchmark_package_runtime import reserve_test_run

            reserve_test_run(FREEZE, OUT, resume=args.resume)
        key = sha(FREEZE)
        target = immutable_snapshot(ROOT, frozen, ROOT / ".runtime/benchmark-snapshots" / key)
        target_suite = target / suite_dir.relative_to(ROOT)
        atomic_json(target_suite / "freeze.json", frozen)
        env = dict(os.environ)
        env["HF_HOME"] = str(Path(os.environ.get("HF_HOME", ROOT / ".runtime/huggingface")).resolve())
        for name, value in settings().model_dump(mode="json").items():
            if name == "storage_dir":
                value = str(settings().storage_dir.resolve())
            env["RAG_" + name.upper()] = (
                json.dumps(value) if isinstance(value, (bool, list, dict)) else str(value)
            )
        command = [
            sys.executable,
            str(target / "scripts/run_unseen_benchmark.py"),
            "--suite-dir",
            str(target_suite),
            "--output",
            str(OUT),
            "--snapshot-worker",
        ]
        if args.resume:
            command.append("--resume")
        # Host load is part of the measurement: probe before and after, keep both.
        from benchmark_runtime import host_throughput_probe

        probe_path = OUT.with_suffix(".host-probe.json")
        probes = json.loads(probe_path.read_text()) if probe_path.exists() else []
        # The first call after idle runs cold; it is recorded but only the second decides.
        probes.append(host_throughput_probe(settings().ollama_url, settings().chat_model) | {"phase": "warmup"})
        before = host_throughput_probe(settings().ollama_url, settings().chat_model) | {"phase": "before"}
        probes.append(before)
        atomic_json(probe_path, probes)
        if not before["meets_baseline"] and not args.allow_slow_host:
            raise SystemExit(
                f"host below throughput baseline (prefill {before['prefill_tok_s']} tok/s, "
                f"decode {before['decode_tok_s']} tok/s); recorded in {probe_path.name}"
            )
        try:
            subprocess.run(command, cwd=target, env=env, check=True)
        finally:
            # The worker appends mid-run probes to the same file; re-read before appending.
            probes = json.loads(probe_path.read_text())
            probes.append(host_throughput_probe(settings().ollama_url, settings().chat_model) | {"phase": "after"})
            atomic_json(probe_path, probes)
        return

    with checkpoint_lock(OUT):
        from app.config import settings
        from app.db import SessionLocal
        from run_agent_hard_benchmark import run_arm

        if is_package:
            from scripts.benchmark_package_runtime import run_package_arm

            run_arm = run_package_arm

        results = {arm: [] for arm in arms}
        checkpoint = CheckpointStore(OUT.with_suffix(".checkpoint.json"), frozen, resume=args.resume)
        resumed_keys, fresh_rows = set(checkpoint.data["completed"]), 0
        labels = task_safety_labels(suite, spec, ROOT)
        original_control = settings().semantic_coverage_control
        original_slot_control = settings().semantic_slot_control_enabled
        with SessionLocal() as db:
            try:
                # Rotate execution order per task; avoid one arm always running cold.
                for index, task in enumerate(
                    suite["tasks"] + (suite.get("safety_tasks", []) if uses_slots else [])
                ):
                    for arm in arms[index % len(arms) :] + arms[: index % len(arms)]:
                        from app.models import User
                        from contextlib import ExitStack

                        settings().semantic_coverage_control = (
                            arm == "hybrid_semantic" and not uses_slots
                        )
                        settings().semantic_slot_control_enabled = (
                            arm == "hybrid_semantic" and uses_slots
                        )
                        scoring = {}
                        if uses_slots:
                            if is_package:
                                from scripts.benchmark_package_scoring import provisional_score
                            else:
                                from unseen_v2_scoring import provisional_score
                            scoring["score_fn"] = provisional_score
                        mode = (
                            matrix[arm]["mode"]
                            if is_package
                            else "hybrid"
                            if arm == "hybrid_semantic"
                            else arm
                        )
                        with ExitStack() as restored:
                            if is_package:
                                for setting, value in matrix[arm]["overrides"].items():
                                    previous = getattr(settings(), setting)
                                    restored.callback(setattr, settings(), setting, previous)
                                    setattr(settings(), setting, value)
                            key = f"{arm}:{task['id']}"
                            if key in checkpoint.data["completed"]:
                                row = checkpoint.data["completed"][key]
                            elif key in checkpoint.data["pending"]:
                                row = recover_checkpoint_attempt(
                                    db,
                                    task,
                                    mode if is_package else arm,
                                    checkpoint.data["pending"][key],
                                    **scoring,
                                )
                                checkpoint.finish(arm, task["id"], row)
                            else:
                                token = str(uuid4())
                                checkpoint.begin(arm, task["id"], token)
                                row = run_arm(
                                    db,
                                    db.get(User, task["user"]),
                                    task,
                                    mode,
                                    on_created=lambda rid: checkpoint.attach(arm, task["id"], rid),
                                    benchmark_run_token=token,
                                    **scoring,
                                )
                                if row.get("run_id"):
                                    row["judge_usage"] = judge_usage(db, row["run_id"])
                                checkpoint.finish(arm, task["id"], row)
                        results[arm].append(row)
                        if key not in resumed_keys:
                            fresh_rows += 1
                            if fresh_rows % 20 == 0:
                                watch_host(OUT)
                        print(f"[{arm}] {task['id']} success={row.get('task_success')}", flush=True)
            finally:
                settings().semantic_coverage_control = original_control
                settings().semantic_slot_control_enabled = original_slot_control
        categories = sorted(
            {
                t["category"]
                for t in suite["tasks"] + (suite.get("safety_tasks", []) if uses_slots else [])
            }
        )
        output = {
            "benchmark": suite["version"],
            "freeze": str(FREEZE.relative_to(ROOT)),
            "arms": arms,
            "summary": {arm: summary(rows, labels) for arm, rows in results.items()},
            "by_category": {
                arm: {c: summary([r for r in rows if r["category"] == c], labels) for c in categories}
                for arm, rows in results.items()
            },
            "results": results,
            "configuration": frozen["configuration"],
            "arm_overrides": {
                arm: {
                    "mode": "hybrid" if arm == "hybrid_semantic" else arm,
                    "semantic_coverage_control": arm == "hybrid_semantic" and not uses_slots,
                    "semantic_slot_control_enabled": arm == "hybrid_semantic" and uses_slots,
                }
                for arm in arms
            },
        }
        if uses_slots:
            if is_package:
                from scripts.benchmark_package_scoring import answer_review_packet, report
                from scripts.research_review import digest

                output.update(
                    split=suite["split"],
                    corpus_sha256=digest(frozen["indexed_corpus"]),
                    arm_overrides=matrix,
                    registration={"suite_sha256": digest(suite), "freeze_sha256": digest(frozen)},
                )
                packet = answer_review_packet(output, suite)
                output["three_layer_report"] = report(output, suite)
                output["summary"] = output["three_layer_report"]["summary"]
                output["by_category"] = {
                    arm: metrics["by_category"] for arm, metrics in output["summary"].items()
                }
            else:
                from unseen_v2_scoring import answer_packet

                packet = answer_packet(output, suite)
            output["quality_status"] = "provisional_pending_semantic_answer_review"
            atomic_json(OUT.with_suffix(".answer-review.json"), packet)
        atomic_json(OUT, output)
        print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


def recover_checkpoint_attempt(db, task, arm, pending, *, score_fn=None):
    """Recover a committed result; an unfinished attempt is a failure, never rerun."""
    from app.models import AgentTask, Answer, User
    from agent.controller import task_payload
    from run_agent_hard_benchmark import _tool_trace, score_task

    run = db.get(AgentTask, pending["run_id"]) if pending["run_id"] else None
    if arm != "rag" and run is None:
        run = db.scalar(
            select(AgentTask).where(
                AgentTask.user_id == task["user"],
                AgentTask.input["benchmark_run_token"].as_string() == pending["token"],
            )
        )
    answer = (
        db.scalar(
            select(Answer).where(
                Answer.user_id == task["user"],
                Answer.payload["trace"]["benchmark_run_token"].as_string() == pending["token"],
            )
        )
        if arm == "rag"
        else None
    )
    payload, observable, trace, steps = (
        {"status": "execution_failed", "claims": [], "citations": []},
        {},
        [],
        0,
    )
    if answer:
        payload = observable = answer.payload
    elif run:
        if run.status not in {"completed", "failed", "cancelled"}:
            run.status, run.error = "failed", "interrupted_benchmark_attempt"
            run.lease_until = run.lease_token = None
            db.commit()
        observable = task_payload(db, db.get(User, task["user"]), run)
        payload = observable.get("result") or payload
        trace, steps = _tool_trace(db, run.id), run.step_no
    row = (score_fn or score_task)(
        task,
        payload,
        trace,
        steps,
        0,
        observable_payload=observable,
        scenario_events=run.input.get("_benchmark_scenario_events", {}) if run else {},
        execution_error_kind=(
            "authorization_change" if run and run.error and "404" in run.error else "unexpected"
        )
        if payload["status"] == "execution_failed"
        else None,
    )
    row.update(
        task_id=task["id"],
        run_id=run.id if run else pending["run_id"],
        latency_ms=None,
        latency_source="unrecoverable",
        recovered_from_checkpoint=True,
    )
    if run and payload["status"] == "execution_failed":
        row["execution_error"] = run.error or "interrupted_benchmark_attempt"
    if run:
        row["judge_usage"] = judge_usage(db, run.id)
        budget = run.input.get("_execution_budget", {})
        row["usage"]["execution_budget"] = budget
        if budget.get("calls", {}).get("judge"):
            row["judge_usage"].update(budget["calls"]["judge"])
    return row


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
