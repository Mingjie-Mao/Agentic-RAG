"""Resume a frozen package run from its immutable code snapshot.

The normal entry point refuses to resume once the working tree has moved on, which is
correct. The snapshot taken at freeze time is the code that must finish the run, so this
launcher starts that snapshot's worker exactly as the entry point would (same settings
environment, same suite copy), with host throughput probes before and after.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-slow-host", action="store_true")
    args = p.parse_args()
    from app.config import settings
    from benchmark_runtime import atomic_json, host_throughput_probe

    suite_dir, out = args.suite_dir.resolve(), args.output.resolve()
    freeze = suite_dir / "freeze.json"
    target = ROOT / ".runtime/benchmark-snapshots" / hashlib.sha256(freeze.read_bytes()).hexdigest()
    target_suite = target / suite_dir.relative_to(ROOT)
    if not (target_suite / "freeze.json").exists() or (target_suite / "freeze.json").read_bytes() != freeze.read_bytes():
        raise SystemExit("frozen snapshot missing or different from the registered freeze")
    if not out.with_suffix(".checkpoint.json").exists():
        raise SystemExit("no checkpoint to resume")
    env = dict(os.environ)
    env["HF_HOME"] = str(Path(os.environ.get("HF_HOME", ROOT / ".runtime/huggingface")).resolve())
    for name, value in settings().model_dump(mode="json").items():
        if name == "storage_dir":
            value = str(settings().storage_dir.resolve())
        env["RAG_" + name.upper()] = json.dumps(value) if isinstance(value, (bool, list, dict)) else str(value)
    probe_path = out.with_suffix(".host-probe.json")
    probes = json.loads(probe_path.read_text()) if probe_path.exists() else []
    probes.append(host_throughput_probe(settings().ollama_url, settings().chat_model) | {"phase": "warmup"})
    before = host_throughput_probe(settings().ollama_url, settings().chat_model) | {"phase": "before_snapshot_resume"}
    probes.append(before)
    atomic_json(probe_path, probes)
    if not before["meets_baseline"] and not args.allow_slow_host:
        raise SystemExit(f"host below throughput baseline: {before}")
    command = [sys.executable, str(target / "scripts/run_unseen_benchmark.py"), "--suite-dir", str(target_suite),
               "--output", str(out), "--snapshot-worker", "--resume"]
    try:
        subprocess.run(command, cwd=target, env=env, check=True)
    finally:
        probes = json.loads(probe_path.read_text())  # keep the worker's mid-run probes
        probes.append(host_throughput_probe(settings().ollama_url, settings().chat_model) | {"phase": "after"})
        atomic_json(probe_path, probes)


if __name__ == "__main__":
    main()
