"""Foreground development supervisor: Ctrl-C stops only children it started."""

import os
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from app.config import settings
from setup_models import runtime_env

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
children, logs = [], []
process_ids = {}
stopping = False


def stop(*_):
    global stopping
    stopping = True


def start(args, label, env=None):
    log = (ROOT / ".runtime" / f"{label}.log").open("a")
    logs.append(log)
    process = subprocess.Popen(args, stdout=log, stderr=log, env=env)
    children.append(process)
    process_ids[label] = process.pid
    (ROOT / ".runtime/processes.json").write_text(
        json.dumps({"supervisor": os.getpid(), "children": process_ids})
    )
    return process


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
(ROOT / ".runtime").mkdir(exist_ok=True)
try:
    if not (ROOT / "web/dist/index.html").exists():
        raise RuntimeError("Build the frontend first: make web")
    cfg = settings()
    configured_target = urlparse(cfg.ollama_url)
    if (cfg.public_demo_runtime and configured_target.hostname in {"127.0.0.1", "localhost"}
            and configured_target.port == 11436):
        raise RuntimeError("Public demo must not share the pinned experiment queue")
    try:
        httpx.get(settings().ollama_url + "/api/version", timeout=2, trust_env=False).raise_for_status()
    except httpx.HTTPError:
        if cfg.public_demo_runtime and cfg.ollama_url.rstrip("/") != "http://127.0.0.1:11438":
            raise RuntimeError("Independent inference endpoint unavailable; refusing local fallback")
        runtime = ROOT / ".runtime/ollama/ollama"
        if platform.system() != "Darwin" or not runtime.exists():
            raise RuntimeError("Start Ollama first: make models (macOS), or make models-cpu")
        target = urlparse(cfg.ollama_url)
        if target.scheme != "http" or target.hostname not in {"127.0.0.1", "localhost"} or not target.port:
            raise RuntimeError("Configured external inference endpoint is unavailable")
        start([str(runtime), "serve"], "public-ollama" if cfg.public_demo_runtime else "ollama",
              runtime_env(f"127.0.0.1:{target.port}"))
        for _ in range(60):
            try:
                httpx.get(cfg.ollama_url + "/api/version", timeout=1, trust_env=False).raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Model server startup timed out")
    if cfg.public_demo_runtime:
        from scripts.run_public_demo import prewarm
        warmup = prewarm(cfg.ollama_url, cfg)
        warmup["endpoint"] = cfg.ollama_url
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        with (ROOT / ".runtime" / f"public-model-warmup-{stamp}.json").open("x") as handle:
            json.dump(warmup, handle, indent=2)
        print("Dedicated demo models prewarmed; starting API and worker", flush=True)
    start([sys.executable, "-m", "app.worker"], "worker")
    start(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"], "api"
    )
    print("Agentic-RAG: http://127.0.0.1:8000 | logs: .runtime/*.log | Ctrl-C to stop", flush=True)
    while not stopping:
        if any(child.poll() is not None for child in children):
            raise RuntimeError("A service stopped unexpectedly. Inspect .runtime/*.log")
        time.sleep(0.5)
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    for log in logs:
        log.close()
