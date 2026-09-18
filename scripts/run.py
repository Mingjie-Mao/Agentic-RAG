"""Foreground development supervisor: Ctrl-C stops only children it started."""

import os
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import httpx

from app.config import settings
from setup_models import runtime_env

ROOT = Path(__file__).resolve().parent.parent
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
    try:
        httpx.get(settings().ollama_url + "/api/version", timeout=2, trust_env=False).raise_for_status()
    except httpx.HTTPError:
        runtime = ROOT / ".runtime/ollama/ollama"
        if platform.system() != "Darwin" or not runtime.exists():
            raise RuntimeError("Start Ollama first: make models (macOS), or make models-cpu")
        start([str(runtime), "serve"], "ollama", runtime_env())
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
