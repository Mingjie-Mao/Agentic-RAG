"""Use a dedicated Ollama request queue for the public demo; prewarm before serving.

The pinned experimental endpoint remains at 11436. A separate local queue does not
isolate physical GPU compute: concurrent experiments still need another machine.
No global process kills or changes to the existing experiment environment.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx

from app.config import settings
from scripts.setup_models import MODELS, VERSION, runtime_env

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_URL = "http://127.0.0.1:11438"


def validate_endpoint(value):
    parsed = urlparse(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError("Use an inference origin without embedded credentials or query parameters")
    local = parsed.hostname in {"127.0.0.1", "localhost"}
    if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
        raise ValueError("External inference must use HTTPS or a localhost SSH tunnel")
    if local and parsed.port == 11436:
        raise ValueError("Public inference must not share the experiment queue")
    return value.rstrip("/")


def prewarm(base, cfg, *, client=None):
    """Official empty chat request loads generation; embed call loads its runtime."""
    owned = client is None
    client = client or httpx.Client(base_url=base, trust_env=False, timeout=cfg.model_timeout_seconds)
    try:
        version = client.get("/api/version")
        version.raise_for_status()
        if version.json()["version"] != VERSION:
            raise ValueError("Public runtime version differs from the pinned baseline")
        tags = client.get("/api/tags")
        tags.raise_for_status()
        installed = {m["name"]: m["digest"] for m in tags.json()["models"]}
        for name in (cfg.embed_model, cfg.chat_model):
            if name not in MODELS or installed.get(name) != MODELS[name]:
                raise ValueError("Public model differs from the pinned baseline: " + name)
        timings = {}
        for stage, path, body in (
            ("embedding", "/api/embed", {"model": cfg.embed_model, "input": ["预热"],
                                       "truncate": False, "keep_alive": "30m"}),
            ("generation", "/api/chat", {"model": cfg.chat_model, "messages": [], "stream": False,
                                       "keep_alive": "30m", "options": {"num_ctx": 8192}}),
        ):
            start = time.monotonic()
            result = client.post(path, json=body)
            result.raise_for_status()
            payload = result.json()
            if payload.get("error"):
                raise RuntimeError("Public model warmup failed: " + stage)
            timings[stage + "_ms"] = round((time.monotonic() - start) * 1000, 1)
        return {"runtime_version": VERSION, "model_digests": {n: installed[n] for n in
                (cfg.embed_model, cfg.chat_model)}, "timings": timings}
    finally:
        if owned:
            client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-only", action="store_true")
    parser.add_argument("--inference-url", default=PUBLIC_URL)
    args = parser.parse_args()
    endpoint = validate_endpoint(args.inference_url)
    if not args.warm_only:
        env = {**os.environ, "RAG_OLLAMA_URL": endpoint, "RAG_PUBLIC_DEMO_RUNTIME": "true",
               "RAG_DEMO_ANSWER_CACHE_ENABLED": "true", "RAG_COMPACT_PROMPT_JSON": "false"}
        os.chdir(ROOT)
        os.execvpe(sys.executable, [sys.executable, "scripts/run.py"], env)
    cfg = settings()
    runtime = ROOT / ".runtime/ollama/ollama"
    child = None
    log = None
    try:
        try:
            httpx.get(endpoint + "/api/version", timeout=2, trust_env=False).raise_for_status()
        except httpx.HTTPError:
            if endpoint != PUBLIC_URL:
                raise RuntimeError("Configured independent inference endpoint is unavailable")
            if not runtime.is_file():
                raise RuntimeError("Pinned runtime missing; run make models first")
            log = (ROOT / ".runtime/public-ollama.log").open("a")
            child = subprocess.Popen([str(runtime), "serve"], env=runtime_env("127.0.0.1:11438"),
                                     stdout=log, stderr=log)
            for _ in range(60):
                if child.poll() is not None:
                    raise RuntimeError("Dedicated model server exited")
                try:
                    httpx.get(endpoint + "/api/version", timeout=1, trust_env=False).raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(0.5)
            else:
                raise RuntimeError("Dedicated model server did not become ready")
        result = prewarm(endpoint, cfg)
        result["endpoint"] = endpoint
        result["physical_gpu_shared"] = endpoint == PUBLIC_URL
        print(json.dumps(result), flush=True)
    finally:
        if child:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if log:
            log.close()


if __name__ == "__main__":
    main()
