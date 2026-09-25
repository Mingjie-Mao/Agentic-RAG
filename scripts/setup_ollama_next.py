"""A second, newer Ollama runtime for the generator comparison, beside the pinned one.

The production runtime stays at 0.11.10 because every frozen result was produced on it,
and a newer llama.cpp can change qwen2.5's outputs. Qwen3.5 needs Ollama >= 0.17.1, so
it gets its own pinned runtime, port and model directory. Nothing the production path
uses is touched: 11436 and `.runtime/ollama-models` stay as they were.

The server is left running for the experiment; stop it with `pkill -f ollama-next`.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import time

import httpx

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime/ollama-next"
VERSION = "0.34.4"
ARCHIVE_SHA = "e9c8fddaab5f48f47f2c4ae3d23d0732f5182417125353faeed2188e34a22799"
HOST = "127.0.0.1:11437"
# Pinned on first pull (2026-09-25); a changed digest means earlier results are not comparable.
MODELS = {"qwen3.5:9b": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"}


def runtime_env():
    return {
        **os.environ,
        "OLLAMA_HOST": HOST,
        "OLLAMA_MODELS": str(ROOT / ".runtime/ollama-next-models"),
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }


def main():
    if platform.system() != "Darwin":
        raise SystemExit("This helper is for macOS.")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    archive = RUNTIME / "ollama-darwin.tgz"
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        print(f"Downloading Ollama {VERSION}…", flush=True)
        response = httpx.get(
            f"https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-darwin.tgz",
            follow_redirects=True,
            timeout=600,
        )
        response.raise_for_status()
        assert hashlib.sha256(response.content).hexdigest() == ARCHIVE_SHA, "Runtime checksum mismatch"
        archive.write_bytes(response.content)
    if not (RUNTIME / "ollama").is_file():
        with tarfile.open(archive) as package:
            package.extractall(RUNTIME, filter="data")
    with httpx.Client(base_url=f"http://{HOST}", trust_env=False, timeout=180) as client:
        try:
            client.get("/api/version").raise_for_status()
        except httpx.HTTPError:
            log = (ROOT / ".runtime/ollama-next.log").open("a")
            subprocess.Popen(
                [str(RUNTIME / "ollama"), "serve"],
                env=runtime_env(),
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            for _ in range(60):
                try:
                    client.get("/api/version").raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(1)
            else:
                raise RuntimeError("Ollama startup failed; inspect .runtime/ollama-next.log")
        assert client.get("/api/version").json()["version"] == VERSION, "Unexpected Ollama version"
        installed = {m["name"]: m["digest"] for m in client.get("/api/tags").json()["models"]}
        for name, digest in MODELS.items():
            if name not in installed or (digest and installed[name] != digest):
                print(f"Pulling {name}", flush=True)
                with client.stream("POST", "/api/pull", json={"model": name, "stream": True}, timeout=None) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line and (event := json.loads(line)).get("error"):
                            raise RuntimeError(event["error"])
            current = {m["name"]: m["digest"] for m in client.get("/api/tags").json()["models"]}
            assert digest is None or current.get(name) == digest, f"{name} digest changed"
            print(f"{name} {current.get(name)}", flush=True)


if __name__ == "__main__":
    main()
