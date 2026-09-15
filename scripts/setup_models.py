"""Install a pinned, project-local macOS runtime and verify the model manifests."""

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
VERSION = "0.11.10"
ARCHIVE_SHA = "6179e85eeca0f731390c8eb4d4bf83791526d4a6072370c312f3968077a5e63f"
MODELS = {
    "bge-m3:567m": "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab",
    "qwen2.5:7b-instruct": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
}


def runtime_env():
    return {
        **os.environ,
        "OLLAMA_HOST": "127.0.0.1:11436",
        "OLLAMA_MODELS": str(ROOT / ".runtime/ollama-models"),
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "2",
    }


def setup():
    if platform.system() != "Darwin":
        raise SystemExit(
            "This helper is for macOS. Use make models-cpu, or configure an existing Ollama endpoint."
        )
    runtime = ROOT / ".runtime/ollama"
    runtime.mkdir(parents=True, exist_ok=True)
    archive = runtime / "ollama-darwin.tgz"
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        print("Downloading pinned Ollama runtime…", flush=True)
        response = httpx.get(
            f"https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-darwin.tgz",
            follow_redirects=True,
            timeout=120,
        )
        response.raise_for_status()
        assert hashlib.sha256(response.content).hexdigest() == ARCHIVE_SHA, "Runtime checksum mismatch"
        archive.write_bytes(response.content)
    if not (runtime / "ollama").is_file():
        with tarfile.open(archive) as package:
            package.extractall(runtime, filter="data")
    child = None
    with httpx.Client(base_url="http://127.0.0.1:11436", trust_env=False, timeout=180) as client:
        try:
            client.get("/api/version").raise_for_status()
        except httpx.HTTPError:
            log = (ROOT / ".runtime/model-setup.log").open("a")
            child = subprocess.Popen(
                [str(runtime / "ollama"), "serve"], env=runtime_env(), stdout=log, stderr=log
            )
            for _ in range(60):
                try:
                    client.get("/api/version").raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(1)
            else:
                child.terminate()
                raise RuntimeError("Ollama startup failed; inspect .runtime/model-setup.log")
        try:
            assert client.get("/api/version").json()["version"] == VERSION, (
                "Unexpected Ollama runtime version"
            )
            installed = {m["name"]: m["digest"] for m in client.get("/api/tags").json()["models"]}
            for name, digest in MODELS.items():
                if installed.get(name) != digest:
                    print("Pulling " + name, flush=True)
                    with client.stream(
                        "POST", "/api/pull", json={"model": name, "stream": True}
                    ) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line:
                                event = json.loads(line)
                                if event.get("error"):
                                    raise RuntimeError(event["error"])
                current = {m["name"]: m["digest"] for m in client.get("/api/tags").json()["models"]}
                assert current.get(name) == digest, (
                    f"{name} digest changed; do not reuse old evaluation results"
                )
                print(name + " verified", flush=True)
        finally:
            if child:
                child.terminate()
                child.wait(timeout=30)


if __name__ == "__main__":
    setup()
