from types import SimpleNamespace

import httpx
import pytest

from scripts.run_public_demo import prewarm, validate_endpoint
from scripts.setup_models import MODELS, VERSION, runtime_env


def test_dedicated_runtime_does_not_change_experiment_environment():
    assert runtime_env()["OLLAMA_HOST"] == "127.0.0.1:11436"
    assert runtime_env("127.0.0.1:11438")["OLLAMA_HOST"] == "127.0.0.1:11438"
    assert runtime_env()["OLLAMA_NUM_PARALLEL"] == "1"


@pytest.mark.parametrize("endpoint", ["http://localhost:11436", "http://127.0.0.1:11436",
                                      "http://external.test:11438", "https://user:secret@external.test",
                                      "https://external.test?token=secret"])
def test_public_endpoint_rejects_shared_queue_cleartext_or_embedded_credentials(endpoint):
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


@pytest.mark.parametrize("failure", [None, "version", "model", "warmup"])
def test_warmup_checks_identity_before_serving_and_does_not_generate_a_question(failure):
    calls = []
    def handle(request):
        import json
        body = json.loads(request.content) if request.content else None
        calls.append((request.url.path, body))
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "different" if failure == "version" else VERSION})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n, "digest": "changed" if failure == "model" else d}
                                                      for n, d in MODELS.items()]})
        return httpx.Response(200, json={"error": "failed"} if failure == "warmup" else {})
    cfg = SimpleNamespace(chat_model="qwen2.5:7b-instruct", embed_model="bge-m3:567m")
    with httpx.Client(base_url="http://demo.test", transport=httpx.MockTransport(handle)) as client:
        if failure:
            with pytest.raises((ValueError, RuntimeError)):
                prewarm("http://demo.test", cfg, client=client)
            if failure in {"version", "model"}:
                assert not any(path in {"/api/chat", "/api/embed"} for path, _ in calls)
        else:
            result = prewarm("http://demo.test", cfg, client=client)
            assert result["model_digests"] == MODELS
            assert calls[-1][1]["messages"] == []
            assert calls[-1][1]["options"]["num_ctx"] == 8192
            assert calls[-1][1]["keep_alive"] == "30m"
