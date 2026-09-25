import json
from pathlib import Path
import sys

from app.clients import AnswerVerdict, ModelAnswer, Models
from app.config import Settings

# Scripts import their helpers the way they do when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generation_ceiling import hosted_schema  # noqa: E402


def test_default_backend_is_the_local_model(monkeypatch):
    calls = []
    models = Models()
    monkeypatch.setattr(models, "_post", lambda path, body: calls.append(path) or {"ok": True})
    assert models._chat({"messages": []}) == {"ok": True}
    assert calls == ["/api/chat"]


def test_agent_policy_model_can_be_changed_without_changing_answer_model(monkeypatch):
    cfg = Settings(agent_policy_model="qwen2.5:3b", chat_model="qwen2.5:7b-instruct")
    monkeypatch.setattr("app.clients.settings", lambda: cfg)
    bodies = []
    models = Models()
    monkeypatch.setattr(models, "_post", lambda _path, body: bodies.append(body) or {
        "message": {"content": '{"action":"final","arguments":{},"purpose":"done"}'},
        "prompt_eval_count": 10, "eval_count": 5,
    })
    assert models.decide_agent_action("goal", [], 1).action == "final"
    assert bodies[0]["model"] == "qwen2.5:3b"
    assert cfg.chat_model == "qwen2.5:7b-instruct"


def test_a_swapped_backend_receives_the_unchanged_request():
    seen = []
    models = Models(chat_backend=lambda body: seen.append(body) or {"message": {"content": "{}"}})
    models._chat({"model": "x", "messages": [{"role": "user", "content": "q"}]})
    assert seen == [{"model": "x", "messages": [{"role": "user", "content": "q"}]}]


def test_hosted_schema_drops_unsupported_constraints_and_closes_every_object():
    schema = hosted_schema(ModelAnswer.model_json_schema())
    text = json.dumps(schema)
    assert "maxItems" not in text and "minLength" not in text and "maxLength" not in text
    claim = schema["$defs"]["ModelClaim"]
    assert claim["additionalProperties"] is False and set(claim["required"]) == {"text", "source_ids"}
    verdict = hosted_schema(AnswerVerdict.model_json_schema())
    assert "maximum" not in json.dumps(verdict)
