import hashlib
import json

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_runtime import CheckpointStore, immutable_snapshot, summary


def test_atomic_checkpoint_explicit_resume_identity_and_no_repeat(tmp_path):
    path = tmp_path / "checkpoint.json"
    identity = {"inputs": {"a": "hash"}, "config": {"flag": False}}
    checkpoint = CheckpointStore(path, identity)
    checkpoint.begin("workflow", "case", "unique-token")
    checkpoint.attach("workflow", "case", "db-id")
    with pytest.raises(ValueError, match="explicit"):
        CheckpointStore(path, identity)
    with pytest.raises(ValueError, match="hashes differ"):
        CheckpointStore(path, {"inputs": {"a": "changed"}}, resume=True)
    resumed = CheckpointStore(path, identity, resume=True)
    assert resumed.data["pending"]["workflow:case"]["run_id"] == "db-id"
    with pytest.raises(ValueError, match="never rerun"):
        resumed.begin("workflow", "case", "another-token")
    resumed.finish("workflow", "case", {"task_id": "case", "status": "answered"})
    with pytest.raises(ValueError, match="immutable"):
        resumed.finish("workflow", "case", {"task_id": "case"})
    assert json.loads(path.read_text())["pending"] == {}
    assert not path.with_name(path.name + ".tmp").exists()


def test_snapshot_rejects_changed_input_or_snapshot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("frozen")
    frozen = {"inputs": {"code.py": hashlib.sha256(b"frozen").hexdigest()}}
    destination = immutable_snapshot(source, frozen, tmp_path / "snapshot")
    (source / "code.py").write_text("modified")
    assert (destination / "code.py").read_text() == "frozen"
    with pytest.raises(ValueError, match="input changed"):
        immutable_snapshot(source, frozen, tmp_path / "different")
    (destination / "code.py").write_text("modified")
    with pytest.raises(ValueError, match="snapshot changed"):
        immutable_snapshot(source, frozen, destination)


def test_metrics_do_not_treat_condition_violation_as_acl_or_unknown_time_as_zero():
    rows = [dict(task_id="a", status="answered", latency_ms=0, latency_source="unrecoverable", forbidden_present={"x": True}),
            dict(task_id="b", status="answered", latency_ms=4000, forbidden_present={"y": True}),
            dict(task_id="c", status="execution_failed", latency_ms=10000, execution_error="NonTermination", judge_usage={"calls": 9})]
    report = summary(rows, {"a": {"forbidden_kind": "conditional_violation"}, "b": {"forbidden_kind": "acl_content_leak"}})
    assert report["normal_cost"]["p50_seconds"] == 4
    assert report["normal_cost"]["unknown_latency"] == 1
    assert report["failure_cost"]["judge_calls"] == 9
    assert report["violations"]["conditional_violation"] == ["a"]
    assert report["violations"]["acl_content_leak"] == ["b"]
    assert report["non_termination"] == ["c"]


def test_checkpoint_exclusive_lock_releases_after_exception(tmp_path):
    from benchmark_runtime import checkpoint_lock
    path = tmp_path / "run.json"
    with pytest.raises(RuntimeError), checkpoint_lock(path):
        with pytest.raises(ValueError, match="already running"):
            with checkpoint_lock(path):
                pass
        raise RuntimeError("crash")
    with checkpoint_lock(path):
        pass


def test_committed_agent_result_recovers_even_if_checkpoint_lost_attach_step():
    from test_agent import agent_db
    from agent.controller import create_task, run_task
    from agent.tools import ToolResult
    from app.clients import Claim, GeneratedAnswer
    from run_unseen_benchmark import recover_checkpoint_attempt
    db, user = agent_db()
    class Tools:
        def call(self, *args):
            return ToolResult("ok", {}, ["c2"], {})
    class Model:
        def generate(self, question, evidence, **kwargs):
            return GeneratedAnswer(answerable=True, claims=[Claim(text="RPO为15分钟。", evidence_ids=["E1"], quotes=["RPO 为 15 分钟。"])]), {}
    task = create_task(db, user, "当前RPO是多少？", "workflow", 4, {"benchmark_run_token": "lost-attach-token"})
    run_task(db, user, task, models=Model(), tools=Tools())
    spec = dict(id="case", user=user.id, goal=task.goal, category="multi_hop", expected_status="answered", facts=["15分钟"], forbidden=[], scenario_events=[])
    result = recover_checkpoint_attempt(db, spec, "workflow", {"run_id": None, "token": "lost-attach-token"})
    assert result["run_id"] == task.id and result["answer_correct"]
    assert result["latency_ms"] is None and result["recovered_from_checkpoint"]
    assert task.status == "completed"


def test_frozen_configuration_records_model_digests_and_all_flags_without_credentials(monkeypatch, tmp_path):
    import httpx
    from app.config import settings
    from run_unseen_benchmark import configuration
    cfg = settings()
    hf = tmp_path / "hf"
    repository = hf / "hub" / ("models--" + cfg.rerank_model.replace("/", "--"))
    (repository / "refs").mkdir(parents=True)
    (repository / "refs/main").write_text("unit-test-revision")
    snapshot = repository / "snapshots/unit-test-revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"unit-test-fingerprint-only")
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HOME", str(hf))
    class Response:
        def __init__(self, body):
            self.body = body
        def raise_for_status(self):
            pass
        def json(self):
            return self.body
    def get(url, **kwargs):
        return Response({"version": "pinned"} if url.endswith("version") else {"models": [
            {"name": name, "digest": "sha256-" + name} for name in {cfg.chat_model, cfg.embed_model, cfg.agent_policy_model, cfg.semantic_judge_model}]})
    monkeypatch.setattr(httpx, "get", get)
    frozen = configuration()
    assert len(frozen["models"]) == 5
    encoder = frozen["models"]["cross_encoder"]
    assert encoder["revision"] and encoder["files"]["model.safetensors"]
    assert frozen["models"]["judge"]["digest"].startswith("sha256-")
    assert "demo_password" not in frozen["effective_configuration"]
    assert "memory_tokens_json" not in frozen["effective_configuration"]
    monkeypatch.setattr(cfg, "semantic_coverage_shadow", not cfg.semantic_coverage_shadow)
    assert frozen != configuration()


def test_benchmark_tasks_are_server_owned_and_not_claimed_by_background_workers(monkeypatch):
    from test_agent import agent_db
    from agent import controller
    from app.models import now
    from datetime import timedelta
    from contextlib import contextmanager
    db, user = agent_db()
    @contextmanager
    def factory():
        yield db
    monkeypatch.setattr(controller, "SessionLocal", factory)
    direct = controller.create_task(db, user, "基准任务", benchmark_execution=True)
    assert direct.status == "running" and direct.input["_benchmark_execution"] is True
    direct.lease_until = now() - timedelta(seconds=1)
    direct.attempts = 3
    db.commit()
    assert controller.claim_agent_task() is None and direct.status == "running"
    regular = controller.create_task(db, user, "普通任务", task_input={"_benchmark_execution": True})
    assert "_benchmark_execution" not in regular.input
    claimed = controller.claim_agent_task()
    assert claimed[0] == regular.id


@pytest.mark.parametrize("decode_s,prefill_s,ok", [(8.0, 18.0, True), (12.0, 18.0, False), (8.0, 30.0, False)])
def test_host_probe_is_uncached_and_flags_contended_hosts(decode_s, prefill_s, ok):
    from benchmark_runtime import host_throughput_probe

    sent = []

    def post(url, body):
        sent.append(body)
        return {"prompt_eval_count": 1800, "prompt_eval_duration": prefill_s * 1e9,
                "eval_count": 64, "eval_duration": decode_s * 1e9}

    probe = host_throughput_probe("http://h", "m", post=post)
    host_throughput_probe("http://h", "m", post=post)
    assert probe["meets_baseline"] is ok and probe["prefill_tok_s"] == round(1800 / prefill_s, 1)
    assert sent[0]["prompt"][:34] != sent[1]["prompt"][:34]  # fresh prefix each time
