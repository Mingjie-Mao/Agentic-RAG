"""Opt-in post-completion slot observations; no production output or state writes."""
import hashlib
import json
from pathlib import Path
import time
from queue import Queue, Full
from threading import Lock, Thread

from app.config import settings
from app.execution_budget import ExecutionBudget, current_budget, use_budget
from app.semantic_evidence import SlotEvidenceEvaluator, aggregate_slots


class ShadowDispatcher:
    """Best-effort diagnostics: one daemon worker, at most one waiting job.

    Only record IDs cross threads; the observer opens its own database session.
    Pending work may be lost on process exit, never treated as a quality result.
    """

    def __init__(self):
        self.queue = Queue(maxsize=1)
        self.lock = Lock()
        self.pending = set()
        self.worker = None

    def submit(self, kind, run_id, user_id, sample_rate):
        if current_budget() is not None:
            return {"status": "skipped", "reason": "primary_budget_active"}
        if kind not in {"agent", "answer"}:
            return {"status": "skipped", "reason": "invalid_kind"}
        fraction = int(hashlib.sha256(f"{kind}:{run_id}".encode()).hexdigest()[:16], 16) / 2**64
        if fraction >= sample_rate:
            return {"status": "skipped", "reason": "not_sampled"}
        key = (kind, run_id, user_id)
        with self.lock:
            if key in self.pending:
                return {"status": "skipped", "reason": "already_pending"}
            if self.worker is None:
                self.worker = Thread(target=self._run, name="slot-shadow", daemon=True)
                self.worker.start()
            self.pending.add(key)
            try:
                self.queue.put_nowait(key)
            except Full:
                self.pending.remove(key)
                return {"status": "skipped", "reason": "queue_full"}
        return {"status": "queued"}

    def _run(self):
        while True:
            key = self.queue.get()
            try:
                observe_completed(*key)
            except Exception:
                # Even a diagnostic setup failure must not kill the worker or
                # propagate into a completed primary request.
                pass
            finally:
                with self.lock:
                    self.pending.remove(key)
                self.queue.task_done()


_dispatcher = ShadowDispatcher()


def schedule_completed(kind, run_id, user_id):
    try:
        return _dispatcher.submit(kind, run_id, user_id, settings().semantic_slot_shadow_sample_rate)
    except Exception:
        return {"status": "skipped", "reason": "dispatcher_unavailable"}


def observation_items(contract, payload):
    from agent.planner import subgoals
    if contract.intent == "ordinary":
        return [{"id": f"s{n}", "text": text} for n, text in enumerate(subgoals(contract.question), 1)]
    trace = payload.get("usage", {}).get("task_contract", {})
    condition = trace.get("condition") or {}
    active = condition.get("active_slot_ids", [])
    settled = condition.get("result") in {"true", "false"}
    history_pending = (contract.intent == "history"
                       and trace.get("contract", {}).get("version_chain_complete") is not True)
    return [dict(s.model_dump(), id=s.slot_id, text=s.query,
                 active=s.applies_if == "always" or not settled or s.slot_id in active,
                 prerequisite_pending=history_pending or (s.applies_if != "always" and not settled))
            for s in contract.slots]


def observe_completed(kind, run_id, user_id):
    # A nested caller's primary deadline must never include shadow work.
    if current_budget() is not None:
        return None
    if kind not in {"agent", "answer"}:
        return None
    from sqlalchemy import select
    from agent.planner import evidence_coverage
    from app.db import SessionLocal
    from app.models import AgentEvent, AgentTask, Answer, User
    from app.security import require_chunk
    from app.task_contract import build_contract

    cfg = settings().model_copy(update={
        "agent_task_timeout_seconds": settings().semantic_slot_shadow_seconds,
        "agent_judge_max_calls": 2, "agent_judge_token_budget": 32000,
    })
    budget = ExecutionBudget(cfg)
    started = time.monotonic()
    observation = {"version": "slot-shadow-v2", "kind": kind, "run_id": run_id,
                   "status": "unknown", "affects_control": False}
    path = Path(".runtime/slot-shadow") / kind / f"{run_id}.json"
    try:
        with SessionLocal() as db, use_budget(budget):
            user = db.get(User, user_id)
            record = db.get(AgentTask if kind == "agent" else Answer, run_id)
            if not user or not record or record.user_id != user.id:
                return None
            if kind == "agent":
                if record.status != "completed":
                    return None
                question, payload = record.goal, record.result
                events = db.scalars(select(AgentEvent).where(
                    AgentEvent.task_id == run_id, AgentEvent.event_type == "generation_context"
                ).order_by(AgentEvent.sequence)).all()
                refs = list(dict.fromkeys(ref for event in events for ref in event.evidence_chunk_ids))
            else:
                question, payload = record.question, record.payload
                refs = payload.get("trace", {}).get("generation_context_chunk_ids")
                if refs is None:
                    # Legacy answers did not retain the actual generation context.
                    # A citation subset cannot stand in for everything the model saw.
                    observation.update(reason="generation_context_unavailable")
                    refs = []
            # Capture bytes before observation: no writes to these records/events.
            baseline_digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            contract = build_contract(question)
            historical = contract.intent == "history"
            def authorize(p):
                chunk, version, doc = require_chunk(db, user, p["chunk_id"], active_only=not historical)
                return chunk.text == p["text"] and version.id == p["version_id"] and doc.id == p["document_id"]
            passages = []
            for cid in refs:
                chunk, version, doc = require_chunk(db, user, cid, active_only=not historical)
                passages.append({"chunk_id": cid, "text": chunk.text, "version_id": version.id,
                                 "document_id": doc.id, "title": doc.title})
            from scripts.run_unseen_benchmark import configuration
            signature = configuration()["models"]
            config_hash = hashlib.sha256(json.dumps({
                "question": question, "user_id": user_id,
                "models": signature, "seconds": cfg.semantic_slot_shadow_seconds,
                "code": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                         for name in ("app/semantic_evidence.py", "app/slot_shadow.py", "app/task_contract.py",
                                      "app/task_contract_runtime.py", "app/clients.py", "app/security.py",
                                      "app/config.py", "app/execution_budget.py", "agent/planner.py")}
            }, sort_keys=True).encode()).hexdigest()
            evidence_hash = hashlib.sha256(json.dumps(passages, sort_keys=True).encode()).hexdigest()
            # Re-authorization happened above. A completed task revisit must not
            # issue another judge call for identical evidence/configuration.
            if path.exists():
                cached = json.loads(path.read_text())
                if (cached.get("status") == "observed" and cached.get("payload_sha256") == baseline_digest
                        and cached.get("evidence_hash") == evidence_hash and cached.get("config_hash") == config_hash):
                    return cached
            # Use actual runtime splitter inputs for ordinary goals, shared contract
            # slots for structured goals. Unknown predicates cannot activate branches.
            items = observation_items(contract, payload)
            evaluator = SlotEvidenceEvaluator(authorize=authorize, model_fingerprint=signature)
            judgments = evaluator.coverage(question, [i for i in items if i.get("active", True)
                                                     and not i.get("prerequisite_pending")], passages)
            lexical = evidence_coverage([i["text"] for i in items], passages, question)
            observation.update(status="unknown" if observation.get("reason") else "observed", payload_sha256=baseline_digest,
                evidence_hash=evidence_hash, config_hash=config_hash, model_signature=signature,
                slots=items, judgments=[j.model_dump() for j in judgments], report=aggregate_slots(items, judgments),
                judge_context=evaluator.last_context, raw_judgments=evaluator.last_raw_judgments,
                lexical_covered={i["id"]: bool(lexical.get(i["text"])) for i in items},
                judge_usage=evaluator.usage.as_dict(), cache_hits=evaluator.cache_hits)
            if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != baseline_digest:
                raise RuntimeError("shadow changed primary payload")
    except Exception as exc:
        # A shadow timeout/ACL revocation/judge failure leaves a committed answer intact.
        observation.update(status="unknown", reason=type(exc).__name__)
    finally:
        observation.update(shadow_wall_ms=round((time.monotonic() - started) * 1000, 1),
                           execution_budget=budget.snapshot())
    try:
        from scripts.benchmark_runtime import atomic_json
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, observation)
    except OSError:
        return observation
    return observation
