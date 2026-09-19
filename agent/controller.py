"""Durable single-Agent controller with a cheap workflow baseline."""

import hashlib
import json
from datetime import timedelta

from fastapi import HTTPException
from sqlalchemy import func, or_, select

from agent.tools import KnowledgeTools, ToolResult
from app.clients import Models
from app.config import settings
from app.db import SessionLocal
from app.models import AgentEvent, AgentTask, ToolExecution, User, now, uid
from app.qa import validate_claims
from app.security import require_chunk
from app.task_analysis import acceptance_items, version_intent


def _event(db, task, event_type, *, tool_name=None, payload=None, refs=None):
    sequence = (
        db.scalar(select(func.max(AgentEvent.sequence)).where(AgentEvent.task_id == task.id)) or 0
    ) + 1
    row = AgentEvent(
        task_id=task.id,
        sequence=sequence,
        event_type=event_type,
        tool_name=tool_name,
        payload=payload or {},
        evidence_chunk_ids=refs or [],
    )
    db.add(row)
    db.commit()
    return row


def create_task(db, user, goal, mode="workflow", max_steps=6, task_input=None):
    task = AgentTask(
        tenant_id=user.tenant_id,
        user_id=user.id,
        goal=goal,
        input=task_input or {},
        mode=mode,
        max_steps=max_steps,
        status="queued",
    )
    db.add(task)
    db.commit()
    _event(db, task, "task_created", payload={"mode": mode, "max_steps": max_steps})
    return task


def require_task(db, user, task_id):
    task = db.get(AgentTask, task_id, populate_existing=True)
    if task is None or task.user_id != user.id or task.tenant_id != user.tenant_id:
        raise HTTPException(404, "任务不存在")
    return task


def _safe_summary(result):
    data = result.data
    return {
        "status": result.status,
        "error_code": result.error_code,
        "evidence_count": len(result.evidence_refs),
        "keys": list(data)[:12],
        "titles": list(dict.fromkeys(row.get("title") for row in data.get("matches", []) if row.get("title")))[:8],
    }


def _execute(db, task, tools, name, arguments, *, reuse=False):
    request_hash = hashlib.sha256(
        json.dumps({"tool": name, "arguments": arguments}, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if name == "compare_versions" and not (
        task.input.get("document_id") or version_intent(task.goal)
    ):
        _event(
            db,
            task,
            "tool_rejected",
            tool_name=name,
            payload={"error_code": "version_intent_required", "request_hash": request_hash},
        )
        task.step_no += 1
        task.state_version += 1
        task.updated_at = now()
        db.commit()
        return None
    previous = db.scalar(
        select(ToolExecution).where(
            ToolExecution.task_id == task.id, ToolExecution.request_hash == request_hash
        )
    )
    if previous and previous.status == "succeeded":
        if reuse:
            for chunk_id in previous.evidence_chunk_ids:
                require_chunk(db, tools.user, chunk_id)
            return ToolResult(
                "ok",
                {},
                previous.evidence_chunk_ids,
                {"checked_now": True, "replayed_from_checkpoint": True},
            )
        _event(
            db,
            task,
            "tool_rejected",
            tool_name=name,
            payload={"error_code": "repeated_call", "request_hash": request_hash},
        )
        task.step_no += 1
        task.state_version += 1
        task.updated_at = now()
        db.commit()
        return None
    execution = previous or ToolExecution(
        task_id=task.id,
        tool_name=name,
        request_hash=request_hash,
        arguments=arguments,
        status="running",
    )
    db.add(execution)
    db.commit()
    result = tools.call(name, arguments)
    execution.status = "succeeded" if result.status == "ok" else "failed"
    execution.result = _safe_summary(result)
    execution.evidence_chunk_ids = result.evidence_refs
    execution.error = result.error_code
    execution.completed_at = now()
    task.step_no += 1
    task.state_version += 1
    task.updated_at = now()
    db.commit()
    _event(
        db,
        task,
        "tool_completed",
        tool_name=name,
        payload={"arguments": arguments, **_safe_summary(result)},
        refs=result.evidence_refs,
    )
    return result


def _uses_historical_versions(db, task):
    if not (task.input.get("document_id") or version_intent(task.goal)):
        return False
    return bool(
        db.scalar(
            select(func.count())
            .select_from(ToolExecution)
            .where(
                ToolExecution.task_id == task.id,
                ToolExecution.tool_name == "compare_versions",
                ToolExecution.status == "succeeded",
            )
        )
    )


def _evidence_from_refs(db, user, refs, *, allow_historical=False):
    authorized = []
    for chunk_id in dict.fromkeys(refs):
        chunk, version, document = require_chunk(
            db, user, chunk_id, active_only=False
        )
        is_active = version.id == document.active_version_id
        if not allow_historical and not is_active:
            continue
        authorized.append(
            {
                "chunk_id": chunk.id,
                "document_id": document.id,
                "version_id": version.id,
                "title": (
                    f"{document.title} · 当前版本"
                    if is_active
                    else f"{document.title} · 历史版本 {version.created_at.date().isoformat()}"
                ),
                "text": chunk.text,
                "locator": chunk.locator,
                "metadata": document.metadata_json,
                "is_active": is_active,
            }
        )
    # Active evidence wins. Within it, round-robin documents so one version or long
    # document cannot consume all eight final slots before a conflicting source.
    buckets = {}
    for row in sorted(authorized, key=lambda item: not item["is_active"]):
        buckets.setdefault(row["document_id"], []).append(row)
    selected = []
    while buckets and len(selected) < 8:
        for document_id in list(buckets):
            selected.append(buckets[document_id].pop(0))
            if not buckets[document_id]:
                del buckets[document_id]
            if len(selected) == 8:
                break
    evidence = []
    for row in selected:
        row = {key: value for key, value in row.items() if key != "is_active"}
        evidence.append({"id": f"E{len(evidence) + 1}", **row})
    return evidence


def _finish(db, user, task, models, refs, memory_context=None):
    allow_historical = _uses_historical_versions(db, task)
    evidence = _evidence_from_refs(db, user, refs, allow_historical=allow_historical)
    if not evidence:
        payload = {
            "status": "insufficient_evidence",
            "claims": [],
            "citations": [],
            "message": "Agent 在预算内没有找到足够依据。",
        }
    else:
        if memory_context:
            generated, usage = models.generate(
                task.goal, evidence, memory_context=memory_context
            )
        else:
            generated, usage = models.generate(task.goal, evidence)
        policy_usage = getattr(models, "agent_policy_usage", None)
        if policy_usage:
            usage["agent_policy_prompt_tokens"] = policy_usage["prompt_tokens"]
            usage["agent_policy_completion_tokens"] = policy_usage["completion_tokens"]
            usage["total_prompt_tokens"] = (usage.get("prompt_tokens") or 0) + policy_usage[
                "prompt_tokens"
            ]
            usage["total_completion_tokens"] = (
                usage.get("completion_tokens") or 0
            ) + policy_usage["completion_tokens"]
        claims, status = validate_claims(generated, evidence)
        if status == "answered" and usage.get("answer_status") == "conflict":
            status = "conflict"
        used = {chunk_id for claim in claims for chunk_id in claim["evidence_ids"]}
        # Final re-authorization is mandatory, including historical-version evidence.
        for chunk_id in used:
            require_chunk(db, user, chunk_id, active_only=not allow_historical)
        citations = [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "version_id": row["version_id"],
                "title": row["title"],
                "locator": row["locator"],
                "text": row["text"],
                "preview_url": f"/api/evidence/{row['chunk_id']}",
                "original_url": f"/api/versions/{row['version_id']}/original",
            }
            for row in evidence
            if row["chunk_id"] in used
        ]
        payload = {
            "status": status,
            "claims": claims,
            "citations": citations,
            "message": "" if claims else "Agent 找到了资料，但最终答案未通过引用校验。",
            "usage": usage,
        }
    db.refresh(task)
    if task.status == "cancelled":
        return task
    task.status = "completed"
    task.result = payload
    task.evidence_chunk_ids = [row["chunk_id"] for row in evidence]
    task.lease_until = None
    task.lease_token = None
    task.updated_at = now()
    db.commit()
    _event(db, task, "task_completed", payload={"status": payload["status"]}, refs=refs)
    return task


def run_task(db, user, task, *, models=None, tools=None):
    if task.status == "cancelled":
        raise HTTPException(409, "任务已取消")
    if task.status == "completed":
        return task
    models = models or Models()
    tools = tools or KnowledgeTools(db, user, models=models)
    task.status = "running"
    task.error = None
    task.updated_at = now()
    db.commit()
    _event(db, task, "task_started", payload={"resumed": task.step_no > 0})
    refs, observations = [], []
    try:
        memory_context = []
        memory_adapter = getattr(tools, "memory", None)
        if task.input.get("use_memory", False) and memory_adapter and memory_adapter.enabled:
            memory = _execute(
                db,
                task,
                tools,
                "search_memory",
                {"query": task.goal, "limit": 3},
                reuse=True,
            )
            if memory and memory.status == "ok":
                memory_context = memory.data.get("memories", [])
                observations.append(
                    {
                        "tool": "search_memory",
                        **_safe_summary(memory),
                        "data": memory.data,
                    }
                )
        if task.mode == "workflow":
            document_id = task.input.get("document_id")
            if document_id:
                version_result = _execute(
                    db,
                    task,
                    tools,
                    "get_document_version",
                    {"document_id": document_id},
                    reuse=True,
                )
                observations.append(_safe_summary(version_result))
                comparison = _execute(
                    db,
                    task,
                    tools,
                    "compare_versions",
                    {
                        "document_id": document_id,
                        "from_version_id": task.input.get("from_version_id"),
                        "to_version_id": task.input.get("to_version_id"),
                    },
                    reuse=True,
                )
                if comparison:
                    refs.extend(comparison.evidence_refs)
            else:
                search = _execute(
                    db,
                    task,
                    tools,
                    "search_documents",
                    {"query": task.goal, "top_k": 6},
                    reuse=True,
                )
                if search:
                    refs.extend(search.evidence_refs)
                    if refs and task.step_no < task.max_steps:
                        detail = _execute(
                            db,
                            task,
                            tools,
                            "retrieve_evidence",
                            {"chunk_ids": refs[:8]},
                            reuse=True,
                        )
                        if detail:
                            refs = detail.evidence_refs
        else:
            while task.step_no < task.max_steps:
                db.refresh(task)
                if task.status == "cancelled":
                    return task
                decision = models.decide_agent_action(task.goal, observations[-6:], task.step_no + 1)
                _event(
                    db,
                    task,
                    "policy_decision",
                    payload={"action": decision.action, "purpose": decision.purpose},
                )
                if decision.action == "final":
                    break
                result = _execute(db, task, tools, decision.action, decision.arguments)
                if result is None:
                    observations.append({"status": "error", "error_code": "repeated_call"})
                    continue
                refs.extend(result.evidence_refs)
                if decision.action in {"retrieve_evidence", "open_document", "compare_versions"}:
                    refs = result.evidence_refs + refs
                observations.append(
                    {
                        "tool": decision.action,
                        **_safe_summary(result),
                        "handles": result.evidence_refs[:8],
                        "data": result.data,
                    }
                )
            # A dynamic policy may stop after answering only the first half of a
            # compound request. Use any remaining step for one deterministic,
            # ACL-scoped coverage retrieval over the original goal. Generation
            # still has to cite and validate the returned chunks.
            if len(acceptance_items(task.goal)) > 1 and task.step_no < task.max_steps:
                coverage = _execute(
                    db,
                    task,
                    tools,
                    "search_documents",
                    {"query": task.goal, "top_k": 8},
                    reuse=True,
                )
                if coverage:
                    refs = coverage.evidence_refs + refs
                    _event(
                        db,
                        task,
                        "coverage_retrieval",
                        payload={
                            "acceptance_items": acceptance_items(task.goal),
                            "evidence_count": len(coverage.evidence_refs),
                        },
                        refs=coverage.evidence_refs,
                    )
        return _finish(db, user, task, models, refs, memory_context)
    except Exception as exc:
        task.status = "failed"
        task.error = str(exc)[:1000]
        task.lease_until = None
        task.lease_token = None
        task.updated_at = now()
        db.commit()
        _event(db, task, "task_failed", payload={"error_type": type(exc).__name__})
        raise


def task_payload(db, user, task):
    require_task(db, user, task.id)
    events = db.scalars(
        select(AgentEvent)
        .where(AgentEvent.task_id == task.id)
        .order_by(AgentEvent.sequence)
    ).all()
    # Search/tool events can retain handles even when a task fails before copying
    # them onto AgentTask. Treat every persisted handle as a dependency so a later
    # revoke also hides the observable trace, not only the final answer.
    dependency_refs = list(task.evidence_chunk_ids)
    for event in events:
        dependency_refs.extend(event.evidence_chunk_ids)
    access_changed = False
    try:
        allow_historical = _uses_historical_versions(db, task)
        for chunk_id in dict.fromkeys(dependency_refs):
            require_chunk(db, user, chunk_id, active_only=not allow_historical)
        result = task.result
    except HTTPException:
        access_changed = True
        result = {
            "status": "access_changed",
            "claims": [],
            "citations": [],
            "message": "任务依赖的资料已删除或访问权限已变化，结果已隐藏。",
        }

    def public_event(row):
        payload = row.payload
        refs = row.evidence_chunk_ids
        if access_changed:
            # Keep operational shape only. Arguments, titles and model-authored
            # purposes can contain source-derived text and must disappear.
            payload = {
                key: payload[key]
                for key in ("status", "error_code", "evidence_count", "action")
                if key in payload
            }
            refs = []
        return {
            "sequence": row.sequence,
            "event_type": row.event_type,
            "tool_name": row.tool_name,
            "payload": payload,
            "evidence_refs": refs,
            "created_at": row.created_at.isoformat(),
        }

    return {
        "id": task.id,
        "goal": task.goal,
        "mode": task.mode,
        "status": task.status,
        "step_no": task.step_no,
        "max_steps": task.max_steps,
        "state_version": task.state_version,
        "result": result,
        "error": None if access_changed else task.error,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "events": [public_event(row) for row in events],
    }


def cancel_task(db, user, task_id):
    task = require_task(db, user, task_id)
    if task.status in {"completed", "cancelled"}:
        raise HTTPException(409, "任务已经结束")
    task.status = "cancelled"
    task.state_version += 1
    task.updated_at = now()
    db.commit()
    _event(db, task, "task_cancelled")
    return task


def resume_task(db, user, task_id):
    task = require_task(db, user, task_id)
    if task.status not in {"failed", "paused"}:
        raise HTTPException(409, "只有失败或暂停的任务可以恢复")
    task.status = "queued"
    task.attempts = 0
    task.error = None
    task.lease_until = None
    task.lease_token = None
    task.state_version += 1
    task.updated_at = now()
    db.commit()
    _event(db, task, "task_queued", payload={"resumed": True})
    return task


def claim_agent_task():
    """Atomically lease one queued task; expired leases may be recovered three times."""
    with SessionLocal() as db, db.begin():
        exhausted = db.scalars(
            select(AgentTask)
            .where(
                AgentTask.status == "running",
                AgentTask.lease_until < now(),
                AgentTask.attempts >= 3,
            )
            .with_for_update(skip_locked=True)
        ).all()
        for stale in exhausted:
            stale.status = "failed"
            stale.error = "任务多次中断，请检查模型和依赖服务后恢复"
            stale.lease_until = None
            stale.lease_token = None
            stale.updated_at = now()
        task = db.scalar(
            select(AgentTask)
            .where(
                or_(
                    AgentTask.status == "queued",
                    (AgentTask.status == "running") & (AgentTask.lease_until < now()),
                ),
                AgentTask.attempts < 3,
            )
            .order_by(AgentTask.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if task is None:
            return None
        task.status = "running"
        task.attempts += 1
        task.lease_token = uid()
        task.lease_until = now() + timedelta(seconds=settings().agent_lease_seconds)
        task.updated_at = now()
        return task.id, task.lease_token


def process_agent_task(task_id: str, token: str):
    with SessionLocal() as db:
        task = db.get(AgentTask, task_id)
        if not task or task.status != "running" or task.lease_token != token:
            return
        user = db.get(User, task.user_id)
        if user is None or not user.active:
            task.status = "failed"
            task.error = "任务用户不存在或已停用"
            task.lease_until = None
            task.lease_token = None
            db.commit()
            return
        run_task(db, user, task)
