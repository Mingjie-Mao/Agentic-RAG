"""Durable single-Agent controller with a cheap workflow baseline."""

import hashlib
import json
import time
from datetime import timedelta

from fastapi import HTTPException
from sqlalchemy import func, or_, select

from agent.planner import (
    carry_forward_query,
    checklist,
    evidence_coverage,
    repair_question,
    subgoals,
    uncovered_items,
    version_depth,
    version_pairs,
    version_points,
)
from agent.tools import KnowledgeTools, ToolResult
from agent.workflow_state import WorkflowState
from app.clients import Models
from app.config import conflict_check_enabled, settings
from app.db import SessionLocal
from app.execution_budget import BudgetExceeded, ExecutionBudget, TaskCancelled, current_budget, use_budget
from app.models import AgentEvent, AgentTask, ToolExecution, User, now, uid
from app.qa import (
    answer_verdict,
    merge_verdict_usage,
    repair_exact_values,
    shadow_scores,
    validate_claims,
)
from app.security import require_chunk, require_document
from app.task_analysis import (
    conflict_intent,
    version_intent,
    historical_route_intent,
)


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


def workflow_like(task):
    """Workflow, or the first (cheap) pass of an adaptive task before any escalation."""
    return task.mode == "workflow" or (task.mode == "adaptive" and not task.input.get("_cascade_escalated"))


def planner_active(task):
    """Planner, or an adaptive task after its escalation to the planner."""
    return task.mode == "planner" or (task.mode == "adaptive" and bool(task.input.get("_cascade_escalated")))


def create_task(db, user, goal, mode="workflow", max_steps=6, task_input=None, *, benchmark_execution=False):
    task = AgentTask(
        tenant_id=user.tenant_id,
        user_id=user.id,
        goal=goal,
        input={key: value for key, value in (task_input or {}).items() if key not in
               {"_workflow_state", "_execution_route", "_execution_budget", "_hybrid_state", "_task_contract", "_benchmark_execution", "_benchmark_scenario_events", "_langgraph_state", "_langgraph_config", "_planner_historical", "_planner_values", "_cascade_escalated", "_configured_max_steps"}},
        mode=mode,
        max_steps=max_steps,
        status="queued",
    )
    if benchmark_execution:
        task.input = {**task.input, "_benchmark_execution": True}
        task.status = "running"
    if mode in {'workflow', 'adaptive'} and settings().adaptive_routing_enabled:
        from app.routing import execution_route
        route = execution_route(goal, document_id=task.input.get('document_id'))
        # Explicit workflow always remains deterministic. Auto can select dynamic
        # in the API; only server-derived direct routing is persisted here.
        task.input = {**task.input, '_execution_route': route if route['mode'] == 'workflow'
                      else {'mode': 'workflow', 'reason': 'explicit_workflow'}}
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
    if current_budget():
        current_budget().check()
    request_hash = hashlib.sha256(
        json.dumps({"tool": name, "arguments": arguments}, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    previous = db.scalar(
        select(ToolExecution).where(
            ToolExecution.task_id == task.id, ToolExecution.request_hash == request_hash
        )
    )
    if task.step_no >= task.max_steps and not (reuse and previous and previous.status == "succeeded"):
        _event(db, task, "tool_rejected", tool_name=name, payload={"error_code": "step_budget_exhausted"})
        return None
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
    if previous and previous.status == "succeeded":
        if reuse:
            if arguments.get("document_id"):
                document = require_document(db, tools.user, arguments["document_id"])
                checkpoint_active = (previous.result or {}).get("checkpoint_data", {}).get("active_version_id")
                if checkpoint_active and checkpoint_active != document.active_version_id:
                    raise HTTPException(409, "任务恢复期间文档版本已变化，请新建任务")
            for chunk_id in previous.evidence_chunk_ids:
                require_chunk(db, tools.user, chunk_id, active_only=name != "compare_versions")
            return ToolResult(
                "ok",
                (previous.result or {}).get("checkpoint_data", {}),
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
    started = time.monotonic()
    result = tools.call(name, arguments)
    wall_ms = round((time.monotonic() - started) * 1000, 1)
    execution.status = "succeeded" if result.status == "ok" else "failed"
    execution.result = {
        **_safe_summary(result), "wall_ms": wall_ms, "usage": result.usage or {},
        "checkpoint_data": {key: result.data[key] for key in (
            "matches", "versions", "active_version_id", "document_id"
        ) if key in result.data},
    }
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
        payload={"arguments": arguments, **_safe_summary(result), "wall_ms": wall_ms},
        refs=result.evidence_refs,
    )
    return result


def _uses_historical_versions(db, task):
    if task.input.get("_task_contract", {}).get("intent") == "history" and task.input["_task_contract"].get("version_chain_complete"):
        return True
    # Set only by the planner after it selected a dated version (never by user input).
    if planner_active(task) and task.input.get("_planner_historical"):
        return True
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


def _evidence_from_refs(db, user, refs, *, allow_historical=False, limit=8):
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
                # Match shared retrieval's source identity. Decorating a current
                # title made the local generator abstain on otherwise identical
                # supported Dev inputs. Historical sources keep their scope label.
                "title": (
                    document.title
                    if is_active
                    else f"{document.title} · 历史版本 {version.created_at.date().isoformat()}"
                ),
                "text": chunk.text,
                "locator": chunk.locator,
                "metadata": document.metadata_json,
                "is_active": is_active,
            }
        )
    if limit is None:
        return [dict(row, id=f"E{i}") for i, row in enumerate(authorized, 1)]
    # Active evidence wins. Within it, round-robin documents so one version or long
    # document cannot consume all eight final slots before a conflicting source.
    buckets = {}
    for row in sorted(authorized, key=lambda item: not item["is_active"]):
        buckets.setdefault(row["document_id"], []).append(row)
    selected = []
    while buckets and len(selected) < limit:
        for document_id in list(buckets):
            selected.append(buckets[document_id].pop(0))
            if not buckets[document_id]:
                del buckets[document_id]
            if len(selected) == limit:
                break
    evidence = []
    for row in selected:
        row = {key: value for key, value in row.items() if key != "is_active"}
        evidence.append({"id": f"E{len(evidence) + 1}", **row})
    return evidence


def _merge_usage(usage, extra):
    for key in (
        "prompt_tokens", "completion_tokens", "model_duration_ms", "model_load_ms",
        "prompt_eval_ms", "completion_eval_ms",
    ):
        if extra.get(key):
            usage[key] = (usage.get(key) or 0) + extra[key]
    return usage


def _repair_coverage(db, user, task, models, tools, items, claims, evidence, pool, usage, *, facts=None):
    """Answer the subgoals the first pass left open, and nothing else.

    A weak policy model tends to restate a conditional request instead of executing it,
    and to stop after the first half of a compound one. Which items are still open is
    decided lexically by `agent.planner`; this step looks again for those items only
    and generates the gap. It runs at most once, keeps every claim the first pass
    already earned, and every added claim goes through the same citation validation.
    """
    missing = uncovered_items(items, claims, task.goal)
    _event(
        db,
        task,
        "coverage_check",
        payload={"items": items, "missing": missing, "covered": len(items) - len(missing)},
    )
    if not missing:
        return claims, usage
    focused = evidence
    if tools is not None and task.step_no < task.max_steps:
        # The second hop of a latent link is not reachable from the question alone,
        # so the first hop's authorized text is carried into the query.
        gap = _execute(
            db,
            task,
            tools,
            "search_documents",
            {"query": carry_forward_query(missing, evidence), "top_k": 6},
            reuse=True,
        )
        if gap and gap.evidence_refs:
            allow_historical = _uses_historical_versions(db, task)
            # The repair generates on what was retrieved *for the missing item*. Mixing
            # it back into the first pass's material is what buried the answer the
            # second search had just found.
            focused = _evidence_from_refs(db, user, gap.evidence_refs, allow_historical=allow_historical)
            pool.update({row["chunk_id"]: row for row in focused})
    if not focused:
        return claims, usage
    if settings().passage_window_enabled:
        from app.passages import prepare_passages
        focused, trace = prepare_passages(db, user, repair_question(items, missing), focused,
                                          settings(), historical=_uses_historical_versions(db, task))
        pool.update({row['chunk_id']: row for row in focused})
        _event(db, task, 'passage_selection', payload=trace,
               refs=[row['chunk_id'] for row in focused])
    _event(db, task, 'generation_context', payload={'stage': 'coverage_repair'},
           refs=[row['chunk_id'] for row in focused])
    options={'acceptance_items_override':checklist(missing),'check_conflict':False,'max_output_tokens':400}
    if settings().answer_quality_enabled:
        from app.answer_quality import recover_answer
        generated,extra=recover_answer(models,repair_question(items,missing),focused,options=options)
    else:
        generated,extra=models.generate(repair_question(items,missing),focused,**options)
    added, status = validate_claims(generated, focused)
    if extra.get('quality_validation_failed'):
        added,status=[],'verification_failed'
    _event(
        db,
        task,
        "coverage_repair",
        payload={"missing": missing, "status": status, "added_claims": len(added)},
    )
    if status != "answered":
        return claims, _merge_usage(usage, extra)
    if facts is not None:
        from app.source_facts import public_facts
        facts.extend(public_facts(generated.facts, added))
    seen = {claim["text"] for claim in claims}
    return claims + [claim for claim in added if claim["text"] not in seen], _merge_usage(usage, extra)


def _workflow_snapshot(db, task, *, evidence=None, claims=None, status=None, phase, blocked=False):
    if not workflow_like(task):
        return None
    items = subgoals(task.goal)
    state = WorkflowState.restore(task.input.get("_workflow_state", {}), items)
    if evidence is not None:
        state.observe(evidence_coverage(items, evidence, task.goal), blocked=blocked)
    if claims is not None:
        # Apply the same coverage proxy to single-item asks too; the older compound
        # checker deliberately skips those and cannot certify a single answer.
        covered = evidence_coverage(items, [
            {"chunk_id": str(index), "text": claim["text"]}
            for index, claim in enumerate(claims, 1)
        ], task.goal)
        missing = set(uncovered_items(items, claims, task.goal))
        missing.update(item for item in items if not covered[item])
        state.check_answers(list(missing), claims, valid=status in {"answered", "conflict"})
    snapshot = state.snapshot()
    task.input = {**task.input, "_workflow_state": snapshot}
    task.state_version += 1
    refs = list(dict.fromkeys(ref for row in state.items for ref in row["evidence_refs"]))
    _event(db, task, "subgoal_state", payload={"phase": phase, **snapshot}, refs=refs)
    return state


def _workflow_versions(db, task, tools, document_id):
    refs = []
    if task.step_no >= task.max_steps:
        return refs
    result = _execute(db, task, tools, "get_document_version", {"document_id": document_id}, reuse=True)
    if result is None or result.status != "ok":
        _event(db, task, "version_route", payload={"status": "blocked", "reason": "version_lookup_failed"})
        return refs
    versions = result.data.get("versions", [])
    selection = None
    if settings().focused_generation_enabled:
        from app.temporal import select_effective_versions
        selection = select_effective_versions(task.goal, versions)
        _event(db, task, "effective_selection", payload=selection)
    pairs = version_pairs(versions, task.goal, task.max_steps - task.step_no)
    selected_pairs = None
    if selection and selection['status'] == 'selected':
        wanted = {row['version_id'] for row in selection['selections']}
        positions = [index for index, row in enumerate(versions) if row['version_id'] in wanted]
        low, high = min(positions), max(positions)
        if low == high:
            low, high = (low, low + 1) if low + 1 < len(versions) else (max(0, low - 1), low)
        selected_pairs = [(versions[index + 1]['version_id'], versions[index]['version_id'])
                          for index in range(low, high)]
        pairs = selected_pairs[:max(0, task.max_steps - task.step_no)]
    if not pairs and len(versions) >= 2:
        pairs = [(task.input.get("from_version_id"), task.input.get("to_version_id"))]
    # Explicit dates/labels must not shrink merely because a requested version is
    # absent. A generic history request can cover the whole available short chain.
    requested = (max(version_points(task.goal) - 1, version_depth(task.goal))
                 if version_points(task.goal) > 1
                 else min(version_depth(task.goal), max(0, len(versions) - 1)))
    if selected_pairs is not None:
        requested = len(selected_pairs)
    _event(db, task, "version_plan", payload={
        "document_id": document_id, "pairs": len(pairs), "requested_pairs": requested,
        "budget_limited": len(pairs) < min(requested, max(0, len(versions) - 1)),
        "chain_limited": max(0, len(versions) - 1) < requested,
    })
    completed_pairs = 0
    for older, newer in pairs:
        if task.step_no >= task.max_steps:
            break
        comparison = _execute(db, task, tools, "compare_versions", {
            "document_id": document_id, "from_version_id": older, "to_version_id": newer,
        }, reuse=True)
        if comparison and comparison.status == "ok":
            refs.extend(comparison.evidence_refs)
            completed_pairs += 1
    _event(db, task, "version_route", payload={
        "status": "complete" if requested and completed_pairs == requested else "unresolved",
        "requested_pairs": requested, "completed_pairs": completed_pairs,
        "reason": "version_chain_checked" if completed_pairs == requested and requested
        else "incomplete_version_chain",
    }, refs=refs)
    return refs


def _finish(db, user, task, models, refs, memory_context=None, tools=None, started=None,
            contract=None, prepared_evidence=None, preparation_trace=None):
    if current_budget():
        current_budget().check()
    allow_historical = _uses_historical_versions(db, task)
    if contract and contract.intent == "history":
        allow_historical = contract.version_chain_complete
    evidence = prepared_evidence if prepared_evidence is not None else _evidence_from_refs(db, user, refs, allow_historical=allow_historical)
    passage_trace = None
    if settings().passage_window_enabled and evidence and not contract:
        from app.passages import prepare_passages
        evidence, passage_trace = prepare_passages(
            db, user, task.goal, evidence, settings(), historical=allow_historical)
        _event(db, task, "passage_selection", payload=passage_trace,
               refs=[row['chunk_id'] for row in evidence])
    _workflow_snapshot(db, task, evidence=evidence, phase="before_generation")
    # The hybrid mode walks versions with the same routine, so it gets the same guard:
    # an incompletely read history is refused, never answered from the current version.
    needs_history = (workflow_like(task) or task.mode == "hybrid") and bool(
        task.input.get("document_id") or historical_route_intent(task.goal)
    )
    version_plan = db.scalar(select(AgentEvent).where(
        AgentEvent.task_id == task.id, AgentEvent.event_type == "version_plan"
    ).order_by(AgentEvent.sequence.desc())) if needs_history else None
    version_route = db.scalar(select(AgentEvent).where(
        AgentEvent.task_id == task.id, AgentEvent.event_type == "version_route"
    ).order_by(AgentEvent.sequence.desc())) if needs_history else None
    incomplete_history = needs_history and (
        not allow_historical or bool(version_plan and version_plan.payload.get("budget_limited"))
        or bool(version_route and version_route.payload.get("status") in {"blocked", "unresolved"})
    )
    effective_selection = db.scalar(select(AgentEvent).where(
        AgentEvent.task_id == task.id, AgentEvent.event_type == 'effective_selection'
    ).order_by(AgentEvent.sequence.desc())) if settings().focused_generation_enabled else None
    if effective_selection:
        selected = effective_selection.payload
        incomplete_history |= selected['status'] == 'unresolved'
        if selected['status'] == 'selected':
            wanted = {row['version_id'] for row in selected['selections']}
            evidence = [dict(row, metadata={**(row.get('metadata') or {}),
                        'requested_effective_dates': [item['date'] for item in selected['selections']
                                                     if item['version_id'] == row['version_id']]})
                        for row in evidence if row['version_id'] in wanted]
    if contract:
        incomplete_history = contract.intent == "history" and not contract.version_chain_complete
    if not evidence or incomplete_history:
        payload = {
            "status": "insufficient_evidence",
            "claims": [],
            "citations": [],
            "message": "未能唯一定位并完整读取所需历史版本，不能用当前资料代替历史答案。"
            if incomplete_history else "Agent 在预算内没有找到足够依据。",
            "usage": {},
        }
    else:
        if settings().answer_quality_enabled and not allow_historical:
            scope_trace=None
            from app.evidence_scope import attach_scope
            evidence, scope_trace = attach_scope(db,user,task.goal,evidence)
        # Evidence can be re-selected by the coverage step; a chunk that was already
        # cited must keep its citation, so every row seen stays available here.
        pool = {row["chunk_id"]: row for row in evidence}
        items = subgoals(task.goal)
        compound = len(items) > 1 and not conflict_intent(task.goal) and not contract
        # The first pass is left exactly as the single-turn path runs it. Handing the
        # model its own subgoal checklist here was measured in a paired A/B
        # (`make checklist-ab`) and changed nothing, while a splitter that cuts an
        # enumeration short would hand the model a list shorter than the question.
        # No measured gain, non-zero risk, so the subgoals only drive the step below.
        options = {"memory_context": memory_context} if memory_context else {}
        if planner_active(task) and task.input.get("_planner_values"):
            # Program-verified bridge values (literal spans), mapped to this context's ids.
            ids = {row["chunk_id"]: row["id"] for row in evidence}
            options["verified_steps"] = [
                {"what": v["what"], "values": v["values"], "source_quote": v["quote"], "evidence_id": ids.get(v["chunk_id"])}
                for v in task.input["_planner_values"] if v["chunk_id"] in ids]
        if not conflict_check_enabled(task.tenant_id):
            options["check_conflict"] = False
        generated_started = time.monotonic()
        _event(db, task, 'generation_context', payload={'stage': 'first'},
               refs=[row['chunk_id'] for row in evidence])
        if contract:
            from app.task_contract import contract_generate
            generated, usage = contract_generate(models, task.goal, evidence, contract=contract, options=options)
            usage["contract_preparation"] = preparation_trace
        elif settings().task_contract_enabled and not planner_active(task):
            # The planner already resolved versions and branches from its own steps;
            # a keyword contract would re-demand a full version chain it never needed.
            from app.task_contract import contract_generate
            generated, usage = contract_generate(models, task.goal, evidence, options=options)
        else:
            if settings().answer_quality_enabled:
                from app.answer_quality import recover_answer
                generated, usage = recover_answer(models, task.goal, evidence, options=options)
            else:
                generated, usage = models.generate(task.goal, evidence, **options)
        if passage_trace is not None:
            usage['passage_selection'] = passage_trace
        if settings().answer_quality_enabled and not allow_historical:
            usage['scope_preparation']=scope_trace
        usage["generation_wall_ms"] = round((time.monotonic() - generated_started) * 1000, 1)
        usage['generation_context_chunk_ids'] = [row['chunk_id'] for row in evidence]
        if planner_active(task) and task.input.get("_planner_values") and generated.answerable:
            from agent.planned import cite_bridges
            generated, usage["bridge_citations_added"] = cite_bridges(
                task.goal, generated, evidence, task.input["_planner_values"])
        claims, status = validate_claims(generated, evidence)
        if usage.get('quality_validation_failed'):
            claims,status=[],'verification_failed'
        usage["answer_validation"] = {
            "generated_answerable": generated.answerable,
            "generated_claim_count": len(generated.claims),
            "validated_claim_count": len(claims),
            "status": status,
            "reason": (('contract_' + usage['task_contract']['status'])
                       if not generated.answerable and contract else
                       "generator_abstained" if not generated.answerable else
                       "claim_validation_failed" if status == "verification_failed" else "validated"),
        }
        facts = list(generated.facts)
        if status == "answered" and usage.get("answer_status") == "conflict":
            status = "conflict"
        _workflow_snapshot(db, task, claims=claims, status=status, phase="first_answer_check")
        if compound and status == "answered" and claims and usage.get('evidence_scope',{}).get('status')!='bound_window_comparison':
            repair_started = time.monotonic()
            claims, usage = _repair_coverage(
                db, user, task, models, tools, items, claims, evidence, pool, usage,
                facts=facts if settings().source_facts_enabled else None,
            )
            usage["coverage_repair_wall_ms"] = round((time.monotonic() - repair_started) * 1000, 1)
            usage['final_coverage'] = {'missing': uncovered_items(items, claims, task.goal),
                'tool_budget_exhausted': task.step_no >= task.max_steps,
                'basis': 'lexical_diagnostic; strict correctness assessed offline'}
        if status == "answered" and claims and not contract:
            exact_started = time.monotonic()
            claims, usage = repair_exact_values(
                models, task.goal, list(pool.values()), claims, usage,
                facts=facts if settings().source_facts_enabled else None,
            )
            if usage.get("exact_value_slots_missing_first_pass"):
                usage["exact_value_repair_wall_ms"] = round((time.monotonic() - exact_started) * 1000, 1)
                _event(
                    db, task, "exact_value_check",
                    payload={
                        "missing": usage["exact_value_slots_missing_first_pass"],
                        "status": usage.get("exact_value_repair_status"),
                        "added_claims": usage.get("exact_value_repair_claims", 0),
                    },
                )
        # A judgment question gets its verdict as a field. It is read out of the claims
        # the pipeline already validated, so it restates a conclusion without adding a
        # fact, and it is the last step: repaired claims are included.
        from app.source_facts import public_facts
        facts = public_facts(facts, claims)
        if contract:
            verdict, verdict_usage = None, {}
        else:
            verdict, verdict_usage = answer_verdict(models, task.goal, claims, status,
                                                    evidence=evidence,
                                                    verified_verdict=usage.get('verified_verdict') or usage.get('evidence_scope',{}).get('verdict'),
                                                    facts=facts if settings().source_facts_enabled
                                                    and not usage.get('extraction_fallback') else None)
        usage = merge_verdict_usage(usage, verdict_usage)
        if settings().focused_generation_enabled or settings().adaptive_routing_enabled:
            usage['execution_route'] = task.input.get('_execution_route', {'mode': task.mode})
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
            for key in ("wall_ms", "model_duration_ms", "model_load_ms", "prompt_eval_ms", "completion_eval_ms"):
                # Unreported server timings stay null, not a fabricated 0 ms.
                usage[f"agent_policy_{key}"] = policy_usage.get(key)
        executions = db.scalars(select(ToolExecution).where(ToolExecution.task_id == task.id)).all()
        usage["tool_wall_ms"] = round(
            sum((row.result or {}).get("wall_ms", 0) for row in executions), 1
        )
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
            for chunk_id, row in pool.items()
            if chunk_id in used
        ]
        evidence = list(pool.values())
        payload = {
            "status": status,
            "claims": claims,
            "verdict": verdict,
            **({"facts": facts} if settings().source_facts_enabled else {}),
            "citations": citations,
            "message": ("" if claims else "未能确认所需条件或历史事实，资料不足以支持完整回答。"
                        if contract and status == 'insufficient_evidence' else "Agent 找到了资料，但生成器未能依据这些资料给出完整回答。"
                        if status == "insufficient_evidence" else "Agent 找到了资料，但最终答案未通过引用校验。"),
            "usage": usage,
            "shadow_scores": (
                shadow_scores(task.goal, evidence, claims, semantic=settings().semantic_shadow_enabled)
                if evidence else None
            ),
        }
    if started is not None:
        payload["usage"]["task_wall_ms"] = round((time.monotonic() - started) * 1000, 1)
    if "tool_wall_ms" not in payload["usage"]:
        executions = db.scalars(select(ToolExecution).where(ToolExecution.task_id == task.id)).all()
        payload["usage"]["tool_wall_ms"] = round(
            sum((row.result or {}).get("wall_ms", 0) for row in executions), 1
        )
    policy_usage = getattr(models, "agent_policy_usage", None)
    if policy_usage and "agent_policy_wall_ms" not in payload["usage"]:
        payload["usage"].update({f"agent_policy_{key}": value for key, value in policy_usage.items()})
    db.refresh(task)
    if task.status == "cancelled":
        return task
    state = _workflow_snapshot(db, task, evidence=evidence, claims=payload.get("claims", []),
                               status=payload["status"], phase="final_answer_check")
    if state:
        _event(db, task, "workflow_stop", payload={
            "reason": "answer_items_covered" if all(row["state"] == "answer_covered" for row in state.items)
            else "bounded_plan_finished_with_unresolved_items",
            "basis": "lexical_coverage_proxy; not semantic correctness",
        })
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
    from agent.graph_runtime import graph_identity
    db.refresh(user)
    graph_identity(task, user)
    result = _budgeted_run_task(db, user, task, models=models, tools=tools)
    if settings().semantic_slot_shadow_enabled and result.status == "completed":
        from app.slot_shadow import schedule_completed
        schedule_completed("agent", result.id, user.id)
    return result


def _budgeted_run_task(db, user, task, *, models=None, tools=None):
    if task.status in {"completed", "cancelled"}:
        return _run_task(db, user, task, models=models, tools=tools)

    def persist(snapshot):
        if task.lease_token and task.status == "running":
            task.lease_until = now() + timedelta(seconds=max(settings().agent_lease_seconds, settings().model_timeout_seconds + 5))
        task.input = {**task.input, "_execution_budget": snapshot}
        db.commit()

    def cancelled():
        return db.scalar(select(AgentTask.status).where(AgentTask.id == task.id)) == "cancelled"

    budget = ExecutionBudget(settings(), task.input.get("_execution_budget"),
                             persist=persist, cancelled=cancelled)
    try:
        with use_budget(budget):
            return _run_task(db, user, task, models=models, tools=tools)
    except TaskCancelled:
        task.status = "cancelled"
        task.lease_until = task.lease_token = None
        db.commit()
        return task
    except BudgetExceeded as exc:
        db.rollback()
        task.status, task.error = "failed", exc.reason
        task.lease_until = task.lease_token = None
        task.updated_at = now()
        db.commit()
        _event(db, task, "budget_exhausted", payload={"reason": exc.reason})
        raise
    finally:
        db.rollback()
        if task.result:
            task.result = {**task.result, "usage": {**task.result.get("usage", {}),
                                                   "execution_budget": budget.snapshot()}}
        persist(budget.snapshot())


def _run_task(db, user, task, *, models=None, tools=None):
    started = time.monotonic()
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
    try:
        from agent.graph import invoke_agent_graph
        from app.retrieval import routing_goal

        # Model-written queries only: Dynamic/Hybrid/Planner, and Adaptive once escalated.
        with routing_goal(task.goal, fuse=lambda: task.mode in {"dynamic", "hybrid"} or planner_active(task)):
            return invoke_agent_graph(db, user, task, models, tools, started)
    except TaskCancelled:
        task.status = "cancelled"
        task.lease_until = None
        task.lease_token = None
        db.commit()
        return task
    except Exception as exc:
        db.rollback()
        db.refresh(task)
        # The answer may already be committed when a checkpoint write fails.
        # Preserve that terminal result rather than enqueueing a second generation.
        if task.status == "completed":
            return task
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
                AgentTask.input["_benchmark_execution"].as_boolean().is_not(True),
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
                AgentTask.input["_benchmark_execution"].as_boolean().is_not(True),
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
