"""LangGraph orchestration; nodes delegate enterprise semantics to existing services."""

from typing import TypedDict

from fastapi import HTTPException
from langgraph.graph import END, START, StateGraph

from agent import controller as c
from agent.graph_runtime import graph_checkpointer, graph_identity, graph_signature
from agent.planner import compact_observation, evidence_gaps, recovery_query, retrieval_quality, subgoals
from app.clients import DependencyError
from app.task_analysis import historical_route_intent, multi_source_intent
from app.task_contract import TaskContract
from app.routing import execution_route


class RunState(TypedDict, total=False):
    signature: str
    refs: list[str]
    memory_context: list[dict]
    observations: list[dict]
    contract: dict | None
    preparation_trace: dict
    hybrid: dict
    planner: dict
    workflow_search: dict
    workflow_ready: bool
    supplement_index: int
    targeted: int
    decision: dict | None
    decision_step: int
    idle_steps: int
    stop: bool


def build_agent_graph(db, user, task, models, tools, started):
    def guard():
        if c.current_budget():
            c.current_budget().check()
        db.refresh(task)
        if task.status == "cancelled":
            raise c.TaskCancelled("task_cancelled")

    def prepare(state):
        guard()
        c._workflow_snapshot(db, task, phase="started_or_resumed")
        # The planner owns version and branch steps itself; a keyword contract must not preempt it.
        if not c.settings().task_contract_enabled or task.mode == "planner":
            return {"contract": None}
        from app.task_contract import build_contract
        from app.task_contract_runtime import prepare_contract
        contract = build_contract(task.goal)
        if contract.intent == "ordinary":
            return {"contract": None}
        initial = []
        if contract.intent != "conditional" and not task.input.get("document_id"):
            found = c._execute(db, task, tools, "search_documents", {"query": task.goal, "top_k": 8}, reuse=True)
            initial = c._evidence_from_refs(db, user, found.evidence_refs) if found and found.status == "ok" else []
        def call(name, args):
            return c._execute(db, task, tools, name, args, reuse=True)
        evidence, trace = prepare_contract(contract, initial,
            search=lambda query: call("search_documents", {"query": query, "top_k": 8}),
            load=lambda refs, historical: c._evidence_from_refs(db, user, refs, allow_historical=historical, limit=None),
            versions=lambda doc: call("get_document_version", {"document_id": doc, "limit": 20}),
            compare=lambda doc, old, new: call("compare_versions", {"document_id": doc, "from_version_id": old, "to_version_id": new}),
            open_version=lambda doc, vid: call("open_document", {"document_id": doc, "version_id": vid, "limit": 8}),
            document_id=task.input.get("document_id"))
        task.input = {**task.input, "_task_contract": contract.model_dump()}
        refs = [row["chunk_id"] for row in evidence]
        c._event(db, task, "task_contract", payload={"contract": contract.model_dump(), "preparation": trace}, refs=refs)
        return {"contract": contract.model_dump(), "preparation_trace": trace, "refs": refs}

    def memory(state):
        guard()
        adapter = getattr(tools, "memory", None)
        if not task.input.get("use_memory", False) or not adapter or not adapter.enabled:
            return {}
        result = c._execute(db, task, tools, "search_memory", {"query": task.goal, "limit": 3}, reuse=True)
        if result and result.status == "ok":
            return {"memory_context": result.data.get("memories", []), "observations": [
                compact_observation("search_memory", c._safe_summary(result), result.data, [])]}
        return {}

    def workflow_search(state):
        guard()
        if task.input.get("document_id"):
            return {"refs": c._workflow_versions(db, task, tools, task.input["document_id"])}
        result = c._execute(db, task, tools, "search_documents", {
            "query": task.goal, "top_k": 8 if multi_source_intent(task.goal) else 6}, reuse=True)
        quality = (retrieval_quality(task.goal, result.evidence_refs, result.data.get("matches", []))
                   if result and result.status == "ok" else "tool_error")
        c._event(db, task, "retrieval_assessment", payload={"quality": quality})
        return {"workflow_search": {"result": result.dump() if result else None, "quality": quality},
                "refs": list(dict.fromkeys((result.evidence_refs if result else []) + state["refs"]))}

    def recovery_route(state):
        info = state.get("workflow_search", {})
        retry = recovery_query(task.goal, task.goal)
        return ("workflow_recover" if info.get("quality") in {"empty", "irrelevant"}
                and task.step_no < task.max_steps and retry else "workflow_assess")

    def workflow_recover(state):
        guard()
        query = recovery_query(task.goal, task.goal)
        c._event(db, task, "query_recovery", payload={"retry_query": query,
                                                    "reason": state["workflow_search"]["quality"]})
        result = c._execute(db, task, tools, "search_documents", {"query": query, "top_k": 6}, reuse=True)
        return {"workflow_search": {"result": result.dump() if result else None},
                "refs": result.evidence_refs if result else state["refs"]}

    def workflow_assess(state):
        guard()
        refs = state["refs"]
        raw = state.get("workflow_search", {}).get("result")
        if not raw:
            return {"workflow_ready": True}
        result = c.ToolResult(**raw)
        historical = historical_route_intent(task.goal)
        if historical and result.status == "ok":
            documents = {row["document_id"] for row in c._evidence_from_refs(db, user, refs)}
            quality = retrieval_quality(task.goal, result.evidence_refs, result.data.get("matches", []))
            if len(documents) == 1 and quality == "candidate" and task.max_steps - task.step_no >= 2:
                document = next(iter(documents))
                c._event(db, task, "version_route", payload={"status": "routed", "document_id": document,
                    "reason": "single_retrieved_authorized_document"}, refs=refs)
                refs = refs + c._workflow_versions(db, task, tools, document)
            else:
                c._event(db, task, "version_route", payload={"status": "unresolved",
                    "reason": "ambiguous_document_or_budget", "document_count": len(documents)}, refs=refs)
        evidence = c._evidence_from_refs(db, user, refs, allow_historical=c._uses_historical_versions(db, task))
        progress = c._workflow_snapshot(db, task, evidence=evidence, phase="retrieval", blocked=result.status != "ok")
        ready = bool(progress and progress.evidence_ready and (not historical or c._uses_historical_versions(db, task)))
        if ready:
            c._event(db, task, "workflow_stop", payload={"reason": "evidence_coverage_ready",
                "basis": "lexical_coverage_proxy", "allowed_next": "retrieve_evidence_and_answer_validation"}, refs=refs)
        return {"refs": refs, "workflow_ready": ready}

    def supplement_route(state):
        raw = state.get("workflow_search", {}).get("result")
        items = subgoals(task.goal)
        return ("workflow_supplement" if raw and raw["status"] == "ok" and len(items) > 1
                and state["refs"] and not state.get("workflow_ready") and state.get("targeted", 0) < 2
                and state.get("supplement_index", 0) < len(items) and task.step_no < task.max_steps - 1
                else "workflow_detail")

    def workflow_supplement(state):
        guard()
        index = state.get("supplement_index", 0)
        item = subgoals(task.goal)[index]
        raw = state["workflow_search"]["result"]
        quality = retrieval_quality(item, raw["evidence_refs"], raw["data"].get("matches", []))
        c._event(db, task, "subgoal_evidence_assessment", payload={"subgoal": item, "quality": quality})
        update = {"supplement_index": index + 1}
        if quality in {"empty", "irrelevant"}:
            result = c._execute(db, task, tools, "search_documents", {"query": item, "top_k": 4}, reuse=True)
            update["targeted"] = state.get("targeted", 0) + 1
            if result and result.status == "ok":
                update["refs"] = list(dict.fromkeys(result.evidence_refs + state["refs"]))
        return update

    def workflow_detail(state):
        guard()
        if (state["refs"] and not task.input.get("document_id") and task.step_no < task.max_steps
                and not c._uses_historical_versions(db, task) and not task.input.get("_execution_route", {}).get("direct")):
            result = c._execute(db, task, tools, "retrieve_evidence", {"chunk_ids": state["refs"][:8]}, reuse=True)
            if result:
                return {"refs": result.evidence_refs}
        return {}

    def dynamic_route(state):
        if state.get('refs') and not historical_route_intent(task.goal):
            evidence = c._evidence_from_refs(db, user, state['refs'])
            if c.settings().answer_quality_enabled:
                from app.evidence_scope import attach_scope, scoped_windows
                scoped,_=attach_scope(db,user,task.goal,evidence)
                bound,info=scoped_windows(task.goal,scoped)
                if bound is not None and bound.answerable:
                    c._event(db,task,'deterministic_stop',payload={'reason':'entity_period_window_bound'},refs=state['refs'])
                    return 'finish'
            if not evidence_gaps(task.goal, evidence):
                c._event(db, task, 'deterministic_stop', payload={
                    'reason': 'evidence_coverage_ready', 'basis': 'candidate_coverage; final citation checks still required'},
                    refs=state['refs'])
                return 'dynamic_coverage'
            if (c.settings().answer_quality_enabled and len(subgoals(task.goal))>1
                    and execution_route(task.goal).get('reason')!='observation_dependent_candidate'):
                # The first policy chose retrieval. Explicit independent missing
                # subgoals already have a bounded shared retrieval policy; another
                # model turn to choose that same supplementation adds no decision.
                return 'dynamic_coverage'
        return "dynamic_coverage" if state.get("stop") or task.step_no >= task.max_steps else "dynamic_policy"

    def dynamic_policy(state):
        guard()
        try:
            decision = models.decide_agent_action(task.goal, state.get("observations", [])[-3:], task.step_no + 1)
        except DependencyError as exc:
            if exc.stage != "agent_policy":
                raise
            c._event(db, task, "policy_rejected", payload={"error_code": "unparseable_action"})
            task.step_no += 1
            task.state_version += 1
            task.updated_at = c.now()
            db.commit()
            idle = state.get("idle_steps", 0) + 1
            if idle >= 2:
                c._event(db, task, "deterministic_stop", payload={"reason": "no_new_evidence"})
            return {"decision": None, "observations": (state.get("observations", []) + [
                {"status": "error", "error_code": "unparseable_action"}])[-3:], "idle_steps": idle, "stop": idle >= 2}
        c._event(db, task, "policy_decision", payload={"action": decision.action, "purpose": decision.purpose})
        return {"decision": decision.model_dump(), "decision_step": task.step_no,
                "stop": decision.action == "final"}

    def dynamic_act(state):
        guard()
        decision = state["decision"]
        # A finished call in this same graph node can be replayed after a crash;
        # a new policy decision repeating an old call still consumes the budget.
        result = c._execute(db, task, tools, decision["action"], decision["arguments"],
                            reuse=task.step_no > state["decision_step"])
        refs = state["refs"]
        observations = state.get("observations", [])
        if result is None:
            observations = observations + [{"status": "error", "error_code": "repeated_call"}]
            idle = state.get("idle_steps", 0) + 1
            stop = idle >= 2
        else:
            known = set(refs)
            refs = refs + result.evidence_refs
            if decision["action"] in {"retrieve_evidence", "open_document", "compare_versions"}:
                refs = result.evidence_refs + refs
            observations = observations + [compact_observation(
                decision["action"], c._safe_summary(result), result.data, result.evidence_refs)]
            idle = 0 if set(refs) - known else state.get("idle_steps", 0) + 1
            stop = bool(idle >= 2 and refs)
        if stop:
            c._event(db, task, "deterministic_stop", payload={"reason": "no_new_evidence"})
        return {"refs": refs, "observations": observations[-3:], "idle_steps": idle, "stop": stop, "decision": None}

    def dynamic_coverage(state):
        guard()
        refs = list(dict.fromkeys(state['refs']))
        missing = evidence_gaps(task.goal, c._evidence_from_refs(db, user, refs))
        for item in missing[:2]:
            if task.step_no >= task.max_steps:
                break
            result = c._execute(db, task, tools, 'search_documents', {'query': item, 'top_k': 4}, reuse=True)
            if result and result.status == 'ok':
                # Keep the targeted lane first, retaining the other lane's facts.
                refs = list(dict.fromkeys(result.evidence_refs + refs))
                c._event(db, task, 'coverage_retrieval', payload={'subgoal': item,
                    'evidence_count': len(result.evidence_refs)}, refs=result.evidence_refs)
        remaining = evidence_gaps(task.goal, c._evidence_from_refs(db, user, refs))
        c._event(db, task, 'coverage_assessment', payload={'missing': remaining,
            'budget_exhausted': bool(remaining and task.step_no >= task.max_steps), 'basis': 'candidate_coverage'})
        return {'refs': refs}

    def finish(state):
        guard()
        contract = TaskContract.model_validate(state["contract"]) if state.get("contract") else None
        evidence = (c._evidence_from_refs(db, user, state["refs"], limit=None,
                    allow_historical=contract.intent == "history" and contract.version_chain_complete)
                    if contract else None)
        if task.mode == "planner":
            # Hop order matters and a selected historical version must survive to generation.
            evidence = c._evidence_from_refs(db, user, state["refs"], limit=12,
                                             allow_historical=bool(state.get("planner", {}).get("historical")))
        c._finish(db, user, task, models, state["refs"], state.get("memory_context"), tools=tools, started=started,
                  contract=contract, prepared_evidence=evidence, preparation_trace=state.get("preparation_trace"))
        return {}

    from agent.hybrid import build_hybrid_graph
    def authorize(state):
        guard()
        held = state.get("hybrid", {})
        refs = state.get("refs", []) + held.get("evidence_refs", []) + [
            ref for done in (state.get("planner") or {}).get("done", {}).values() for ref in done]
        for ref in dict.fromkeys(refs):
            c.require_chunk(db, user, ref, active_only=not c._uses_historical_versions(db, task))
        for document_id in held.get("known_documents", {}):
            c.require_document(db, user, document_id)

    def checked(node):
        def run(state):
            authorize(state)
            return node(state)
        return run
    graph = StateGraph(RunState)
    nodes = {"prepare": prepare, "memory": memory, "workflow_search": workflow_search,
             "workflow_recover": workflow_recover, "workflow_assess": workflow_assess,
             "workflow_supplement": workflow_supplement, "workflow_detail": workflow_detail,
             "dynamic_policy": dynamic_policy, "dynamic_act": dynamic_act,
             "dynamic_coverage": dynamic_coverage, "finish": finish}
    for name, node in nodes.items():
        graph.add_node(name, checked(node))
    if task.mode == "hybrid":
        hybrid = build_hybrid_graph(db, user, task, models, tools, execute=c._execute, event=c._event,
            evidence_from_refs=c._evidence_from_refs, walk_versions=c._workflow_versions,
            uses_history=c._uses_historical_versions, authorize=authorize)
        graph.add_node("hybrid", hybrid.compile())
    if task.mode == "planner":
        from agent.planned import build_planned_graph
        planned = build_planned_graph(db, user, task, models, tools, execute=c._execute, event=c._event,
                                      evidence_from_refs=c._evidence_from_refs)
        graph.add_node("planner", planned.compile())
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", lambda s: "finish" if s.get("contract") else "memory")
    graph.add_conditional_edges("memory", lambda s: "hybrid" if task.mode == "hybrid"
                                else "planner" if task.mode == "planner"
                                else "workflow_search" if task.mode == "workflow" else dynamic_route(s))
    graph.add_conditional_edges("workflow_search", lambda s: "finish" if task.input.get("document_id") else recovery_route(s))
    graph.add_edge("workflow_recover", "workflow_assess")
    graph.add_conditional_edges("workflow_assess", supplement_route)
    graph.add_conditional_edges("workflow_supplement", supplement_route)
    graph.add_edge("workflow_detail", "finish")
    graph.add_conditional_edges("dynamic_policy", lambda s: dynamic_route(s) if not s.get("decision") or s.get("stop") else "dynamic_act")
    graph.add_conditional_edges("dynamic_act", dynamic_route)
    graph.add_edge("dynamic_coverage", "finish")
    if task.mode == "hybrid":
        graph.add_edge("hybrid", "finish")
    if task.mode == "planner":
        graph.add_edge("planner", "finish")
    graph.add_edge("finish", END)
    return graph


def invoke_agent_graph(db, user, task, models, tools, started):
    config = graph_identity(task, user)
    signature = graph_signature(task)
    with graph_checkpointer(db) as saver:
        graph = build_agent_graph(db, user, task, models, tools, started).compile(checkpointer=saver)
        snapshot = graph.get_state(config)
        if snapshot.values:
            if snapshot.values.get("signature") != signature:
                raise HTTPException(409, "任务恢复期间代码、配置或输入已变化，请新建任务")
            refs = snapshot.values.get("refs", []) + snapshot.values.get("hybrid", {}).get("evidence_refs", []) + [
                ref for done in snapshot.values.get("planner", {}).get("done", {}).values() for ref in done]
            for ref in dict.fromkeys(refs):
                c.require_chunk(db, user, ref, active_only=not c._uses_historical_versions(db, task))
            initial = None
        else:
            if task.step_no:
                raise HTTPException(409, "旧任务没有 LangGraph 检查点，请新建任务")
            initial = {"signature": signature, "refs": [], "observations": [], "memory_context": [],
                       "idle_steps": 0, "stop": False, "supplement_index": 0, "targeted": 0}
        # Let the framework finish each checkpoint before advancing to the next
        # node; no application-managed checkpoint flush or write queue is needed.
        graph.invoke(initial, config, durability="sync")
    return task
