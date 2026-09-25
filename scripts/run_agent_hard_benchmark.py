"""Run and score the 30-task Hard Agent Benchmark."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from fastapi import HTTPException
from sqlalchemy import select

from agent.controller import create_task, run_task, task_payload
from agent.tools import KnowledgeTools, ToolResult
from app.db import SessionLocal
from app.models import Chunk, Document, DocumentVersion, ToolExecution, User
from app.qa import answer_question


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = "fixtures/agent/hard_tasks.json"
CATEGORIES = {
    "multi_hop", "temporal_version", "conditional_planning",
    "query_recovery", "security_state", "efficiency_stopping",
}
CONTROLLED_FIELDS = {"fault_script", "state_change", "memory_fixture"}


# ISO timestamps carry hours, minutes and seconds that look exactly like forbidden
# values: a task that runs at 15:20 UTC puts "t15:" into every event it writes, and a
# boundary match for the forbidden value "15" hits it. Leaving them in made the
# security metric depend on the wall clock, so they are removed before any scan.
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|z)?")


def normalized(value):
    return "".join(str(value).lower().split())


def scannable(value):
    return TIMESTAMP.sub(" ", normalized(value))


def _literal_pattern(value):
    escaped = re.escape(normalized(value))
    left = r"(?<![0-9.])" if escaped and escaped[0].isdigit() else ""
    right = r"(?![0-9.])" if escaped and escaped[-1].isdigit() else ""
    return left + escaped + right


def fact_matches(matcher, text):
    """Match a fact without allowing numeric substrings such as 50 in 250."""
    matcher = {"id": str(matcher), "aliases": [str(matcher)]} if isinstance(matcher, str) else matcher
    aliases = matcher.get("aliases", [])
    patterns = matcher.get("patterns", [])
    return any(re.search(_literal_pattern(alias), text) for alias in aliases) or any(
        re.search(pattern, text, re.IGNORECASE) for pattern in patterns
    )


def _task_matchers(task):
    return task.get("fact_matchers") or task.get("facts", [])


def is_controlled(task):
    return any(field in task for field in CONTROLLED_FIELDS)


def validate_suite(suite, manifest, hard_documents=None):
    errors = []
    tasks = suite.get("tasks", [])
    hard_documents = hard_documents or []
    manifest_ids = {row["id"] for row in manifest}
    hard_ids = {row["document_id"] for row in hard_documents}
    known_ids = manifest_ids | hard_ids
    counts = Counter()
    ids = [row.get("id") for row in tasks]
    if len(tasks) != 30:
        errors.append(f"expected 30 tasks, found {len(tasks)}")
    if len(set(ids)) != len(ids):
        errors.append("task ids must be unique")
    if set(suite.get("categories", {})) != CATEGORIES:
        errors.append("suite category declaration does not match the six required categories")
    for document in hard_documents:
        if len(document.get("versions", [])) < 3:
            errors.append(f"{document.get('document_id')}: hard document needs at least 3 versions")
    for task in tasks:
        task_id = task.get("id", "<missing>")
        category = task.get("category")
        counts[category] += 1
        if category not in CATEGORIES:
            errors.append(f"{task_id}: unknown category {category}")
            continue
        for field in ("user", "goal", "expected_status", "facts", "forbidden", "scenario_events"):
            if field not in task:
                errors.append(f"{task_id}: missing {field}")
        for matcher in task.get("fact_matchers", []):
            if not isinstance(matcher, dict) or not matcher.get("id") or not (
                matcher.get("aliases") or matcher.get("patterns")
            ):
                errors.append(f"{task_id}: invalid fact matcher")
        document_ids = set(task.get("expected_evidence_documents", []))
        transition = task.get("conditional_transition") or {}
        document_ids.update(
            value for value in (transition.get("observe_document"), transition.get("then_document")) if value
        )
        state_change = task.get("state_change") or {}
        if state_change.get("document_id"):
            document_ids.add(state_change["document_id"])
        unknown = document_ids - known_ids
        if unknown:
            errors.append(f"{task_id}: unknown documents {sorted(unknown)}")
        expected_events = set()
        if task.get("fault_script"):
            expected_events.add("first_search_miss")
        if task.get("state_change"):
            expected_events.add("state_change")
        if task.get("memory_fixture"):
            expected_events.add("memory_fixture")
        if set(task.get("scenario_events", [])) != expected_events:
            errors.append(f"{task_id}: scenario_events do not match configured scenario")
        if task.get("fault_script") not in {None, "first_search_empty"}:
            errors.append(f"{task_id}: unsupported fault script")
        if category == "query_recovery" and not (
            task.get("fault_script") == "first_search_empty" and task.get("recovery_required")
        ):
            errors.append(f"{task_id}: recovery task must inject a first-search miss")
        if category == "security_state" and not task.get("security_event_required"):
            errors.append(f"{task_id}: security task must require security handling")
        if category == "efficiency_stopping" and "expected_evidence_documents" not in task:
            errors.append(f"{task_id}: stopping task needs gold evidence")
    for category in CATEGORIES:
        if counts[category] != 5:
            errors.append(f"{category}: expected 5 tasks, found {counts[category]}")
        if suite.get("categories", {}).get(category) != counts[category]:
            errors.append(f"{category}: declared count differs from actual")
    version_tasks = [
        row for row in tasks
        if set(row.get("expected_evidence_documents", [])) & hard_ids
        and {"get_document_version", "compare_versions"} <= set(row.get("required_tools", []))
    ]
    if hard_ids and len(version_tasks) < 3:
        errors.append("at least three temporal tasks must exercise the hard-only version chain")
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "tasks": len(tasks),
        "categories": dict(counts),
        "documents": len(known_ids),
        "shared_comparable_subset": sum(not is_controlled(row) for row in tasks),
        "controlled_agent_only_subset": sum(is_controlled(row) for row in tasks),
    }


class _MemoryFlag:
    def __init__(self, enabled):
        self.enabled = enabled


class ScenarioTools:
    """Keep deterministic benchmark faults outside production tool code."""

    def __init__(self, db, user, task, models):
        self.db, self.user, self.task = db, user, task
        self.base = KnowledgeTools(db, user, models=models)
        self.memory = _MemoryFlag(bool(task.get("memory_fixture")))
        self.calls = []
        self.scenario_events = {
            "first_search_miss": False,
            "state_change": False,
            "memory_fixture": False,
        }
        self._original = None

    def _apply_state_change(self, tool_name):
        change = self.task.get("state_change")
        if not change or self.scenario_events["state_change"] or change.get("after_tool") != tool_name:
            return
        if sum(row["tool"] == tool_name for row in self.calls) != change.get("occurrence", 1):
            return
        if change.get("action") != "revoke_document":
            raise ValueError(f"unsupported state change: {change.get('action')}")
        document = self.db.get(Document, change["document_id"], populate_existing=True)
        if document is None:
            raise RuntimeError(f"missing state-change document {change['document_id']}")
        self._original = (document, list(document.read_groups), document.tenant_public, document.revision)
        document.read_groups, document.tenant_public, document.revision = [], False, document.revision + 1
        self.db.commit()
        self.scenario_events["state_change"] = True

    def call(self, name, arguments):
        self.calls.append({"tool": name, "arguments": arguments})
        if name == "search_memory" and self.task.get("memory_fixture"):
            rows = [
                {"id": f"fixture-{index}", "content": content, "status": "active"}
                for index, content in enumerate(self.task["memory_fixture"], 1)
            ]
            self.scenario_events["memory_fixture"] = True
            result = ToolResult("ok", {"memories": rows, "candidates_considered": len(rows)}, [], {
                "scope": "synthetic_benchmark_namespace", "checked_now": True
            })
        elif (
            name == "search_documents"
            and self.task.get("fault_script") == "first_search_empty"
            and sum(row["tool"] == name for row in self.calls) == 1
        ):
            self.scenario_events["first_search_miss"] = True
            result = ToolResult("ok", {"query": arguments.get("query"), "matches": [], "candidate_count": 0}, [], {
                "scope": "injected_first_search_miss", "checked_now": True
            })
        else:
            result = self.base.call(name, arguments)
        self._apply_state_change(name)
        return result

    def restore(self):
        if not self._original:
            return
        document, groups, public, revision = self._original
        document.read_groups, document.tenant_public, document.revision = groups, public, revision
        self.db.commit()


def _expected_statuses(task):
    status = task["expected_status"]
    return set(status if isinstance(status, list) else [status])


def _documents_for_refs(db, refs):
    if not refs:
        return []
    rows = db.execute(
        select(Chunk.id, Document.id)
        .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Chunk.id.in_(refs))
    ).all()
    mapping = dict(rows)
    return list(dict.fromkeys(mapping[ref] for ref in refs if ref in mapping))


def _tool_trace(db, task_id):
    rows = db.scalars(
        select(ToolExecution).where(ToolExecution.task_id == task_id)
        .order_by(ToolExecution.created_at, ToolExecution.id)
    ).all()
    return [{
        "tool": row.tool_name,
        "arguments": row.arguments,
        "status": row.status,
        "result": row.result,
        "error": row.error,
        "evidence_count": len(row.evidence_chunk_ids),
        "evidence_documents": _documents_for_refs(db, row.evidence_chunk_ids),
    } for row in rows]


def _transition_satisfied(trace, transition):
    if not transition:
        return None
    observed_at = next(
        (index for index, row in enumerate(trace)
         if transition["observe_document"] in row.get("evidence_documents", [])), None
    )
    followed_at = next(
        (index for index, row in enumerate(trace)
         if transition["then_document"] in row.get("evidence_documents", [])
         and observed_at is not None and index > observed_at), None
    )
    return observed_at is not None and followed_at is not None


def _sufficiency(task, trace):
    if "expected_evidence_documents" not in task or not trace:
        return None, None, None
    expected = set(task["expected_evidence_documents"])
    accumulated = set()
    sufficient = None
    for index, row in enumerate(trace, 1):
        accumulated.update(row.get("evidence_documents", []))
        if expected <= accumulated and (expected or row["tool"] == "search_documents"):
            sufficient = index
            break
    if sufficient is None:
        return None, None, True
    allowances = Counter(task.get(
        "allowed_tools_after_sufficient", ["retrieve_evidence", "verify_chunk_access"]
    ))
    extra = 0
    for row in trace[sufficient:]:
        tool = row["tool"]
        if allowances[tool]:
            allowances[tool] -= 1
        else:
            extra += 1
    return sufficient, extra, False


def score_task(
    task, payload, trace, steps, latency_ms, *, observable_payload=None,
    scenario_events=None, execution_error_kind=None
):
    claims_text = normalized(" ".join(row.get("text", "") for row in payload.get("claims", [])))
    matchers = _task_matchers(task)
    facts = {
        (row.get("id") if isinstance(row, dict) else str(row)): fact_matches(row, claims_text)
        for row in matchers
    }
    observable_text = scannable(json.dumps(
        observable_payload if observable_payload is not None else payload, ensure_ascii=False
    ))
    forbidden = {value: fact_matches(value, observable_text) for value in task["forbidden"]}
    citation_documents = list(dict.fromkeys(
        row.get("document_id") for row in payload.get("citations", []) if row.get("document_id")
    ))
    tools = [row["tool"] for row in trace]
    status_ok = payload.get("status") in _expected_statuses(task)
    execution_ok = execution_error_kind in {None, "authorization_change"}
    answer_correct = status_ok and execution_ok and all(facts.values()) and not any(forbidden.values())
    capabilities = {}
    if task.get("min_source_documents"):
        capabilities["cross_document_evidence"] = len(set(citation_documents)) >= task["min_source_documents"]
    if "expected_evidence_documents" in task and task["expected_evidence_documents"]:
        capabilities["gold_evidence_coverage"] = set(task["expected_evidence_documents"]) <= set(citation_documents)
    transition = _transition_satisfied(trace, task.get("conditional_transition"))
    if transition is not None:
        capabilities["conditional_transition"] = transition
    for tool in task.get("required_tools", []):
        capabilities[f"required_tool:{tool}"] = tool in tools
    recovery_applicable = bool(task.get("recovery_required"))
    if recovery_applicable:
        searches = [row for row in trace if row["tool"] == "search_documents"]
        queries = {normalized(row.get("arguments", {}).get("query", "")) for row in searches} - {""}
        capabilities["query_recovery"] = (
            bool(searches) and searches[0]["evidence_count"] == 0
            and any(row["evidence_count"] > 0 for row in searches[1:]) and len(queries) >= 2
        )
    scenario_events = scenario_events or {}
    for event in task.get("scenario_events", []):
        capabilities[f"{event}_exercised"] = bool(scenario_events.get(event))
    if task.get("security_event_required") and not task.get("scenario_events"):
        capabilities["security_boundary_respected"] = not any(forbidden.values())
    sufficient_step, extra_calls, early_stop = _sufficiency(task, trace)
    forbidden_tools = sorted(set(tools) & set(task.get("forbidden_tools", [])))
    over_planned = None
    if task["category"] == "efficiency_stopping":
        over_planned = bool(forbidden_tools) or bool(extra_calls)
        capabilities["efficient_stop"] = not over_planned and not early_stop
    capability_success = all(capabilities.values())
    return {
        "category": task["category"], "controlled": is_controlled(task),
        "status": payload.get("status"), "expected_status": task["expected_status"],
        "status_ok": status_ok, "execution_error_kind": execution_error_kind,
        "facts": facts, "forbidden_present": forbidden,
        "answer_correct": answer_correct, "capabilities": capabilities,
        "capability_success": capability_success,
        "task_success": answer_correct and capability_success,
        "citation_documents": citation_documents, "tool_trace": trace, "steps": steps,
        "forbidden_tool_calls": forbidden_tools, "over_planned": over_planned,
        "early_stop": early_stop, "evidence_sufficient_step": sufficient_step,
        "extra_calls_after_sufficient_evidence": extra_calls,
        "recovery_applicable": recovery_applicable,
        "recovered": capabilities.get("query_recovery"),
        "scenario_events": scenario_events,
        "latency_ms": round(latency_ms, 1), "usage": payload.get("usage", {}),
    }


def run_arm(db, user, task, arm):
    if arm == "rag" and is_controlled(task):
        return {"task_id": task["id"], "category": task["category"], "controlled": True,
                "skipped": True, "reason": "controlled scenario requires the agent tool boundary"}
    started = time.monotonic()
    execution_error = execution_error_kind = None
    if arm == "rag":
        payload = answer_question(db, user, task["goal"])
        observable, trace, steps, run_id, scenario_events = payload, [], 1, None, {}
    else:
        task_input = {"use_memory": bool(task.get("memory_fixture")), "benchmark_id": task["id"]}
        task_input.update(task.get("task_input", {}))
        row = create_task(db, user, task["goal"], arm, task.get("max_steps", 8), task_input)
        tools = ScenarioTools(db, user, task, models=None)
        try:
            run_task(db, user, row, tools=tools)
        except Exception as exc:
            execution_error = f"{type(exc).__name__}: {exc}"
            execution_error_kind = (
                "authorization_change"
                if isinstance(exc, HTTPException) and exc.status_code in {403, 404, 409}
                else "unexpected"
            )
            db.rollback()
            db.expire_all()
        try:
            observable = task_payload(db, user, row)
            payload = observable["result"] or {
                "status": "execution_failed", "claims": [], "citations": []
            }
            steps, trace = observable["step_no"], _tool_trace(db, row.id)
            scenario_events = dict(tools.scenario_events)
        finally:
            tools.restore()
        run_id = row.id
    result = score_task(
        task, payload, trace, steps, (time.monotonic() - started) * 1000,
        observable_payload=observable, scenario_events=scenario_events,
        execution_error_kind=execution_error_kind,
    )
    result.update({"task_id": task["id"], "run_id": run_id})
    if execution_error:
        result["execution_error"] = execution_error
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)] if ordered else None


def summarize(rows):
    scored = [row for row in rows if not row.get("skipped")]
    recoveries = [row for row in scored if row["recovery_applicable"]]
    stopping = [row for row in scored if row["over_planned"] is not None]
    early_stops = [row for row in scored if row["early_stop"] is not None]
    prompt_tokens = [
        row["usage"].get("total_prompt_tokens", row["usage"].get("prompt_tokens", 0)) or 0
        for row in scored
    ]
    count = len(scored)
    phase_keys = (
        "agent_policy_wall_ms", "tool_wall_ms", "generation_wall_ms",
        "coverage_repair_wall_ms", "exact_value_repair_wall_ms", "verdict_model_duration_ms",
        "task_wall_ms",
    )
    phase_latency = {
        key: {
            "recorded": len(values),
            "p50_ms": round(statistics.median(values), 1) if values else None,
            "p95_ms": round(percentile(values, .95), 1) if values else None,
        }
        for key in phase_keys
        if (values := [row["usage"][key] for row in scored if key in row.get("usage", {})])
    }
    def ratio(numerator, denominator):
        return round(numerator / denominator, 3) if denominator else None
    return {
        "requested": len(rows), "scored": count, "skipped": len(rows) - count,
        "answer_correct": sum(row["answer_correct"] for row in scored),
        "answer_correct_rate": ratio(sum(row["answer_correct"] for row in scored), count),
        "task_success": sum(row["task_success"] for row in scored),
        "task_success_rate": ratio(sum(row["task_success"] for row in scored), count),
        "recovery_rate": ratio(sum(bool(row["recovered"]) for row in recoveries), len(recoveries)),
        "over_planning_rate": ratio(sum(bool(row["over_planned"]) for row in stopping), len(stopping)),
        "early_stop_rate": ratio(
            sum(bool(row["early_stop"]) for row in early_stops), len(early_stops)
        ),
        "security_leaks": sum(any(row["forbidden_present"].values()) for row in scored),
        "unexpected_execution_failures": sum(row["execution_error_kind"] == "unexpected" for row in scored),
        "mean_steps": round(statistics.mean(row["steps"] for row in scored), 2) if scored else None,
        "p50_latency_ms": round(statistics.median(row["latency_ms"] for row in scored), 1) if scored else None,
        "p95_latency_ms": round(percentile([row["latency_ms"] for row in scored], .95), 1) if scored else None,
        "mean_prompt_tokens": round(statistics.mean(prompt_tokens), 1) if scored else None,
        "phase_latency": phase_latency,
    }


def summarize_arm(rows):
    scored = [row for row in rows if not row.get("skipped")]
    shared = [row for row in rows if not row.get("controlled")]
    controlled = [row for row in rows if row.get("controlled")]
    return {
        "shared_comparable_subset": summarize(shared),
        "controlled_agent_only_subset": summarize(controlled),
        "all_scored_tasks": summarize(rows),
        "by_category": {
            category: summarize([row for row in scored if row["category"] == category])
            for category in sorted(CATEGORIES)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--arms", nargs="+", choices=["rag", "workflow", "dynamic"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--out", default="artifacts/agent-hard-benchmark-v2.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    arms = args.arms or ["rag", "workflow", "dynamic"]
    raw = (ROOT / args.tasks).read_bytes()
    suite = json.loads(raw)
    manifest = json.loads((ROOT / "fixtures/manifest.json").read_text())
    hard_documents = json.loads((ROOT / "fixtures/agent/hard_documents.json").read_text())
    validation = validate_suite(suite, manifest, hard_documents)
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    tasks = suite["tasks"]
    if args.ids:
        wanted = set(args.ids)
        tasks = [row for row in tasks if row["id"] in wanted]
        missing = wanted - {row["id"] for row in tasks}
        if missing:
            raise SystemExit(f"unknown task ids: {', '.join(sorted(missing))}")
    tasks = tasks[:args.limit]
    results = {arm: [] for arm in arms}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    def output_payload():
        return {
            "benchmark": suite["version"], "task_sha256": hashlib.sha256(raw).hexdigest(),
            "validation": validation, "arms": arms, "results": results,
            "summary": {arm: summarize_arm(rows) for arm, rows in results.items() if rows},
            "requested_tasks": len(tasks),
            "note": (
                "开发 benchmark，不是训练集或独立留出集。跨 arm 只能比较 "
                "shared_comparable_subset；受控场景仅用于 Agent 内部诊断。"
            ),
        }

    with SessionLocal() as db:
        for task in tasks:
            user = db.get(User, task["user"])
            if user is None:
                raise RuntimeError(f"missing seeded user {task['user']}")
            for arm in arms:
                print(f"{task['id']} {arm} ...", flush=True)
                results[arm].append(run_arm(db, user, task, arm))
            out.write_text(json.dumps(output_payload(), ensure_ascii=False, indent=2) + "\n")
    output = output_payload()
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
