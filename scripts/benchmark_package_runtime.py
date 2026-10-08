"""Benchmark-only adapters over the existing runner and shared retrieval boundary."""
from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import patch

from scripts.benchmark_package_scoring import provisional_score
from scripts.research_review import digest


def method_matrix(suite):
    from scripts.benchmark_package_scoring import METHODS
    rows = suite.get("method_matrix", [{"name": m, "mode": m, "overrides": {}} for m in METHODS])
    if [r["name"] for r in rows] != suite["arms"] or len({r["name"] for r in rows}) != len(rows):
        raise ValueError("registered matrix and arm names differ")
    mapping = {r["name"]: r for r in rows}
    if not set(METHODS) <= set(mapping) or any(mapping[m] != {"name": m, "mode": m, "overrides": {}} for m in METHODS):
        raise ValueError("four main methods must share unchanged conditions")
    for row in rows:
        # "planner" is an experimental extra arm, never one of the four headline methods.
        if row["mode"] not in METHODS + ("planner",):
            raise ValueError("invalid execution mode")
        for key, value in row["overrides"].items():
            if key in {"passage_rerank", "task_contract_enabled"}:
                valid = type(value) is bool
            elif key == "retrieval_mode":
                valid = value in {"bm25", "dense", "hybrid"}
            elif key == "rewrite_mode":
                valid = value in {"off", "rule"}
            else:
                valid = False
            if not valid:
                raise ValueError("unregistered/invalid ablation setting")
    return mapping


def reserve_test_run(freeze, output, *, resume=False):
    from pathlib import Path
    import json
    from scripts.benchmark_runtime import atomic_json, checkpoint_lock
    freeze, output = Path(freeze), Path(output).resolve()
    ledger = freeze.with_name("registered-run.json")
    identity = {"freeze_sha256": digest(json.loads(freeze.read_text())), "output": str(output)}
    with checkpoint_lock(ledger):
        if ledger.exists():
            if json.loads(ledger.read_text()) != identity or not resume:
                raise ValueError("test matrix already reserved; only explicit resume of the same output is allowed")
        else:
            if resume:
                raise ValueError("no registered test run to resume")
            atomic_json(ledger, identity)


def validate_test_worker(root, freeze, output):
    import hashlib
    import json
    from pathlib import Path
    root, freeze = Path(root).resolve(), Path(freeze).resolve()
    if root.parent.name != "benchmark-snapshots" or root.parent.parent.name != ".runtime" or root.name != hashlib.sha256(freeze.read_bytes()).hexdigest():
        raise ValueError("test worker must execute the immutable registered snapshot")
    origin = root.parents[2] / freeze.relative_to(root)
    ledger = origin.with_name("registered-run.json")
    identity = {"freeze_sha256": digest(json.loads(freeze.read_text())), "output": str(Path(output).resolve())}
    if not ledger.exists() or json.loads(ledger.read_text()) != identity:
        raise ValueError("test worker has no matching original run registration")


def indexed_corpus(spec):
    """Freeze actual parsed/indexed IDs, source hashes, ACLs, users and version order.

    Raw bytes alone cannot identify a corpus re-ingested under different chunk IDs.
    Bind actual search text/vector hashes as well as the PostgreSQL records.
    """
    from sqlalchemy import select
    from app.db import SessionLocal
    from app.models import Chunk, Document, DocumentVersion, User
    import hashlib
    from pathlib import Path
    from app.clients import Search
    root = Path(__file__).resolve().parents[1]
    records = []
    with SessionLocal() as db:
        for item in sorted(spec["documents"], key=lambda d: d["document_id"]):
            doc = db.get(Document, item["document_id"])
            if not doc or not doc.active_version_id:
                raise ValueError("benchmark corpus not published: " + item["document_id"])
            versions = list(db.scalars(select(DocumentVersion).where(DocumentVersion.document_id == doc.id).order_by(DocumentVersion.created_at, DocumentVersion.id)))
            expected = {hashlib.sha256((root / v["path"]).read_bytes()).hexdigest() for v in item["versions"]}
            if {v.content_hash for v in versions} != expected:
                raise ValueError("indexed source versions differ from package: " + doc.id)
            if (doc.tenant_id != item.get("tenant_id", spec["tenant"]["id"])
                    or doc.owner_id != item.get("owner", spec["owner"])
                    or set(doc.read_groups) != set(item["groups"])
                    or doc.tenant_public != item["tenant_public"] or doc.deleted):
                raise ValueError("indexed ACL/tenant differs from package: " + doc.id)
            records.append({"document": doc.id, "tenant": doc.tenant_id, "active_version": doc.active_version_id,
                "owner": doc.owner_id, "acl": [doc.read_groups, doc.tenant_public, doc.revision], "versions": [
                    {"id": v.id, "source_sha256": v.content_hash, "created_at": v.created_at.isoformat(),
                     "chunks": [{"id": c.id, "ordinal": c.ordinal, "text_sha256": digest(c.text), "locator": c.locator}
                         for c in db.scalars(select(Chunk).where(Chunk.version_id == v.id).order_by(Chunk.ordinal))]}
                    for v in versions]})
        users = []
        for item in sorted(spec["users"], key=lambda u: u["id"]):
            user = db.get(User, item["id"])
            if not user:
                raise ValueError("benchmark user missing: " + item["id"])
            if (user.tenant_id != item.get("tenant_id", spec["tenant"]["id"])
                    or set(user.groups) != set(item["groups"])
                    or user.role != item["role"] or not user.active):
                raise ValueError("indexed user permissions differ from package: " + user.id)
            users.append({"id": user.id, "tenant": user.tenant_id, "groups": user.groups,
                          "role": user.role, "active": user.active})
    search = Search()
    ids = [c["id"] for d in records for v in d["versions"] for c in v["chunks"]]
    indexed = []
    for offset in range(0, len(ids), 256):
        found = search.request("POST", f"/{search.index}/_mget", json={"ids": ids[offset:offset+256]})["docs"]
        if any(not d.get("found") for d in found):
            raise ValueError("benchmark search chunks missing")
        indexed.extend({"id": d["_id"], "source_sha256": digest(d["_source"])} for d in found)
    return {"documents": records, "users": users, "search_documents": indexed,
            "search_mapping": search.request("GET", f"/{search.index}/_mapping")}


def run_package_arm(db, user, task, arm, **options):
    from fastapi import HTTPException
    from sqlalchemy import select
    from app.models import Chunk, Document, DocumentVersion
    from app.qa import answer_question
    from app.retrieval import retrieve_authorized
    from scripts.run_agent_hard_benchmark import ScenarioTools, run_arm

    observed, first, rag_trace, error = {}, None, [], {}
    scenario = ScenarioTools(db, user, task, models=None) if arm == "rag" else None

    def record_retrieval(*args, **kwargs):
        nonlocal first
        found = retrieve_authorized(*args, **kwargs)
        if arm == "rag" and task.get("fault_script") == "first_search_empty" and first is None:
            from dataclasses import replace
            found = replace(found, evidence=[], candidates=[])
            scenario.scenario_events["first_search_miss"] = True
        for e in found.evidence:
            observed[e["chunk_id"]] = deepcopy(e)
        docs = list(dict.fromkeys(e["document_id"] for e in found.evidence))
        if first is None:
            first = docs
        if arm == "rag":
            rag_trace.append({"tool": "search_documents", "arguments": {"query": args[2]}, "status": "ok",
                              "evidence_documents": docs, "evidence_count": len(found.evidence)})
            # Same event boundary in RAG and Agents: first shared retrieval returns,
            # then revoke before any subsequent evidence use/generation.
            scenario.calls.append({"tool": "search_documents", "arguments": {"query": args[2]}})
            scenario._apply_state_change("search_documents")
        return found

    def capture_answer(*args, **kwargs):
        try:
            return answer_question(*args, **kwargs)
        except HTTPException as exc:
            if exc.status_code in {403, 404, 409}:
                error["kind"] = "authorization_change"
            raise

    def score(task_spec, payload, trace, steps, latency_ms, **kwargs):
        trace = rag_trace if arm == "rag" else trace
        if arm == "rag":
            kwargs["scenario_events"] = dict(scenario.scenario_events)
            if error:
                kwargs["execution_error_kind"] = error["kind"]
        row = provisional_score(task, payload, trace, steps, latency_ms, **kwargs)
        row.update(trace_complete=True, retrieved_evidence=list(observed.values()), first_retrieval_documents=first)
        refs = set(observed)
        # Passage windows and contract preparation may add authorized evidence
        # after a tool search. Bind the chunks actually supplied to generation.
        refs.update(payload.get("usage", {}).get("generation_context_chunk_ids", []))
        if arm != "rag":
            # Include historically opened versions; the shared retrieval hook covers
            # search, while these IDs are from successfully returned tool results.
            from app.models import ToolExecution
            if options.get("_run_id"):
                refs.update(cid for tool in db.scalars(select(ToolExecution).where(
                    ToolExecution.task_id == options["_run_id"])) for cid in tool.evidence_chunk_ids)
        for chunk, version, doc in db.execute(select(Chunk, DocumentVersion, Document).join(
            DocumentVersion, Chunk.version_id == DocumentVersion.id).join(Document, DocumentVersion.document_id == Document.id).where(Chunk.id.in_(refs))):
            observed.setdefault(chunk.id, {"chunk_id": chunk.id, "document_id": doc.id, "version_id": version.id,
                "title": doc.title, "text": chunk.text, "locator": chunk.locator})
            observed[chunk.id].update(source_sha256=version.content_hash, effective_from=version.created_at.isoformat(),
                                      is_active=doc.active_version_id == version.id)
        row["retrieved_evidence"] = list(observed.values())
        return row

    original_callback = options.pop("on_created", None)
    options.pop("score_fn", None)
    def created(rid):
        options["_run_id"] = rid
        if original_callback:
            original_callback(rid)
    # run_arm's historical RAG limitation is bypassed only here: deterministic
    # revocation is now injected at the shared retrieval boundary for every method.
    runnable = {k: v for k, v in task.items() if arm != "rag" or k not in {"state_change", "fault_script", "memory_fixture"}}
    try:
        with ExitStack() as stack:
            for target in ("app.retrieval.retrieve_authorized", "app.qa.retrieve_authorized", "agent.tools.retrieve_authorized"):
                stack.enter_context(patch(target, record_retrieval))
            stack.enter_context(patch("scripts.run_agent_hard_benchmark.answer_question", capture_answer))
            return run_arm(db, user, runnable, arm, on_created=created, score_fn=score, **options)
    finally:
        if scenario:
            scenario.restore()
