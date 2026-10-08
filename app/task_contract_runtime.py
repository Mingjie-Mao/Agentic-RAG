"""Bounded authorized evidence preparation shared by RAG and every Agent mode."""
from app.task_contract import bind_values, evaluate_condition


def resolve_history_document(contract, evidence):
    """Unique authorized source matching every requested attribute, never top-hit ID."""
    documents = {r['document_id'] for r in evidence}
    if len(documents) == 1:
        return next(iter(documents))
    explicit = {r['document_id'] for r in evidence if r.get('title') and
                r['title'].split(' [')[0] in contract.question}
    if len(explicit) == 1:
        return next(iter(explicit))
    matching = [doc for doc in documents if all(bind_values(slot,
        [r for r in evidence if r['document_id'] == doc]) for slot in contract.slots)]
    return matching[0] if len(matching) == 1 else None


def prepare_contract(contract, initial, *, search, load, versions, compare, open_version,
                     document_id=None):
    trace = {"searches": [], "recovery_attempted": False, "version_chain_complete": False}
    evidence = list(initial)

    def lookup(query, *, recover=False):
        nonlocal evidence
        result = search(query)
        trace["searches"].append({"query": query, "status": result.status if result else "budget_exhausted"})
        rows = load(result.evidence_refs, False) if result and result.status == "ok" else []
        # Only a successful empty/off-topic search can trigger recovery; never ACL or
        # transport failure. One recovery for the entire preparation, not per slot.
        if recover and result and result.status == "ok" and not trace["recovery_attempted"]:
            from agent.planner import recovery_query, retrieval_quality
            quality = retrieval_quality(query, result.evidence_refs, result.data.get("matches", []))
            if quality in {"empty", "irrelevant"}:
                retry = recovery_query(query, query)
                if retry and retry != query:
                    trace["recovery_attempted"] = True
                    rows += lookup(retry)
        return rows

    if contract.intent == "conditional":
        operand = next(s for s in contract.slots if s.applies_if == "always")
        evidence = list(initial) if len(bind_values(operand, initial)) == 1 else lookup(operand.query, recover=True)
        state = evaluate_condition(contract, evidence)
        trace["condition"] = state.model_dump()
        # Unknown does not activate either branch. At most two explicit branch slots
        # are supported by this candidate; compound predicates stay unknown.
        active = [s for s in contract.slots if s.applies_if == state.result]
        if len(active) > 2:
            trace["branch_budget_exhausted"] = True
            evidence = []
        elif state.result != "unknown":
            for slot in active:
                if not slot.query.startswith(("只回答", "只给出")):
                    evidence += lookup(slot.query)
    elif contract.intent == "comparison":
        for slot in contract.slots:
            if len(bind_values(slot, evidence)) != 1:
                evidence += lookup(slot.query, recover=True)
    elif contract.intent == "history":
        if not evidence and not document_id:
            evidence = lookup(contract.question, recover=True)
        doc = document_id or resolve_history_document(contract, evidence)
        if not doc:
            trace["history_reason"] = "ambiguous_or_missing_document"
        else:
            manifest = versions(doc)
            records = manifest.data.get("versions", []) if manifest and manifest.status == "ok" else []
            ready = sorted([v for v in records if v.get('status') == 'ready' and v.get('version_id')
                            and v.get('created_at') and v.get('filename')],
                           key=lambda v: (v['created_at'], v['version_id']))
            complete = bool(ready and len(ready) == len(records)
                            and manifest.data.get("total_version_count", len(records)) == len(records)
                            and len(ready) <= 4)
            refs = []
            if complete:
                if len(ready) == 1:
                    result = open_version(doc, ready[0]["version_id"])
                    complete = bool(result and result.status == "ok")
                    if complete:
                        refs += result.evidence_refs
                for older, newer in zip(ready, ready[1:]):
                    result = compare(doc, older["version_id"], newer["version_id"])
                    if not result or result.status != "ok":
                        complete = False
                        break
                    refs += result.evidence_refs
            contract.version_ids = [v["version_id"] for v in ready]
            contract.version_chain_complete = complete
            trace["version_chain_complete"] = complete
            # Keep a separately bound fact for each version; no active-first cap can
            # silently discard old versions. Missing values still refuse generation.
            evidence = load(refs, True) if complete else []
            from app.temporal import complete_version_intervals
            texts = {v['version_id']: '\n'.join(r['text'] for r in evidence if r['version_id'] == v['version_id'])
                     for v in ready}
            ready = complete_version_intervals(ready, texts)
            # Source-effective chronology takes precedence over upload chronology.
            if ready and all(v['effective_interval'].get('valid_from') for v in ready):
                ready.sort(key=lambda v: v['effective_interval']['valid_from'])
                if len({v['effective_interval']['valid_from'] for v in ready}) != len(ready):
                    contract.version_chain_complete = False
            elif contract.history_kind == 'first':
                contract.version_chain_complete = False
                trace['history_reason'] = 'effective_chronology_unknown'
            contract.version_ids = [v['version_id'] for v in ready]
            contract.versions = ready
            trace['version_chain_complete'] = contract.version_chain_complete
            if not contract.version_chain_complete:
                evidence = []
            by_version = {v["version_id"]: v for v in ready}
            evidence = [dict(row, effective_interval=by_version[row["version_id"]].get("effective_interval", {}),
                             version_label=by_version[row["version_id"]]["filename"])
                        for row in evidence if row["version_id"] in by_version]
    unique = {row["chunk_id"]: row for row in evidence}
    return [dict(row, id=f"E{i}") for i, row in enumerate(unique.values(), 1)], trace
