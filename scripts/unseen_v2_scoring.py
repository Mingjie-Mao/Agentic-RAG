"""Fair v2 diagnostics and independent semantic answer review (no string-only pass)."""
import json
from pathlib import Path

from scripts.research_review import digest, packet, validated_labels, paired_ci


def reviewed_suite(suite, root):
    source = json.loads((root / suite["review_packet"]).read_text())
    spec = json.loads((root / Path(suite["review_packet"]).parent / "documents.json").read_text())
    import hashlib
    corpus_sha = digest({v["path"]: hashlib.sha256((root / v["path"]).read_bytes()).hexdigest()
                         for d in spec["documents"] for v in d["versions"]})
    if any(r.get("corpus_sha256") != corpus_sha for r in source["items"]):
        raise ValueError("review corpus bytes changed")
    reviewed_path = root / suite["reviewed_packet"]
    if not reviewed_path.exists():
        raise ValueError("P4 requires independent task/necessary-evidence/scorer review before freeze or ingestion")
    reviewed = json.loads(reviewed_path.read_text())
    labels = validated_labels(source, reviewed)
    expected = {t["id"]: t for t in suite["tasks"] + suite.get("safety_tasks", [])}
    if {r["id"]: r["task"] for r in source["items"]} != expected:
        raise ValueError("task review no longer matches this suite")
    if any(label != "approved" for label in labels.values()):
        raise ValueError("holdout task review rejected/unclear; revise and create a fresh review packet")
    for row in reviewed["reviews"]:
        required = next(r["required_review"] for r in source["items"] if r["id"] == row["id"])
        if any(row.get("checks", {}).get(key) is not True for key in required):
            raise ValueError("question, evidence necessity, condition/version scope and scorer checks required")
    return {"source": digest(source), "review": digest(reviewed)}


def provisional_score(task, payload, trace, steps, latency_ms, **kwargs):
    from scripts.run_agent_hard_benchmark import score_task
    # All evidence may arrive together; neither hop count nor forced sequential
    # transitions is a success condition. Keep the trace solely for diagnostics.
    fair = {k: v for k, v in task.items() if k not in {"conditional_transition", "required_tools"}}
    if task["category"] == "conditional_planning":
        # Forbidden inactive branch belongs to final claims. Authorized input
        # quoting that branch is legal, and is not an ACL leak.
        fair = {**fair, "forbidden": []}
    row = score_task(fair, payload, trace, steps, latency_ms, **kwargs)
    if task["category"] == "conditional_planning":
        from scripts.run_agent_hard_benchmark import fact_matches, normalized
        text = normalized(" ".join(c.get("text", "") for c in payload.get("claims", [])))
        row["condition_violation"] = any(fact_matches(f, text) for f in task["forbidden"])
        row["answer_correct"] &= not row["condition_violation"]
        row["task_success"] &= not row["condition_violation"]
    row.update(quality_provisional=True, quality_basis="string_diagnostics_only_pending_independent_semantic_review",
               answer_payload=payload)
    return row


def answer_packet(result, suite):
    tasks = {t["id"]: t for t in suite["tasks"] + suite.get("safety_tasks", [])}
    items = []
    # Arm identities are concealed from the reviewer; the hash maps back later.
    for arm, rows in result["results"].items():
        for row in rows:
            if row.get("skipped"):
                continue
            task = tasks[row["task_id"]]
            items.append({"id": digest([arm, row["task_id"], row.get("run_id")]), "kind": "answer",
                          "task": task, "answer": row["answer_payload"], "tool_trace": row.get("tool_trace", []),
                          "required_review": ["slot_verdicts", "citation_valid", "condition_compliant", "version_scope_valid", "status_valid"]})
    return packet(items, "blinded independent completeness, cited entailment and scope review")


def quality_gate(raw, suite, reviewed, *, candidate="hybrid", baseline="workflow"):
    source = answer_packet(raw, suite)
    labels = validated_labels(source, reviewed)
    reviews = {r["id"]: r for r in reviewed["reviews"]}
    correctness, outcomes, completeness = {}, {}, {}
    for arm, rows in raw["results"].items():
        correctness[arm], outcomes[arm], completeness[arm] = {}, {}, {}
        for row in rows:
            if row.get("skipped"):
                continue
            rid = digest([arm, row["task_id"], row.get("run_id")])
            review, task = reviews[rid], next(r["task"] for r in source["items"] if r["id"] == rid)
            slots = {s["id"] for s in task["required_slots"]}
            verdicts = review.get("slot_verdicts", {})
            if set(verdicts) != slots or any(v not in {"supported", "partial", "unsupported", "contradicted", "unclear"} for v in verdicts.values()):
                raise ValueError("every required answer slot needs an independent semantic verdict")
            expected = all(v == "supported" for v in verdicts.values()) and all(
                review.get(k) is True for k in ("citation_valid", "condition_compliant", "version_scope_valid", "status_valid"))
            correct = labels[rid] == "correct"
            if correct != expected:
                raise ValueError("answer label disagrees with completeness/citation/scope verdicts")
            correctness[arm][row["task_id"]] = correct
            completeness[arm][row["task_id"]] = {"required_slots": len(slots),
                 "supported_slots": sum(v == "supported" for v in verdicts.values()),
                 "early_stop_error": review.get("early_stop_error"),
                 "condition_compliant": review.get("condition_compliant")}
            # Capabilities are diagnostic except actually exercised controlled safety
            # and execution status. One retrieval may legally satisfy two hops.
            safe = not any(row.get("forbidden_present", {}).values()) and row.get("execution_error_kind") in {None, "authorization_change"}
            outcomes[arm][row["task_id"]] = correct and safe and all(
                row.get("scenario_events", {}).get(e) for e in task.get("scenario_events", []))
    main = {t["id"] for t in suite["tasks"]}
    if any(main - set(correctness[arm]) for arm in (candidate, baseline)):
        raise ValueError("paired main tasks incomplete")
    reports = {}
    for name, data in (("answer_correct", correctness), ("task_success", outcomes)):
        delta = [int(data[candidate][tid]) - int(data[baseline][tid]) for tid in sorted(main)]
        reports[name] = {"paired_n": len(delta), "difference": sum(delta)/len(delta), "ci": paired_ci(delta)}
    passed = all(r["difference"] >= .05 and r["ci"][0] is not None and r["ci"][0] > 0 for r in reports.values())
    null_ids = {t["id"] for t in suite["tasks"] if t["category"] == "null_insufficient"}
    condition_ids = {t["id"] for t in suite["tasks"] if t["category"] == "conditional_planning"}
    checks = {"quality": passed,
              "null_no_regression": sum(not correctness[candidate][t] for t in null_ids) <= sum(not correctness[baseline][t] for t in null_ids),
              "condition_no_regression": sum(completeness[candidate][t]["condition_compliant"] is False for t in condition_ids)
                 <= sum(completeness[baseline][t]["condition_compliant"] is False for t in condition_ids),
              "acl_zero": not any(any(r.get("forbidden_present", {}).values()) for rows in raw["results"].values() for r in rows),
              "termination_zero": not any(r.get("status") == "execution_failed" and r.get("execution_error_kind") != "authorization_change"
                                          for rows in raw["results"].values() for r in rows)}
    by_category = {}
    for arm, rows in raw["results"].items():
        by_category[arm] = {}
        for category in sorted({t["category"] for t in suite["tasks"] + suite.get("safety_tasks", [])}):
            selected = [r for r in rows if r["category"] == category and not r.get("skipped")]
            coverage = [completeness[arm][r["task_id"]] for r in selected]
            stop = [c["early_stop_error"] for c in coverage if c["early_stop_error"] is not None]
            by_category[arm][category] = {"tasks": len(selected),
                "answer_correct": sum(correctness[arm][r["task_id"]] for r in selected),
                "task_success": sum(outcomes[arm][r["task_id"]] for r in selected),
                "required_fact_slots": sum(c["required_slots"] for c in coverage),
                "supported_fact_slots": sum(c["supported_slots"] for c in coverage),
                "early_stop_errors": sum(bool(v) for v in stop), "early_stop_applicable": len(stop),
                "early_stop_unknown_or_na": len(selected)-len(stop),
                "over_planning": sum(bool(r.get("over_planned")) for r in selected),
                "over_planning_applicable": sum(r.get("over_planned") is not None for r in selected),
                "recovered": sum(bool(r.get("recovered")) for r in selected),
                "recovery_applicable": sum(bool(r.get("recovery_applicable")) for r in selected),
                "required_documents": sum(len(next(t.get("expected_evidence_documents", []) for t in suite["tasks"]+suite.get("safety_tasks", []) if t["id"]==r["task_id"])) for r in selected),
                "actual_cited_documents": sum(len(r.get("citation_documents", [])) for r in selected),
                "condition_violations": sum(c["condition_compliant"] is False for c in coverage),
                "execution_failures": sum(r.get("status") == "execution_failed" for r in selected)}
    return {"version": "unseen-v2-quality-gate", "passed": all(checks.values()), "checks": checks,
            "by_category": by_category,
            "reports": reports, "raw_sha256": digest(raw), "review_sha256": digest(reviewed),
            "candidate": candidate, "baseline": baseline, "decision": "quality_passed_cost_gate_still_required" if all(checks.values()) else "do_not_enable"}
