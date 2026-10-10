"""Pure offline, source-bound delivery proxies; never semantic or answer scoring.

``supporting_facts`` returns in-memory reference text. Only ``diagnose``,
``summarize`` and ``pair_profiles`` results are suitable for derived artifacts:
they contain identifiers, fingerprints and counts, never source/question text.
Native spans and upstream supporting facts use separate proxy denominators.
"""

from collections import Counter
import hashlib
import json
import re

from scripts.multihop_retrieval_eval import document_id, fact_delivered

PHASES = ("candidates", "admitted", "context")
UPSTREAM = "upstream_word4_proxy"
SOURCE_SPAN = "source_span_char4_proxy"
SOURCE_ATOM = "source_atom_exact_proxy"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _basis(task):
    return UPSTREAM if "upstream_gold_evidence" in task else SOURCE_SPAN


def supporting_facts(task):
    """Extract unique references, binding upstream facts to actual gold source SHA.

    Invalid/missing/ambiguous bindings raise ValueError, rather than manufacturing
    a reference that can never match. Text must remain in memory/local raw input.
    """
    facts = []
    if _basis(task) == UPSTREAM:
        sources = {}
        for row in task.get("gold_evidence", []):
            sources.setdefault(row.get("document_id"), set()).add(row.get("source_sha256"))
        for row in task["upstream_gold_evidence"]:
            did = document_id(row["title"])
            hashes = sources.get(did, set())
            if len(hashes) != 1 or not all(hashes):
                raise ValueError("upstream reference requires unique source binding")
            source_sha = next(iter(hashes))
            fact = {"document_id": did, "source_sha256": source_sha,
                    "text": row["fact"], "basis": UPSTREAM}
            fact["id"] = "RF-" + _digest(fact)
            if fact not in facts:
                facts.append(fact)
    else:
        for row in task.get("gold_evidence", []):
            fact = {"document_id": row.get("document_id"),
                    "source_sha256": row.get("source_sha256"),
                    "text": row.get("quote"), "basis": SOURCE_SPAN}
            fact["id"] = row.get("id") or "RF-" + _digest(fact)
            facts.append(fact)
    _validate_facts(facts)
    return facts


def _validate_facts(facts):
    ids = [fact["id"] for fact in facts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate reference IDs")
    for fact in facts:
        if not fact.get("document_id") or not fact.get("source_sha256"):
            raise ValueError("reference requires source binding")
        if not isinstance(fact.get("text"), str) or not fact["text"].strip():
            raise ValueError("reference requires nonempty text")
        if fact.get("basis") not in (UPSTREAM, SOURCE_SPAN, SOURCE_ATOM):
            raise ValueError("unsupported reference basis")


def _character_grams(value):
    text = "".join(value.casefold().split())
    return {text[i:i + 4] for i in range(max(0, len(text) - 3))} or ({text} if text else set())


def _delivered(fact, texts):
    if fact["basis"] == SOURCE_ATOM:
        # Complete reviewed content blocks, conjunctively; Markdown headings are
        # formatting. This allows scope/value blocks in separate authorized
        # chunks without rewarding a partial sentence or an unrelated number.
        blocks = []
        for block in re.split(r"\n\s*\n", fact["text"]):
            body = "\n".join(line for line in block.splitlines()
                             if not re.fullmatch(r"\s*#{1,6}\s+.+", line))
            normalized = "".join(body.casefold().split())
            if normalized:
                blocks.append(normalized)
        present = ["".join(text.casefold().split()) for text in texts]
        return bool(blocks) and all(any(block in text for text in present) for block in blocks)
    if fact["basis"] == UPSTREAM:
        return fact_delivered(fact["text"], texts)
    wanted = _character_grams(fact["text"])
    present = set().union(*(_character_grams(text) for text in texts)) if texts else set()
    return bool(wanted) and len(wanted & present) / len(wanted) >= 0.5


def _rank(row):
    for field in ("rank", "retrieval_rank"):
        value = row.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def phase_coverage(facts, evidence):
    """Count delivered references using only matching document AND source SHA.

    None means unavailable telemetry; [] is a measured empty phase. Supporting
    rank is the first ranked prefix reaching the 0.5 proxy threshold, including
    unions across chunks. Missing rank metadata remains null, never guessed.
    """
    if evidence is None:
        return None
    _validate_facts(facts)
    if not facts:
        return {"status": "not_applicable", "total": 0, "delivered": None,
                "matched_reference_ids": [], "all_delivered": None,
                "first_supporting_rank": None, "first_supporting_ranks": {}}
    matches, ranks = [], {}
    for fact in facts:
        rows = [row for row in evidence if row.get("document_id") == fact["document_id"]
                and row.get("source_sha256") == fact["source_sha256"]]
        rows.sort(key=lambda row: _rank(row) if _rank(row) is not None else float("inf"))
        texts = []
        for row in rows:
            texts.append(row.get("text") or "")
            if _delivered(fact, texts):
                matches.append(fact["id"])
                ranks[fact["id"]] = _rank(row)
                break
    known_ranks = [rank for rank in ranks.values() if rank is not None]
    return {"status": "measured", "total": len(facts), "delivered": len(matches),
            "matched_reference_ids": sorted(matches), "all_delivered": len(matches) == len(facts),
            "first_supporting_rank": min(known_ranks) if known_ranks else None,
            "first_supporting_ranks": dict(sorted(ranks.items()))}


def diagnose(task, phases):
    """Return text-free phase diagnostics for an active-only component assessment.

    Any unavailable phase gives error/null cause, while available measurements
    remain visible. Loss counts retain partial upstream and downstream loss.
    """
    result = {"task_id": task["id"], "category": task.get("category"), "basis": _basis(task),
              "eligible": True, "status": "error", "cause": None, "reason": None,
              "losses": None, "phases": dict.fromkeys(PHASES),
              "strict_task_success": None, "semantic_recall": None,
              "reference_fingerprint": _digest([task.get("requires_version", False),
                  task.get("gold_evidence", []), task.get("upstream_gold_evidence")])}
    try:
        facts = supporting_facts(task)
    except (ValueError, KeyError, TypeError):
        result.update(reason="invalid_references", reference_count=None)
        return result
    result["reference_fingerprint"] = _digest([
        bool(task.get("requires_version")), sorted(facts, key=lambda fact: fact["id"])])
    result["reference_count"] = len(facts)
    if task.get("requires_version") or not facts:
        result.update(eligible=False, status="not_applicable", cause="not_applicable",
                      reason="requires_version" if task.get("requires_version") else "no_references")
        return result
    result["phases"] = {name: phase_coverage(facts, (phases or {}).get(name)) for name in PHASES}
    if any(value is None for value in result["phases"].values()):
        result["reason"] = "missing_phase_telemetry"
        return result
    candidates, admitted, context = [
        set(result["phases"][name]["matched_reference_ids"]) for name in PHASES]
    losses = {"candidate_missing": len(facts) - len(candidates),
              "admission_loss": len(candidates - admitted), "context_loss": len(admitted - context)}
    result.update(status="measured", losses=losses,
                  cause=next((name for name, count in losses.items() if count), "proxy_complete"))
    return result


def _counts(rows):
    return {"tasks": len(rows), "eligible_tasks": sum(row["eligible"] for row in rows),
            "measured_tasks": sum(row["status"] == "measured" for row in rows),
            "missing_tasks": sum(row["eligible"] and row["status"] != "measured" for row in rows),
            "not_applicable_tasks": sum(not row["eligible"] for row in rows)}


def _group_summary(rows):
    result = _counts(rows)
    result["causes"] = dict(Counter(row["cause"] for row in rows if row["cause"] is not None))
    result["phases"] = {}
    for name in PHASES:
        eligible = [row for row in rows if row["eligible"]]
        measured = [row["phases"][name] for row in eligible
                    if row["phases"][name] is not None and row["phases"][name]["status"] == "measured"]
        denominator = sum(value["total"] for value in measured)
        delivered = sum(value["delivered"] for value in measured)
        result["phases"][name] = {
            "eligible_tasks": len(eligible), "measured_tasks": len(measured),
            "missing_tasks": len(eligible) - len(measured), "reference_denominator": denominator,
            "delivered": delivered, "all_delivered_tasks": sum(value["all_delivered"] is True for value in measured),
            "proxy_coverage": delivered / denominator if denominator else None,
        }
        if any("annotation" in row for row in rows):
            result["phases"][name].update(
                completion_measured_tasks=sum(v["all_delivered"] is not None for v in measured),
                completion_pending_tasks=sum(v["all_delivered"] is None for v in measured))
    return result


def summarize(rows):
    """Aggregate separately by proxy basis/category, preserving eligibility counts."""
    rows = list(rows)
    result = {**_counts(rows), "strict_task_success": None, "semantic_recall": None, "by_basis": {}}
    for basis in sorted({row["basis"] for row in rows}):
        subset = [row for row in rows if row["basis"] == basis]
        group = _group_summary(subset)
        group["by_category"] = {
            category: _group_summary([row for row in subset if row["category"] == category])
            for category in sorted({row["category"] for row in subset}, key=str)}
        result["by_basis"][basis] = group
    return result


def pair_profiles(left, right):
    """Pair diagnostics by task ID and exact gold/source fingerprints.

    Duplicate IDs or mismatched sets/references raise ValueError. Failures remain
    pairs with missing measurement counts; they never shrink the denominator.
    """
    def indexed(rows):
        found = {}
        for row in rows:
            if row["task_id"] in found:
                raise ValueError("duplicate task IDs")
            found[row["task_id"]] = row
        return found

    left, right = indexed(left), indexed(right)
    if left.keys() != right.keys():
        raise ValueError("mismatched task sets")
    pairs = []
    for task_id in sorted(left):
        a, b = left[task_id], right[task_id]
        if (not a.get("reference_fingerprint") or any(a[key] != b[key] for key in
                ("reference_fingerprint", "basis", "category", "eligible"))):
            raise ValueError("mismatched task references or applicability")
        status = ("not_applicable" if not a["eligible"] else
                  "complete" if a["status"] == b["status"] == "measured" else "missing")
        measurements = {}
        for name in PHASES:
            x, y = a["phases"][name], b["phases"][name]
            phase_status = ("not_applicable" if not a["eligible"] else
                            "complete" if x is not None and y is not None else "missing")
            measurements[name] = {"status": phase_status, "left": x, "right": y}
        pairs.append({"task_id": task_id, "basis": a["basis"], "category": a["category"],
                      "status": status, "phases": measurements})
    return {"total_pairs": len(pairs), "eligible_pairs": sum(p["status"] != "not_applicable" for p in pairs),
            "complete_pairs": sum(p["status"] == "complete" for p in pairs),
            "missing_pairs": sum(p["status"] == "missing" for p in pairs),
            "not_applicable_pairs": sum(p["status"] == "not_applicable" for p in pairs),
            "phases": {name: {"complete_pairs": sum(p["phases"][name]["status"] == "complete" for p in pairs),
                              "missing_pairs": sum(p["phases"][name]["status"] == "missing" for p in pairs)}
                       for name in PHASES}, "pairs": pairs}
