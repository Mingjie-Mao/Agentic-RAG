"""Gap-driven escalation: answer cheaply first, escalate to the planner only on a gap.

Lexical coverage cannot see a bridge gap: the asked attribute ("超时阈值") also occurs
in the wrong documents. Two observable gaps are used instead, both measured on Dev:
the first pass abstained, or its answer cites no document that mentions the question's
distinctive subject (an incident id, a nickname, a rule id). Conflicts are not escalated:
the planner did not resolve them better on Dev.
"""

from agent.planned import _terms

ABSTAINED = {"insufficient_evidence", "verification_failed", "execution_failed"}


def escalation_reason(goal, payload, evidence):
    """Why the first pass needs the planner, or None when its answer can stand."""
    status = (payload or {}).get("status")
    if status in ABSTAINED:
        return {"reason": "first_pass_abstained", "status": status}
    if status != "answered":
        return None
    documents = {}
    for row in evidence:
        documents[row["document_id"]] = documents.get(row["document_id"], row["title"]) + " " + row["text"]
    cited = {c["chunk_id"] for c in payload.get("citations", [])}
    cited_docs = {row["document_id"] for row in evidence if row["chunk_id"] in cited}
    if not documents or not cited_docs:
        return None
    terms = {d: _terms(text) for d, text in documents.items()}
    wanted = _terms(goal)
    distinct = [t for t in wanted if 0 < sum(t in terms[d] for d in terms) <= max(1, len(terms) // 4)]
    cited_terms = set().union(*(terms[d] for d in cited_docs))
    missing = sorted(t for t in distinct if t not in cited_terms)
    if missing:
        return {"reason": "answer_not_grounded_in_question_subject", "terms": missing[:6]}
    return None
