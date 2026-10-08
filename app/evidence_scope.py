"""Explicit entity and period bindings for operating-window comparisons.

This is a narrow enterprise contract, not a general entity extractor. Unknown or
overlapping applicability is refused; publication and upload dates never apply.
"""

import calendar
from datetime import date
import re

from app.clients import Claim, GeneratedAnswer

_WINDOW = re.compile(
    r"(?:收货|营业|办公|服务|支持|接待|发货|配送)(?:窗口|时段|时间)|(?:opening|operating|support) hours",
    re.I,
)
_RANGE = re.compile(r"(?P<start>\d{1,2}:\d{2})\s*(?:至|到|[-–—])\s*(?P<end>\d{1,2}:\d{2})")
_QUARTER = re.compile(r"(\d{4})\s*年\s*第?([一二三四1-4])\s*季度")
_MONTH = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?!\s*\d)")
# A requested entity must end where its name ends: "华东仓二号库" or "华东仓运营组"
# is a different unit, not the 华东仓 itself.
_ENTITY_TAIL = (
    r"(?=$|[\s，。；：、,.;:()（）《》“”\"'\-–—]|的|(?:收货|营业|办公|服务|支持|接待|发货|配送|工作日|周末|"
    r"窗口|时间|时段|临时|通知|规则|安排|调整|说明))"
)


def names_entity(entity, text):
    return re.search(re.escape(entity) + _ENTITY_TAIL, text) is not None


def period(text):
    matches = []
    for m in _QUARTER.finditer(text):
        year = int(m[1])
        q = "一二三四".find(m[2]) + 1 if m[2] in "一二三四" else int(m[2])
        month = q * 3
        matches.append(
            (date(year, month - 2, 1), date(year, month, calendar.monthrange(year, month)[1]))
        )
    for m in _MONTH.finditer(text):
        year, month = map(int, m.groups())
        if 1 <= month <= 12:
            matches.append(
                (date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]))
            )
    return matches[0] if matches and len(set(matches)) == 1 else None


def window_request(question):
    if not (_WINDOW.search(question) and re.search(r"是否相同|是否一致|一样|相同吗", question)):
        return None
    match = re.match(
        r"\s*([^，。？?]{1,32}?)(?:和|与|及)([^，。？?]{1,32}?)(?=\s*\d{4}\s*年|的)", question
    )
    if not match:
        return None
    entities = [x.strip() for x in match.groups()]
    if not all(entities) or entities[0] == entities[1]:
        return None
    return {
        "entities": entities,
        "period": period(question),
        "attribute": _WINDOW.search(question).group(),
        "weekday": bool(re.search(r"工作日|weekdays?", question, re.I)),
    }


def source_scope(text):
    """Read only an explicit applicability clause, never an incidental date."""
    clauses = [m.group() for m in re.finditer(r"(?:本[^。\n]{0,12})?适用于[^。\n]+", text)]
    p = period(" ".join(clauses))
    return {
        "period": [d.isoformat() for d in p] if p else None,
        "clauses": clauses,
        "status": "explicit" if p else "unknown",
    }


def attach_scope(db, user, question, evidence, *, historical=False):
    request = window_request(question)
    if not request:
        return evidence, {"status": "not_applicable"}
    from sqlalchemy import select
    from app.models import Chunk
    from app.security import require_chunk
    from app.chunking import token_count
    from app.config import settings

    def cost(row):
        return token_count(row["text"]) + token_count(row["title"]) + 100

    tokens = sum(cost(row) for row in evidence)
    by_version = {}
    for row in evidence:
        by_version.setdefault(row["version_id"], []).append(row)
    enriched = []
    added = []
    for vid, rows in by_version.items():
        # The same version, tenant and document are checked again at use time.
        require_chunk(db, user, rows[0]["chunk_id"], active_only=not historical)
        chunks = list(
            db.scalars(select(Chunk).where(Chunk.version_id == vid).order_by(Chunk.ordinal).limit(65))
        )
        scope = (
            source_scope("\n".join(c.text for c in chunks))
            if len(chunks) <= 64
            else {"status": "unknown", "reason": "scope_scan_limit"}
        )
        scope["source_chunk_ids"] = (
            [c.id for c in chunks if "适用于" in c.text] if len(chunks) <= 64 else []
        )
        rows = list(rows)
        known = {r["chunk_id"] for r in rows}
        # Applicability and its original citation must remain together. No synthetic
        # quote is made from metadata; only literal authorized source chunks enter.
        for c in chunks if len(chunks) <= 64 else []:
            if (
                "适用于" in c.text or (_WINDOW.search(c.text) and _RANGE.search(c.text))
            ) and c.id not in known:
                require_chunk(db, user, c.id, active_only=not historical)
                candidate = dict(rows[0], chunk_id=c.id, text=c.text, locator=c.locator)
                if tokens + cost(candidate) > settings().context_token_budget:
                    scope = {"status": "unknown", "reason": "scope_context_budget"}
                    break
                rows.append(candidate)
                tokens += cost(candidate)
                added.append(c.id)
        enriched.extend(
            dict(row, metadata={**(row.get("metadata") or {}), "document_scope": scope}) for row in rows
        )
    return [dict(r, id=f"E{i}") for i, r in enumerate(enriched, 1)], {
        "status": "scope_attached",
        "added_chunk_ids": added,
        "context_tokens": tokens,
    }


def scoped_windows(question, evidence):
    request = window_request(question)
    if not request:
        return None, {"status": "not_applicable"}
    candidates = []
    rejected = []
    for row in evidence:
        match = _RANGE.search(row["text"])
        if not match or not _WINDOW.search(row["text"]):
            continue
        scope = (row.get("metadata") or {}).get("document_scope") or source_scope(row["text"])
        entities = [
            e
            for e in request["entities"]
            if names_entity(e, row["title"]) or any(names_entity(e, c) for c in scope.get("clauses", []))
        ]
        reason = None
        if scope.get("reason") in {"scope_scan_limit", "scope_context_budget"}:
            reason = scope["reason"]
        elif len(entities) != 1:
            reason = "entity_missing_or_ambiguous"
        elif request["weekday"] and not re.search(r"工作日|weekdays?", row["text"], re.I):
            reason = "weekday_scope_missing"
        elif request["period"]:
            p = scope.get("period")
            wanted = [d.isoformat() for d in request["period"]]
            if not p or p[0] > wanted[0] or p[1] < wanted[1]:
                reason = "period_not_covering_request"
        if reason:
            rejected.append({"chunk_id": row["chunk_id"], "reason": reason})
            continue
        # Weekday qualifiers belong to the same operating-window statement.
        # Collect every explicit range, so a second conflicting window is not hidden.
        for line in row["text"].splitlines():
            if not _WINDOW.search(line) or (
                request["weekday"] and not re.search(r"工作日|weekdays?", line, re.I)
            ):
                continue
            for window in _RANGE.finditer(line):
                value = tuple(tuple(map(int, window[key].split(":"))) for key in ("start", "end"))
                if any(hour > 23 or minute > 59 for hour, minute in value):
                    rejected.append({"chunk_id": row["chunk_id"], "reason": "invalid_clock_range"})
                    continue
                candidates.append(
                    {
                        "entity": entities[0],
                        "range": (window["start"], window["end"]),
                        "range_key": value,
                        "row": row,
                        "quote": line,
                    }
                )
    selected = []
    for entity in request["entities"]:
        found = [c for c in candidates if c["entity"] == entity]
        if not found or len({c["range_key"] for c in found}) != 1:
            return GeneratedAnswer(answerable=False, claims=[]), {
                "status": "ambiguous_or_missing_window",
                "rejected": rejected,
            }
        selected.append(found[0])
    claims = [
        Claim(
            text=f"{c['entity']}的{request['attribute']}为 {c['range'][0]} 至 {c['range'][1]}。",
            evidence_ids=[c["row"]["id"]],
            quotes=[c["quote"]],
        )
        for c in selected
    ]
    # Bind applicability to its original version's quoted text as well as the
    # operating-window line. Metadata alone is never a synthetic citation.
    for claim, c in zip(claims, selected):
        scope = (c["row"].get("metadata") or {}).get("document_scope") or {}
        for scope_row in evidence:
            if (
                scope_row["version_id"] == c["row"]["version_id"]
                and scope_row["chunk_id"] in scope.get("source_chunk_ids", [])
                and scope_row["id"] not in claim.evidence_ids
            ):
                claim.evidence_ids.append(scope_row["id"])
                claim.quotes.append(scope_row["text"])
    equal = selected[0]["range_key"] == selected[1]["range_key"]
    claims.append(
        Claim(
            text=f"两者的{request['attribute']}{'相同' if equal else '不同'}。",
            evidence_ids=[c["row"]["id"] for c in selected],
            quotes=[c["quote"] for c in selected],
        )
    )
    return GeneratedAnswer(answerable=True, claims=claims), {
        "status": "bound_window_comparison",
        "rejected": rejected,
        "verdict": {
            "value": "yes" if equal else "no",
            "claim_index": 3,
            "evidence_ids": [c["row"]["chunk_id"] for c in selected],
            "method": "explicit_entity_period_window_binding",
        },
    }
