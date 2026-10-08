"""Question-bound attributes and narrow source-scope checks, without an LLM call."""

import re


def source_scoped_evidence(question: str, evidence: list[dict]):
    """Reuse publication aliases; retain original dates for relation evaluation.

    One publication can be asked about on several dates in the same sentence.
    Filtering by only the first date would drop another required report.
    """
    from types import SimpleNamespace
    from app.retrieval import route_sources

    documents = list(
        {
            (r.get("document_id", r["id"]), r.get("version_id", r["id"])): SimpleNamespace(
                id=r.get("document_id", r["id"]),
                active_version_id=r.get("version_id", r["id"]),
                metadata_json={"source": (r.get("metadata") or {}).get("source")},
            )
            for r in evidence
        }.values()
    )
    groups = route_sources(question, documents, strict_dates=True)
    allowed = {key for group in groups for key in group["key"]}
    scoped = [r for r in evidence if r.get("document_id", r["id"]) in allowed] if groups else evidence
    return scoped, {"groups": groups, "input_chunks": len(evidence), "scoped_chunks": len(scoped)}


def bind_verdict_source_groups(question: str, verdict: dict, evidence: list[dict], claims=None):
    """A comparison cannot be established by proof from only one named source.

    Reuse publication routing and authorized chunk identity, not lexical entailment.
    AND=no and OR=yes may be established by one decisive proposition; comparisons
    require both sides for either result. This only rejects incomplete proof.
    """
    if verdict.get("value") not in {"yes", "no"}:
        return verdict
    _, scope = source_scoped_evidence(question, evidence)
    groups = scope["groups"]
    if len(groups) < 2:
        return verdict
    operator = verdict.get("operator")
    if (operator == "all" and verdict["value"] == "no") or (
        operator == "any" and verdict["value"] == "yes"
    ):
        return verdict
    by_chunk = {row["chunk_id"]: row for row in evidence}
    documents = {
        by_chunk[ref]["document_id"]
        for ref in verdict.get("evidence_ids", []) if ref in by_chunk
    }
    # A third-party excerpt can explicitly report the requested publication.
    # Preserve that alternative proof path; semantic support is still judged later.
    cited = set(verdict.get("evidence_ids", []))
    for claim in claims or []:
        for ref, quote in zip(claim["evidence_ids"], claim.get("quotes", [])):
            if ref not in cited:
                continue
            _, attributed = source_scoped_evidence(quote, evidence)
            documents.update(key for group in attributed["groups"] for key in group["key"])
    missing = [group["mention"] for group in groups if not documents.intersection(group["key"])]
    if not missing:
        return verdict
    return {
        **verdict, "value": "unclear", "claim_index": None, "claim_indices": [],
        "evidence_ids": [], "question_complete": False,
        "validation_issues": sorted(set(verdict.get("validation_issues", []) + ["missing_source_group_proof"])),
        "missing_source_groups": missing,
    }


def verdict_proof_groups(question: str, evidence: list[dict], claims: list[dict]) -> list[dict]:
    """Offer source-bound claim choices, without deciding their semantic truth.

    Only single propositions use this transport. Multi-proposition AND/OR keeps its
    own short-circuit semantics. A single claim may supply several groups, and an
    explicitly attributed third-party quotation remains a legal proof path.
    """
    from app.verdict import question_slots

    if len(question_slots(question)) != 1 or not re.search(
        r"\b(?:compar(?:ed|ing|ison)|consistent|consistency|same|different|difference|align)\b"
        r"|\b(?:more|less|greater|lower|higher)\b.{0,80}\bthan\b"
        r"|比较|对比|相比|一致|相同|不同",
        question, re.I,
    ):
        return []
    _, scope = source_scoped_evidence(question, evidence)
    groups = scope["groups"]
    if not 2 <= len(groups) <= 4:
        return []
    by_chunk = {row["chunk_id"]: row for row in evidence}
    claim_documents = []
    for claim in claims:
        documents = {
            by_chunk[ref]["document_id"]
            for ref in claim["evidence_ids"] if ref in by_chunk
        }
        for ref, quote in zip(claim["evidence_ids"], claim.get("quotes", [])):
            if ref not in by_chunk:
                continue
            _, attributed = source_scoped_evidence(quote, evidence)
            documents.update(key for group in attributed["groups"] for key in group["key"])
        claim_documents.append(documents)
    return [
        {"key": f"g{number}", "source": group["mention"],
         "claim_indices": [index for index, documents in enumerate(claim_documents, 1)
                           if documents.intersection(group["key"])]}
        for number, group in enumerate(groups, 1)
    ]


def publication_binding_issues(text: str, source_ids: list[str], evidence: list[dict], quotes=None) -> list[str]:
    """A claim explicitly attributed to a publication needs that provenance.

    Only compare existing metadata and explicit publication mentions. Claims that
    name no publication, or sources with unknown metadata, remain semantic cases.
    """
    from app.retrieval import readable_source_names

    # A readable publication that was never retrieved still counts as named; a
    # citation from another publication cannot stand in for it.
    present = {(row.get("metadata") or {}).get("source") for row in evidence}
    named = evidence + [
        {"id": f"absent:{name}", "document_id": f"absent:{name}", "version_id": f"absent:{name}",
         "metadata": {"source": name}}
        for name in sorted(readable_source_names(evidence) - present)
    ]
    _, scope = source_scoped_evidence(text, named)
    if not scope["groups"]:
        return []
    allowed = {document for group in scope["groups"] for document in group["key"]}
    by_id = {row["id"]: row for row in evidence}
    for index, ref in enumerate(source_ids):
        if (
            ref not in by_id
            or not (by_id[ref].get("metadata") or {}).get("source")
            or by_id[ref].get("document_id", by_id[ref]["id"]) in allowed
        ):
            continue
        quote = quotes[index] if quotes and index < len(quotes) else ""
        _, attributed = source_scoped_evidence(quote, named)
        attribution = {key for group in attributed["groups"] for key in group["key"]}
        if not attribution.intersection(allowed):
            return ["publication_reference_mismatch"]
    return []


def comparison_relevant_evidence(question: str, evidence: list[dict], scope: dict):
    """Reuse the cross-encoder for a single-attribute, single-article comparison.

    Relevance is not entailment. All original candidates remain in telemetry; a
    model must still bind both selected articles and establish the relation.
    Aggregates, unknown facets and multi-article groups retain their full context.
    """
    groups = scope["groups"]
    facet = comparison_dimension(question)
    whole = question.strip().rstrip("?？")[:600]
    if (
        facet == whole
        or not 2 <= len(groups) <= 4
        or any(len(group["key"]) != 1 for group in groups)
        or re.search(r"\b(?:counts?|totals?|sums?|aggregates?|all events)\b|统计|汇总|累计|总计", question, re.I)
    ):
        return evidence, {"status": "full_context", "reason": "not_single_article_single_attribute"}
    from app.rerank import rerank
    from app.clients import DependencyError, evidence_spans

    selected, scores = [], []
    for group in groups:
        candidates = [row for row in evidence if row.get("document_id", row["id"]) in group["key"]]
        prose = {
            row["id"]: "\n".join(
                span["quote"] for span in evidence_spans([row], split_english=True)[0].values()
            )
            for row in candidates
        }
        candidates = [row for row in candidates if prose[row["id"]]]
        if not candidates:
            return evidence, {"status": "full_context", "reason": "missing_group"}
        ranked = (
            rerank(
                facet + "\n" + (group.get("clause") or question),
                candidates,
                text_of=lambda row: row["title"] + "\n" + prose[row["id"]],
                window=len(candidates),
            )
            if len(candidates) > 1
            else [{**candidates[0], "rerank_score": None}]
        )
        if not ranked or ranked[0]["id"] not in {row["id"] for row in candidates}:
            raise DependencyError("比较重排未返回原始来源", stage="answer_composition")
        chosen = next(row for row in candidates if row["id"] == ranked[0]["id"])
        selected.append(chosen)
        scores.append(
            {
                "source": group["mention"],
                "selected_chunk_id": chosen["chunk_id"],
                "scores": [
                    {"chunk_id": row["chunk_id"], "relevance_score": row["rerank_score"]}
                    for row in ranked
                ],
            }
        )
    selected_ids = {row["id"] for row in selected}
    return [row for row in evidence if row["id"] in selected_ids], {
        "status": "focused",
        "facet": facet,
        "ranking_only_not_entailment": True,
        "candidate_chunk_ids": [row["chunk_id"] for row in evidence],
        "source_groups": scores,
    }


# Each value is copied from the question. Unknown attributes retain the whole ask;
# no news entity, publication, benchmark ID or expected answer is encoded here.
_DIMENSIONS = (
    r"(?:fantasy football|fantasy) strategy",
    r"(?:launch|opening|release|effective|publication|event) dates?",
    r"dates? (?:of|for) [^,?]{1,60}",
    r"(?:opening|operating|support) hours",
    r"\b(?:control over (?:personal )?)?privacy\b",
    r"(?:个人)?隐私(?:控制权)?",
    r"(?:生效|发布|上线|开业|事件|实施)(?:日期|时间)",
    r"(?:营业|支持|服务)时段",
    r"(?:幻想足球|梦幻足球|选人|替补|投资|应对)策略",
)
_REPORTING = re.compile(
    r"\b(?:according to|argued|testified|defen[cs]e|prosecution|alleged|said|claimed)\b"
    r"|辩方|检方|律师|证人|声称|据.{1,16}表示|指出|认为",
    re.IGNORECASE,
)
_BOUNDED = re.compile(
    r"(?:provided|retrieved|available|cited|current) (?:excerpts?|passages?|snippets?|chunks?)"
    r"|(?:现有|当前|所给|已检索|引用的)(?:摘录|片段|节选)",
    re.IGNORECASE,
)
_DOCUMENT_ABSENCE = re.compile(
    r"\b(?:article|report|document|policy|notice|publication|paper)\b[^.!?]{0,300}"
    r"\b(?:did not|does not|do not|never|without) (?:claim|mention|stat|report|attribut|discuss|specif|address)\w*"
    r"|(?:整篇|整份|全文|报道|文章|通知|文档|政策)[^。！？]{0,50}"
    r"(?:从未|没有|未曾|未)(?:提及|提到|说明|声称|归因|讨论)",
    re.IGNORECASE,
)


def comparison_dimension(question: str) -> str:
    focus = question
    if re.match(r"\s*(?:(?:despite|although|while)\b|尽管|虽然)", focus, re.IGNORECASE):
        parts = re.split(r"[,，]", focus, maxsplit=1)
        if len(parts) == 2:
            focus = parts[1]
    focus = re.split(
        r"\b(?:despite|whereas|while)\b|注意|不要比较", focus, maxsplit=1, flags=re.IGNORECASE
    )[0]
    matches = []
    for pattern in _DIMENSIONS:
        matches.extend(re.finditer(pattern, focus, re.IGNORECASE))
    matches.sort(key=lambda match: match.start())
    if len(matches) > 1:
        between = focus[matches[0].end() : matches[-1].start()]
        if re.search(r"\band\b|和|以及|及", between, re.IGNORECASE):
            # Shared comparisons with several attributes stay a whole ask.
            return question.strip().rstrip("?？")[:600]
    markers = list(
        re.finditer(r"consistent|consistency|same|agree|align|一致|相同", focus, re.IGNORECASE)
    )
    if matches:
        if markers:
            marker = markers[-1]
            preceding = [match for match in matches if 0 <= marker.start() - match.end() <= 40]
            return min(
                preceding or matches,
                key=lambda match: min(
                    abs(match.end() - marker.start()), abs(match.start() - marker.end())
                ),
            ).group()
        return matches[0].group()
    return question.strip().rstrip("?？")[:600]


def question_contract(question: str) -> dict:
    return {
        "comparison_target": comparison_dimension(question),
        "comparison_target_extracted": comparison_dimension(question)
        != question.strip().rstrip("?？")[:600],
        "attribution": "Preserve who asserted a statement and whether it is an allegation/opinion; a publication reporting a speaker is not independently endorsing them.",
        "absence_scope": "Only retrieved excerpts are available. Missing a statement here cannot establish absence from a whole document. Explicit source denials remain ordinary cited facts.",
    }


def attribution_cues(text: str) -> list[str]:
    return list(dict.fromkeys(match.group() for match in _REPORTING.finditer(text)))[:6]


def claim_scope_issues(text: str, quotes: list[str] | None = None) -> list[str]:
    # Conservative syntactic guard only. It does not prove entailment, check ordinary
    # factual negation, or grant completeness because an excerpt is long.
    for sentence in re.split(r"(?<=[.!?。！？])\s*", text):
        if not _DOCUMENT_ABSENCE.search(sentence):
            continue
        normalized = "".join(sentence.casefold().split()).strip(".!?。！？")
        if any(normalized in "".join(quote.casefold().split()) for quote in (quotes or [])):
            # A source may explicitly state that a document omits something. This
            # accepts that literal assertion, not completeness inferred from a gap.
            continue
        global_scope = re.search(
            r"\b(?:entire|whole|throughout|never|anywhere)\b|整篇|整份|全文|从未",
            sentence,
            re.IGNORECASE,
        )
        if global_scope or not _BOUNDED.search(sentence):
            return ["unbounded_document_absence"]
    return []


def source_audit(text: str, quotes: list[str], *, check_polarity: bool = False) -> list[str]:
    issues = claim_scope_issues(text, quotes)
    has_reported_source = any(attribution_cues(quote) for quote in quotes)
    publication_assertion = re.search(
        r"\b(?:article|report|publication)\s+(?:claims?|claimed|states?|stated|asserts?|suggests?|confirmed?|confirms?)\b"
        r"|(?:报道|文章)(?:声称|认为|断言|确认|证实|认定)",
        text,
        re.IGNORECASE,
    )
    # Only a diagnostic: reporting language is not reliable enough to gate every
    # attributed claim. Explicit document-wide absence is gated separately.
    speaker_retained = re.search(
        r"\b(?:defen[cs]e|prosecution|lawyer|witness|according to|said|argued|testified)\b"
        r"|律师|检方|辩方|证人|表示|转述|主张",
        text,
        re.IGNORECASE,
    )
    if has_reported_source and publication_assertion and not speaker_retained:
        issues.append("possible_speaker_attribution_loss")
    uncertain = any(
        re.search(r"\b(?:may|might|could|possibly)\b|可能|或许", quote, re.I) for quote in quotes
    )
    definite = re.search(
        r"\b(?:confirmed|definitely|certainly|proved|already)\b|确认|证实|肯定|必然|已经|已实现",
        text,
        re.I,
    )
    if (
        uncertain
        and definite
        and not re.search(
            r"\b(?:may|might|could|possibly|unconfirmed|not confirm(?:ed)?)\b|可能|或许|尚未确认|未确认",
            text,
            re.I,
        )
    ):
        issues.append("possible_modality_loss")
    if check_polarity:
        from app.claim_consistency import source_polarity_issue

        if source_polarity_issue(text, quotes):
            issues.append("explicit_source_negation_lost")
    return issues


def quantitative_support_issues(text: str, quotes: list[str], *, source_titles=()) -> list[str]:
    """Literal quantity/date support in this claim's bound quotes only.

    This deliberately does not claim general semantic entailment. Unit conversions
    and explicit differences are supported, but a date/number somewhere else in
    retrieved context cannot rescue an unsupported citation.
    """
    from app.task_contract import _VALUE, unit_value
    from app.source_facts import dates

    for title in source_titles:
        if text.startswith(title + "：") or text.startswith(title + ":"):
            text = text[len(title) + 1 :]
            break

    def quantities(value):
        for raw, _ in dates(value):
            value = value.replace(raw, "")
        # A calendar year before a month/quarter is not a duration in years.
        # Partial temporal scope remains subject to the semantic/version review.
        value = re.sub(
            r"(?<!\d)(?:19|20)\d{2}\s*年(?=\s*(?:第?[一二三四1-4]\s*季度|[\d一二三四五六七八九十]{1,3}\s*月))",
            "",
            value,
        )
        combined = set()
        for match in re.finditer(
            r"(\d+(?:\.\d+)?|[零一二三四五六七八九十百]+)\s*小时\s*(\d+(?:\.\d+)?|[零一二三四五六七八九十百]+)\s*分钟",
            value,
        ):
            combined.add(("seconds", unit_value(match[1], "小时")[1] + unit_value(match[2], "分钟")[1]))
            value = value.replace(match.group(), "")
        return combined | ({unit_value(m["number"], m["unit"]) for m in _VALUE.finditer(value)} - {None})

    cited = set().union(*(quantities(q) for q in quotes))
    wanted = quantities(text)
    unsupported = wanted - cited
    # Natural comparative forms are equivalent to an explicit difference, but
    # require the comparison marker so ordinary quantities stay literal checks.
    difference_pattern = (
        r"相差|差值|difference|多出|少了|额外(?:增加|减少)|\b(?:extra|additional)\b|"
        r"(?:比|较)[^。；;]{1,60}?(?:多|少|长|短)(?:了|出)?(?=\s*[\d零一二三四五六七八九十百])|"
        r"(?:longer|shorter|more|less)\s+than\b[^.!?;]{1,60}?\bby\b"
    )
    difference = re.search(difference_pattern, text, re.I)
    if difference:
        result_values = quantities(re.split(r"[；;。]", text[difference.end() :], maxsplit=1)[0])
        for dimension, value in result_values:
            inputs = {v for d, v in cited if d == dimension}
            direct_difference = any(
                re.search(difference_pattern, q, re.I) and (dimension, value) in quantities(q)
                for q in quotes
            )
            same_inputs = len(inputs) == 1 and value == 0 and len(quotes) >= 2
            if (
                not direct_difference
                and not same_inputs
                and (len(inputs) != 2 or value != abs(max(inputs) - min(inputs)))
            ):
                return ["unsupported_cited_calculation"]
    if unsupported and difference:
        # Bounded arithmetic only: the two cited inputs must share a dimension.
        for dimension, value in list(unsupported):
            inputs = {v for d, v in cited if d == dimension}
            if (len(inputs) == 2 and value == abs(max(inputs) - min(inputs))) or (
                len(inputs) == 1 and value == 0 and len(quotes) >= 2
            ):
                unsupported.remove((dimension, value))
    quoted_dates = {day for q in quotes for _, day in dates(q)}
    missing_dates = [(raw, day) for raw, day in dates(text) if day not in quoted_dates]
    if missing_dates:
        from app.temporal import effective_interval, select_effective_versions

        intervals = [
            {"version_id": str(i), "effective_interval": effective_interval(q)}
            for i, q in enumerate(quotes)
        ]
        if any(
            select_effective_versions(raw, intervals)["status"] != "selected" for raw, _ in missing_dates
        ):
            return ["unsupported_cited_date"]
    return ["unsupported_cited_quantity"] if unsupported else []
