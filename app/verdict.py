"""Structural checks for claims-only judgments; this is not an entailment judge."""

from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field
from app.task_analysis import acceptance_items


def question_slots(question: str) -> list[str]:
    """Expose explicit independent clauses; keep unsplittable asks as one slot."""
    slots = acceptance_items(question)
    if len(slots) == 1:
        slots = [part.strip(" ,;?？") for part in re.split(
            r"\s+(?:and|or)\s+(?=(?:did|do|does|is|are|was|were|has|have|can)\b)",
            question, flags=re.IGNORECASE,
        )]
        # In 'either East ... or West ...', each subject names an alternative.
        if len(slots) == 1 and re.search(r"\beither\b", question, re.IGNORECASE):
            slots = [part.strip(" ,;?？") for part in re.split(
                r"\s+or\s+", question, flags=re.IGNORECASE
            )]
    return slots if 1 <= len(slots) <= 4 else [question]


def constrained_schema(question: str) -> dict:
    schema = StructuredVerdict.model_json_schema()
    slots = question_slots(question)
    schema["$defs"]["PropositionVerdict"]["properties"]["question_span"]["enum"] = slots
    schema["properties"]["propositions"].update(minItems=len(slots), maxItems=len(slots))
    if len(slots) == 1:
        schema["properties"]["operator"]["enum"] = ["atomic", "comparison"]
    else:
        schema["properties"]["operator"]["enum"] = ["all", "any"]
    return schema


class PropositionVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_span: str = Field(min_length=1, max_length=600)
    truth: Literal["supported", "contradicted", "unknown"]
    claim_indices: list[int] = Field(max_length=8)


class StructuredVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operator: Literal["atomic", "all", "any", "comparison"]
    question_complete: bool
    comparison_dimension: str = Field(max_length=200)
    propositions: list[PropositionVerdict] = Field(min_length=1, max_length=4)


def resolve_verdict(
    decision: StructuredVerdict, question: str, claims: list[dict], *, constrained: bool = False
) -> dict:
    """Combine supported propositions, rejecting incomplete or dangling references.

    The model still owns semantic interpretation and completeness. Exact spans and
    indices only prevent malformed decisions from becoming confident conclusions.
    """
    issues = []
    if not decision.question_complete:
        issues.append("incomplete_question")
    if decision.operator in {"atomic", "comparison"} and len(decision.propositions) != 1:
        issues.append("invalid_operator_arity")
    if decision.operator in {"all", "any"} and len(decision.propositions) < 2:
        issues.append("invalid_operator_arity")
    spans = [item.question_span.casefold() for item in decision.propositions]
    if len(set(spans)) != len(spans):
        issues.append("duplicate_proposition")
    if constrained and set(spans) != {slot.casefold() for slot in question_slots(question)}:
        issues.append("question_slots_not_covered")
    if any(span not in question.casefold() for span in spans):
        issues.append("question_span_not_found")
    if decision.operator == "comparison" and (
        not decision.comparison_dimension.strip()
        or decision.comparison_dimension.casefold() not in question.casefold()
    ):
        issues.append("comparison_dimension_not_found")
    for item in decision.propositions:
        if any(index < 1 or index > len(claims) for index in item.claim_indices):
            issues.append("invalid_claim_reference")
        if item.truth != "unknown" and not item.claim_indices:
            issues.append("missing_claim_reference")
    truths = [item.truth for item in decision.propositions]
    value = "unclear"
    if not issues:
        if decision.operator == "all":
            value = "no" if "contradicted" in truths else (
                "yes" if all(truth == "supported" for truth in truths) else "unclear"
            )
        elif decision.operator == "any":
            value = "yes" if "supported" in truths else (
                "no" if all(truth == "contradicted" for truth in truths) else "unclear"
            )
        else:
            value = {"supported": "yes", "contradicted": "no", "unknown": "unclear"}[truths[0]]
    indices = sorted({
        index for item in decision.propositions for index in item.claim_indices
        if 1 <= index <= len(claims)
    }) if value != "unclear" else []
    return {
        "value": value,
        "claim_index": indices[0] if indices else None,
        "claim_indices": indices,
        "evidence_ids": list(dict.fromkeys(
            evidence for index in indices for evidence in claims[index - 1]["evidence_ids"]
        )),
        "method": "structured_claims_classifier",
        "scope": "仅组合已校验 claims；结构校验不等于原文蕴含校验",
        "operator": decision.operator,
        "question_complete": decision.question_complete,
        "comparison_dimension": decision.comparison_dimension,
        "propositions": [item.model_dump() for item in decision.propositions],
        "validation_issues": sorted(set(issues)),
    }
