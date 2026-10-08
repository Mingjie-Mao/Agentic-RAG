"""Structural checks for claims-only judgments; this is not an entailment judge."""

from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field
from app.task_analysis import acceptance_items
from app.answer_contract import comparison_dimension


def source_assertion_question(span: str) -> bool:
    """Reporting/characterization is a different proposition from actual world truth.

    This signals the scope only; the model must still bind a decisive relevant
    source. It never turns missing passages or a document-wide absence into no.
    """
    if re.search(r"\b(?:never|not)\s+(?:mention|discuss|report|claim|say)|未提及|从未", span, re.I):
        return False
    return bool(
        re.search(
            r"\b(?:article|report|source|passage|excerpt)\b.{0,90}\b(?:claims?|suggests?|impl(?:y|ies)|states?|says?|reports?|asserts?)\b"
            r"|^\s*(?:does|did|do)\b.{1,85}\b(?:claim|suggest|imply|state|say|report|assert)\b"
            r"|(?:文章|报道|文档|片段)(?:声称|认为|说明|指出|暗示)",
            span,
            re.I,
        )
    )


def conditional_only_support(span: str, quotes: list[str]) -> bool:
    """Explicit conditional excerpts alone cannot establish an unconditional fact.

    This is a syntactic boundary, not a general entailment classifier. Questions
    about the condition itself and literal conditional statements remain valid.
    """
    hypothetical = (
        r"\b(?:if|unless|would|could|might|conditional|possib\w*|prospect\w*)\b|如果|若|可能|条件|前景"
    )
    return (
        bool(quotes)
        and not re.search(hypothetical, span, re.I)
        and all(re.match(r"\s*[\"“']?(?:if\b|unless\b|如果|若)", quote, re.I) for quote in quotes)
    )


def question_slots(question: str) -> list[str]:
    """Expose explicit independent clauses; keep unsplittable asks as one slot."""
    slots = acceptance_items(question)
    if len(slots) == 1:
        # A new source with its own predicate is an independent proposition.
        # Keep the exact question spans, including negation and qualifications.
        slots = [
            part.strip(" ,;?？")
            for part in re.split(
                r",\s+(?:while|and\s+then)\s+",
                question,
                flags=re.IGNORECASE,
            )
        ]
    if len(slots) == 1:
        slots = [
            part.strip(" ,;?？")
            for part in re.split(
                r"\s+(?:and|or)\s+(?=(?:did|do|does|is|are|was|were|has|have|can)\b)",
                question,
                flags=re.IGNORECASE,
            )
        ]
        # In 'either East ... or West ...', each subject names an alternative.
        if len(slots) == 1 and re.search(r"\beither\b", question, re.IGNORECASE):
            slots = [
                part.strip(" ,;?？") for part in re.split(r"\s+or\s+", question, flags=re.IGNORECASE)
            ]
    return slots if 1 <= len(slots) <= 4 else [question]


def constrained_schema(question: str) -> dict:
    schema = StructuredVerdict.model_json_schema()
    slots = question_slots(question)
    schema["$defs"]["PropositionVerdict"]["properties"]["question_span"]["enum"] = slots
    schema["properties"]["propositions"].update(minItems=len(slots), maxItems=len(slots))
    schema["properties"]["comparison_dimension"].update(
        enum=["", comparison_dimension(question)], maxLength=600
    )
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
    comparison_dimension: str = Field(max_length=600)
    propositions: list[PropositionVerdict] = Field(min_length=1, max_length=4)


class IndexedPropositionVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_index: int = Field(ge=1, le=4)
    truth: Literal["supported", "contradicted", "unknown"]
    claim_indices: list[int] = Field(max_length=8)


class IndexedStructuredVerdict(BaseModel):
    """Compact transport only; public decisions retain exact question spans.

    The local Ollama grammar rejects escaped quotes inside string enums. Binding
    short indices avoids that provider gap and repeated question decoding without
    relaxing the server's completeness or evidence checks.
    """
    model_config = ConfigDict(extra="forbid")
    operator: Literal["atomic", "all", "any", "comparison"]
    question_complete: bool
    comparison_dimension: Literal["", "target"]
    propositions: list[IndexedPropositionVerdict] = Field(min_length=1, max_length=4)


class SourceBoundPropositionVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_index: int = Field(ge=1, le=4)
    truth: Literal["supported", "contradicted", "unknown"]
    source_proofs: dict[str, list[int]]


class DecisiveSourceBoundPropositionVerdict(SourceBoundPropositionVerdict):
    truth: Literal["supported", "contradicted"]


class UnknownSourceBoundPropositionVerdict(SourceBoundPropositionVerdict):
    truth: Literal["unknown"]


class SourceBoundIndexedVerdict(BaseModel):
    """Select proof for each named source; no truth or conclusion is forced."""
    model_config = ConfigDict(extra="forbid")
    operator: Literal["atomic", "comparison"]
    question_complete: bool
    comparison_dimension: Literal["", "target"]
    propositions: list[
        DecisiveSourceBoundPropositionVerdict | UnknownSourceBoundPropositionVerdict
    ] = Field(min_length=1, max_length=1)


def source_bound_schema(proof_groups: list[dict]) -> dict:
    schema = SourceBoundIndexedVerdict.model_json_schema()
    for name, minimum in (("DecisiveSourceBoundPropositionVerdict", 1),
                          ("UnknownSourceBoundPropositionVerdict", 0)):
        props = schema["$defs"][name]["properties"]
        props["slot_index"]["enum"] = [1]
        props["source_proofs"] = {
            "type": "object", "additionalProperties": False,
            "required": [group["key"] for group in proof_groups],
            "properties": {
                group["key"]: {
                    "type": "array", "minItems": minimum, "maxItems": 8,
                    "items": {"type": "integer", **(
                        {"enum": group["claim_indices"]} if group["claim_indices"] else {}
                    )},
                    **({"maxItems": 0} if not group["claim_indices"] else {}),
                }
                for group in proof_groups
            },
        }
    if any(not group["claim_indices"] for group in proof_groups):
        # Missing provenance cannot support either comparative conclusion. Keep a
        # satisfiable unknown branch instead of an impossible provider grammar.
        schema["properties"]["propositions"]["items"] = {
            "$ref": "#/$defs/UnknownSourceBoundPropositionVerdict"
        }
        del schema["$defs"]["DecisiveSourceBoundPropositionVerdict"]
    return schema


def expand_source_bound_verdict(
    decision: SourceBoundIndexedVerdict, question: str, proof_groups: list[dict]
) -> StructuredVerdict:
    """Use only model-selected source proofs, checking the schema again in Python."""
    allowed = {group["key"]: set(group["claim_indices"]) for group in proof_groups}
    propositions = []
    complete = decision.question_complete
    for item in decision.propositions:
        if set(item.source_proofs) != set(allowed):
            raise ValueError("source_proof_groups_not_covered")
        if any(not set(indices).issubset(allowed[key]) or len(indices) > 8
               for key, indices in item.source_proofs.items()):
            raise ValueError("invalid_source_proof_reference")
        if item.truth != "unknown" and any(not indices for indices in item.source_proofs.values()):
            complete = False
        propositions.append(IndexedPropositionVerdict(
            slot_index=item.slot_index, truth=item.truth,
            claim_indices=sorted({index for indices in item.source_proofs.values() for index in indices}),
        ))
    return expand_indexed_verdict(IndexedStructuredVerdict(
        operator=decision.operator, question_complete=complete,
        comparison_dimension=decision.comparison_dimension, propositions=propositions,
    ), question)


def indexed_schema(question: str, claim_count: int) -> dict:
    schema = IndexedStructuredVerdict.model_json_schema()
    count = len(question_slots(question))
    props = schema["$defs"]["IndexedPropositionVerdict"]["properties"]
    props["slot_index"]["enum"] = list(range(1, count + 1))
    props["claim_indices"]["items"].update(minimum=1, maximum=claim_count)
    schema["properties"]["propositions"].update(minItems=count, maxItems=count)
    schema["properties"]["operator"]["enum"] = (
        ["atomic", "comparison"] if count == 1 else ["all", "any"]
    )
    return schema


def expand_indexed_verdict(decision: IndexedStructuredVerdict, question: str) -> StructuredVerdict:
    slots = question_slots(question)
    indices = [item.slot_index for item in decision.propositions]
    if sorted(indices) != list(range(1, len(slots) + 1)):
        raise ValueError("question_slots_not_covered")
    return StructuredVerdict(
        operator=decision.operator,
        question_complete=decision.question_complete,
        comparison_dimension=(
            comparison_dimension(question) if decision.comparison_dimension == "target" else ""
        ),
        propositions=[
            PropositionVerdict(
                question_span=slots[item.slot_index - 1],
                truth=item.truth,
                claim_indices=item.claim_indices,
            )
            for item in decision.propositions
        ],
    )


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
    if (
        constrained
        and decision.operator == "comparison"
        and (decision.comparison_dimension != comparison_dimension(question))
    ):
        issues.append("comparison_dimension_not_bound")
    for item in decision.propositions:
        if any(index < 1 or index > len(claims) for index in item.claim_indices):
            issues.append("invalid_claim_reference")
        if item.truth != "unknown" and not item.claim_indices:
            issues.append("missing_claim_reference")
    truths = [item.truth for item in decision.propositions]
    value = "unclear"
    if not issues:
        if decision.operator == "all":
            value = (
                "no"
                if "contradicted" in truths
                else ("yes" if all(truth == "supported" for truth in truths) else "unclear")
            )
        elif decision.operator == "any":
            value = (
                "yes"
                if "supported" in truths
                else ("no" if all(truth == "contradicted" for truth in truths) else "unclear")
            )
        else:
            value = {"supported": "yes", "contradicted": "no", "unknown": "unclear"}[truths[0]]
    indices = (
        sorted(
            {
                index
                for item in decision.propositions
                for index in item.claim_indices
                if 1 <= index <= len(claims)
            }
        )
        if value != "unclear"
        else []
    )
    return {
        "value": value,
        "claim_index": indices[0] if indices else None,
        "claim_indices": indices,
        "evidence_ids": list(
            dict.fromkeys(
                evidence for index in indices for evidence in claims[index - 1]["evidence_ids"]
            )
        ),
        "method": "structured_claims_classifier",
        "scope": "仅组合已校验 claims；结构校验不等于原文蕴含校验",
        "operator": decision.operator,
        "question_complete": decision.question_complete,
        "comparison_dimension": decision.comparison_dimension,
        "propositions": [item.model_dump() for item in decision.propositions],
        "validation_issues": sorted(set(issues)),
    }
