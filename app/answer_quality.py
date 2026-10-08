"""Bounded answer recovery on LangGraph; citation checks remain business rules."""

from typing import TypedDict
import hashlib
import inspect
import json
from pathlib import Path
import time

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.clients import Claim, DependencyError, GeneratedAnswer
from app.config import settings
from app.task_analysis import judgment_intent
from app.execution_budget import current_budget


class AnswerState(TypedDict, total=False):
    answer: dict
    usage: dict
    issues: list[dict]
    attempts: int
    first_issues: list[dict]
    stages: dict
    audit: dict
    repair_stalled: bool
    composed_audit: dict
    composition_all_evidence: bool


def answer_state(answer):
    """Only primitives enter framework checkpoints; no session/model objects."""
    return {
        "answerable": answer.answerable,
        "claims": [c.model_dump() for c in answer.claims],
        "facts": answer.facts,
    }


def bind_source_text(answer, evidence):
    """Render selected original spans instead of an unproved paraphrase.

    Selection/relevance still require task evaluation. This proves only literal
    claim provenance, and never creates an inferred conclusion or numeric result.
    """
    by_id = {r["id"]: r for r in evidence}
    selected = []
    seen = set()
    for claim in answer.claims:
        if len(claim.evidence_ids) != len(claim.quotes):
            return answer, {"status": "unrenderable_source_span"}
        for eid, quote in zip(claim.evidence_ids, claim.quotes):
            if (
                eid not in by_id
                or quote not in by_id[eid]["text"]
                or not quote.strip()
                or len(quote) > 1000
            ):
                return answer, {"status": "unrenderable_source_span"}
            key = (eid, quote)
            if key not in seen:
                selected.append(Claim(text=quote, evidence_ids=[eid], quotes=[quote]))
                seen.add(key)
    if len(selected) > 8:
        return answer, {"status": "source_claim_limit"}
    return GeneratedAnswer(answerable=answer.answerable, claims=selected, facts=answer.facts), {
        "status": "literal_source_claims",
        "source_claim_count": len(selected),
        "guarantee": "verbatim_provenance_only; relevance_and_completeness_require_evaluation",
    }


def audit_signature(*, literal=False):
    from app.clients import BoundAnswerAudit, BoundConclusion, Models
    from app.clients import evidence_spans, citation_context
    from app.verdict import question_slots, resolve_verdict
    from app.verdict import source_assertion_question, conditional_only_support
    from app.answer_contract import source_scoped_evidence, question_contract, comparison_dimension
    from app.retrieval import route_sources
    from app.answer_contract import _DIMENSIONS, comparison_relevant_evidence

    value = {
        "implementation": inspect.getsource(
            Models.compose_literal_answer if literal else Models.audit_bound_answer
        ),
        "schema": (BoundConclusion if literal else BoundAnswerAudit).model_json_schema(),
    }
    if literal:
        from app.rerank import rerank, _model

        value["comparison_reranker"] = inspect.getsource(rerank)
        value["reranker_loading"] = inspect.getsource(_model)
        value["source_spans"] = inspect.getsource(evidence_spans)
        value["citation_context"] = inspect.getsource(citation_context)
        value["question_slots"] = inspect.getsource(question_slots)
        value["composition"] = inspect.getsource(resolve_verdict)
        value["source_scope"] = inspect.getsource(source_scoped_evidence)
        value["question_contract"] = inspect.getsource(question_contract)
        value["comparison_dimension"] = inspect.getsource(comparison_dimension)
        value["comparison_patterns"] = _DIMENSIONS
        value["comparison_focus"] = inspect.getsource(comparison_relevant_evidence)
        value["publication_routing"] = inspect.getsource(route_sources)
        value["proposition_scope"] = inspect.getsource(source_assertion_question)
        value["condition_boundary"] = inspect.getsource(conditional_only_support)
        value["configuration_binding"] = inspect.getsource(composition_configuration)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def composition_configuration():
    """Bind the actual inference mode and span selection to component validation."""
    cfg = settings()
    result = {
        key: getattr(cfg, key)
        for key in (
            "answer_comparison_focus_enabled",
            "answer_literal_judgment_enabled",
            "answer_composition_model",
            "answer_composition_reasoning",
            "answer_composition_max_tokens",
            "answer_extractive_enabled",
            "answer_quality_enabled",
        )
    }

    if cfg.answer_comparison_focus_enabled:
        import os
        from scripts.run_unseen_benchmark import fingerprint_file

        base = Path(os.environ.get("HF_HOME", Path.cwd() / ".runtime/huggingface"))
        repo = base / "hub" / ("models--" + cfg.rerank_model.replace("/", "--"))
        revision = (repo / "refs/main").read_text().strip()
        files = {}
        for path in sorted((repo / "snapshots" / revision).iterdir()):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".model", ".txt"}:
                stat = path.stat()
                files[path.name] = fingerprint_file(str(path), stat.st_size, stat.st_mtime_ns)
        if not any(name.endswith(".safetensors") for name in files):
            raise ValueError("comparison focus cross-encoder weights missing")
        result["comparison_reranker"] = {
            "model": cfg.rerank_model,
            "revision": revision,
            "max_tokens": cfg.rerank_max_tokens,
            "batch": cfg.rerank_batch,
            "files": files,
        }
    return result


def validation_summary(rows, thresholds):
    positive = [r for r in rows if r["label"] == "supported"]
    negative = [r for r in rows if r["label"] != "supported"]
    rates = {
        "good_claim_acceptance": sum(r["prediction"] == "supported" for r in positive) / len(positive)
        if positive
        else None,
        "bad_claim_rejection": sum(r["prediction"] in {"unsupported", "contradicted"} for r in negative)
        / len(negative)
        if negative
        else None,
        "transport_or_format_failures": sum(r["prediction"] is None for r in rows),
    }
    rates["passed"] = bool(
        positive
        and negative
        and rates["good_claim_acceptance"] >= thresholds["good_claim_acceptance_min"]
        and rates["bad_claim_rejection"] >= thresholds["bad_claim_rejection_min"]
        and rates["transport_or_format_failures"] <= thresholds["transport_or_format_failure_max"]
    )
    return rates


def require_audit_gate():
    cfg = settings()
    if cfg.answer_literal_judgment_enabled and cfg.answer_semantic_audit_enabled:
        raise ValueError("literal judgment and arbitrary-claim audit require separate validation protocols")
    if not cfg.answer_semantic_audit_gate_path:
        raise ValueError("bound answer audit component validation gate required")
    gate = json.loads(Path(cfg.answer_semantic_audit_gate_path).read_text())
    from scripts.research_review import digest

    literal = gate.get("protocol") == "literal-answer-composition-gate-v1"
    expected_model = cfg.answer_composition_model if literal else cfg.semantic_judge_model
    if (
        gate.get("protocol") not in {"bound-answer-audit-gate-v1", "literal-answer-composition-gate-v1"}
        or gate.get("passed") is not True
        or (cfg.answer_literal_judgment_enabled and not literal)
        or (literal and not (cfg.answer_extractive_enabled or cfg.answer_literal_judgment_enabled))
        or gate.get("implementation_sha256") != audit_signature(literal=literal)
        or gate.get("model") != expected_model
        or (literal and gate.get("composition_configuration") != composition_configuration())
    ):
        raise ValueError("bound answer audit gate/model/protocol mismatch")
    result_path = Path(gate["result_path"])
    results = json.loads(result_path.read_text())
    cases = json.loads(Path(gate["cases_path"]).read_text())
    if (
        digest(results) != gate.get("result_sha256")
        or digest(cases) != gate.get("cases_packet_sha256")
        or results.get("completed") != len(cases["cases"])
        or results.get("cases_sha256") != cases["cases_sha256"]
        or digest(cases["cases"]) != cases["cases_sha256"]
        or results.get("implementation_sha256") != gate["implementation_sha256"]
        or results.get("model_digest") != gate["model_digest"]
        or results.get("registered_gates") != cases["gates"]
        or (literal and results.get("composition_configuration") != gate["composition_configuration"])
        or [(r["id"], r["label"]) for r in results["rows"]]
        != [(r["id"], r["label"]) for r in cases["cases"]]
        or (not literal and validation_summary(results["rows"], cases["gates"])["passed"] is not True)
        or (
            literal
            and (not results["rows"] or not all(row.get("passed") is True for row in results["rows"]))
        )
    ):
        raise ValueError("bound answer audit validation results changed or incomplete")
    # Name aliases can be repointed; verify the actual installed digest at use.
    import httpx

    endpoint = cfg.answer_composition_url if literal else cfg.semantic_judge_url or cfg.ollama_url
    response = httpx.get(endpoint + "/api/tags", timeout=10, trust_env=False)
    response.raise_for_status()
    model = next((m for m in response.json()["models"] if m["name"] == expected_model), {})
    if model.get("digest") != gate.get("model_digest"):
        raise ValueError("bound answer audit model digest changed")


def provenance_hint(answer, issue, evidence):
    """Name the evidence that can carry a claim's named publication; never widen it."""
    if "publication_reference_mismatch" not in issue["codes"] or issue["claim_index"] is None:
        return ""
    from app.answer_contract import source_scoped_evidence

    scoped, scope = source_scoped_evidence(answer.claims[issue["claim_index"]].text, evidence)
    named = "、".join(group["mention"] for group in scope["groups"])
    ids = [row["id"] for row in scoped] if scope["groups"] else []
    if not ids:
        return "断言点名的媒体没有可用原文；不得用其他媒体的引文代替，无法支持时删除该断言。"
    return f"断言点名{named}，只能引用来自该媒体的原文：{'、'.join(ids)}。"


def recover_answer(models, question, evidence, *, options=None):
    """One local cited repair; no blind retries, no permission or budget bypass."""
    from app.qa import claim_validation_issues

    options = dict(options or {})
    from app.clients import evidence_spans

    if not evidence_spans(evidence)[0]:
        return GeneratedAnswer(answerable=False, claims=[]), {
            "answer_recovery": {
                "framework": "langgraph",
                "repair_attempts": 0,
                "first_issues": [],
                "remaining_issues": [],
                "stages": {},
            },
            "answer_status": "insufficient_evidence",
            "reason": "no_eligible_source_spans",
        }
    if settings().answer_semantic_audit_enabled or (
        settings().answer_literal_judgment_enabled and judgment_intent(question)
    ):
        require_audit_gate()
    # Cross-document consistency is explicitly requested; an ordinary multi-hop
    # question about different attributes is not a rule conflict. Deterministic
    # conflict contracts and the existing tenant control still take precedence.
    from app.task_analysis import conflict_intent

    if not conflict_intent(question):
        options["check_conflict"] = False

    def generate(state):
        started = time.monotonic()
        if (
            (settings().answer_literal_judgment_enabled or (
                settings().answer_extractive_enabled and settings().answer_semantic_audit_enabled
            ))
            and judgment_intent(question)
        ):
            checked, usage = models.compose_literal_answer(
                question, GeneratedAnswer(answerable=True, claims=[]), evidence
            )
            answer = GeneratedAnswer(
                answerable=checked.question_complete, claims=checked.selected_claims or []
            )
            return {
                "answer": answer_state(answer),
                "usage": usage,
                "attempts": 0,
                "composed_audit": checked.model_dump(),
                "composition_all_evidence": True,
                "stages": {"evidence_composition_ms": round((time.monotonic() - started) * 1000, 1)},
            }
        answer, usage = models.generate(question, evidence, **options)
        if settings().answer_extractive_enabled:
            answer, usage["source_rendering"] = bind_source_text(answer, evidence)
            if usage["source_rendering"]["status"] != "literal_source_claims":
                usage["source_rendering_failed"] = True
        return {
            "answer": answer_state(answer),
            "usage": usage,
            "attempts": 0,
            "stages": {"initial_generation_ms": round((time.monotonic() - started) * 1000, 1)},
        }

    def validate(state):
        if current_budget():
            current_budget().check()
        issues = claim_validation_issues(GeneratedAnswer.model_validate(state["answer"]), evidence)
        if state["usage"].get("source_rendering_failed"):
            issues.append({"claim_index": None, "codes": ["unrenderable_source_span"]})
        if state["usage"].get("selection_missing_items"):
            issues.append(
                {
                    "claim_index": None,
                    "codes": ["selection_slots_missing"],
                    "reason": "；".join(state["usage"]["selection_missing_items"]),
                }
            )
        if state.get("repair_stalled"):
            issues.append({"claim_index": None, "codes": ["no_new_cited_evidence"]})
        return {"issues": issues, "first_issues": state.get("first_issues", issues)}

    def audit(state):
        try:
            if state.get("composition_all_evidence"):
                from app.clients import BoundAnswerAudit

                checked, extra = BoundAnswerAudit.model_validate(state["composed_audit"]), {}
            else:
                method = (
                    models.compose_literal_answer
                    if settings().answer_extractive_enabled
                    else models.audit_bound_answer
                )
                checked, extra = method(
                    question, GeneratedAnswer.model_validate(state["answer"]), evidence
                )
        except DependencyError:
            return {
                "issues": [{"claim_index": None, "codes": ["semantic_audit_unavailable"]}],
                "attempts": 1,
                "audit": {"status": "unavailable"},
            }
        issues = [
            {"claim_index": r.claim_index - 1, "codes": ["bound_claim_" + r.support], "reason": r.reason}
            for r in checked.checks
            if r.support != "supported"
        ]
        if not checked.question_complete or (judgment_intent(question) and checked.verdict == "unclear"):
            issues.append({"claim_index": None, "codes": ["question_not_settled"]})
        usage = dict(state["usage"])
        usage["answer_semantic_audit"] = checked.model_dump() | extra
        generated = GeneratedAnswer.model_validate(state["answer"])
        if not issues and judgment_intent(question) and checked.verdict != "unclear":
            selected = [generated.claims[i - 1] for i in checked.conclusion_claim_indices]
            refs = list(
                dict.fromkeys(
                    (eid, quote) for c in selected for eid, quote in zip(c.evidence_ids, c.quotes)
                )
            )
            if len(refs) > 6 or len(generated.claims) >= 8:
                issues.append({"claim_index": None, "codes": ["conclusion_citation_budget"]})
            else:
                conclusion = Claim(
                    text=checked.conclusion,
                    evidence_ids=[r[0] for r in refs],
                    quotes=[r[1] for r in refs],
                )
                generated = generated.model_copy(update={"claims": generated.claims + [conclusion]})
                mechanical = claim_validation_issues(generated, evidence)
                if mechanical:
                    issues += mechanical
                else:
                    ids = {r["id"]: r["chunk_id"] for r in evidence}
                    usage["verified_verdict"] = {
                        "value": checked.verdict,
                        "claim_index": len(generated.claims),
                        "evidence_ids": list(dict.fromkeys(ids[r[0]] for r in refs)),
                        "method": "bound_claim_audit_composition",
                        "independent": False,
                    }
        return {
            "issues": issues,
            "usage": usage,
            "answer": answer_state(generated),
            "audit": {"status": "checked"},
        }

    def repair(state):
        started = time.monotonic()
        answer = GeneratedAnswer.model_validate(state["answer"])
        failed = {r["claim_index"] for r in state["issues"]}
        valid = [
            c
            for i, c in enumerate(answer.claims)
            if i not in failed
            and (
                not settings().answer_extractive_enabled
                or (len(c.quotes) == 1 and c.text == c.quotes[0])
            )
        ]
        context = {
            "instruction": "仅重写失败的断言，选择真正支持每个断言的原文编号。错误草稿不是事实；不得照抄错误值。不要重复已保留的断言。question_not_settled表示必须补齐问题要求的关系或判断。",
            "failed_claims": [
                {
                    "text": answer.claims[r["claim_index"]].text if r["claim_index"] is not None else "",
                    "errors": r["codes"],
                    "reason": r.get("reason") or provenance_hint(answer, r, evidence),
                }
                for r in state["issues"]
            ],
            "retained_claims": [c.text for c in valid],
        }
        try:
            added, extra = models.generate(
                question,
                evidence,
                **(
                    options
                    | {"repair_context": context, "check_conflict": False, "max_output_tokens": 350}
                ),
            )
            if settings().answer_extractive_enabled:
                added, extra["source_rendering"] = bind_source_text(added, evidence)
                extra["source_rendering_failed"] = (
                    extra["source_rendering"]["status"] != "literal_source_claims"
                )
            seen = {"".join(c.text.split()) for c in valid}
            additions = []
            for claim in added.claims:
                key = "".join(claim.text.split())
                if key not in seen:
                    additions.append(claim)
                    seen.add(key)
            replacement = GeneratedAnswer(
                answerable=added.answerable,
                claims=(valid + additions)[:8] if additions else answer.claims,
                facts=answer.facts + added.facts,
            )
            stalled = settings().answer_extractive_enabled and not additions
        except DependencyError:
            extra, replacement = {}, answer
            stalled = settings().answer_extractive_enabled
        usage = dict(state["usage"])
        if "source_rendering" in extra:
            usage["source_rendering"] = extra["source_rendering"]
            usage["source_rendering_failed"] = extra["source_rendering_failed"]
        if "selection_missing_items" in extra:
            usage["selection_missing_items"] = extra["selection_missing_items"]
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "model_duration_ms",
            "model_load_ms",
            "prompt_eval_ms",
            "completion_eval_ms",
        ):
            usage[key] = (usage.get(key) or 0) + (extra.get(key) or 0)
        return {
            "answer": answer_state(replacement),
            "usage": usage,
            "attempts": state["attempts"] + 1,
            "repair_stalled": stalled,
            "stages": state["stages"]
            | {"local_repair_ms": round((time.monotonic() - started) * 1000, 1)},
        }

    graph = StateGraph(AnswerState)
    graph.add_node(
        "generate",
        generate,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_interval=0.2,
            jitter=False,
            retry_on=lambda error: isinstance(error, DependencyError) and error.retryable,
        ),
    )
    graph.add_node("validate", validate)
    graph.add_node("repair", repair)
    graph.add_node("audit", audit)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda s: "repair"
        if s["issues"] and s["attempts"] < 1 and not s.get("composition_all_evidence")
        else END
        if s["issues"] or not s["answer"]["answerable"] or not (
            settings().answer_semantic_audit_enabled or s.get("composition_all_evidence")
        )
        else "audit",
    )
    graph.add_conditional_edges(
        "audit",
        lambda s: "repair"
        if s["issues"] and s["attempts"] < 1 and not s.get("composition_all_evidence")
        else END,
    )
    graph.add_edge("repair", "validate")
    result = graph.compile().invoke({}, {"recursion_limit": 12})
    result["usage"]["answer_recovery"] = {
        "framework": "langgraph",
        "repair_attempts": result["attempts"],
        "first_issues": result["first_issues"],
        "remaining_issues": result["issues"],
        "stages": result["stages"],
    }
    result["usage"]["quality_validation_failed"] = bool(result["issues"])
    return GeneratedAnswer.model_validate(result["answer"]), result["usage"]
