"""Source-bound semantic audit component, with shadow observations only.

The registered runner below assembles complete Dev shadow observations; no
production gate is generated. ``reauthorize`` is caller-owned readonly authority (not serializable config):
it must return the complete ordered evidence reauthorized under the caller's scope,
or raise. Both pre-dispatch and post-audit records must match the supplied trusted
records. The caller supplies full-source SHA identities; chunk text is never
hashed as a substitute for a source identity.
"""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import time
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from app.chunking import token_count
from app.clients import DependencyError, GeneratedAnswer, Models
from app.execution_budget import BudgetExceeded, ExecutionBudget, TaskCancelled, use_budget
from app.qa import validate_claims
from app.task_analysis import judgment_intent

import argparse
import importlib.util
from importlib.metadata import version
import math
from pathlib import Path
import re
import sys

from app.config import settings
from scripts.benchmark_runtime import atomic_json
from scripts.research_review import digest
from scripts.retrieval_quality_index import active_scope, index_ledger
from scripts.run_evidence_selection_dev import rebind_rows
from scripts.run_evidence_supplement_dev import active_model_identity
from scripts.run_external_repair_dev import private_root
from scripts.run_retrieval_quality_dev import ROOT, _publish, code_fingerprints, reserve_output, sha


_SOURCE_FIELDS = ("chunk_id", "document_id", "version_id", "source_sha256", "title", "text", "locator")
_CONTEXT = 8192
_SECONDS = 180


class _AuditStop(RuntimeError):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason
        super().__init__(reason)


def _serialized(value):
    # JSON bytes distinguish 1, true and 1.0; Python record equality does not.
    # Encode first to reject nonfinite values/cycles, then disallow coercions
    # json.dumps otherwise permits (tuple arrays and non-string object keys).
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise TypeError("non_json_object_key")
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise TypeError("non_json_field")
    return encoded


def _authorized(reauthorize, trusted):
    if not callable(reauthorize):
        raise _AuditStop("authorization_failure", "reauthorization_required")
    try:
        refreshed = reauthorize(deepcopy(trusted))
        refreshed_json = _serialized(refreshed)
    except (BudgetExceeded, TaskCancelled):
        raise
    except Exception as exc:
        raise _AuditStop("authorization_failure", "reauthorization_failed") from exc
    if refreshed_json != _serialized(trusted):
        raise _AuditStop("authorization_failure", "reauthorization_changed_sources")


def _bind(payload, evidence):
    """Bind original serialized citations before reusing production validation."""
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("missing_trusted_evidence")
    by_chunk = {}
    for row in evidence:
        if not isinstance(row, dict) or any(key not in row for key in _SOURCE_FIELDS):
            raise ValueError("missing_source_field")
        for key in ("chunk_id", "document_id", "version_id", "source_sha256", "text"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("invalid_source_field")
        if not isinstance(row["title"], str) or not isinstance(row["locator"], dict):
            raise ValueError("invalid_original_fields")
        if "metadata" in row and not isinstance(row["metadata"], dict):
            raise ValueError("invalid_original_metadata")
        if row["chunk_id"] in by_chunk:
            raise ValueError("duplicate_trusted_chunk")
        by_chunk[row["chunk_id"]] = row
    citations = payload.get("citations")
    if not isinstance(citations, list):
        raise ValueError("missing_citations")
    cited = set()
    for citation in citations:
        if not isinstance(citation, dict) or any(key not in citation for key in _SOURCE_FIELDS):
            raise ValueError("missing_original_citation_field")
        cid = citation["chunk_id"]
        if not isinstance(cid, str) or cid not in by_chunk or cid in cited:
            raise ValueError("unbound_or_duplicate_citation")
        original = by_chunk[cid]
        # Optional supplied source fields are checked too; audit metadata always
        # comes from trusted evidence, never a generated citation's replacement.
        if any(key not in original for key in citation) or _serialized(citation) != _serialized(
            {key: original[key] for key in citation}
        ):
            raise ValueError("original_citation_mismatch")
        cited.add(cid)
    supplied = [dict(deepcopy(row), id=f"E{i}") for i, row in enumerate(evidence, 1)]
    aliases = {row["chunk_id"]: row["id"] for row in supplied}
    claims, seen, used = [], set(), set()
    for claim in payload["claims"]:
        if not isinstance(claim, dict) or set(claim) != {"text", "evidence_ids", "quotes"}:
            raise ValueError("malformed_claim")
        key = _serialized(claim)
        if key in seen:
            raise ValueError("duplicate_claim")
        seen.add(key)
        ids, quotes = claim["evidence_ids"], claim["quotes"]
        if not isinstance(ids, list) or not isinstance(quotes, list) or len(ids) != len(quotes):
            raise ValueError("citation_arity")
        local = set()
        for cid, quote in zip(ids, quotes):
            if not isinstance(cid, str) or cid not in cited:
                raise ValueError("unbound_claim_chunk")
            if not isinstance(quote, str) or not quote.strip() or quote not in by_chunk[cid]["text"]:
                raise ValueError("nonoriginal_quote")
            if (cid, quote) in local:
                raise ValueError("duplicate_claim_source")
            local.add((cid, quote))
            used.add(cid)
        claims.append(dict(claim, evidence_ids=[aliases[cid] for cid in ids]))
    if cited != used:
        raise ValueError("uncited_original_source")
    generated = GeneratedAnswer(answerable=True, claims=claims)
    _, status = validate_claims(generated, supplied)
    if status != "answered":
        raise ValueError("production_claim_validation_failed")
    return generated, supplied


def _counter(value):
    return value if type(value) is int and value >= 0 else None


class _MeteredJudge:
    """One experiment meter around the existing judge transport, not an executor."""

    def __init__(self, models, cfg, budget, stack, cost):
        self.models, self.cfg, self.budget, self.stack, self.cost = models, cfg, budget, stack, cost
        self.usage = None
        self.output_limit = None

    def _record_usage(self, prompt, completion):
        cost = self.cost
        cost["provider_prompt_tokens"], cost["provider_completion_tokens"] = prompt, completion
        known = prompt is not None and completion is not None
        cost["token_accounting"] = "provider_observed" if known else "conservative_estimate"
        if self.usage is not None:
            # ExecutionBudget requires numeric accounting. Missing provider
            # counters retain the prepaid conservative upper bound instead of
            # settling to 0. These numeric entries are explicitly estimates;
            # observed provider counters above and in the record remain null.
            self.usage.update(
                prompt_tokens=prompt if prompt is not None else _CONTEXT,
                completion_tokens=completion if completion is not None else self.output_limit,
                token_accounting=cost["token_accounting"],
                provider_prompt_tokens=prompt,
                provider_completion_tokens=completion,
            )

    def _judge_chat(self, body):
        body = deepcopy(body)
        body["model"] = self.cfg.semantic_judge_model
        self.output_limit = body["options"]["num_predict"]
        estimate = token_count(_serialized(body))
        self.cost.update(
            input_token_estimate=estimate,
            output_token_limit=self.output_limit,
            conservative_reserved_tokens=_CONTEXT + self.output_limit,
        )
        if estimate + self.output_limit > _CONTEXT:
            raise _AuditStop("preflight_block", "serialized_request_estimate_exceeds_context")
        # Enter immediately before physical dispatch, but close only after the
        # official audit API has parsed and checked its complete output schema.
        self.usage = self.stack.enter_context(
            self.budget.call("judge", reserve_tokens=_CONTEXT + self.output_limit)
        )
        self.cost["dispatches"] += 1
        self._record_usage(None, None)
        try:
            result = Models._judge_chat.__wrapped__(
                self.models, body, base=self.cfg.semantic_judge_url or self.cfg.ollama_url
            )
        except (BudgetExceeded, TaskCancelled):
            raise
        except Exception as exc:
            extra = getattr(exc, "usage", {}) or {}
            self._record_usage(
                _counter(extra.get("prompt_tokens")), _counter(extra.get("completion_tokens"))
            )
            self.cost["transport_failures"] += 1
            raise _AuditStop("transport_failure", "judge_transport_failed") from exc
        self.cost["transport_successes"] += 1
        if not isinstance(result, dict):
            raise _AuditStop("schema_failure", "invalid_transport_response")
        self._record_usage(_counter(result.get("prompt_eval_count")), _counter(result.get("eval_count")))
        for field, target in (
            ("total_duration", "server_ms"),
            ("load_duration", "load_ms"),
            ("prompt_eval_duration", "prompt_eval_ms"),
            ("eval_duration", "completion_eval_ms"),
        ):
            value = result.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                self.cost[target] = self.usage[target] = round(value / 1e6, 1)
        prompt, completion = self.cost["provider_prompt_tokens"], self.cost["provider_completion_tokens"]
        if (
            (prompt is not None and prompt > _CONTEXT)
            or (completion is not None and completion > self.output_limit)
            or (prompt is not None and completion is not None and prompt + completion > _CONTEXT)
        ):
            raise _AuditStop("ceiling_violation", "provider_reported_token_ceiling_violation")
        return result


def audit_answer(question, payload, evidence, models, cfg, *, reauthorize):
    """Observe an existing answer without changing its claims/status/verdict.

    Dict payloads use a disclosed canonical JSON digest; bytes/string payloads use
    their original UTF-8 bytes. Evidence is trusted original source material and
    must be reauthorized by the explicit caller callback before and after audit.
    Provider counts and conservative numeric budget bookkeeping are separate.
    """
    started = time.monotonic()
    raw = (
        payload
        if isinstance(payload, bytes)
        else payload.encode("utf-8")
        if isinstance(payload, str)
        else None
    )
    digest = hashlib.sha256(raw).hexdigest() if raw is not None else None
    limits = {
        "generation": 0,
        "policy": 0,
        "search": 0,
        "embedding": 0,
        "judge": 1,
        "judge_tokens": cfg.agent_judge_token_budget,
    }
    budget = ExecutionBudget(cfg, saved={"limits": limits})
    budget.state["deadline"] = min(budget.state["deadline"], budget.clock() + _SECONDS)
    cost = dict(
        dispatches=0,
        transport_successes=0,
        transport_failures=0,
        schema_failures=0,
        provider_prompt_tokens=0,
        provider_completion_tokens=0,
        token_accounting="no_dispatch",
        input_token_estimate=None,
        token_counter="cl100k_base_estimate_not_provider_exact",
        serialized_request_includes="schema_messages_options_and_json_framing",
        provider_framing_and_truncation_verified=False,
        conservative_reserved_tokens=0,
        elapsed_limit_seconds=_SECONDS,
        context_token_limit=_CONTEXT,
        model=cfg.semantic_judge_model,
    )
    result = dict(
        mode="shadow",
        status=None,
        reason=None,
        original_payload_sha256=digest,
        independent=False,
        scope="each_claim_bound_quotes_only",
        payload_digest_encoding="original_bytes" if raw is not None else "canonical_json_utf8",
        original_status=None,
        original_verdict=None,
        assertions_supported=None,
        question_complete=None,
        model_judgment_relation=None,
        original_verdict_mismatch=None,
        semantic_pass=None,
        audit=None,
        cost=cost,
    )
    try:
        frozen = json.loads(raw) if raw is not None else deepcopy(payload)
        trusted = deepcopy(evidence)
        frozen_json = _serialized(frozen)
        if raw is None:
            result["original_payload_sha256"] = hashlib.sha256(frozen_json.encode()).hexdigest()
        _serialized(trusted)
        if not isinstance(frozen, dict):
            raise ValueError("malformed_answer_payload")
        result.update(
            original_status=frozen.get("status"), original_verdict=deepcopy(frozen.get("verdict"))
        )
        if frozen.get("status") not in {"answered", "conflict"} or frozen.get("claims") == []:
            raise _AuditStop("not_applicable", "no_applicable_answer")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(frozen.get("claims"), list)
        ):
            raise ValueError("malformed_question_or_claims")
        generated, supplied = _bind(frozen, trusted)
        with use_budget(budget), ExitStack() as stack:
            _authorized(reauthorize, trusted)
            meter = _MeteredJudge(models, cfg, budget, stack, cost)
            # Call the official implementation for exhaustive claim checks,
            # original excerpts and conclusion references; no entailment fork.
            audit, _ = Models.audit_bound_answer(meter, question, generated, supplied)
            if any(
                len(check.source_indices) != len(set(check.source_indices)) for check in audit.checks
            ):
                raise _AuditStop("schema_failure", "duplicate_source_proof_indices")
            if len(audit.conclusion_claim_indices) != len(set(audit.conclusion_claim_indices)):
                raise _AuditStop("schema_failure", "duplicate_conclusion_proof_indices")
            if not judgment_intent(question) and (
                audit.verdict != "unclear" or audit.conclusion or audit.conclusion_claim_indices
            ):
                raise _AuditStop("schema_failure", "unexpected_nonjudgment_relation")
            _authorized(reauthorize, trusted)
        supported = all(check.support == "supported" for check in audit.checks)
        checked_indices = {check.claim_index for check in audit.checks if check.support == "supported"}
        relation_bound = (
            (
                bool(audit.conclusion_claim_indices)
                and set(audit.conclusion_claim_indices).issubset(checked_indices)
            )
            if audit.verdict != "unclear"
            else None
        )
        original = frozen.get("verdict")
        original_value = original.get("value") if isinstance(original, dict) else original
        expects_verdict = judgment_intent(question)
        mismatch = (
            (original_value != audit.verdict)
            if expects_verdict and original_value in ("yes", "no", "unclear")
            else None
        )
        judgment_pass = (
            (relation_bound is True and original_value in ("yes", "no") and mismatch is False)
            if expects_verdict
            else True
        )
        result.update(
            status="observed",
            assertions_supported=supported,
            question_complete=audit.question_complete,
            audit=audit.model_dump(),
            model_judgment_relation=dict(
                value=audit.verdict,
                conclusion=audit.conclusion,
                claim_indices=audit.conclusion_claim_indices,
                supported_by_checked_claims=relation_bound,
            ),
            original_verdict_mismatch=mismatch,
            semantic_pass=supported and audit.question_complete and judgment_pass,
        )
    except _AuditStop as exc:
        result.update(status=exc.status, reason=exc.reason)
        if exc.status == "schema_failure":
            cost["schema_failures"] += 1
    except (TaskCancelled, KeyboardInterrupt) as exc:
        result.update(status="interrupted", reason=str(exc))
    except BudgetExceeded as exc:
        result.update(status="budget_exhausted", reason=str(exc))
    except DependencyError:
        result.update(status="schema_failure", reason="official_audit_schema_validation_failed")
        cost["schema_failures"] += 1
    except (ValueError, KeyError, TypeError, ValidationError):
        if cost["dispatches"]:
            result.update(status="schema_failure", reason="invalid_audit_response")
            cost["schema_failures"] += 1
        else:
            result.update(status="mechanical_block", reason="invalid_original_binding")
    finally:
        cost["wall_ms"] = round((time.monotonic() - started) * 1000, 1)
        cost["business_budget"] = budget.snapshot()
        cost["business_budget_token_fields"] = (
            "numeric_bookkeeping_see_per_record_token_accounting_not_provider_totals"
        )
    return result


# The replay layer freezes business identities; orchestration remains a bounded
# sequential experiment. No production policy or semantic threshold is changed.

_ARMS = ("shared_contract", "literal_fallback")
_RUNNER_SOURCE = "scripts/run_bound_citation_dev.py"
_HELPER_SOURCE = ".runtime/evidence-first-20261011/generation_review_v2.py"
_OUTCOMES = (
    "observed",
    "not_applicable",
    "mechanical_block",
    "authorization_failure",
    "preflight_block",
    "transport_failure",
    "schema_failure",
    "ceiling_violation",
    "interrupted",
    "budget_exhausted",
    "not_observed",
)
_BOOLEAN_FIELDS = (
    "assertions_supported",
    "question_complete",
    "original_verdict_mismatch",
    "semantic_pass",
)
_COST_COUNTERS = (
    "dispatches",
    "transport_successes",
    "transport_failures",
    "schema_failures",
    "provider_prompt_tokens",
    "provider_completion_tokens",
    "input_token_estimate",
    "output_token_limit",
    "conservative_reserved_tokens",
)
_COST_TIMES = ("wall_ms", "server_ms", "load_ms", "prompt_eval_ms", "completion_eval_ms")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _safe_id(value):
    _require(type(value) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value), "unsafe_identifier")
    return value


def _read(path):
    return json.loads(Path(path).read_bytes())


def _review_helper():
    path = ROOT / _HELPER_SOURCE
    spec = importlib.util.spec_from_file_location("bound_citation_stage2_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _review_paths(args, helper):
    run = private_root(args.run)
    paths = helper.paths_for(run.parents[3], run)
    for argument, key in (
        ("review_root", "out"),
        ("generation_registration", "generation_registration"),
        ("generation_public", "source_artifact"),
        ("review_registration", "registration"),
        ("review_public", "public"),
    ):
        _require(
            Path(getattr(args, argument)).resolve() == paths[key].resolve(),
            "nonstandard_review_lineage_path",
        )
    return paths


def _expected_frozen_paths(args, helper, generation):
    """The exact Stage2 v2 prepare() declaration: 30 business + 205 source files."""
    p = _review_paths(args, helper)
    expected = {
        "raw": p["run"] / "raw.json",
        "suite": p["run"] / "suite.json",
        "packet": p["run"] / "review-packet.json",
        "runtime_registration": p["run"] / "registration.json",
        "generation_registration": p["generation_registration"],
        "source_artifact": p["source_artifact"],
        "v1_registration": p["v1_registration"],
        "v1_artifact": p["v1_artifact"],
        "v1_raw": p["v1_run"] / "raw.json",
        "v1_suite": p["v1_run"] / "suite.json",
        "v1_packet": p["v1_run"] / "review-packet.json",
        "v1_mapping": p["v1_out"] / "mapping.json",
        "v1_instructions": p["base"] / "generation-review-instructions-v1.md",
        "helper": ROOT / _HELPER_SOURCE,
        "instructions": p["instructions"],
        "mapping": p["out"] / "mapping.json",
        "canonical": p["out"] / "canonical-evidence.json",
    }
    for name in ("prepare_generation_review_v1.py", "score_generation_review_v1.py"):
        expected["v1_helper_" + name] = p["base"] / name
    for kind, count in (("answer", 4), ("evidence", 2)):
        for number in range(1, count + 1):
            name = f"{kind}-part-{number}.json"
            expected["v1_input_" + name] = p["v1_out"] / name
            if kind == "evidence":
                name = f"{kind}-part-{number}-reviews.json"
                expected["v1_review_" + name] = p["v1_out"] / name
    for number in (1, 2, 3):
        expected["answer_part_" + str(number)] = p["out"] / f"answer-part-{number}.json"
    for index, relative in enumerate(sorted(generation["code"])):
        expected[f"production_code_{index:04d}"] = ROOT / relative
    _require(len(expected) == 235, "exact_235_upstream_files_required")
    return {key: path.resolve() for key, path in expected.items()}


def _review_registration_metadata(registration, generation, raw, helper):
    fixed = {
        "version": "slot-generation-review-input-v2",
        "split": "dev",
        "comparison": "fallback",
        "source_run_id": raw["run_id"],
        "protocols": list(_ARMS),
        "candidate": _ARMS[1],
        "baseline": _ARMS[0],
        "positions": 94,
        "strict_scoreable_tasks": 40,
        "canonical_evidence_tasks": 47,
        "settings_sha256": generation["settings_sha256"],
        "production_code_sha256": generation["code"],
    }
    _require(
        all(_serialized(registration.get(key)) == _serialized(value) for key, value in fixed.items()),
        "review_registration_metadata_changed",
    )
    _require(
        registration["pending_task_ids"] == generation["review"]["pending_task_ids"]
        and set(registration["pending_task_ids"]) == helper.PENDING_IDS
        and len(registration["files"]) == 3
        and digest(settings().model_dump(mode="json")) == generation["settings_sha256"],
        "review_registration_pending_or_settings_changed",
    )


def _validate_review_bindings(args, helper, generation, registration):
    expected = _expected_frozen_paths(args, helper, generation)
    bindings = registration["frozen_files"]
    _require(
        type(bindings) is dict and set(bindings) == set(expected),
        "complete_upstream_key_path_byte_bindings_required",
    )
    for key, path in expected.items():
        record = bindings[key]
        _require(
            type(record) is dict
            and set(record) == {"path", "bytes_sha256"}
            and type(record["path"]) is str
            and type(record["bytes_sha256"]) is str
            and Path(record["path"]).resolve(strict=True) == path.resolve(strict=True)
            and path.is_file()
            and record["bytes_sha256"] == sha(path.read_bytes()),
            "complete_upstream_key_path_byte_bindings_required",
        )
    p = _review_paths(args, helper)
    anchor = _read(p["out"] / "registration-sha256.json")
    expected_anchor = {
        "registration_path": str(p["registration"].resolve()),
        "registration_bytes_sha256": sha(p["registration"].read_bytes()),
    }
    _require(
        type(anchor) is dict
        and set(anchor) == set(expected_anchor)
        and type(anchor["registration_path"]) is str
        and Path(anchor["registration_path"]).resolve(strict=True)
        == p["registration"].resolve(strict=True)
        and type(anchor["registration_bytes_sha256"]) is str
        and anchor["registration_bytes_sha256"] == expected_anchor["registration_bytes_sha256"],
        "review_registration_anchor_changed",
    )
    _require(
        registration["rubric_sha256"] == digest(helper.SYSTEM)
        and registration["instructions_sha256"] == digest(p["instructions"].read_text())
        and registration["assembly_sha256"] == {p["helper"].name: digest(p["helper"].read_text())},
        "review_rubric_instructions_helper_changed",
    )
    return expected, anchor


_ATTESTATION_KEYS = {
    "version",
    "source_run_id",
    "header",
    "review_registration_bytes_sha256",
    "generation_registration_bytes_sha256",
    "raw_bytes_sha256",
    "suite_bytes_sha256",
    "packet_bytes_sha256",
    "reviewed_semantic_sha256",
    "reference_file_bytes_sha256",
    "frozen_files",
    "anchor",
    "parts",
    "upstream_code",
    "runner_code_sha256",
    "stage2_helper_bytes_sha256",
}


def _reference_paths(args):
    out = private_root(args.review_root)
    return dict(
        review_public=Path(args.review_public).resolve(),
        reviewed=out / "reviewed.json",
        scored=out / "scored-primary.json",
        canonical=out / "canonical-evidence.json",
        **{f"reviewer_{i}": out / f"answer-part-{i}-reviews.json" for i in (1, 2, 3)},
    )


def _reference_header(public, registration, raw, helper):
    expected = {
        "version": "slot-generation-review-dev-v2",
        "split": "dev",
        "comparison": "fallback",
        "valid": True,
        "status": "complete",
        "run_id": raw["run_id"],
        "reviewer_model": "GPT-6 / Codex",
        "reviewer_human": False,
        "exact_serving_variant": None,
        "positions": 94,
        "strict_scoreable_tasks": 40,
        "canonical_evidence_tasks": 47,
        "reviewer_independent_fresh_context": True,
        "reviewer_independent_of_implementation_context": True,
        "model_family_independent_of_generator": True,
        "independent_of_model_gold_audit": False,
        "historical_answer_review_reused": False,
        "historical_evidence_review_reused": True,
        "headline_eligible": False,
        "historical_retrieval_gain_claim": False,
    }
    _require(
        all(
            type(public.get(key)) is type(value) and public.get(key) == value
            for key, value in expected.items()
        ),
        "review_public_header_changed",
    )
    _require(
        public["review_registration_sha256"] == digest(registration)
        and public["source_run_sha256"] == digest(raw)
        and public["pending_annotation_task_ids"] == sorted(helper.PENDING_IDS),
        "review_public_provenance_changed",
    )
    return expected


def prepare_review_attestation(args):
    """Separately invoked, model-free Stage2 lineage verification before registration.

    This sole preparation step may parse review files. Its exclusive output contains
    fixed header metadata and byte provenance only, never verdicts or reasons.
    Register/execute subsequently treat every reference artifact as opaque bytes.
    """
    helper = _review_helper()
    p = _review_paths(args, helper)
    raw, suite, packet = [
        _read(p["run"] / name) for name in ("raw.json", "suite.json", "review-packet.json")
    ]
    helper.require_full_run(raw, suite, packet)
    generation = _read(args.generation_registration)
    registration = _read(args.review_registration)
    _upstream_sources(generation, registration)
    _require(generation == raw["registration"], "generation_registration_changed")
    helper.source_bindings(p, raw, suite, packet)
    _review_registration_metadata(registration, generation, raw, helper)
    expected, anchor = _validate_review_bindings(args, helper, generation, registration)
    reference_before = {k: sha(path.read_bytes()) for k, path in _reference_paths(args).items()}
    registration_before = Path(args.review_registration).read_bytes()
    canonical, old_bindings = helper.canonical_evidence(p, packet)
    _require(
        _read(p["out"] / "canonical-evidence.json") == {"reviews": canonical}, "canonical_review_changed"
    )
    parts, mapping = helper.answer_parts(packet, generation["generator_identity"]["name"])
    _require(
        _read(p["out"] / "mapping.json") == mapping
        and digest(mapping) == registration["mapping_sha256"],
        "review_mapping_changed",
    )
    public = _read(args.review_public)
    header = _reference_header(public, registration, raw, helper)
    from scripts.benchmark_package_scoring import validate_reviews

    reviewed = _read(p["out"] / "reviewed.json")
    _reviews = validate_reviews(packet, reviewed)
    _require(digest(reviewed) == public["reviewed_sha256"], "reviewed_hash_changed")
    fresh = {}
    hashes = {}
    for number, part in enumerate(parts, 1):
        metadata = helper.reviewer_metadata(number, generation["generator_identity"]["name"])
        record = {
            "name": f"answer-part-{number}.json",
            "sha256": digest(part),
            "items": len(part["items"]),
            **metadata,
        }
        _require(
            registration["files"][number - 1] == record and _read(p["out"] / record["name"]) == part,
            "review_part_or_reviewer_changed",
        )
        name = record["name"].replace(".json", "-reviews.json")
        wrapper = _read(p["out"] / name)
        _require(
            set(wrapper) == {"reviews"}
            and len(wrapper["reviews"]) == len(part["items"])
            and {r["id"] for r in wrapper["reviews"]} == {i["id"] for i in part["items"]},
            "fresh_review_incomplete",
        )
        hashes[name] = {
            "bytes_sha256": sha((p["out"] / name).read_bytes()),
            "reviews_sha256": digest(wrapper["reviews"]),
        }
        items = {i["id"]: i for i in part["items"]}
        for review in wrapper["reviews"]:
            item = items[review["id"]]
            cid = mapping["task_to_canonical_evidence"][item["task"]["id"]]
            checked = helper.answer_decision(review, item, canonical[cid]["retrieved_fact_verdicts"])
            _require(review["id"] not in fresh, "duplicate_fresh_review")
            fresh[review["id"]] = {**checked, **metadata, "independent": True}
    _require(hashes == public["review_file_hashes"], "fresh_review_byte_provenance_changed")
    expected_reviewed = deepcopy(packet)
    expected_reviewed["reviews"] = []
    for arm in _ARMS:
        for row in raw["results"][arm]:
            rid = digest([arm, row["task_id"], row["run_id"]])
            uid = mapping["packet_to_unique_answer"][rid]
            cid = mapping["task_to_canonical_evidence"][row["task_id"]]
            expanded = deepcopy(fresh[uid])
            expanded.update(
                id=rid,
                identical_answer_review_id=uid,
                canonical_evidence_review_id=cid,
                canonical_evidence_reused=True,
                canonical_evidence_provenance=deepcopy(canonical[cid]),
            )
            expected_reviewed["reviews"].append(expanded)
    _require(reviewed == expected_reviewed, "expanded_fresh_review_changed")
    scored = _read(p["out"] / "scored-primary.json")
    _require(
        helper.public_details(scored["details"]) == public["details"], "scored_public_reference_changed"
    )
    for arm in _ARMS:
        _require(
            set(scored["details"][arm]) == set(generation["selected_task_ids"]),
            "scored_denominator_changed",
        )
        for tid in generation["selected_task_ids"]:
            strict = scored["details"][arm][tid]["answer"]["strict_task_success"]
            _require(
                strict is None if tid in helper.PENDING_IDS else type(strict) is bool,
                "scored_pending_primary_changed",
            )
    attestation = dict(
        version="bound-citation-review-metadata-v1",
        source_run_id=_safe_id(raw["run_id"]),
        header=header,
        review_registration_bytes_sha256=sha(Path(args.review_registration).read_bytes()),
        generation_registration_bytes_sha256=sha(Path(args.generation_registration).read_bytes()),
        raw_bytes_sha256=sha((p["run"] / "raw.json").read_bytes()),
        suite_bytes_sha256=sha((p["run"] / "suite.json").read_bytes()),
        packet_bytes_sha256=sha((p["run"] / "review-packet.json").read_bytes()),
        reviewed_semantic_sha256=public["reviewed_sha256"],
        reference_file_bytes_sha256=reference_before,
        frozen_files=deepcopy(registration["frozen_files"]),
        anchor=anchor,
        parts=deepcopy(registration["files"]),
        upstream_code=deepcopy(generation["code"]),
        runner_code_sha256=sha((ROOT / _RUNNER_SOURCE).read_bytes()),
        stage2_helper_bytes_sha256=sha((ROOT / _HELPER_SOURCE).read_bytes()),
    )
    _validate_review_bindings(args, helper, generation, registration)
    _review_registration_metadata(registration, generation, raw, helper)
    _upstream_sources(generation, registration)
    _require(
        Path(args.review_registration).read_bytes() == registration_before
        and reference_before
        == {k: sha(path.read_bytes()) for k, path in _reference_paths(args).items()},
        "reference_changed_during_attestation",
    )
    _serialized(attestation)
    destination = private_root(args.review_attestation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _publish(destination, attestation)
    return attestation


def _read_attestation(args, helper, generation, registration, raw):
    attestation = _read(args.review_attestation)
    _require(
        type(attestation) is dict and set(attestation) == _ATTESTATION_KEYS,
        "incomplete_or_unregistered_attestation_fields",
    )
    canonical = (json.dumps(attestation, ensure_ascii=False, indent=2) + "\n").encode()
    _require(Path(args.review_attestation).read_bytes() == canonical, "attestation_bytes_changed")
    _, anchor = _validate_review_bindings(args, helper, generation, registration)
    _require(
        attestation["version"] == "bound-citation-review-metadata-v1"
        and attestation["source_run_id"] == raw["run_id"]
        and attestation["frozen_files"] == registration["frozen_files"]
        and attestation["anchor"] == anchor
        and attestation["parts"] == registration["files"]
        and attestation["upstream_code"] == generation["code"]
        and attestation["runner_code_sha256"] == sha((ROOT / _RUNNER_SOURCE).read_bytes())
        and attestation["stage2_helper_bytes_sha256"] == sha((ROOT / _HELPER_SOURCE).read_bytes()),
        "attestation_source_identity_changed",
    )
    for key, path in (
        ("review_registration", Path(args.review_registration)),
        ("generation_registration", Path(args.generation_registration)),
        ("raw", Path(args.run) / "raw.json"),
        ("suite", Path(args.run) / "suite.json"),
        ("packet", Path(args.run) / "review-packet.json"),
    ):
        _require(
            attestation[key + "_bytes_sha256"] == sha(path.read_bytes()),
            "attested_upstream_bytes_changed",
        )
    _require(
        attestation["reference_file_bytes_sha256"]
        == {k: sha(path.read_bytes()) for k, path in _reference_paths(args).items()},
        "attested_reference_bytes_changed",
    )
    # Header is fixed business metadata, with no reference predictions permitted.
    expected_header = {
        "version": "slot-generation-review-dev-v2",
        "split": "dev",
        "comparison": "fallback",
        "valid": True,
        "status": "complete",
        "run_id": raw["run_id"],
        "reviewer_model": "GPT-6 / Codex",
        "reviewer_human": False,
        "exact_serving_variant": None,
        "positions": 94,
        "strict_scoreable_tasks": 40,
        "canonical_evidence_tasks": 47,
        "reviewer_independent_fresh_context": True,
        "reviewer_independent_of_implementation_context": True,
        "model_family_independent_of_generator": True,
        "independent_of_model_gold_audit": False,
        "historical_answer_review_reused": False,
        "historical_evidence_review_reused": True,
        "headline_eligible": False,
        "historical_retrieval_gain_claim": False,
    }
    _require(
        _serialized(attestation["header"]) == _serialized(expected_header), "attestation_header_changed"
    )
    _require(
        type(attestation["reviewed_semantic_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", attestation["reviewed_semantic_sha256"]),
        "invalid_attested_review_hash",
    )
    return attestation


def _input_paths(args, review_registration, attestation):
    run, reviewed = private_root(args.run), private_root(args.review_root)
    paths = {
        key: Path(getattr(args, key)).resolve()
        for key in (
            "generation_registration",
            "generation_public",
            "review_registration",
            "review_public",
            "review_attestation",
            "parent",
            "fixture",
        )
    }
    paths.update(
        {
            key: run / name
            for key, name in (
                ("raw", "raw.json"),
                ("suite", "suite.json"),
                ("packet", "review-packet.json"),
                ("runtime_registration", "registration.json"),
            )
        }
    )
    paths.update(_reference_paths(args))
    paths["mapping"] = reviewed / "mapping.json"
    paths["review_anchor"] = reviewed / "registration-sha256.json"
    paths["stage2_helper"] = ROOT / _HELPER_SOURCE
    for key, record in review_registration["frozen_files"].items():
        paths["upstream_" + sha(key.encode())] = Path(record["path"])
    return paths


def _upstream_sources(generation, review_registration):
    expected, actual = generation["code"], code_fingerprints()
    _require(
        type(expected) is dict
        and len(expected) == 205
        and review_registration["production_code_sha256"] == expected,
        "upstream_205_source_binding_required",
    )
    _require(
        set(actual) - set(expected) <= {_RUNNER_SOURCE} and set(expected) <= set(actual),
        "unregistered_new_or_missing_source",
    )
    for relative, fingerprint in expected.items():
        path = (ROOT / relative).resolve()
        _require(
            path.is_relative_to(ROOT)
            and path.is_file()
            and actual[relative] == fingerprint
            and sha(path.read_bytes()) == fingerprint,
            "upstream_source_bytes_changed",
        )
    return actual


def _validate_authored_fixture(fixture):
    pairs = fixture.get("pairs")
    _require(type(pairs) is list and len(pairs) == 12, "authored_exact_twelve_pairs_required")
    contrasts = set()
    for pair in pairs:
        _require(
            type(pair) is dict and set(pair) == {"contrast", "faithful", "contrast_case"},
            "authored_pair_schema_changed",
        )
        contrast = pair["contrast"]
        _require(
            type(contrast) is str
            and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", contrast)
            and contrast not in contrasts,
            "authored_unique_contrast_required",
        )
        contrasts.add(contrast)
        for role in ("faithful", "contrast_case"):
            case = pair[role]
            _require(
                type(case) is dict
                and set(case) == {"question", "evidence", "payload", "expected", "label_basis"},
                "authored_case_schema_changed",
            )
            expected = case["expected"]
            _require(
                type(expected) is dict
                and set(expected) == {"assertions_supported", "question_complete", "stage"}
                and type(expected["assertions_supported"]) is bool
                and type(expected["question_complete"]) is bool
                and type(expected["stage"]) is str
                and expected["stage"] in ("semantic_observation", "mechanical_or_semantic"),
                "authored_expected_schema_requires_strict_booleans",
            )


def _load_inputs(args):
    helper = _review_helper()
    run = private_root(args.run)
    raw, suite, packet = [_read(run / name) for name in ("raw.json", "suite.json", "review-packet.json")]
    pending = helper.require_full_run(raw, suite, packet)
    generation, review_registration, parent, fixture = [
        _read(getattr(args, name))
        for name in ("generation_registration", "review_registration", "parent", "fixture")
    ]
    helper.source_bindings(
        {
            "run": run,
            "generation_registration": Path(args.generation_registration),
            "source_artifact": Path(args.generation_public),
        },
        raw,
        suite,
        packet,
    )
    _require(generation == raw["registration"], "upstream_generation_registration_changed")
    _require(
        review_registration["version"] == "slot-generation-review-input-v2"
        and review_registration["split"] == "dev"
        and review_registration["comparison"] == "fallback"
        and review_registration["source_run_id"] == raw["run_id"]
        and review_registration["protocols"] == list(_ARMS)
        and review_registration["positions"] == 94
        and review_registration["strict_scoreable_tasks"] == 40
        and set(review_registration["pending_task_ids"]) == set(pending),
        "review_registration_changed",
    )
    for name, value in (("raw", raw), ("suite", suite), ("packet", packet)):
        _require(
            review_registration[name + "_sha256"] == digest(value), "upstream_semantic_hash_changed"
        )
    _review_registration_metadata(review_registration, generation, raw, helper)
    attestation = _read_attestation(args, helper, generation, review_registration, raw)
    mapping = _read(private_root(args.review_root) / "mapping.json")
    _, expected_mapping = helper.answer_parts(packet, generation["generator_identity"]["name"])
    _require(
        mapping == expected_mapping and digest(mapping) == review_registration["mapping_sha256"],
        "review_mapping_changed",
    )
    _require(len(review_registration["files"]) == 3, "incomplete_registered_reviewer_parts")
    parts, _ = helper.answer_parts(packet, generation["generator_identity"]["name"])
    for i, (record, part) in enumerate(zip(review_registration["files"], parts, strict=True), 1):
        _require(
            record
            == {
                "name": f"answer-part-{i}.json",
                "sha256": digest(part),
                "items": len(part["items"]),
                **helper.reviewer_metadata(i, generation["generator_identity"]["name"]),
            },
            "reviewer_identity_changed",
        )
        _require(_read(private_root(args.review_root) / record["name"]) == part, "review_part_changed")
    code = _upstream_sources(generation, review_registration)
    cfg = settings()
    _require(
        digest(cfg.model_dump(mode="json"))
        == generation["settings_sha256"]
        == review_registration["settings_sha256"],
        "upstream_settings_changed",
    )
    identity = parent["identity"]
    _require(
        digest(identity["corpus"]) == identity["corpus_sha256"] == raw["corpus_sha256"]
        and digest(identity["source_index"]) == identity["source_index_sha256"],
        "source_parent_changed",
    )
    _require(
        sha(Path(args.parent).read_bytes()) in generation["input_bytes_sha256"],
        "source_parent_not_byte_bound",
    )
    _require(
        fixture["human_gold"] is False
        and fixture["label_author"] == "assistant"
        and fixture["inference_status"] == "not_run"
        and len(fixture["pairs"]) == 12
        and fixture["source_identity"]
        == "synthetic_caller_owned_identity_not_verified_full_source_bytes",
        "authored_fixture_provenance_changed",
    )
    _validate_authored_fixture(fixture)
    paths = _input_paths(args, review_registration, attestation)
    return dict(
        raw=raw,
        suite=suite,
        packet=packet,
        generation=generation,
        review_registration=review_registration,
        review_attestation=attestation,
        parent=parent,
        fixture=fixture,
        paths=paths,
        code=code,
        cfg=cfg,
        pending=pending,
    )


def _audit_configuration(cfg):
    return dict(
        seconds=_SECONDS,
        context_tokens=_CONTEXT,
        output_rule="min_750_80_times_claims_plus_100",
        judge_calls=1,
        judge_tokens=cfg.agent_judge_token_budget,
        generation=0,
        policy=0,
        search=0,
        embedding=0,
        token_counter="cl100k_base_estimate_not_provider_exact",
        provider_framing_and_truncation_verified=False,
        independent=False,
        shadow=True,
        full_question_required=True,
    )


def _positions(inputs):
    rows, configuration = [], _audit_configuration(inputs["cfg"])
    for task_index, task in enumerate(inputs["suite"]["tasks"]):
        _safe_id(task["id"])
        _safe_id(task["user"])
        for arm in _ARMS:
            row = inputs["raw"]["results"][arm][task_index]
            rows.append(
                dict(
                    id=digest([arm, row["task_id"], row["run_id"]]),
                    kind="dev",
                    arm=arm,
                    task_id=task["id"],
                    task_index=task_index,
                    user=task["user"],
                    question=task["goal"],
                    payload=deepcopy(row["answer_payload"]),
                    evidence=deepcopy(row["retrieved_evidence"]),
                    pending=task["id"] in inputs["pending"],
                )
            )
    for index, pair in enumerate(inputs["fixture"]["pairs"]):
        for role in ("faithful", "contrast_case"):
            case = pair[role]
            rows.append(
                dict(
                    id=digest(["authored", index, role]),
                    kind="authored",
                    arm=role,
                    task_id=None,
                    task_index=index,
                    user=None,
                    question=case["question"],
                    payload=deepcopy(case["payload"]),
                    evidence=deepcopy(case["evidence"]),
                    pending=False,
                    expected=deepcopy(case["expected"]),
                )
            )
    for row in rows:
        row["payload_sha256"] = sha(_serialized(row["payload"]).encode())
        row["evidence_sha256"] = sha(_serialized(row["evidence"]).encode())
        # Scope user is part of complete input identity; no cross-user sharing.
        row["audit_input_sha256"] = digest(
            [row["question"], row["payload"], row["evidence"], configuration, row["user"]]
        )
    return rows


def _position_identity(row):
    return {
        key: deepcopy(row[key])
        for key in (
            "id",
            "kind",
            "arm",
            "task_id",
            "task_index",
            "pending",
            "payload_sha256",
            "evidence_sha256",
            "audit_input_sha256",
        )
    }


def registration_for(args):
    inputs = _load_inputs(args)
    cfg = inputs["cfg"]
    judge_cfg = cfg.model_copy(
        update={
            "agent_policy_model": cfg.semantic_judge_model,
            "agent_policy_url": cfg.semantic_judge_url,
        }
    )
    model = active_model_identity(judge_cfg)
    rows = _positions(inputs)
    return dict(
        version="bound-citation-input-v1",
        split="dev",
        mode="shadow",
        source_run_id=_safe_id(inputs["raw"]["run_id"]),
        dev_positions=94,
        authored_positions=24,
        primary_tasks=40,
        pending_tasks=7,
        protocols=list(_ARMS),
        corpus_sha256=inputs["parent"]["identity"]["corpus_sha256"],
        index_sha256=inputs["parent"]["identity"]["source_index_sha256"],
        pending_task_ids=sorted(inputs["pending"]),
        positions=[_position_identity(row) for row in rows],
        input_bytes_sha256={key: sha(path.read_bytes()) for key, path in inputs["paths"].items()},
        upstream_code=deepcopy(inputs["generation"]["code"]),
        code=inputs["code"],
        settings_sha256=digest(cfg.model_dump(mode="json")),
        model_identity_sha256=digest(model["policy"]),
        libraries_sha256=digest(model["libraries"]),
        tokenizer_library_sha256=digest(version("tiktoken")),
        configuration=_audit_configuration(cfg),
        deduplicate=args.deduplicate is True,
        duplicate_policy="exact_complete_input_and_scope_user",
        payload_encoding="canonical_json_utf8_nested_original_payload_with_raw_file_byte_binding",
        cost_rule="one_physical_record_per_unique_input_shared_costs_not_independent_arm_latency",
        accuracy_basis="assistant_authored_contrasts_and_GPT_Codex_model_reference_agreement_not_human_accuracy",
        retrieval_gain_claim=False,
        answer_correctness_gain_claim=False,
        headline_eligible=False,
    )


def make_models():
    return Models()


def public_cost(cost):
    result = {key: _counter(cost.get(key)) for key in _COST_COUNTERS}
    for key in _COST_TIMES:
        value = cost.get(key)
        result[key] = (
            value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None
        )
    result["provider_tokens_known"] = all(
        result[k] is not None for k in ("provider_prompt_tokens", "provider_completion_tokens")
    )
    return result


def _public_position(row):
    out = _position_identity(row)
    for key in ("id", "payload_sha256", "evidence_sha256", "audit_input_sha256"):
        value = out[key]
        out[key] = value if type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) else None
    out["kind"] = out["kind"] if out["kind"] in ("authored", "dev") else None
    out["arm"] = out["arm"] if out["arm"] in (*_ARMS, "faithful", "contrast_case") else None
    tid = out["task_id"]
    out["task_id"] = tid if type(tid) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", tid) else None
    out["task_index"] = _counter(out["task_index"])
    out["pending"] = out["pending"] is True
    observation = row.get("observation") or {}
    out["status"] = (
        observation.get("status") if observation.get("status") in _OUTCOMES else "not_observed"
    )
    for key in _BOOLEAN_FIELDS:
        value = observation.get(key)
        out[key] = value if type(value) is bool else None
    relation = observation.get("model_judgment_relation")
    relation = relation if type(relation) is dict else {}
    out["judgment_relation"] = (
        relation.get("value") if relation.get("value") in ("yes", "no", "unclear") else None
    )
    supported = relation.get("supported_by_checked_claims")
    out["relation_supported_by_checked_claims"] = supported if type(supported) is bool else None
    physical_id = row.get("physical_audit_id")
    out["physical_audit_id"] = (
        physical_id if type(physical_id) is str and re.fullmatch(r"[0-9a-f]{64}", physical_id) else None
    )
    out["shared_physical_cost"] = row.get("shared_physical_cost", False) is True
    return out


def reference_support(verdicts):
    if not verdicts or any(type(v) is not str for v in verdicts):
        return None
    if any(v in ("partial", "unsupported", "contradicted") for v in verdicts):
        return False
    return True if all(v == "supported" for v in verdicts) else None


def _agreement(prediction, reference):
    return prediction == reference if type(prediction) is bool and type(reference) is bool else None


def compare_references(positions, inputs, args):
    """Post-runtime model-reference comparison; never a production accuracy gate."""
    from scripts.benchmark_package_scoring import validate_reviews

    review_public = _read(args.review_public)
    reviewed = _read(private_root(args.review_root) / "reviewed.json")
    _require(
        digest(reviewed) == inputs["review_attestation"]["reviewed_semantic_sha256"],
        "reviewed_hash_changed",
    )
    reviews = validate_reviews(inputs["packet"], reviewed)
    scored = _read(private_root(args.review_root) / "scored-primary.json")
    comparisons = []
    for row in positions:
        observation = row.get("observation") or {}
        if row["kind"] == "authored":
            expected = row["expected"] if type(row.get("expected")) is dict else {}
            support, complete = expected.get("assertions_supported"), expected.get("question_complete")
            expected_pass = (
                support and complete if type(support) is bool and type(complete) is bool else None
            )
            comparisons.append(
                dict(
                    id=row["id"],
                    kind="authored",
                    pending=False,
                    support_agreement=_agreement(observation.get("assertions_supported"), support),
                    completeness_agreement=_agreement(observation.get("question_complete"), complete),
                    semantic_detection=observation.get("status") == "observed"
                    and observation.get("semantic_pass") is False
                    and expected_pass is False,
                    expected_pass=expected_pass,
                    predicted_pass=observation.get("semantic_pass")
                    if type(observation.get("semantic_pass")) is bool
                    else None,
                    mechanical_block=observation.get("status") == "mechanical_block",
                )
            )
            continue
        review = reviews[row["id"]]
        strict = scored["details"][row["arm"]][row["task_id"]]["answer"]["strict_task_success"]
        _require(
            strict is None if row["pending"] else type(strict) is bool, "primary_pending_score_changed"
        )
        _require(
            strict
            == review_public["details"][row["arm"]][row["task_id"]]["answer"]["strict_task_success"],
            "strict_score_public_reference_changed",
        )
        support = reference_support(review["claim_verdicts"] + review["citation_verdicts"])
        # Completeness is a separate rubric observation, not support in context.
        complete = reference_support(list(review["fact_verdicts"].values()))
        critical = review.get("no_critical_error")
        if critical is False and complete is True:
            complete = None
        comparisons.append(
            dict(
                id=row["id"],
                kind="dev",
                pending=row["pending"],
                support_reference=support,
                completeness_reference=complete,
                support_agreement=None
                if row["pending"]
                else _agreement(observation.get("assertions_supported"), support),
                completeness_agreement=None
                if row["pending"]
                else _agreement(observation.get("question_complete"), complete),
                strict_task_agreement=None
                if row["pending"]
                else _agreement(observation.get("semantic_pass"), strict),
                original_verdict_mismatch=observation.get("original_verdict_mismatch")
                if type(observation.get("original_verdict_mismatch")) is bool
                else None,
            )
        )
    authored = [row for row in comparisons if row["kind"] == "authored"]
    positives = [row for row in authored if row["expected_pass"] is True]
    negatives = [row for row in authored if row["expected_pass"] is False]
    metrics = dict(
        registered=len(authored),
        positive_registered=len(positives),
        negative_registered=len(negatives),
        semantic_false_accept_denominator=sum(row["predicted_pass"] is not None for row in negatives),
        semantic_false_reject_denominator=sum(row["predicted_pass"] is not None for row in positives),
        false_accepts=sum(row["predicted_pass"] is True for row in negatives),
        false_rejects=sum(row["predicted_pass"] is False for row in positives),
        negative_unknown=sum(row["predicted_pass"] is None for row in negatives),
        positive_unknown=sum(row["predicted_pass"] is None for row in positives),
        mechanical_blocked=sum(row["mechanical_block"] for row in authored),
        semantic_detections=sum(row["semantic_detection"] for row in authored),
    )
    return dict(
        basis="model_reference_agreement_not_human_accuracy",
        authored=metrics,
        primary_positions=80,
        pending_positions=14,
        review_cost_tokens=None,
        review_cost_currency=None,
        rows=comparisons,
    )


def _summaries(rows, physical):
    groups = {}
    for kind in ("authored", "dev"):
        selected = [r for r in rows if r["kind"] == kind]
        groups[kind] = dict(
            registered=len(selected),
            observed=sum(r["status"] == "observed" for r in selected),
            unknown=sum(r["semantic_pass"] is None for r in selected),
            operational_observed=sum(r["status"] == "observed" for r in selected),
            mechanical_blocked=sum(r["status"] == "mechanical_block" for r in selected),
            semantic_accept=sum(
                r["status"] == "observed" and r["semantic_pass"] is True for r in selected
            ),
            semantic_reject=sum(
                r["status"] == "observed" and r["semantic_pass"] is False for r in selected
            ),
            statuses={status: sum(r["status"] == status for r in selected) for status in _OUTCOMES},
        )
    costs = [record["cost"] for record in physical]
    groups["physical"] = dict(
        unique_records=len(costs),
        dispatches=sum(c["dispatches"] or 0 for c in costs),
        provider_prompt_tokens=sum(c["provider_prompt_tokens"] for c in costs)
        if all(c["provider_prompt_tokens"] is not None for c in costs)
        else None,
        provider_completion_tokens=sum(c["provider_completion_tokens"] for c in costs)
        if all(c["provider_completion_tokens"] is not None for c in costs)
        else None,
        unknown_provider_records=sum(not c["provider_tokens_known"] for c in costs),
    )
    return groups


def execute(args, reservation):
    """Fresh readonly replay, checkpointing observations before loading grades."""
    from sqlalchemy import text
    from app.db import SessionLocal
    from app.clients import Search

    runtime = private_root(args.private_root) / _safe_id(reservation["run_id"])
    runtime.mkdir(parents=True, exist_ok=False)
    registration, inputs, positions, physical = None, None, [], []
    public = dict(
        version="bound-citation-dev-v1",
        split="dev",
        mode="shadow",
        run_id=reservation["run_id"],
        valid=False,
        status="failed",
        positions=[],
        physical=[],
        comparison=None,
        drift={},
        error=None,
        headline_eligible=False,
        retrieval_gain_claim=False,
        answer_correctness_gain_claim=False,
    )
    stage, finished = "preflight", False
    try:
        registration_bytes = Path(args.registration).read_bytes()
        registration = json.loads(registration_bytes)
        # Registered denominator remains visible even if current inputs are corrupt.
        positions = [dict(row) for row in registration["positions"]]
        current = registration_for(args)
        _require(current == registration, "registration_changed")
        inputs = _load_inputs(args)
        positions = _positions(inputs)
        atomic_json(runtime / "registration.json", registration)
        source = Search(index=inputs["parent"]["identity"]["source_index"]["index"])
        with SessionLocal() as db:
            db.execute(text("SET TRANSACTION READ ONLY"))
            users, records, corpus = active_scope(db)
            index = index_ledger(source, records)
            _require(
                digest(corpus) == inputs["parent"]["identity"]["corpus_sha256"]
                and digest(index) == inputs["parent"]["identity"]["source_index_sha256"],
                "live_source_changed",
            )
            cache = {}
            stage = "audit"
            try:
                for row in positions:
                    key = row["audit_input_sha256"] if registration["deduplicate"] else row["id"]
                    physical_id = digest(["audit", key])
                    row.update(physical_audit_id=physical_id, shared_physical_cost=key in cache)
                    if row["kind"] == "dev":
                        user_key = row["user"]

                        def reauthorize(evidence, user_key=user_key):
                            fresh_users, _, fresh_corpus = active_scope(db)
                            return rebind_rows(db, fresh_users[user_key], evidence, fresh_corpus)
                    else:
                        # Authored identities are synthetic caller-owned records;
                        # never claim these came from the real corpus ledger.
                        def reauthorize(evidence):
                            return deepcopy(evidence)

                    if key in cache:
                        # Sharing inference never shares stale permission authority.
                        row["observation"] = deepcopy(cache[key])
                        row["observation"].update(
                            status="not_observed",
                            reason="cached_reauthorization_pending",
                            audit=None,
                            model_judgment_relation=None,
                            **dict.fromkeys(_BOOLEAN_FIELDS, None),
                        )
                        try:
                            _authorized(reauthorize, row["evidence"])
                        except BaseException as exc:
                            status = (
                                "budget_exhausted"
                                if isinstance(exc, BudgetExceeded)
                                else "interrupted"
                                if isinstance(exc, TaskCancelled) or not isinstance(exc, Exception)
                                else "authorization_failure"
                            )
                            row["observation"].update(
                                status=status, reason="cached_position_reauthorization_stopped"
                            )
                            raise
                        else:
                            row["observation"] = deepcopy(cache[key])
                        finally:
                            atomic_json(
                                runtime / "raw.json", dict(positions=positions, physical=physical)
                            )
                        continue
                    observation = audit_answer(
                        row["question"],
                        row["payload"],
                        row["evidence"],
                        make_models(),
                        inputs["cfg"],
                        reauthorize=reauthorize,
                    )
                    row["observation"] = observation
                    cache[key] = deepcopy(observation)
                    physical.append(dict(id=physical_id, observation=deepcopy(observation)))
                    atomic_json(runtime / "raw.json", dict(positions=positions, physical=physical))
                    if observation["status"] == "interrupted":
                        raise KeyboardInterrupt()
                finished = True
            finally:
                _, final_records, final_corpus = active_scope(db)
                public["drift"]["authorization"] = any(
                    (row.get("observation") or {}).get("status") == "authorization_failure"
                    for row in positions
                )
                public["drift"].update(
                    corpus=final_corpus != corpus, index=index_ledger(source, final_records) != index
                )
        stage = "post_runtime_comparison"
        _require(
            registration_for(args) == registration
            and Path(args.registration).read_bytes() == registration_bytes
            and all(v is False for v in public["drift"].values()),
            "runtime_identity_changed_before_reference_access",
        )
        public["comparison"] = compare_references(positions, inputs, args)
    except BaseException as exc:
        interrupted = isinstance(exc, TaskCancelled) or not isinstance(exc, Exception)
        public.update(
            status="interrupted" if interrupted else "failed",
            error=dict(stage=stage, type="Interrupted" if interrupted else "RuntimeFailure"),
        )
    finally:
        if registration is not None:
            try:
                public["drift"]["registration"] = (
                    Path(args.registration).read_bytes() != registration_bytes
                )
                public["drift"]["identity"] = registration_for(args) != registration
            except Exception:
                public["drift"]["identity"] = True
        public["positions"] = [_public_position(row) for row in positions]
        public["physical"] = [
            dict(id=record["id"], cost=public_cost(record["observation"]["cost"])) for record in physical
        ]
        public["summary"] = _summaries(public["positions"], public["physical"])
        public["valid"] = (
            finished
            and public["error"] is None
            and len(positions) == 118
            and bool(public["drift"])
            and all(v is False for v in public["drift"].values())
        )
        if public["valid"]:
            public["status"] = "complete"
        elif any(v is True for v in public["drift"].values()):
            public["status"] = "invalid_drift"
        raw = dict(
            version=public["version"],
            positions=positions,
            physical=physical,
            valid=public["valid"],
            status=public["status"],
            drift=public["drift"],
            comparison=public["comparison"],
        )
        if not public["valid"]:
            public["comparison"] = None
        atomic_json(runtime / "raw.json", raw)
        public["private_raw_sha256"] = digest(raw)
        _publish(args.output, public)
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Register or execute complete Dev47 bound citation shadow replay."
    )
    parser.add_argument("command", choices=("prepare-review-attestation", "register", "execute"))
    for name in (
        "run",
        "review-root",
        "generation-registration",
        "generation-public",
        "review-registration",
        "review-public",
        "review-attestation",
        "parent",
        "fixture",
        "registration",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--deduplicate", action="store_true", help="Preregister exact complete-input sharing."
    )
    parser.add_argument("--private-root", help="Ignored .runtime directory; required for execute.")
    parser.add_argument("--output", help="Fresh public artifact; required for execute.")
    args = parser.parse_args(argv)
    if args.command == "prepare-review-attestation":
        prepare_review_attestation(args)
        print(json.dumps(dict(attested=True, dev_positions=94, primary_tasks=40, pending_tasks=7)))
        return 0
    if args.command == "register":
        result = registration_for(args)
        Path(args.registration).parent.mkdir(parents=True, exist_ok=True)
        _publish(args.registration, result)
        print(json.dumps(dict(registered=True, dev_positions=94, authored_positions=24)))
        return 0
    if not args.private_root or not args.output:
        parser.error("execute requires --private-root and --output")
    result = execute(args, reserve_output(args.output))
    print(json.dumps(dict(valid=result["valid"], status=result["status"])))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
