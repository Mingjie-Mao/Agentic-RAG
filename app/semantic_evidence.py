"""Layered semantic evidence evaluation: relevance, coverage, sufficiency, faithfulness.

The Agent's control decisions — keep searching, what to search next, which evidence to
pass on, whether a subgoal is done — were driven by two lexical proxies: character
bigram overlap for "covers the subgoal" and keyword overlap for "is relevant". Both read
"looks alike" as "answers it". This module replaces them with four separate judgements,
because they fail in different directions and cost different amounts when wrong:

    relevance      (subgoal, passage) -> score           cross-encoder, recall-oriented
    coverage       (subgoal/slot, evidence) -> status     cascade: relevance, then one
                                                          batched structured judge
    sufficiency    judgements -> complete/partial/        pure code
                   missing/contradicted
    faithfulness   (claim, cited evidence) -> support     the calibrated §14A scorer

A passage can be highly relevant and still answer only half of a compound subgoal, so
relevance never marks a subgoal covered. The evaluator reports relations and confidence
only; it names no tool, writes no argument and changes no state. The controller reads
its output and decides what the system does.
"""

from dataclasses import dataclass, field
from contextlib import nullcontext
import json
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.execution_budget import BudgetExceeded, TaskCancelled, current_budget

Status = Literal["supported", "partial", "unsupported", "contradicted"]
STATUSES = ("supported", "partial", "unsupported", "contradicted")


class RelevanceJudgment(BaseModel):
    subgoal_id: str
    chunk_id: str
    score: float
    label: Literal["relevant", "uncertain", "irrelevant"]


class EvidenceJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subgoal_id: str
    status: Status
    support_refs: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    reason_code: str = Field(default="", max_length=40)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Literal["cascade_no_relevant_passage", "judge", "judge_unavailable", "literal_binding"] = "judge"
    evaluation_validity: Literal["evaluated", "unknown"] = "evaluated"
    quote_spans: list[dict[str, str]] = Field(default_factory=list)


@dataclass
class Thresholds:
    """Relevance cut points on the cross-encoder logit, fitted on the calibration dev
    split only. `keep` is set for recall (a useful passage must not be ranked out);
    `high` for precision. Coverage never uses them to say `supported`."""

    # Fitted on the dev split of artifacts/semantic-evidence-eval.json (keep: dev recall
    # >= 0.95; high: dev precision >= 0.90); test results are reported there once.
    keep: float = -5.0339
    high: float = 2.9672
    version: str = "semantic-evidence-v1"


@dataclass
class JudgeUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_ms: float = 0.0
    skipped_by_cascade: int = 0
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0

    def as_dict(self):
        return self.__dict__.copy()


class _JudgeRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subgoal_id: str
    status: Status
    support_refs: list[str]
    missing_slots: list[str]
    reason_code: str
    confidence: float


class _JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    judgments: list[_JudgeRow]


JUDGE_SYSTEM = (
    "你是证据覆盖评估器。对每个子目标，只依据给出的证据判断它是否已被回答，不使用外部知识，"
    "不决定下一步操作。status：supported=证据直接给出了子目标要求的全部内容；"
    "partial=只给出一部分（在 missing_slots 写缺的部分）；unsupported=证据没有回答它，"
    "只是话题相关也算 unsupported；contradicted=针对同一对象、同一指标、同一适用时间或范围，"
    "证据给出互不相容的值。不同时间、不同版本、不同对象的不同数值是变化或差异，不是 contradicted。"
    "support_refs 只能填证据编号。reason_code 用简短英文代码（如 value_present、value_missing、"
    "topic_only、scope_mismatch、incompatible_values）。confidence 取 0 到 1。只输出指定 JSON。"
)


class SemanticEvidenceEvaluator:
    """One interface for the four judgements. Scorers are injectable for tests."""

    def __init__(self, relevance_scorer=None, judge=None, thresholds: Thresholds | None = None, max_passages=8):
        self._relevance_scorer = relevance_scorer
        self._judge = judge
        self.thresholds = thresholds or Thresholds()
        self.max_passages = max_passages
        self.usage = JudgeUsage()

    # --- relevance -------------------------------------------------------------------
    def _scorer(self):
        if self._relevance_scorer is None:
            from app.evidence_selection import cross_encoder_scorer

            self._relevance_scorer = cross_encoder_scorer()
        return self._relevance_scorer

    def relevance(self, items: list[dict], passages: list[dict]) -> list[RelevanceJudgment]:
        """Score every (subgoal, passage) pair. Labels rank; they never delete."""
        out = []
        if not items or not passages:
            return out
        texts = [f"{p.get('header') or p.get('title', '')}\n{p['text']}" for p in passages]
        for item in items:
            scores = self._scorer()(item["text"], texts)
            for passage, score in zip(passages, scores, strict=True):
                label = (
                    "relevant" if score >= self.thresholds.high
                    else "uncertain" if score >= self.thresholds.keep
                    else "irrelevant"
                )
                out.append(RelevanceJudgment(
                    subgoal_id=item["id"], chunk_id=passage["chunk_id"], score=round(score, 4), label=label
                ))
        return out

    # --- coverage --------------------------------------------------------------------
    def coverage(self, question: str, items: list[dict], passages: list[dict],
                 relevance: list[RelevanceJudgment] | None = None) -> list[EvidenceJudgment]:
        """Cascade: a subgoal with no passage above the recall cut is unsupported without
        a model call; all others go to one batched structured judgement."""
        relevance = relevance if relevance is not None else self.relevance(items, passages)
        by_item: dict[str, list[RelevanceJudgment]] = {}
        for row in relevance:
            if row.label != "irrelevant":
                by_item.setdefault(row.subgoal_id, []).append(row)
        judgments, pending = [], []
        for item in items:
            if by_item.get(item["id"]):
                pending.append(item)
            else:
                judgments.append(EvidenceJudgment(
                    subgoal_id=item["id"], status="unsupported", reason_code="no_relevant_passage",
                    confidence=0.9, source="cascade_no_relevant_passage",
                ))
        if not pending:
            self.usage.skipped_by_cascade += 1
            return judgments
        # Candidate evidence for the judge: the best-scoring passages across the pending
        # subgoals, deduplicated, capped. Low-scoring passages stay available to the
        # system; they are only not shown to this one call.
        wanted = sorted(
            (row for item in pending for row in by_item[item["id"]]), key=lambda r: -r.score
        )
        chosen = list(dict.fromkeys(row.chunk_id for row in wanted))[: self.max_passages]
        passage_by_id = {p["chunk_id"]: p for p in passages}
        labelled = {f"E{n}": passage_by_id[cid] for n, cid in enumerate(chosen, 1)}
        judged = self._judge_batch(question, pending, labelled)
        known = {item["id"] for item in pending}
        seen = set()
        for row in judged:
            if row.subgoal_id not in known or row.subgoal_id in seen:
                continue
            seen.add(row.subgoal_id)
            refs = [labelled[key]["chunk_id"] for key in row.support_refs if key in labelled]
            status = row.status
            best = max((r.score for r in by_item.get(row.subgoal_id, [])), default=float("-inf"))
            # Supported must cite something the judge was shown; otherwise it is not
            # evidence of coverage, whatever the label says.
            if status in {"supported", "partial"} and not refs:
                status, reason = "unsupported", "no_cited_support"
            # Rule B (chosen on dev, pre-registered): a judge's "supported" also needs a
            # passage the cross-encoder rates highly. The judge's own confidence carried
            # no information (almost always 0.95-1.0), so it is not used as the gate.
            elif status == "supported" and best < self.thresholds.high:
                status, reason = "partial", "low_relevance_support"
            else:
                reason = row.reason_code[:40]
            judgments.append(EvidenceJudgment(
                subgoal_id=row.subgoal_id, status=status, support_refs=refs,
                missing_slots=row.missing_slots[:6], reason_code=reason,
                confidence=max(0.0, min(1.0, row.confidence)), source="judge",
            ))
        for item in pending:
            if item["id"] not in seen:
                judgments.append(EvidenceJudgment(
                    subgoal_id=item["id"], status="unsupported", reason_code="judge_omitted",
                    confidence=0.0, source="judge_unavailable",
                ))
        return judgments

    def _judge_batch(self, question, items, labelled) -> list[_JudgeRow]:
        payload = self._payload(question, items, labelled)
        system = getattr(self, "judge_system", JUDGE_SYSTEM)
        output_type = getattr(self, "output_type", _JudgeOutput)
        started = time.monotonic()
        budget = current_budget()
        reserve = len((system + json.dumps(payload, ensure_ascii=False)).encode()) + 600
        guard = budget.call("judge", reserve_tokens=reserve) if budget else nullcontext({})
        try:
            with guard as usage:
                self.usage.calls += 1
                self.usage.attempted += 1
                try:
                    try:
                        result = (self._judge(payload) if self._judge else _ollama_judge(
                            payload, schema=output_type.model_json_schema(), system=system))
                    except Exception as exc:
                        usage.update(getattr(exc, "token_usage", {}))
                        raise
                    usage.update(prompt_tokens=result.get("prompt_tokens", 0) or 0,
                                 completion_tokens=result.get("completion_tokens", 0) or 0)
                    rows = output_type.model_validate(result["output"]).judgments
                finally:
                    self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
                    self.usage.completion_tokens += usage.get("completion_tokens", 0)
            self.usage.succeeded += 1
            return rows
        except (BudgetExceeded, TaskCancelled):
            raise
        except Exception:
            self.usage.failed += 1
            return []
        finally:
            self.usage.wall_ms += round((time.monotonic() - started) * 1000, 1)

    def _payload(self, question, items, labelled):
        return {
            "question": question,
            "subgoals": [{"id": item["id"], "text": item["text"]} for item in items],
            "evidence": [
                {"id": key, "source": p.get("header") or p.get("title", ""), "text": p["text"][:1200]}
                for key, p in labelled.items()
            ],
        }

    # --- faithfulness ----------------------------------------------------------------
    @staticmethod
    def claim_support(evidence: list[dict], claims: list[dict]) -> list[dict]:
        """Claim against its cited evidence with the §14A-calibrated scorer (threshold
        chosen for precision on frozen v3 labels). Two states only: that scorer was
        calibrated as supported vs not, and a four-way label here would be invented."""
        from app.semantic_shadow import THRESHOLD, score_claims

        scored = score_claims(evidence, claims)["claim_support_scores"]
        return [
            {"claim_index": n, "score": s,
             "status": None if s is None else ("supported" if s >= THRESHOLD else "not_supported")}
            for n, s in enumerate(scored, 1)
        ]


def _ollama_judge(payload: dict, *, schema=None, system=None) -> dict:
    """The structured judge on a configured Ollama endpoint, thinking off."""
    import httpx

    cfg = settings()
    body = {
        "model": cfg.semantic_judge_model,
        "stream": False,
        "keep_alive": "30m",
        "think": False,
        "format": schema or _JudgeOutput.model_json_schema(),
        "messages": [
            {"role": "system", "content": system or JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "options": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 600},
    }
    if not cfg.semantic_judge_url:
        body.pop("think")
    budget = current_budget()
    response = httpx.post(
        (cfg.semantic_judge_url or cfg.ollama_url) + "/api/chat", json=body,
        timeout=budget.timeout(cfg.model_timeout_seconds) if budget else cfg.model_timeout_seconds, trust_env=False,
    )
    response.raise_for_status()
    data = response.json()
    try:
        output = json.loads(data["message"]["content"])
    except (ValueError, KeyError) as exc:
        exc.token_usage = {"prompt_tokens": data.get("prompt_eval_count", 0) or 0,
                           "completion_tokens": data.get("eval_count", 0) or 0}
        raise
    return {
        "output": output,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
    }


@dataclass
class CoverageReport:
    complete: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    contradicted: list[str] = field(default_factory=list)
    supported_confidence_floor: float = 0.0

    def as_dict(self):
        return self.__dict__.copy()


def aggregate_coverage(items: list[dict], judgments: list[EvidenceJudgment], *,
                       min_supported_confidence: float = 0.6) -> CoverageReport:
    """Pure code: judgements in, sufficiency state out.

    `supported` below the confidence floor counts as partial — a premature stop costs
    more than one more targeted search. A subgoal the judge never reported is missing.
    """
    report = CoverageReport(supported_confidence_floor=min_supported_confidence)
    by_id = {j.subgoal_id: j for j in judgments}
    for item in items:
        judgment = by_id.get(item["id"])
        if judgment is None or judgment.status == "unsupported":
            report.missing.append(item["id"])
        elif judgment.status == "contradicted":
            report.contradicted.append(item["id"])
        elif judgment.status == "supported" and judgment.confidence >= min_supported_confidence:
            report.complete.append(item["id"])
        else:
            report.partial.append(item["id"])
    return report


class QuoteSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str
    quote: str = Field(min_length=1)


class _SlotJudgeRow(_JudgeRow):
    quote_spans: list[QuoteSpan]
    confidence: float = Field(ge=0.0, le=1.0)
    evaluation_validity: Literal["evaluated", "unknown"] = "evaluated"


class _SlotJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    judgments: list[_SlotJudgeRow]


SLOT_JUDGE_SYSTEM = JUDGE_SYSTEM + (
    "评估单位是输入的 slot，必须满足它的主体、属性、单位、版本及适用期间。"
    "每个 supported/partial/contradicted 判断必须在 quote_spans 引用证据中的连续原文，"
    "ref 与 support_refs 对应。不得只凭标题或同主题判定支持。"
    "purpose=claim 时逐条检查断言是否被它自己的引用蕴含；明确反证为 contradicted，"
    "无反证但无支持为 unsupported。不要用其他断言的引用补足。"
    "历史差异不等于同范围冲突。模型 confidence 仅记录，不用于完成判定。"
)


class SlotEvidenceEvaluator(SemanticEvidenceEvaluator):
    """V2 development candidate. Authorization precedes scoring, calls and cache hits.

    It shares the legacy scorer, judge transport and accounting. No threshold in this
    class is claimed calibrated: activation requires an independent review gate.
    Cache lifetime is one evaluator (one task), never a cross-user global cache.
    """

    version = "semantic-slot-v2"
    output_type = _SlotJudgeOutput
    judge_system = SLOT_JUDGE_SYSTEM

    def __init__(self, *args, authorize, model_fingerprint="development-unfrozen", **kwargs):
        super().__init__(*args, **kwargs)
        self.authorize = authorize
        self.model_fingerprint = model_fingerprint
        self._cache = {}
        self._relevance_cache = {}
        self.cache_hits = 0
        self.last_context = {}
        self.last_raw_judgments = []
        self.context_error = None

    def _authorize(self, passages):
        # Callers must check the original chunk, version, user and content, rather
        # than approve an arbitrary supplied ID. Offline frozen inputs use a hash
        # manifest check instead of pretending the database ACL was consulted.
        for passage in passages:
            if self.authorize(passage) is not True:
                raise PermissionError("evidence authorization failed")

    def relevance(self, items, passages):
        self._authorize(passages)
        from scripts.research_review import digest
        key = digest([items, passages, self.thresholds.__dict__, self.version, self.model_fingerprint])
        if key not in self._relevance_cache:
            rows = [r.model_dump() for r in super().relevance(items, passages)]
            self._authorize(passages)
            self._relevance_cache[key] = rows
        return [RelevanceJudgment.model_validate(r) for r in self._relevance_cache[key]]

    def _payload(self, question, items, labelled):
        return {
            "question": question,
            "subgoals": [dict(item, purpose=item.get("purpose", "coverage")) for item in items],
            "evidence": [
                {"id": key, "source": p.get("header") or p.get("title", ""),
                 "text": p["text"], "version_id": p.get("version_id", ""),
                 "scope": p.get("scope", {}), "effective_interval": p.get("effective_interval")}
                for key, p in labelled.items()
            ],
        }

    def _judge_batch(self, question, items, labelled):
        # Byte-level conservative upper bound for the fixed 8192-token Qwen
        # context, including completion/template headroom. Never rely on the
        # server silently truncating evidence while we validate against full text.
        self.context_error = None
        try:
            size = len((self.judge_system + json.dumps(
                self._payload(question, items, labelled), ensure_ascii=False)).encode()) + 800
        except (TypeError, ValueError):
            self.context_error = "context_unserializable"
            return []
        if size > 8192:
            self.context_error = "context_budget_exceeded"
            self.usage.skipped_by_cascade += 1
            return []
        return super()._judge_batch(question, items, labelled)

    def coverage(self, question, items, passages, relevance=None):
        import hashlib
        from collections import Counter

        self._authorize(passages)
        self.last_context = {}
        self.last_raw_judgments = []
        self.context_error = None
        if len({p["chunk_id"] for p in passages}) != len(passages) or len({i["id"] for i in items}) != len(items):
            return [self._unknown(item, "duplicate_input_id") for item in items]
        key = hashlib.sha256(json.dumps({
            "question": question, "items": items, "passages": passages,
            "thresholds": self.thresholds.__dict__, "evaluator": self.version,
            "model": self.model_fingerprint, "max_passages": self.max_passages,
            "relevance": None if relevance is None else [r.model_dump() for r in relevance],
        }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            cached = self._cache[key]
            self.last_context = cached["context"]
            self.last_raw_judgments = cached["raw_judgments"]
            return [EvidenceJudgment.model_validate(row) for row in cached["judgments"]]
        literal = {item["id"]: judgment for item in items
                   if (judgment := self._literal_short_circuit(item, passages)) is not None}
        if len(literal) == len(items):
            output = [literal[item["id"]] for item in items]
            self.usage.skipped_by_cascade += 1
            self._remember(key, output)
            return output
        try:
            relevance = relevance if relevance is not None else self.relevance(items, passages)
        except (BudgetExceeded, TaskCancelled, PermissionError):
            raise
        except Exception:
            return [self._unknown(item, "relevance_unavailable") for item in items]
        by_id = {p["chunk_id"]: p for p in passages}
        ranked = {item["id"]: sorted(
            (r for r in relevance if r.subgoal_id == item["id"] and r.label != "irrelevant"
             and r.chunk_id in by_id and ("allowed_refs" not in item or r.chunk_id in item["allowed_refs"])),
            key=lambda r: (-r.score, r.chunk_id)) if item["id"] not in literal else [] for item in items}
        # Round robin gives every slot a candidate before any gets a second.
        chosen = []
        for depth in range(max((len(rows) for rows in ranked.values()), default=0)):
            for item in items:
                rows = ranked[item["id"]]
                if depth < len(rows) and rows[depth].chunk_id not in chosen:
                    chosen.append(rows[depth].chunk_id)
                    if len(chosen) == self.max_passages:
                        break
            if len(chosen) >= self.max_passages:
                break
        labelled = {f"E{n}": by_id[cid] for n, cid in enumerate(chosen, 1)}
        self.last_context = labelled
        pending = [item for item in items if ranked[item["id"]]]
        if pending:
            self._authorize(list(labelled.values()))
            rows = self._judge_batch(question, pending, labelled)
        else:
            self.usage.skipped_by_cascade += 1
            rows = []
        self.last_raw_judgments = [row.model_dump() for row in rows]
        counts = Counter(r.subgoal_id for r in rows)
        rows_by_id = {r.subgoal_id: r for r in rows}
        output = []
        for item in items:
            sid = item["id"]
            if sid in literal:
                output.append(literal[sid])
                continue
            shown = {r.chunk_id: r.score for r in ranked[sid] if r.chunk_id in chosen}
            if not ranked[sid]:
                # Until relevance recall is calibrated, ranking out is no proof
                # that evidence is absent. This is an unknown, not a fact verdict.
                output.append(self._unknown(item, "relevance_cut_unvalidated"))
                continue
            if not shown:
                output.append(self._unknown(item, "context_omitted"))
                continue
            row = rows_by_id.get(sid)
            if row is None or counts[sid] != 1:
                output.append(self._unknown(item, self.context_error or "judge_unavailable" if row is None else "duplicate_slot"))
                continue
            if row.evaluation_validity == "unknown":
                output.append(self._unknown(item, "judge_ambiguous"))
                continue
            refs = row.support_refs
            quotes = row.quote_spans
            invalid = len(refs) != len(set(refs)) or any(ref not in labelled for ref in refs)
            invalid |= any(q.ref not in refs or q.ref not in labelled
                           or q.quote not in labelled[q.ref]["text"] for q in quotes)
            if row.status in {"supported", "partial", "contradicted"}:
                invalid |= not refs or set(refs) != {q.ref for q in quotes}
            chunk_refs = [labelled[ref]["chunk_id"] for ref in refs if ref in labelled]
            invalid |= any(cid not in shown for cid in chunk_refs)
            if invalid:
                output.append(self._unknown(item, "citation_or_quote_invalid"))
                continue
            selected = [labelled[ref] for ref in refs]
            scope_invalid = any(
                item.get(field) and p.get(field) != item[field]
                for p in selected for field in ("version_id", "effective_interval")
            ) or any(item.get("subject") and p.get("scope", {}).get("subject")
                     and p["scope"]["subject"] != item["subject"] for p in selected)
            if scope_invalid:
                output.append(self._unknown(item, "scope_mismatch"))
                continue
            if row.status == "contradicted" and item.get("purpose") != "claim":
                # Require explicit matching scope metadata for conflicts. Versions
                # with absent metadata cannot be asserted to contradict each other.
                scopes = [p.get("scope") for p in selected]
                if len(selected) < 2 or not all(scopes) or any(scope != scopes[0] for scope in scopes[1:]):
                    output.append(self._unknown(item, "conflict_scope_unproven"))
                    continue
            status, reason = row.status, row.reason_code[:40]
            if status == "supported" and (row.missing_slots or not chunk_refs):
                status, reason = "partial", "missing_required_slots"
            if status == "supported" and not all(shown[cid] >= self.thresholds.high for cid in chunk_refs):
                status, reason = "partial", "quoted_support_low_relevance"
            output.append(EvidenceJudgment(
                subgoal_id=sid, status=status, support_refs=chunk_refs,
                quote_spans=[{"chunk_id": labelled[q.ref]["chunk_id"], "quote": q.quote} for q in quotes],
                missing_slots=row.missing_slots, reason_code=reason, confidence=row.confidence))
        # A revocation while the scorer/judge was running invalidates its result.
        # Never use or cache a verdict based on evidence no longer authorized.
        self._authorize(passages)
        if all(j.evaluation_validity == "evaluated" for j in output):
            self._remember(key, output)
        return output

    def _remember(self, key, judgments):
        self._cache[key] = {"judgments": [j.model_dump() for j in judgments],
                            "context": self.last_context, "raw_judgments": self.last_raw_judgments}

    @staticmethod
    def _literal_short_circuit(item, passages):
        """Reuse P1 bindings only for an entire, explicit single-attribute request.

        No substring matching of compound obligations; claims and prose always go
        through the judge. This optimization is reported separately in calibration.
        """
        import re
        from app.task_contract import SlotSpec, bind_values
        if item.get("purpose") == "claim":
            return None
        match = re.fullmatch(
            r"(?:(?P<subject>[^，,。？?；;]{1,60})的)?(?P<attribute>RTO|RPO|重试次数|超时|保留时间|首次响应时限|状态|等级|生效日期)"
            r"(?:是多少|是什么|为多少|多少)[？?。]?", item["text"].strip(), re.I)
        if not match or re.search(r"和|与|如果|只有|各版本|最早|截至|历史|至少|最多|除外|不适用", match["subject"] or ""):
            return None
        if item.get("subject") and item["subject"] != (match["subject"] or ""):
            return None
        attribute = match["attribute"]
        slot = SlotSpec(slot_id=item["id"], query=item["text"], subject=match["subject"] or "",
                        attribute=attribute, value_type="enum" if attribute in {"状态", "等级"}
                        else "date" if attribute == "生效日期" else "number")
        selected = [p for p in passages if all(not item.get(field) or p.get(field) == item[field]
                    for field in ("version_id", "effective_interval"))
                    and ("allowed_refs" not in item or p["chunk_id"] in item["allowed_refs"])
                    and (not item.get("subject") or not p.get("scope", {}).get("subject")
                         or p["scope"]["subject"] == item["subject"])]
        values = bind_values(slot, selected)
        identities = {(v.value, v.unit, v.version_id) for v in values}
        if len(identities) != 1:
            return None
        value = values[0]
        return EvidenceJudgment(subgoal_id=item["id"], status="supported", source="literal_binding",
            support_refs=[value.chunk_id], quote_spans=[{"chunk_id": value.chunk_id, "quote": value.quote}],
            reason_code="unique_literal_attribute")

    @staticmethod
    def _unknown(item, reason):
        return EvidenceJudgment(subgoal_id=item["id"], status="unsupported",
                                evaluation_validity="unknown", source="judge_unavailable", reason_code=reason)

    @staticmethod
    def aggregate(items, judgments):
        return aggregate_slots(items, judgments)


def aggregate_slots(items, judgments):
    """One completion rule for v2 controller, diagnostics and gates. Ignore confidence."""
    buckets = {key: [] for key in ("complete", "partial", "missing", "contradicted", "unknown", "inactive")}
    groups = {}
    for judgment in judgments:
        groups.setdefault(judgment.subgoal_id, []).append(judgment)
    for item in items:
        sid = item["id"]
        if not item.get("required", True) or item.get("active") is False:
            buckets["inactive"].append(sid)
            continue
        rows = groups.get(sid, [])
        if len(rows) != 1 or rows[0].evaluation_validity != "evaluated":
            buckets["unknown"].append(sid)
            continue
        row = rows[0]
        target = {"supported": "complete", "unsupported": "missing"}.get(row.status, row.status)
        buckets[target].append(sid)
    buckets["ready"] = not any(buckets[key] for key in ("partial", "missing", "contradicted", "unknown"))
    return buckets
