"""How much of the remaining MultiHop-RAG error is the generator's?

Generators × evidence on the used batches 1 and 2 (development data; batch 3 is never
read):

    evidence   system       what the current retrieval admits for the question
               rerank       the same, with each lane reordered by the cross-encoder
               mixed        oracle passages plus the reranked retrieval around them (max 10)
               compact      the reranked retrieval capped at four passages
               oracle       the chunks that contain the gold facts, and nothing else
    generator  local        qwen2.5:7b-instruct, the production model (frozen baseline)
               qwen35       qwen3.5:9b, thinking off — a current local model of the same size
               qwen35think  qwen3.5:9b, thinking on — yes/no questions only
               claude       a strong hosted model, the ceiling

qwen3.5 needs a newer Ollama than the pinned production runtime, so it is served by
`scripts/setup_ollama_next.py` on its own port; the baseline runtime is untouched.

Only the generator is swapped. Retrieval, the prompt, the answer schema, citation
validation, the conflict gate, the claims-only verdict step and the scoring rule are the
production code paths, called unchanged. So:

    local/system  → claude/system   generator headroom with the evidence we actually get
    local/system  → local/oracle    retrieval headroom for the production model
    claude/oracle                   the ceiling of this pipeline on these questions

Null questions (no gold) run only with system evidence: their correct answer is to
refuse, and there is no oracle chunk to give.

Results are appended to a JSONL file per arm, keyed by question, so an interrupted run
resumes. The hosted arms need `ANTHROPIC_API_KEY` in `.env` and the `anthropic` package
(`uv pip install anthropic`); the key is read from `.env`, never logged or written out.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time

from typing import Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.clients import DependencyError, Models, Search
from app.config import settings
from app.db import SessionLocal
from app.models import Chunk, Document, User
from app.qa import answer_verdict, validate_claims
from app.retrieval import retrieve_authorized
from app.security import require_chunk
from app.task_analysis import multi_source_intent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multihop_retrieval_eval import fact_delivered  # noqa: E402
from run_multihop_eval import answer_matches, normalized  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".runtime/multihop"
SUBSETS = {"b1": ROOT / "fixtures/multihop/subset.json", "b2": ROOT / "fixtures/multihop/subset-r2.json"}
OUT = ROOT / "artifacts/generation-ceiling"
USER = "mh-eval"
_UNSUPPORTED = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "minItems", "maxItems", "pattern",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def document_id(title: str) -> str:
    return "mh-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:24]


def hosted_schema(schema):
    """Drop the constraints structured outputs do not accept. They are still enforced:
    the production Pydantic models validate every response afterwards, exactly as they
    do for the local model."""
    if isinstance(schema, dict):
        cleaned = {k: hosted_schema(v) for k, v in schema.items() if k not in _UNSUPPORTED}
        if cleaned.get("type") == "object":
            cleaned["additionalProperties"] = False
            cleaned["required"] = list(cleaned.get("properties", {}))
        return cleaned
    if isinstance(schema, list):
        return [hosted_schema(item) for item in schema]
    return schema


def infrastructure_failure(exc) -> bool:
    """A timeout or connection failure, as opposed to an answer the system rejected."""
    import httpx

    cause = exc.__cause__
    return isinstance(cause, (httpx.TimeoutException, httpx.TransportError)) or any(
        marker in str(exc) for marker in ("unreachable", "unavailable", "error 5", "error 429")
    )


class ClaudeChat:
    """Answers an Ollama /api/chat body with the Claude API, in Ollama's response shape."""

    def __init__(self, model: str):
        import anthropic

        key = next(
            (
                line.split("=", 1)[1].strip().strip("\"'")
                for line in (ROOT / ".env").read_text().splitlines()
                if line.startswith("ANTHROPIC_API_KEY=")
            ),
            "",
        )
        if not key:
            raise SystemExit("hosted arms need ANTHROPIC_API_KEY=... in .env")
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com", max_retries=6)
        self.model = model

    def __call__(self, body: dict) -> dict:
        system = "\n".join(m["content"] for m in body["messages"] if m["role"] == "system")
        user = [{"role": m["role"], "content": m["content"]} for m in body["messages"] if m["role"] == "user"]
        started = time.monotonic()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=user,
                output_config={"format": {"type": "json_schema", "schema": hosted_schema(body["format"])}},
            )
        except self.anthropic.APIStatusError as exc:
            raise DependencyError(f"hosted model error {exc.status_code}", stage="generation") from exc
        except self.anthropic.APIConnectionError as exc:
            raise DependencyError("hosted model unreachable", stage="generation") from exc
        if response.stop_reason in {"refusal", "max_tokens"}:
            raise DependencyError(f"hosted model stopped: {response.stop_reason}", stage="generation")
        text = next((block.text for block in response.content if block.type == "text"), "")
        return {
            "message": {"content": text},
            "prompt_eval_count": response.usage.input_tokens,
            "eval_count": response.usage.output_tokens,
            "total_duration": int((time.monotonic() - started) * 1e9),
        }


class OllamaChat:
    """Sends the unchanged request to a different local model on the second runtime."""

    def __init__(self, model: str, think: bool, url: str = "http://127.0.0.1:11437"):
        import httpx

        self.httpx, self.model, self.think, self.url = httpx, model, think, url

    def __call__(self, body: dict) -> dict:
        request = {**body, "model": self.model, "think": self.think}
        if self.think:
            # Reasoning tokens count against num_predict; the answer must still fit.
            options = body.get("options", {})
            request["options"] = {
                **options,
                "num_predict": 8192,
                "num_ctx": max(options.get("num_ctx", 0), 16384),
            }
        try:
            response = self.httpx.post(
                self.url + "/api/chat", json=request, timeout=900, trust_env=False
            )
            response.raise_for_status()
            return response.json()
        except (self.httpx.HTTPError, ValueError) as exc:
            raise DependencyError(f"{self.model} unavailable", stage="generation") from exc


def backend(generator: str, claude_model: str):
    if generator == "claude":
        return Models(ClaudeChat(claude_model))
    if generator == "qwen35":
        return Models(OllamaChat("qwen3.5:9b", think=False))
    if generator == "qwen35think":
        return Models(OllamaChat("qwen3.5:9b", think=True))
    return Models()  # local and localjudge: the production model


def system_evidence(db, user, question, models, rerank=False, top_k=None):
    cfg = settings().model_copy(update={"passage_rerank": rerank})
    top_k = top_k or (8 if multi_source_intent(question) else 6 if len(question) <= 30 else cfg.top_k)
    found = retrieve_authorized(db, user, question, cfg=cfg, models=models, search=Search(), top_k=top_k)
    return found.evidence


def oracle_evidence(db, user, record):
    """The active chunks that contain each gold fact, in evidence-list order."""
    evidence, seen = [], set()
    for row in record["evidence_list"]:
        document = db.get(Document, document_id(row["title"]))
        chunks = db.scalars(
            select(Chunk).where(Chunk.version_id == document.active_version_id).order_by(Chunk.ordinal)
        ).all()
        for chunk in chunks:
            if chunk.id in seen or not fact_delivered(row["fact"], [chunk.text]):
                continue
            chunk, version, document = require_chunk(db, user, chunk.id, active_only=True)
            seen.add(chunk.id)
            evidence.append(
                {
                    "id": f"E{len(evidence) + 1}",
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "version_id": version.id,
                    "title": document.title,
                    "text": chunk.text,
                    "locator": chunk.locator,
                    "metadata": document.metadata_json,
                }
            )
    return evidence


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[dict]
    verdict: Literal["yes", "no", "unclear"]


def judge_direct(models, question, evidence):
    """Protocol variant for yes/no questions: one call that reads the evidence itself.

    The production path writes claims first and lets a claims-only step read a verdict
    out of them, so a claim that restates the question ("there was inconsistency…")
    becomes a yes. Here the model states, per source, what that source actually says,
    and only then decides — with the premise of the question explicitly not assumed.
    """
    schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}, "says": {"type": "string"}},
                    "required": ["source", "says"],
                },
            },
            "verdict": {"type": "string", "enum": ["yes", "no", "unclear"]},
        },
        "required": ["findings", "verdict"],
    }
    context = [
        {
            "source": f"{item['title']} ({(item.get('metadata') or {}).get('source', '')}, "
            f"{str((item.get('metadata') or {}).get('published_at', ''))[:10]})",
            "text": item["text"],
        }
        for item in evidence
    ]
    result = models._chat(
        {
            "model": settings().chat_model,
            "stream": False,
            "keep_alive": "30m",
            "format": schema,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你要回答一个是非问题，只能依据给出的原文。不要默认问题里的说法成立：问题中的描述"
                        "可能是真的，也可能是被故意说反的。先逐个来源写出它的原文实际说了什么（findings），"
                        "再比较这些内容，判断问题的命题是否成立：成立 yes，不成立 no，"
                        "原文确实无法判断时 unclear。只输出指定 JSON。"
                    ),
                },
                {"role": "user", "content": json.dumps({"question": question, "evidence": context}, ensure_ascii=False)},
            ],
            "options": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 700},
        }
    )
    try:
        judged = Judgment.model_validate_json(result["message"]["content"])
    except (ValueError, KeyError, TypeError) as exc:
        raise DependencyError("模型未返回有效的判断", stage="generation") from exc
    findings = [str(row.get("says", "")) for row in judged.findings if row.get("says")]
    return {
        "status": "answered" if judged.verdict != "unclear" else "insufficient_evidence",
        "claims": [{"text": text} for text in findings] or [{"text": judged.verdict}],
        "verdict": {"value": judged.verdict},
    }, {
        "prompt_tokens": result.get("prompt_eval_count", 0),
        "completion_tokens": result.get("eval_count", 0),
    }


def answer(models, question, evidence):
    """The production single-turn answer path from generation onward."""
    if not evidence:
        return {"status": "insufficient_evidence", "claims": [], "verdict": None}, {}
    generated, usage = models.generate(question, evidence)
    claims, status = validate_claims(generated, evidence)
    if status == "answered" and usage.get("answer_status") == "conflict":
        status = "conflict"
    verdict, verdict_usage = answer_verdict(models, question, claims, status)
    usage = dict(usage)
    usage["verdict_prompt_tokens"] = verdict_usage.get("prompt_tokens", 0)
    usage["verdict_completion_tokens"] = verdict_usage.get("completion_tokens", 0)
    return {"status": status, "claims": claims, "verdict": verdict}, usage


def score(item, record, payload):
    claims_text = normalized(" ".join(claim["text"] for claim in payload["claims"]))
    gold = normalized(record["answer"])
    return {
        "answer_correct": answer_matches(
            item["question_type"], record["answer"], payload["status"], claims_text, payload["verdict"], "v2"
        ),
        "gold_binary": gold if gold in {"yes", "no"} else None,
        "verdict": (payload["verdict"] or {}).get("value"),
    }


def run_arm(arm, generator, evidence_mode, items, records, models, lock, binary_only=False):
    path = OUT / f"{arm}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}
    def wanted(item):
        if evidence_mode == "oracle" and item["question_type"] == "null_query":
            return False
        if generator.endswith(("think", "judge")) or binary_only:
            return normalized(records[item["query_sha256"]]["answer"]) in {"yes", "no"}
        return True

    todo = [item for item in items if item["id"] not in done and wanted(item)]
    print(f"[{arm}] {len(done)} done, {len(todo)} to run", flush=True)
    with SessionLocal() as db:
        user = db.get(User, USER)
        for index, item in enumerate(todo, 1):
            record = records[item["query_sha256"]]
            embedder = Models()
            if evidence_mode == "oracle":
                evidence = oracle_evidence(db, user, record)
            elif evidence_mode == "mixed":
                # Gold passages plus what retrieval brings in besides them: separates
                # "the right passage is missing" from "too much else is there".
                gold = oracle_evidence(db, user, record)
                retrieved = system_evidence(db, user, record["query"], embedder, rerank=True)
                gold_ids = {item["chunk_id"] for item in gold}
                others = [item for item in retrieved if item["chunk_id"] not in gold_ids]
                others = others[: max(0, 10 - len(gold))]  # trim distractors, never gold
                kept = {item["chunk_id"] for item in others} | gold_ids
                # Retrieval order; gold passages retrieval missed go last, the position
                # least favourable to them.
                ordered = [item for item in retrieved if item["chunk_id"] in kept]
                ordered += [item for item in gold if item["chunk_id"] not in {r["chunk_id"] for r in retrieved}]
                evidence = [dict(item, id=f"E{n}") for n, item in enumerate(ordered, 1)]
            elif evidence_mode == "compact":
                # Fewer, better passages: the reranked retrieval capped at four.
                evidence = system_evidence(db, user, record["query"], embedder, rerank=True, top_k=4)
            else:
                evidence = system_evidence(
                    db, user, record["query"], embedder, rerank=evidence_mode == "rerank"
                )
            started = time.monotonic()
            for attempt in range(3):
                try:
                    run = judge_direct if generator.endswith("judge") else answer
                    payload, usage = run(models, record["query"], copy.deepcopy(evidence))
                    error = None
                    break
                except DependencyError as exc:
                    if not infrastructure_failure(exc):
                        # The model answered, but not in a form the system accepts: that
                        # is the system failing the question, and it is scored as such.
                        payload, usage, error = (
                            {"status": "execution_failed", "claims": [], "verdict": None}, {}, str(exc)
                        )
                        break
                    time.sleep(30 * (attempt + 1))
            else:
                # A timeout or a refused connection says nothing about the question.
                # It is not recorded, so the next run retries it.
                print(f"[{arm}] {item['id']} skipped after infrastructure failures", flush=True)
                continue
            row = {
                "id": item["id"],
                "batch": item["batch"],
                "question_type": item["question_type"],
                "arm": arm,
                "generator": generator,
                "evidence": evidence_mode,
                "evidence_chunks": len(evidence),
                "status": payload["status"],
                "claim_count": len(payload["claims"]),
                # Kept locally for failure review; the directory is git-ignored because
                # claims paraphrase third-party articles.
                "claims": [claim["text"] for claim in payload["claims"]],
                "evidence_titles": [item["title"] for item in evidence],
                "error": error,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "prompt_tokens": usage.get("prompt_tokens", 0) + usage.get("verdict_prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0)
                + usage.get("verdict_completion_tokens", 0),
                **score(item, record, payload),
            }
            with lock:
                with path.open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                print(f"[{arm}] {index}/{len(todo)}", flush=True)


def summarize(rows):
    def rate(values):
        values = list(values)
        return round(sum(values) / len(values), 3) if values else None

    answerable = [r for r in rows if r["question_type"] != "null_query"]
    null = [r for r in rows if r["question_type"] == "null_query"]
    binary = [r for r in rows if r["gold_binary"]]
    no_gold = [r for r in binary if r["gold_binary"] == "no"]
    return {
        "questions": len(rows),
        "answer_correct": sum(r["answer_correct"] for r in rows),
        "answer_correct_rate": rate(r["answer_correct"] for r in rows),
        "answerable_accuracy": rate(r["answer_correct"] for r in answerable),
        "answerable_refusal_rate": rate(r["status"] == "insufficient_evidence" for r in answerable),
        "null_false_answer_rate": rate(r["status"] != "insufficient_evidence" for r in null),
        "binary_questions": len(binary),
        "binary_accuracy": rate(r["answer_correct"] for r in binary),
        "binary_always_yes_baseline": rate(r["gold_binary"] == "yes" for r in binary),
        "binary_gold_no_accuracy": rate(r["answer_correct"] for r in no_gold),
        "binary_verdict_yes_share": rate(r["verdict"] == "yes" for r in binary if r["verdict"] in {"yes", "no"}),
        "execution_failed": sum(r["status"] == "execution_failed" for r in rows),
        "median_latency_ms": round(statistics.median(r["latency_ms"] for r in rows), 1) if rows else None,
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rows), 1) if rows else None,
        "by_type": {
            kind: {
                "questions": len(group),
                "answer_correct": sum(r["answer_correct"] for r in group),
                "rate": rate(r["answer_correct"] for r in group),
            }
            for kind in sorted({r["question_type"] for r in rows})
            if (group := [r for r in rows if r["question_type"] == kind])
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arms",
        default=(
            "local-system,local-oracle,qwen35-system,qwen35-oracle,"
            "qwen35think-oracle,claude-system,claude-oracle"
        ),
    )
    parser.add_argument("--claude-model", default="claude-opus-5")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--per-type",
        type=int,
        help="first N answerable questions of each type per batch, in the frozen hash order",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--binary-only", action="store_true", help="yes/no questions only")
    parser.add_argument("--nulls", type=int, default=0, help="with --per-type: N null questions per batch")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = {digest(row["query"]): row for row in json.loads((CACHE / "MultiHopRAG.json").read_bytes())}
    items = [
        {**item, "batch": batch}
        for batch, path in SUBSETS.items()
        for item in json.loads(path.read_text())["items"][: args.limit]
    ]
    if args.per_type:
        taken = {}
        sample = []
        for item in items:
            key = (item["batch"], item["question_type"])
            quota = args.nulls if item["question_type"] == "null_query" else args.per_type
            if taken.get(key, 0) < quota:
                taken[key] = taken.get(key, 0) + 1
                sample.append(item)
        items = sample
    arms = args.arms.split(",")
    if not args.summary_only:
        lock = threading.Lock()
        # Arms that share a model server run one after another: queued behind each
        # other, their requests would spend the timeout waiting, not generating.
        servers = {
            "local": "11436", "localjudge": "11436", "qwen35": "11437", "qwen35think": "11437", "claude": "api"
        }
        by_server = {}
        for arm in arms:
            by_server.setdefault(servers[arm.split("-")[0]], []).append(arm)

        def run_server(server_arms):
            for arm in server_arms:
                generator, evidence_mode = arm.split("-")
                models = backend(generator, args.claude_model)
                run_arm(arm, generator, evidence_mode, items, records, models, lock, args.binary_only)

        with ThreadPoolExecutor(max_workers=len(by_server)) as pool:
            for job in [pool.submit(run_server, group) for group in by_server.values()]:
                job.result()
    summary = {}
    for arm in arms:
        path = OUT / f"{arm}.jsonl"
        if path.exists():
            wanted_ids = {item["id"] for item in items}
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            rows = [row for row in rows if row["id"] in wanted_ids]
            summary[arm] = summarize(rows)
            for batch in SUBSETS:
                summary[arm][f"{batch}_answer_correct"] = sum(
                    r["answer_correct"] for r in rows if r["batch"] == batch
                )
    output = {
        "design": (
            "generator (qwen2.5:7b baseline, qwen3.5:9b think off/on, hosted Claude) x "
            "evidence (system retrieval vs gold-fact oracle); think-on runs yes/no questions only"
        ),
        "batches": list(SUBSETS),
        "claude_model": args.claude_model,
        "local_model": settings().chat_model,
        "scoring": "run_multihop_eval.answer_matches, rule v2",
        "note": "Batches 1 and 2 are development data after their one-shot runs; batch 3 is not read.",
        "arms": summary,
    }
    (OUT / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
