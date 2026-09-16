"""One structured line per request, with nothing in it that a reader is not entitled to (S7).

Operating this system means answering "what did that request cost and where did the
time go", and answering it without a log that quietly becomes a second, unguarded
copy of the corpus. So the record carries identifiers, timings and counts — never
document text, never a quote, never a question, never a credential.

The question text is excluded deliberately: questions routinely contain the very
values a user was not supposed to learn, and logs outlive access grants.
"""

import json
import logging
import time
import uuid

logger = logging.getLogger("enterprise_rag.request")

# Only these may ever be copied out of an answer payload into a log line.
TRACE_FIELDS = (
    "method",
    "prompt_version",
    "embed_ms",
    "retrieval_ms",
    "generation_ms",
    "total_ms",
    "context_tokens_estimate",
    "query_rewritten",
)
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "model_duration_ms", "api_cost", "currency")


def request_id():
    return uuid.uuid4().hex[:16]


def answer_record(payload):
    """Numbers and identifiers from one answer. No text of any kind."""
    trace = payload.get("trace") or {}
    usage = payload.get("usage") or {}
    record = {key: trace[key] for key in TRACE_FIELDS if key in trace}
    record |= {key: usage[key] for key in USAGE_FIELDS if key in usage}
    record["status"] = payload.get("status")
    record["claims"] = len(payload.get("claims") or [])
    record["citations"] = len(payload.get("citations") or [])
    record["candidates"] = len(trace.get("candidates") or [])
    # Identifiers are safe to log: they are meaningless without an authorised read.
    record["evidence_chunk_ids"] = [c.get("chunk_id") for c in (payload.get("citations") or [])]
    return record


def log_request(**fields):
    logger.info(json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str))


class Timer:
    def __enter__(self):
        self.started = time.monotonic()
        return self

    def __exit__(self, *_):
        self.ms = round((time.monotonic() - self.started) * 1000, 1)
