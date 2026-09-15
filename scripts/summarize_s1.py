"""Summarize development smoke checks; this is not a frozen quality benchmark."""

import json
import math
from pathlib import Path
from statistics import median

from app.db import SessionLocal
from app.models import User
from app.security import require_chunk

questions = json.loads(Path("fixtures/questions.json").read_text())["questions"]
rows, latencies = [], []
with SessionLocal() as db:
    for question in questions:
        data = json.loads(Path(f"artifacts/s1-questions/{question['id']}.json").read_text())
        result, checks = data["response"], data["checks"]
        assert result["trace"]["prompt_version"] == "grounded-v4-source-spans", (
            "Mixed prompt versions; rerun older question artifacts"
        )
        user = db.get(User, question["user"])
        for candidate in result["trace"]["candidates"]:
            require_chunk(db, user, candidate["chunk_id"], active_only=True)
        for citation in result["citations"]:
            chunk, version, doc = require_chunk(db, user, citation["chunk_id"])
            assert chunk.text == citation["text"] and version.id == citation["version_id"]
        text = "\n".join(c["text"] for c in result["claims"]) + result["message"]
        if question["expected"] == "conflict":
            checks["explicit_conflict"] = any(
                word in text for word in ["冲突", "不一致", "无法确定", "两份"]
            )
        latencies.append(result["trace"]["total_ms"] / 1000)
        rows.append(
            {
                "id": question["id"],
                "status": result["status"],
                "passed": all(checks.values()),
                "checks": checks,
                "seconds": round(latencies[-1], 2),
            }
        )

result = {
    "run_id": "s1-complete-7b-source-spans-v4",
    "data": "s0-dev-v1",
    "scope": "20 authored development questions; keyword/source/status checks, not held-out quality or human entailment scoring",
    "questions": len(rows),
    "passed": sum(row["passed"] for row in rows),
    "candidate_acl_and_citation_identity_checks": "passed",
    "latency_seconds": {
        "p50": round(median(latencies), 2),
        "p95_nearest_rank": round(sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1], 2),
        "min": round(min(latencies), 2),
        "max": round(max(latencies), 2),
    },
    "model_api_cost_aud": 0,
    "rows": rows,
}
Path("artifacts/s1-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        {key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2
    )
)
