"""Local paired generation check for attribution, absence scope and comparison targets."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.answer_contract import source_audit
from app.clients import DependencyError, Models
from app.config import settings
from app.qa import validate_claims


def score(item, claims, status):
    text = " ".join(claim["text"] for claim in claims)
    if item["category"] == "absence":
        bounded = bool(
            re.search(
                r"provided excerpts?|retrieved passages?|available excerpts?|当前片段|现有片段|现有摘录|所给片段",
                text,
                re.I,
            )
        )
        return {
            "criterion_met": status in {"insufficient_evidence", "verification_failed"} or bounded,
            "scope_guard_refusal": status == "verification_failed",
        }
    facts = all(re.search(pattern, text, re.I) for pattern in item["must_match"])
    refs = {ref for claim in claims for ref in claim["evidence_ids"]}
    all_sources = {row["chunk_id"] for row in item["evidence"]} <= refs
    contradictory = item["category"] == "comparison" and bool(
        re.search(r"\binconsistent\b|不一致|不相同", text, re.I)
    )
    return {
        "criterion_met": bool(
            status in {"answered", "conflict"} and facts and all_sources and not contradictory
        ),
        "scope_guard_refusal": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=Path("fixtures/source_contract/development-v1.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/source-contract-development-v1.json")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raw = args.fixture.read_bytes()
    fixture = json.loads(raw)
    cfg = settings()
    metadata = {
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "model": cfg.chat_model,
        "scope": fixture["scope"],
        "api_cost": 0,
        "check_conflict": False,
        "num_predict": 320,
        "code_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ["app/clients.py", "app/answer_contract.py", "app/qa.py"]
        },
    }
    result = {"metadata": metadata, "records": []}
    if args.out.exists():
        if not args.resume:
            raise SystemExit("Refuse to overwrite; use --resume or a new artifact")
        result = json.loads(args.out.read_text())
        if result["metadata"] != metadata:
            raise SystemExit("Code/configuration changed; use a new artifact")
    done = {(r["id"], r["contract_enabled"]) for r in result["records"]}
    models = Models()
    original = cfg.answer_contract_enabled
    try:
        for number, item in enumerate(fixture["items"]):
            for enabled in [False, True] if number % 2 == 0 else [True, False]:
                if (item["id"], enabled) in done:
                    continue
                cfg.answer_contract_enabled = enabled
                started = time.monotonic()
                try:
                    generated, usage = models.generate(
                        item["question"], item["evidence"], check_conflict=False, max_output_tokens=320
                    )
                    claims, status = validate_claims(generated, item["evidence"])
                    draft_audit = [
                        {"claim_index": index, "issues": source_audit(claim.text, claim.quotes)}
                        for index, claim in enumerate(generated.claims, 1)
                    ]
                    error = None
                except DependencyError as exc:
                    claims, status, usage, draft_audit, error = [], "execution_failed", {}, [], exc.stage
                row = {
                    "id": item["id"],
                    "category": item["category"],
                    "contract_enabled": enabled,
                    "claims": claims,
                    "status": status,
                    "usage": usage,
                    "draft_audit": draft_audit,
                    "error": error,
                    "latency_ms": round((time.monotonic() - started) * 1000, 1),
                    **score(item, claims, status),
                }
                result["records"].append(row)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.out.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                temporary.replace(args.out)
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in [
                                "id",
                                "contract_enabled",
                                "criterion_met",
                                "status",
                                "latency_ms",
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        result["summary"] = {}
        for enabled in [False, True]:
            rows = [row for row in result["records"] if row["contract_enabled"] == enabled]
            result["summary"][str(enabled).lower()] = {
                "criterion_met": sum(row["criterion_met"] for row in rows),
                "total": len(rows),
                "scope_guard_refusals": sum(row["scope_guard_refusal"] for row in rows),
                "p50_ms": round(statistics.median(row["latency_ms"] for row in rows), 1),
                "by_category": {
                    category: {
                        "met": sum(row["criterion_met"] for row in rows if row["category"] == category),
                        "total": sum(row["category"] == category for row in rows),
                    }
                    for category in sorted({row["category"] for row in rows})
                },
            }
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    finally:
        cfg.answer_contract_enabled = original


if __name__ == "__main__":
    main()
