"""Record real, fictional-corpus outputs for the offline public demo.

This is product capture, not a benchmark run. No Core tasks or tuning involved.
Writes a new timestamped file and refuses to overwrite an existing capture.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "site" / f"replay-{stamp}.json"
    data = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "kind": "recorded_real_execution",
        "notice": "虚构星桥资料的真实历史执行；点击回放不会调用模型，也不代表当前实时执行或正式评测成绩。",
        "model": None,
        "scenarios": [],
    }
    with httpx.Client(
        base_url=args.base, headers={"X-Requested-With": "AgenticRAG"}, timeout=240, trust_env=False
    ) as client:
        response = client.post("/api/auth/visitor")
        response.raise_for_status()
        system = client.get("/api/system")
        system.raise_for_status()
        data["model"] = system.json()["generation_model"]
        goal = "星桥标准套餐每分钟允许多少次 API 请求？超过配额会返回什么 HTTP 状态码？"
        response = client.post(
            "/api/agent/tasks", json={"goal": goal, "mode": "workflow", "max_steps": 4}
        )
        response.raise_for_status()
        task = response.json()
        deadline = time.monotonic() + 240
        while task["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(2)
            response = client.get("/api/agent/tasks/" + task["id"])
            response.raise_for_status()
            task = response.json()
        if task["status"] != "completed":
            raise RuntimeError(f"capture incomplete: {task['status']}")
        result = task["result"]
        evidence = []
        allowed = {
            "seed-" + doc["id"]
            for doc in json.loads((ROOT / "fixtures/catalog.json").read_text())["documents"]
        }
        for citation in result.get("citations", []):
            if citation["document_id"] not in allowed:
                raise RuntimeError("Only fictional seed documents may be published")
            response = client.get("/api/evidence/" + citation["chunk_id"])
            response.raise_for_status()
            source = response.json()
            evidence.append(
                {"title": source["title"], "text": source["text"], "locator": source["locator"]}
            )
        data["scenarios"].append(
            {
                "title": "Agent 多步查证",
                "goal": goal,
                "mode": task["mode"],
                "task_id": task["id"],
                "created_at": task["created_at"],
                "updated_at": task["updated_at"],
                "events": [
                    {key: event[key] for key in ("sequence", "event_type", "tool_name", "created_at")}
                    | {
                        "evidence_count": len(event["evidence_refs"]),
                        "wall_ms": event["payload"].get("wall_ms"),
                    }
                    for event in task["events"]
                ],
                "status": result["status"],
                "claims": result.get("claims", []),
                "evidence": evidence,
            }
        )
        for title, question in [
            ("回答与原文核验", "星桥标准套餐每分钟可以调用多少次 API？"),
            ("依据不足时拒答", "公司去火星出差的补贴是多少？"),
        ]:
            response = client.post("/api/chat", json={"question": question})
            response.raise_for_status()
            result = response.json()
            if any(c["document_id"] not in allowed for c in result["citations"]):
                raise RuntimeError("Unexpected corpus in capture")
            data["scenarios"].append(
                {
                    "title": title,
                    "goal": question,
                    "status": result["status"],
                    "claims": result["claims"],
                    "message": result["message"],
                    "elapsed_seconds": result["trace"]["total_ms"] / 1000,
                    "evidence": [
                        {"title": c["title"], "text": c["text"], "locator": c["locator"]}
                        for c in result["citations"]
                    ],
                    "events": [],
                }
            )
    with out.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(out)
    print("Agent task:", task["id"], flush=True)


if __name__ == "__main__":
    main()
