"""Refresh the Agent block of site/data.json from the stored run artifacts.

The rest of the page has always been backed by artifacts; the Agent rows were typed
by hand and went stale. This reads them back instead: the Hard Benchmark shared
subset and the one-shot external MultiHop-RAG run. It only rewrites the keys it owns,
so nothing else on the page moves, and it refuses to write a row whose artifact is
missing rather than leaving a remembered number in place.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site/data.json"
HARD = ROOT / "artifacts/agent-hard-benchmark-v2_1.json"
EXTERNAL = sorted((ROOT / "artifacts").glob("multihop-external*.json"))
LABELS = {"rag": "单次 RAG", "workflow": "固定工作流", "dynamic": "动态 Agent"}


def collected_tests():
    """The page used to claim a hand-typed test count; read it from pytest instead."""
    result = subprocess.run(
        [".venv/bin/pytest", "-q", "--collect-only"], capture_output=True, text=True, cwd=ROOT
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if not match:
        raise SystemExit("无法从 pytest 读出用例数")
    return int(match.group(1))


def seconds(value):
    return round((value or 0) / 1000, 1)


def hard_rows():
    data = json.loads(HARD.read_text())
    rows = []
    for arm in ("rag", "workflow", "dynamic"):
        summary = data["summary"][arm]["shared_comparable_subset"]
        rows.append(
            {
                "name": f"{LABELS[arm]} · Hard 共享 21 题",
                "correct": f"{summary['task_success']}/{summary['scored']}",
                "median_seconds": seconds(summary["p50_latency_ms"]),
                "mean_steps": summary["mean_steps"],
                "leaks": summary["security_leaks"],
            }
        )
    return rows, data["benchmark"]


def external_rows():
    """One row per arm per frozen batch, newest batch last."""
    rows, names = [], []
    batches = []
    for path in EXTERNAL:
        data = json.loads(path.read_text())
        batch = data.get("subset", "subset.json").replace(".json", "")
        number = 1 if batch == "subset" else int(batch.rsplit("-r", 1)[-1])
        batches.append((number, data))
    for number, data in sorted(batches):
        label = f"第 {number} 批"
        names.append(data["benchmark"])
        for arm in data["arms"]:
            summary = data["summary"][arm]["all"]
            rows.append(
                {
                    "name": f"{LABELS[arm]} · MultiHop-RAG 外部 {label}（{summary['questions']} 题）",
                    "correct": f"{summary['answer_correct']}/{summary['questions']}",
                    "median_seconds": seconds(summary["p50_latency_ms"]),
                    "mean_steps": None,
                    "leaks": None,
                    "gold_document_recall": summary["gold_document_recall"],
                    "scoring_rule": data.get("scoring_rule", "v1"),
                }
            )
    return rows, ", ".join(dict.fromkeys(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(SITE))
    args = parser.parse_args()
    if not HARD.exists() or not EXTERNAL:
        raise SystemExit("缺少 Hard 或外部评测产物，先运行对应评测")
    site = json.loads(SITE.read_text())
    site["tests"] = {"collected": collected_tests(), "source": "pytest --collect-only"}
    hard, hard_name = hard_rows()
    external, external_name = external_rows()
    site["agent"]["hard"] = {
        "benchmark": hard_name,
        "subset": "shared_comparable_subset",
        "rows": hard,
        "note": (
            "Hard 30 是参与调试的开发 benchmark，跨策略只比较同分母的 21 题共享子集；"
            "不是泛化成绩。"
        ),
    }
    site["agent"]["external"] = {
        "benchmark": external_name,
        "rows": external,
        "note": (
            "外部 MultiHop-RAG 固定子集，每批只运行一次，未用于调 prompt、规则或阈值。"
            "第 1 批按旧口径（要求答案里出现判断词）评分，第 2 批按显式结论字段评分，"
            "因此两批的答案列不可直接相减；Yes/No 题的判断正确率低于「一律答 yes」的平凡基线，"
            "结论字段解决的是可读性，不是判断力。"
        ),
    }
    Path(args.out).write_text(json.dumps(site, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"hard": hard, "external": external}, ensure_ascii=False, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
