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
EXTERNAL = ROOT / "artifacts/multihop-external.json"
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
    data = json.loads(EXTERNAL.read_text())
    rows = []
    for arm in data["arms"]:
        summary = data["summary"][arm]["all"]
        rows.append(
            {
                "name": f"{LABELS[arm]} · MultiHop-RAG 外部 {summary['questions']} 题",
                "correct": f"{summary['answer_correct']}/{summary['questions']}",
                "median_seconds": seconds(summary["p50_latency_ms"]),
                "mean_steps": None,
                "leaks": None,
                "gold_document_recall": summary["gold_document_recall"],
            }
        )
    return rows, data["benchmark"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(SITE))
    args = parser.parse_args()
    for path in (HARD, EXTERNAL):
        if not path.exists():
            raise SystemExit(f"缺少产物 {path}，先运行对应评测")
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
            "外部 MultiHop-RAG 固定子集，只运行一次，未用于调 prompt、规则或阈值。"
            "答案指标是字面匹配：Yes/No 题要求答案里出现判断词，本系统输出的是带引用的 claim，"
            "两者口径不同，这一栏会低估语义正确率。"
        ),
    }
    Path(args.out).write_text(json.dumps(site, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"hard": hard, "external": external}, ensure_ascii=False, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
