"""Fail when the READMEs describe something the repository does not actually have.

Documentation drifts silently: a number that was true three runs ago still reads as
a measurement. This checks the claims that can be checked mechanically — commands,
paths, accounts, headline metrics — against the repository and the stored artifacts.
It cannot check prose, so it is a floor, not a guarantee.
"""

import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DOCS = ["README.md", "PROJECT_REPORT.md", "RAG_AND_MEMORY.md"]
problems = []


def collected_tests():
    result = subprocess.run([".venv/bin/pytest", "-q", "--collect-only"], capture_output=True, text=True)
    match = re.search(r"(\d+) tests? collected", result.stdout)
    return int(match.group(1)) if match else None


def metric(path, reader):
    try:
        return reader(json.loads(Path(path).read_text()))
    except (OSError, KeyError, ValueError):
        return None


def main():
    targets = set(re.findall(r"^([a-z0-9-]+):", Path("Makefile").read_text(), re.M))
    accounts = {u["username"] for u in json.loads(Path("fixtures/catalog.json").read_text())["users"]}
    tests = collected_tests()

    # Every figure in the README results table, resolved from the artifact it came from.
    def pct(numerator, denominator):
        return f"{numerator / denominator:.1%}"

    hard = "artifacts/agent-hard-benchmark-v2_1.json"
    readme_figures = {
        "holdout answer-state accuracy": metric(
            "artifacts/s7-holdout-summary.json",
            lambda d: pct(d["results"]["hybrid"]["status_correct"]["count"], d["results"]["hybrid"]["status_correct"]["n"]),
        ),
        "holdout literal fact coverage": metric(
            "artifacts/s7-holdout-summary.json",
            lambda d: pct(
                d["results"]["hybrid"]["literal_fact_coverage"]["matches"],
                d["results"]["hybrid"]["literal_fact_coverage"]["total"],
            ),
        ),
        "MultiHop-RAG batch 2 workflow": metric(
            "artifacts/multihop-external-r2.json",
            lambda d: f"{d['summary']['workflow']['all']['answer_correct']}/{d['summary']['workflow']['all']['questions']}",
        ),
        **{
            f"Hard v2.1 {arm}": metric(
                hard,
                lambda d, arm=arm: "{task_success}/{scored}".format(
                    **d["summary"][arm]["shared_comparable_subset"]
                ),
            )
            for arm in ("rag", "workflow", "dynamic")
        },
    }
    # Retrieval headlines now live in the report, which must still quote them exactly.
    report_figures = {
        "s3-v2 hybrid Recall@5": metric(
            "artifacts/s3-v2-hybrid-summary.json",
            lambda d: f'{round(d["results"]["hybrid"]["recall_at_5"]["mean"], 3):.3f}',
        ),
        "public hybrid Recall@5": metric(
            "artifacts/s3-public-summary.json",
            lambda d: f'{round(d["results"]["hybrid"]["recall_at_5"]["mean"], 3):.3f}',
        ),
    }

    for doc in DOCS:
        text = Path(doc).read_text()
        for target in sorted(set(re.findall(r"make ([a-z0-9-]+)", text))):
            if target not in targets:
                problems.append(f"{doc}: 文档里的 `make {target}` 在 Makefile 中不存在")
        for path in sorted(
            set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:py|md|json|ts|tsx|yaml|xml))`", text))
        ):
            if not Path(path).exists():
                problems.append(f"{doc}: 引用的路径 {path} 不存在")
        for account in sorted(set(re.findall(r"`([a-z]+@[a-z]+\.demo)`", text))):
            if account not in accounts:
                problems.append(f"{doc}: 演示账号 {account} 不在 fixtures/catalog.json 中")
        for claimed in re.findall(r"(\d+)\s*项(?:单元与逻辑)?检查", text):
            if tests is not None and int(claimed) != tests:
                problems.append(f"{doc}: 声称 {claimed} 项测试，实际收集 {tests} 项")
    # Headline figures must appear where they are quoted, so "written but stale" and
    # "measured but never written down" both fail rather than pass silently.
    for doc, figures in (("PROJECT_REPORT.md", {**readme_figures, **report_figures}),):
        text = Path(doc).read_text()
        for label, value in figures.items():
            if value is None:
                problems.append(f"{doc}: 无法从产物读取 {label}")
            elif value not in text:
                problems.append(f"{doc}: 缺少或不匹配 {label} = {value}")

    # Headline metrics now have a single denominator; legacy numbers remain in
    # PROJECT_REPORT.md §22 rather than being required in the README main table.
    from scripts.benchmark_package import validate_package
    try:
        counts = validate_package(require_private=False)["counts"]
        text = Path("README.md").read_text()
        expected = f"Dev {counts['dev']} / Core Test {counts['core']} / Security {counts['security']} / External {counts['external']}"
        if expected not in text or "Strict Task Success" not in text:
            problems.append("README.md: 主 benchmark 分母或指标不匹配")
        if any(value in text for value in readme_figures.values() if value):
            problems.append("README.md: 历史分数不应混入当前主表")
    except (ValueError, OSError) as exc:
        problems.append(f"benchmark package: {exc}")

    config = Path("app/config.py").read_text()
    for setting, doc_claim in [
        ('retrieval_mode: str = "hybrid"', "混合检索为默认"),
        ('rewrite_mode: str = "rule"', "规则改写为默认"),
    ]:
        if setting not in config:
            problems.append(f"app/config.py 与文档不符：{doc_claim}")

    if problems:
        print(f"文档与实际不一致，共 {len(problems)} 处：")
        for item in problems:
            print("  -", item)
        sys.exit(1)
    print(f"文档一致性检查通过：命令、路径、账号、测试数（{tests}）与关键指标均与仓库一致")


if __name__ == "__main__":
    main()
