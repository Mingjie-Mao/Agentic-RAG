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

DOCS = ["README.md", "PROJECT_REPORT.md"]
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

    # Headline figures the READMEs quote, resolved from the artifacts they came from.
    expected = {
        "s3-v2 hybrid Recall@5": metric(
            "artifacts/s3-v2-hybrid-summary.json",
            lambda d: round(d["results"]["hybrid"]["recall_at_5"]["mean"], 3),
        ),
        "public hybrid Recall@5": metric(
            "artifacts/s3-public-summary.json",
            lambda d: round(d["results"]["hybrid"]["recall_at_5"]["mean"], 3),
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
    # Headline figures must appear in the README, so "written but stale" and
    # "measured but never written down" both fail rather than pass silently.
    readme = Path("README.md").read_text()
    for label, value in expected.items():
        if value is not None and f"{value:.3f}" not in readme:
            problems.append(f"README.md: 缺少或不匹配 {label} = {value:.3f}")

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
