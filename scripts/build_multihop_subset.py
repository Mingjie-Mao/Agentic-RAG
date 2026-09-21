"""Freeze a fixed MultiHop-RAG subset for one-shot external validation.

The dataset (https://huggingface.co/datasets/yixuantt/MultiHopRAG, ODC-BY) is
downloaded at build time and never committed: the article bodies are news text from
third-party publishers, and the repository has no business redistributing them. What
is committed is the *selection*: the pinned file hashes, the selection rule, and the
question ids with the sha256 of each question and gold answer. That is enough to
rebuild exactly the same subset from the original dataset, and not enough to quietly
edit a question after seeing a result.

Selection is deterministic and content-addressed: within each question type, the
questions are ordered by sha256 of the query text and the first N are taken. It does
not depend on file order, on a random seed, or on anything that could be re-rolled
until the numbers look better.
"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".runtime/multihop"
OUT = ROOT / "fixtures/multihop/subset.json"
BASE = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"
FILES = {
    "MultiHopRAG.json": "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
    "corpus.json": "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
}
# Proportional to the full 2556-question distribution, rounded to 150 total.
QUOTA = {"comparison_query": 50, "inference_query": 48, "temporal_query": 34, "null_query": 18}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fetch(name, expected, download):
    path = CACHE / name
    if not path.exists():
        if not download:
            raise SystemExit(f"{path} 不存在；先运行 --download 获取原始数据集")
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/{name}", path)
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise SystemExit(f"{name} 哈希不符：期望 {expected}，实际 {actual}")
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    queries = fetch("MultiHopRAG.json", FILES["MultiHopRAG.json"], args.download)
    corpus = fetch("corpus.json", FILES["corpus.json"], args.download)
    titles = {row["title"] for row in corpus}
    selected = []
    for question_type, quota in sorted(QUOTA.items()):
        pool = [row for row in queries if row["question_type"] == question_type]
        # Every gold document must exist in the corpus we actually index, otherwise a
        # miss would measure the dataset rather than the system.
        pool = [
            row for row in pool
            if all(item["title"] in titles for item in row.get("evidence_list", []))
        ]
        ordered = sorted(pool, key=lambda row: digest(row["query"]))
        if len(ordered) < quota:
            raise SystemExit(f"{question_type}: 只有 {len(ordered)} 题可用，少于配额 {quota}")
        for row in ordered[:quota]:
            selected.append(
                {
                    "id": f"MH-{digest(row['query'])[:12]}",
                    "question_type": question_type,
                    "query_sha256": digest(row["query"]),
                    "answer_sha256": digest(row["answer"]),
                    "gold_document_count": len({item["title"] for item in row["evidence_list"]}),
                }
            )
    payload = {
        "name": "multihop-rag-external-v1",
        "source": {
            "dataset": "yixuantt/MultiHopRAG",
            "url": "https://huggingface.co/datasets/yixuantt/MultiHopRAG",
            "license": "ODC-BY",
            "paper": "arXiv:2401.15391",
            "files": FILES,
        },
        "selection": (
            "按题型配额分层，题内按 sha256(query) 升序取前 N；配额与 2556 题的原始分布成比例。"
            "只保留 gold 证据文档全部存在于 609 篇语料中的题目。"
        ),
        "quota": QUOTA,
        "questions": len(selected),
        "purpose": (
            "一次性外部验证：与自建 Hard 30 分开报告，不用于调 prompt、规则或阈值。"
            "题面与金标答案不入库，只保存 ID 与哈希，运行时从原始数据集按哈希还原。"
        ),
        "items": selected,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"questions": len(selected), "quota": QUOTA, "out": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
