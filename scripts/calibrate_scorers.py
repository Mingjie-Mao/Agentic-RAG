"""A-Q1: can anything cheap tell relevant material from irrelevant, and supported
claims from unsupported ones?

The ladder is deliberate — rule, then embedding, then cross-encoder — and each rung is
only justified if the one below it could not separate the labels. A scorer that cannot
separate them is reported as such and left off, exactly as the subject-consistency
gate was.

Relevance and entailment are scored separately because they fail separately: material
can be about the right subject and still not answer the question, and a claim can be
present verbatim in material that answers a different question.
"""

import argparse
import json
from pathlib import Path
import re
import statistics

CJK = re.compile(r"[一-鿿]+")


def spans(text, n=2):
    joined = "".join(CJK.findall(text or ""))
    return {joined[i : i + n] for i in range(max(0, len(joined) - n + 1))}


def rule_score(question, material):
    """Character-bigram overlap. The cheapest thing that could possibly work."""
    a, b = spans(question), spans(material)
    return len(a & b) / len(a) if a else 0.0


def embedding_scorer():
    from app.clients import Models

    models = Models()

    def score(question, material):
        if not material.strip():
            return 0.0
        left, right = models.embed([question, material[:2000]])
        norm = sum(x * x for x in left) ** 0.5 * sum(y * y for y in right) ** 0.5
        return sum(x * y for x, y in zip(left, right, strict=True)) / norm if norm else 0.0

    return score


def cross_encoder_scorer():
    from app.rerank import rerank

    def score(question, material):
        if not material.strip():
            return -12.0
        ranked = rerank(question, [{"text": material[:2000]}], text_of=lambda item: item["text"])
        return ranked[0]["rerank_score"]

    return score


def separation(positive, negative):
    """How well the two label groups separate, and the best threshold if they do.

    AUC is reported because a scorer can shift a whole distribution without ever
    ordering the two groups correctly, and a single threshold would hide that.
    """
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (
        len(positive) * len(negative)
    )
    best = max(
        (
            (
                sum(p >= t for p in positive) / len(positive)
                + sum(n < t for n in negative) / len(negative)
            )
            / 2,
            t,
        )
        for t in sorted(set(positive + negative))
    )
    return {
        "auc": round(wins, 3),
        "balanced_accuracy": round(best[0], 3),
        "threshold": round(best[1], 4),
        "positive_median": round(statistics.median(positive), 4),
        "negative_median": round(statistics.median(negative), 4),
        "n_positive": len(positive),
        "n_negative": len(negative),
    }


def material_for(case, task):
    if task == "relevance":
        # What was actually put in front of the model, or what retrieval offered when
        # the system refused and therefore cited nothing.
        source = case["evidence"] or case["retrieved"]
        return "\n".join(item["text"] for item in source[:4])
    return "\n".join(item["text"] for item in case["evidence"][:4])


def run(cases, scorers, task):
    labelled = [c for c in cases if c["label"][task] is not None]
    results = {}
    for name, score in scorers.items():
        values = {c["case_id"]: score(c["question"], material_for(c, task)) for c in labelled}
        positive = [values[c["case_id"]] for c in labelled if c["label"][task] == "yes"]
        negative = [values[c["case_id"]] for c in labelled if c["label"][task] == "no"]
        results[name] = {"separation": separation(positive, negative), "scores": values}
    return {"cases": len(labelled), "scorers": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="fixtures/calibration/a-q1.json")
    parser.add_argument("--out", default="artifacts/a-q1-calibration.json")
    parser.add_argument("--scorers", default="rule,embedding,cross_encoder")
    args = parser.parse_args()

    payload = json.loads(Path(args.set).read_text())
    cases = payload["cases"]
    assert all(c["label"]["relevance"] for c in cases), "校准集尚未标注"

    available = {
        "rule": rule_score,
        "embedding": embedding_scorer,
        "cross_encoder": cross_encoder_scorer,
    }
    scorers = {}
    for name in args.scorers.split(","):
        factory = available[name]
        scorers[name] = factory() if name != "rule" else factory

    report = {
        "calibration_set": args.set,
        "label_provenance": payload["meta"].get("caveat"),
        "independent_human_review": payload["meta"].get("independent_human_review", 0),
        "relevance": run(cases, scorers, "relevance"),
        "entailment": run(cases, scorers, "entailment"),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for task in ["relevance", "entailment"]:
        print(f"\n== {task}（{report[task]['cases']} 例有标签）")
        print(
            f"{'评分器':14s}{'AUC':>7s}{'均衡准确率':>10s}{'阈值':>9s}{'正例中位':>9s}{'负例中位':>9s}"
        )
        for name, value in report[task]["scorers"].items():
            s = value["separation"]
            if not s:
                print(f"{name:14s}  正负样本不全，无法判断")
                continue
            print(
                f"{name:14s}{s['auc']:7.3f}{s['balanced_accuracy']:10.3f}"
                f"{s['threshold']:9.3f}{s['positive_median']:9.3f}{s['negative_median']:9.3f}"
            )


if __name__ == "__main__":
    main()
