"""P2: which cheap scorer best reproduces the frozen v3 gold?

Implements `artifacts/semantic-calibration/PREREGISTRATION-P2.md` and nothing else:
thresholds are fitted on dev only, test is reported once, every interval is a
bootstrap over questions (claims of one question move together), and the selection
rule is applied mechanically. It refuses to run unless the gold files still match the
freeze manifest, so a label edited after freezing cannot quietly change the result.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "artifacts" / "semantic-calibration"
CONTINUOUS = ("rule", "embedding", "cross_encoder")
ORDER = ("rule", "embedding", "cross_encoder", "llm_judge")


def verify_freeze(directory):
    manifest = directory / "gold-v3-manifest.json"
    if not manifest.exists():
        raise SystemExit("gold 尚未冻结（缺 gold-v3-manifest.json），按预注册不得评估")
    frozen = json.loads(manifest.read_text())
    for name, digest in frozen["sha256"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"{name} 在冻结后被修改（sha256 不符），拒绝评估")
    return frozen


def split_of(item_id):
    return "dev" if int(hashlib.sha256(f"p2-split|{item_id}".encode()).hexdigest(), 16) % 10 < 4 else "test"


def load_gold(directory):
    with (directory / "gold-v3-claims.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "key": f"{row['item_id']}#{row['claim_index']}",
            "item": row["item_id"],
            "split": split_of(row["item_id"]),
            "entailment": row["entailment"],
            "relevance": row["relevance"],
            "weight": float(row["weight"]),
        }
        for row in rows
    ]


# ---- weighted metrics ----------------------------------------------------------


def rate(rows, hit, among):
    total = sum(r["w"] for r in rows if among(r))
    return sum(r["w"] for r in rows if among(r) and hit(r)) / total if total else None


def binary_metrics(rows):
    """rows: [{w, gold: bool (positive class), pred: bool, score?, gold3?}]"""
    recall_pos = rate(rows, lambda r: r["pred"], lambda r: r["gold"])
    recall_neg = rate(rows, lambda r: not r["pred"], lambda r: not r["gold"])
    balanced = (recall_pos + recall_neg) / 2 if None not in (recall_pos, recall_neg) else None
    total = sum(r["w"] for r in rows)
    observed = sum(r["w"] for r in rows if r["gold"] == r["pred"]) / total
    p_gold = sum(r["w"] for r in rows if r["gold"]) / total
    p_pred = sum(r["w"] for r in rows if r["pred"]) / total
    expected = p_gold * p_pred + (1 - p_gold) * (1 - p_pred)
    out = {
        "balanced_accuracy": balanced,
        "recall_positive": recall_pos,
        "recall_negative": recall_neg,
        "kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
    }
    if rows and "gold3" in rows[0]:
        out["pass_rate_unsupported"] = rate(rows, lambda r: r["pred"], lambda r: r["gold3"] == "unsupported")
        out["pass_rate_partial"] = rate(rows, lambda r: r["pred"], lambda r: r["gold3"] == "partial")
    if rows and "score" in rows[0]:
        out["auc"] = auc(rows)
    return out


def auc(rows):
    positive = [r for r in rows if r["gold"]]
    negative = [r for r in rows if not r["gold"]]
    if not positive or not negative:
        return None
    wins = sum(
        p["w"] * n["w"] * ((p["score"] > n["score"]) + 0.5 * (p["score"] == n["score"]))
        for p in positive
        for n in negative
    )
    return wins / (sum(p["w"] for p in positive) * sum(n["w"] for n in negative))


def best_threshold(rows):
    """Dev-only threshold maximising balanced accuracy; ties go to the higher (stricter) one."""
    best = None
    for threshold in sorted({r["score"] for r in rows}):
        scored = [{**r, "pred": r["score"] >= threshold} for r in rows]
        value = binary_metrics(scored)["balanced_accuracy"]
        if value is not None and (best is None or value >= best[0]):
            best = (value, threshold)
    return best[1] if best else None


def bootstrap(rows, metric_names, draws=2000, seed=0):
    by_item = {}
    for row in rows:
        by_item.setdefault(row["item"], []).append(row)
    items = sorted(by_item)
    rng = random.Random(seed)
    samples = {name: [] for name in metric_names}
    for _ in range(draws):
        drawn = [row for item in (rng.choice(items) for _ in items) for row in by_item[item]]
        metrics = binary_metrics(drawn)
        for name in metric_names:
            if metrics.get(name) is not None:
                samples[name].append(metrics[name])
    intervals = {}
    for name, values in samples.items():
        if len(values) < draws * 0.9:
            intervals[name] = None
            continue
        values.sort()
        intervals[name] = [round(values[int(0.025 * len(values))], 3), round(values[int(0.975 * len(values)) - 1], 3)]
    return intervals


# ---- evaluation ------------------------------------------------------------------


def rows_for(gold, predictions, name, task, split, weighted):
    positive = "supported" if task == "entailment" else "answers"
    out = []
    for row in gold:
        if row["split"] != split or (task == "entailment" and row["weight"] <= 0):
            continue
        prediction = predictions[name][row["key"]]
        record = {
            "item": row["item"],
            "w": (row["weight"] if weighted and task == "entailment" else 1.0),
            "gold": row[task] == positive,
        }
        if task == "entailment":
            record["gold3"] = row["entailment"]
        if name in CONTINUOUS:
            record["score"] = prediction[task]
        else:
            record["pred"] = prediction[task] == positive
            record["pred3"] = prediction[task]
        out.append(record)
    return out


def rounded(metrics):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}


def evaluate(gold, predictions, task, weighted):
    report = {}
    for name in ORDER:
        if name not in predictions:
            continue
        dev = rows_for(gold, predictions, name, task, "dev", weighted)
        test = rows_for(gold, predictions, name, task, "test", weighted)
        entry = {"n_dev": len(dev), "n_test": len(test)}
        if name in CONTINUOUS:
            threshold = best_threshold(dev)
            entry["threshold_from_dev"] = threshold
            test = [{**r, "pred": r["score"] >= threshold} for r in test]
        metrics = binary_metrics(test)
        entry["test"] = rounded(metrics)
        entry["test_ci95"] = bootstrap(test, ["balanced_accuracy", "pass_rate_unsupported", "kappa"])
        unsupported = sum(r.get("gold3") == "unsupported" for r in test)
        if task == "entailment" and metrics.get("pass_rate_unsupported") == 0 and unsupported:
            # With zero observed passes the bootstrap interval collapses to [0, 0]; the
            # rule of three gives the honest 95% upper bound on the true pass rate.
            entry["pass_rate_upper95_rule_of_three"] = round(3 / unsupported, 3)
        clock = "relevance_ms" if task == "relevance" and name in CONTINUOUS else "ms"
        timings = [p[clock] for p in predictions[name].values() if clock in p]
        entry["median_ms_per_claim"] = round(statistics.median(timings), 1) if timings else None
        if name == "llm_judge" and task == "entailment":
            entry["test_confusion_3way"] = confusion(test)
        report[name] = entry
    return report


def confusion(rows):
    table = {}
    for row in rows:
        cell = table.setdefault(row["gold3"], {})
        cell[row["pred3"]] = round(cell.get(row["pred3"], 0) + row["w"], 2)
    return table


def select(report):
    """Preregistration §5, applied mechanically. The LLM judge is reported, never ranked:
    its labels come from the same kind of model as the gold's final reviewer."""
    candidates = {name: entry for name, entry in report.items() if name in CONTINUOUS}
    alive = {
        name: entry
        for name, entry in candidates.items()
        if entry["test_ci95"]["balanced_accuracy"] and entry["test_ci95"]["balanced_accuracy"][0] > 0.5
    }
    if not alive:
        return {"selected": None, "reason": "没有打分器的均衡准确率 CI 下界 > 0.5：没有打分器能复现 v3 判据"}
    base_name = max(alive, key=lambda n: alive[n]["test"]["balanced_accuracy"])
    base = alive[base_name]["test"]
    eligible = [
        name
        for name, entry in alive.items()
        if entry["test_ci95"]["balanced_accuracy"][1] >= base["balanced_accuracy"]
        and (entry["test"]["pass_rate_unsupported"] or 0) <= (base["pass_rate_unsupported"] or 0) + 0.10
    ]
    chosen = min(eligible, key=lambda n: alive[n]["median_ms_per_claim"])
    return {
        "baseline": base_name,
        "eligible": eligible,
        "eliminated_as_random": sorted(set(candidates) - set(alive)),
        "reported_not_ranked": sorted(set(report) - set(candidates)),
        "selected": chosen,
        "deployment": "shadow only — 记录不拦截；升级为拦截需要另一次冻结评估",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DIR))
    parser.add_argument("--predictions", default=str(DIR / "scorer-predictions.json"))
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "semantic-scorer-calibration.json"))
    args = parser.parse_args()
    directory = Path(args.dir)
    frozen = verify_freeze(directory)
    gold = load_gold(directory)
    predictions = json.loads(Path(args.predictions).read_text())["predictions"]

    report = {
        "preregistration": "artifacts/semantic-calibration/PREREGISTRATION-P2.md",
        "gold_frozen_at": frozen["frozen_at"],
        "final_reviewer": frozen["final_reviewer"],
        "split": {
            split: {
                "items": len({r["item"] for r in gold if r["split"] == split and r["weight"] > 0}),
                "claims": sum(r["split"] == split and r["weight"] > 0 for r in gold),
            }
            for split in ("dev", "test")
        },
        "entailment_weighted": evaluate(gold, predictions, "entailment", weighted=True),
        "entailment_unweighted": evaluate(gold, predictions, "entailment", weighted=False),
        "relevance": evaluate(gold, predictions, "relevance", weighted=False),
    }
    report["selection"] = select(report["entailment_weighted"])
    report["selection_unweighted_sensitivity"] = select(report["entailment_unweighted"])
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def cell(value, width):
        return f"{value:{width}.3f}" if isinstance(value, (int, float)) else f"{'—':>{width}s}"

    for block in ("entailment_weighted", "entailment_unweighted", "relevance"):
        print(f"\n== {block}（test）")
        print(f"{'打分器':14s}{'均衡准确率':>10s}{'95% CI':>16s}{'放行率':>8s}{'AUC':>7s}{'kappa':>7s}{'ms':>9s}")
        for name, entry in report[block].items():
            t, ci = entry["test"], entry["test_ci95"]["balanced_accuracy"]
            print(
                f"{name:14s}{cell(t['balanced_accuracy'], 10)}{str(ci):>16s}"
                f"{cell(t.get('pass_rate_unsupported'), 8)}{cell(t.get('auc'), 7)}"
                f"{cell(t['kappa'], 7)}{entry['median_ms_per_claim'] or 0:9.1f}"
            )
    print("\n选择：", json.dumps(report["selection"], ensure_ascii=False))
    print("不加权敏感性：", json.dumps(report["selection_unweighted_sensitivity"], ensure_ascii=False))
    print(args.out)


if __name__ == "__main__":
    main()
