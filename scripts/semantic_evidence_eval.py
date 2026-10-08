"""Component benchmark for the semantic evidence evaluator, before it touches the Agent.

Two labelled sets, both from development data only — the Chinese S3-v2 development
split and the already-used MultiHop-RAG batches 1 and 2. Neither overlaps the Hard 30
regression set or the unseen task set that will be frozen later.

Relevance  (question, passage) -> relevant iff the passage contains a gold fact
Coverage   (question, evidence set) -> supported / partial / unsupported / contradicted
           built per question from three evidence sets:
             full     every gold passage (+ distractors)  supported, or contradicted for
                      a true cross-document conflict question
             partial  one gold passage removed while another fact remains   partial
             none     same-document passages without the facts + distractors unsupported

Labels are mechanical (does the evidence contain the gold fact strings), so "supported"
here means the needed values are present, not that a human read the passage as an
answer. Questions are split by hash: thresholds are fitted on dev, test is reported.

Compared on the same items: the production lexical proxies, cross-encoder alone, and the
cascade (cross-encoder recall cut, then one batched structured judgement).
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agent.planner import _content, _grams, evidence_coverage  # noqa: E402
from app.semantic_evidence import SemanticEvidenceEvaluator, Thresholds  # noqa: E402

S3_SNAPSHOT = ROOT / ".runtime/evaluation/192bc89271af4b1f61d6766b9c369e3d5bea7362765736fa20e109812f318fce/chunks.json"
S3_QUESTIONS = ROOT / "fixtures/s3-v2/questions.json"
S3_RETRIEVAL = ROOT / "artifacts/s3-v2-hybrid/development-hybrid-retrieval.jsonl"
MH_CACHE = ROOT / ".runtime/multihop"
OUT = ROOT / "artifacts/semantic-evidence-eval.json"
CACHE = ROOT / ".runtime/semantic-evidence-cache"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_of(qid):
    return "dev" if int(digest(qid)[-1], 16) < 6 else "test"  # 6/16 = 37.5% dev


def compact(text):
    return re.sub(r"\s+", "", str(text))


def build_s3():
    chunks = json.loads(S3_SNAPSHOT.read_text())
    by_doc = {}
    for chunk in chunks:
        if len(compact(chunk["text"])) > 40:
            by_doc.setdefault(chunk["document_id"], []).append(chunk)
    hits = {}
    for line in S3_RETRIEVAL.read_text().splitlines():
        row = json.loads(line)
        hits[row["id"]] = [h["chunk_id"] for h in row["hits"]]
    chunk_by_id = {c["id"]: c for c in chunks}
    items = []
    for q in json.loads(S3_QUESTIONS.read_text()):
        if q["split"] != "development" or not q["facts"] or q["kind"] in {"permission", "unanswerable"}:
            continue
        facts = [compact(f) for f in q["facts"]]
        pool = [c for doc in q["source_ids"] for c in by_doc.get(doc, [])]
        gold = [c for c in pool if any(f in compact(c["text"]) for f in facts)]
        hard = [c for c in pool if c not in gold]
        distractors = [chunk_by_id[cid] for cid in hits.get(q["id"], [])
                       if cid in chunk_by_id and chunk_by_id[cid]["document_id"] not in q["source_ids"]][:2]
        items.extend(evidence_sets("s3", q["id"], q["question"], q["kind"], facts, gold, hard, distractors,
                                   contains=lambda f, c: f in compact(c["text"])))
    return items


def build_multihop():
    sys.path.insert(0, str(ROOT / "scripts"))
    from multihop_retrieval_eval import document_id, fact_delivered
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Chunk, Document

    records = {digest(r["query"]): r for r in json.loads((MH_CACHE / "MultiHopRAG.json").read_bytes())}
    items = []
    with SessionLocal() as db:
        def chunks_of(doc_id):
            doc = db.get(Document, doc_id)
            rows = db.scalars(select(Chunk).where(Chunk.version_id == doc.active_version_id).order_by(Chunk.ordinal)).all()
            return [{"id": c.id, "document_id": doc_id, "title": doc.title, "text": c.text} for c in rows]

        for name in ("subset.json", "subset-r2.json"):
            taken = {}
            for item in json.loads((ROOT / "fixtures/multihop" / name).read_text())["items"]:
                kind = item["question_type"]
                if kind == "null_query" or taken.get(kind, 0) >= 10:
                    continue
                taken[kind] = taken.get(kind, 0) + 1
                record = records[item["query_sha256"]]
                facts = [row["fact"] for row in record["evidence_list"]]
                gold_docs = list(dict.fromkeys(document_id(row["title"]) for row in record["evidence_list"]))
                pool = [c for doc in gold_docs for c in chunks_of(doc)]
                gold = [c for c in pool if any(fact_delivered(f, [c["text"]]) for f in facts)]
                hard = [c for c in pool if c not in gold and len(c["text"]) > 80][:6]
                items.extend(evidence_sets("mh", item["id"], record["query"], kind, facts, gold, hard, [],
                                           contains=lambda f, c: fact_delivered(f, [c["text"]])))
    return items


def evidence_sets(corpus, qid, question, kind, facts, gold, hard, distractors, *, contains):
    def covered(chunks):
        return [f for f in facts if any(contains(f, c) for c in chunks)]

    # Minimal gold set: greedily add passages until every fact is present.
    minimal, remaining = [], list(facts)
    for chunk in sorted(gold, key=lambda c: -len([f for f in remaining if contains(f, c)])):
        if any(contains(f, chunk) for f in remaining):
            minimal.append(chunk)
            remaining = [f for f in remaining if not contains(f, chunk)]
    if remaining or not minimal:
        return []
    base = {"corpus": corpus, "qid": qid, "question": question, "kind": kind, "split": split_of(qid)}
    out = []
    full_label = "contradicted" if kind == "conflict" else "supported"
    out.append({**base, "variant": "full", "label": full_label, "evidence": minimal + distractors[:2] + hard[:1]})
    if len(minimal) >= 2 and kind != "conflict":
        reduced = minimal[1:]
        if 0 < len(covered(reduced)) < len(facts):
            out.append({**base, "variant": "partial", "label": "partial", "evidence": reduced + distractors[:1] + hard[:1]})
    negatives = hard[:3] + distractors[:2]
    if negatives and not covered(negatives):
        out.append({**base, "variant": "none", "label": "unsupported", "evidence": negatives})
    for row in out:
        row["evidence"] = [{"chunk_id": c["id"], "title": c["title"], "text": c["text"],
                            "relevant": bool(covered([c]))} for c in row["evidence"]]
    return out


def lexical_relevance(question, text):
    wanted = _grams(_content(question))
    return len(wanted & _grams(text)) / len(wanted) if wanted else 0.0


def metrics_binary(rows):
    tp = sum(r["pred"] and r["gold"] for r in rows)
    fp = sum(r["pred"] and not r["gold"] for r in rows)
    fn = sum(not r["pred"] and r["gold"] for r in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"n": len(rows), "precision": _r(precision), "recall": _r(recall), "f1": _r(f1),
            "false_negative_rate": _r(fn / (tp + fn)) if tp + fn else None}


def auc(rows):
    pos = [r["score"] for r in rows if r["gold"]]
    neg = [r["score"] for r in rows if not r["gold"]]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return _r(wins / (len(pos) * len(neg)))


def _r(value):
    return None if value is None else round(value, 4)


def per_class(rows, labels):
    report = {}
    for label in labels:
        tp = sum(r["pred"] == label and r["gold"] == label for r in rows)
        fp = sum(r["pred"] == label and r["gold"] != label for r in rows)
        fn = sum(r["pred"] != label and r["gold"] == label for r in rows)
        p = tp / (tp + fp) if tp + fp else None
        rc = tp / (tp + fn) if tp + fn else None
        report[label] = {"precision": _r(p), "recall": _r(rc),
                         "f1": _r(2 * p * rc / (p + rc)) if p and rc else 0.0, "support": tp + fn}
    present = [v["f1"] for k, v in report.items() if v["support"]]
    confusion = {}
    for r in rows:
        confusion.setdefault(r["gold"], {}).setdefault(r["pred"], 0)
        confusion[r["gold"]][r["pred"]] += 1
    return {"per_class": report, "macro_f1": _r(statistics.mean(present)) if present else None,
            "accuracy": _r(sum(r["pred"] == r["gold"] for r in rows) / len(rows)) if rows else None,
            "confusion": confusion}


def select_rule(dev_rows, labels):
    """Pre-registered: among rules with dev SUPPORTED precision >= 0.85 pick the highest
    dev macro-F1; if none qualifies, the highest dev SUPPORTED precision."""
    if not dev_rows or "rule_a" not in dev_rows[0]:
        return None
    scored = {}
    for name in ("rule_a", "rule_b", "rule_c"):
        m = per_class([{"pred": r[name], "gold": r["gold"]} for r in dev_rows], labels)
        scored[name] = {"supported_precision": m["per_class"]["supported"]["precision"] or 0.0,
                        "macro_f1": m["macro_f1"] or 0.0}
    eligible = [n for n, v in scored.items() if v["supported_precision"] >= 0.85]
    chosen = (max(eligible, key=lambda n: scored[n]["macro_f1"]) if eligible
              else max(scored, key=lambda n: scored[n]["supported_precision"]))
    return {"dev": scored, "eligible": eligible, "chosen": chosen,
            "rule": "supported precision >= 0.85 on dev, then highest dev macro-F1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--splits", default="dev,test")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    items = build_s3() + build_multihop()
    evaluator = SemanticEvidenceEvaluator()
    scorer = evaluator._scorer()

    # ---- relevance: unique (question, passage) pairs -----------------------------------
    pairs = {}
    for row in items:
        for ev in row["evidence"]:
            pairs[(row["qid"], ev["chunk_id"])] = (row, ev)
    by_question = {}
    for (qid, _), (row, ev) in pairs.items():
        by_question.setdefault(qid, (row, []))[1].append(ev)
    rel_rows = []
    for qid, (row, evs) in by_question.items():
        scores = scorer(row["question"], [f"{e['title']}\n{e['text']}" for e in evs])
        for ev, score in zip(evs, scores, strict=True):
            rel_rows.append({"qid": qid, "split": row["split"], "corpus": row["corpus"], "chunk_id": ev["chunk_id"],
                             "gold": ev["relevant"], "ce": score,
                             "lex": lexical_relevance(row["question"], f"{ev['title']} {ev['text']}")})
    dev = [r for r in rel_rows if r["split"] == "dev"]
    test = [r for r in rel_rows if r["split"] == "test"]

    def fit(rows, key, *, recall_target=None, precision_target=None):
        candidates = sorted({r[key] for r in rows})
        best = candidates[0]
        for threshold in candidates:
            m = metrics_binary([{"pred": r[key] >= threshold, "gold": r["gold"]} for r in rows])
            if recall_target is not None and (m["recall"] or 0) >= recall_target:
                best = threshold
            if precision_target is not None and (m["precision"] or 0) >= precision_target:
                return threshold
        return best

    keep = fit(dev, "ce", recall_target=0.95)
    high = fit(dev, "ce", precision_target=0.90)
    lex_dev_best = max(sorted({r["lex"] for r in dev}),
                       key=lambda t: metrics_binary([{"pred": r["lex"] >= t, "gold": r["gold"]} for r in dev])["f1"] or 0)
    thresholds = Thresholds(keep=keep, high=high, version="semantic-evidence-v1")

    def rel_report(rows):
        return {
            "lexical@0.25 (production)": metrics_binary([{"pred": r["lex"] >= 0.25, "gold": r["gold"]} for r in rows]),
            f"lexical@dev-best {lex_dev_best:.3f}": metrics_binary(
                [{"pred": r["lex"] >= lex_dev_best, "gold": r["gold"]} for r in rows]),
            f"cross-encoder keep>={keep:.3f} (recall cut)": metrics_binary(
                [{"pred": r["ce"] >= keep, "gold": r["gold"]} for r in rows]),
            f"cross-encoder high>={high:.3f} (precision cut)": metrics_binary(
                [{"pred": r["ce"] >= high, "gold": r["gold"]} for r in rows]),
            "auc_lexical": auc([{"score": r["lex"], "gold": r["gold"]} for r in rows]),
            "auc_cross_encoder": auc([{"score": r["ce"], "gold": r["gold"]} for r in rows]),
        }

    relevance = {"dev": rel_report(dev), "test": rel_report(test),
                 "test_by_corpus": {c: rel_report([r for r in test if r["corpus"] == c]) for c in ("s3", "mh")}}

    # ---- coverage: test split only ---------------------------------------------------
    ce = {(r["qid"], r["chunk_id"]): r["ce"] for r in rel_rows}
    labels = ["supported", "partial", "unsupported", "contradicted"]
    cov_rows = []
    cache_file = CACHE / "judge.jsonl"
    cached = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            row = json.loads(line)
            cached[row["key"]] = row
    judged = SemanticEvidenceEvaluator(relevance_scorer=scorer, thresholds=thresholds)
    for row in [r for r in items if r["split"] in args.splits.split(",")]:
        item = [{"id": "s1", "text": row["question"]}]
        passages = [{"chunk_id": e["chunk_id"], "title": e["title"], "text": e["text"]} for e in row["evidence"]]
        lex_cov = evidence_coverage([row["question"]], [{"chunk_id": e["chunk_id"], "title": e["title"], "text": e["text"]}
                                                        for e in row["evidence"]], row["question"])[row["question"]]
        best_ce = max(ce[(row["qid"], e["chunk_id"])] for e in row["evidence"])
        result = {"qid": row["qid"], "split": row["split"], "best_ce": round(best_ce, 4),
                  "corpus": row["corpus"], "kind": row["kind"], "variant": row["variant"],
                  "gold": row["label"], "lexical": "supported" if lex_cov else "unsupported",
                  "cross_encoder_only": "supported" if best_ce >= high else "unsupported"}
        key = digest(json.dumps([row["qid"], row["variant"], [p["chunk_id"] for p in passages]]))
        if not args.skip_judge:
            if key not in cached:
                before = judged.usage.as_dict()
                rel = judged.relevance(item, passages)
                verdict = judged.coverage(row["question"], item, passages, rel)[0]
                after = judged.usage.as_dict()
                cached[key] = {"key": key, "status": verdict.status, "confidence": verdict.confidence,
                               "source": verdict.source, "reason": verdict.reason_code,
                               "calls": after["calls"] - before["calls"],
                               "wall_ms": round(after["wall_ms"] - before["wall_ms"], 1),
                               "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"]}
                with cache_file.open("a") as handle:
                    handle.write(json.dumps(cached[key], ensure_ascii=False) + "\n")
            got = cached[key]
            raw, confidence = got["status"], got["confidence"]
            # Pre-listed rules; one is chosen on dev, then reported once on test.
            rule_a = "partial" if raw == "supported" and confidence < 0.6 else raw
            rule_b = "partial" if rule_a == "supported" and best_ce < high else rule_a
            rule_c = "partial" if raw == "supported" and confidence < 0.8 else raw
            status = rule_a
            result.update(rule_a=rule_a, rule_b=rule_b, rule_c=rule_c)
            result.update(cascade=status, judge_calls=got["calls"], judge_ms=got["wall_ms"],
                          judge_prompt_tokens=got["prompt_tokens"], judge_source=got["source"], reason=got["reason"])
        cov_rows.append(result)
        if len(cov_rows) % 20 == 0:
            print(f"  coverage {len(cov_rows)}", flush=True)

    def cov_report(rows):
        out = {}
        for name in ("lexical", "cross_encoder_only", "cascade", "rule_a", "rule_b", "rule_c"):
            if rows and name in rows[0]:
                out[name] = per_class([{"pred": r[name], "gold": r["gold"]} for r in rows], labels)
        return out

    calls = [r["judge_calls"] for r in cov_rows if "judge_calls" in r]
    output = {
        "data": "S3-v2 development split + MultiHop-RAG batches 1-2 (used); hash split dev 37.5% / test 62.5%",
        "label_rule": "mechanical: gold fact strings present in the evidence; not human-reviewed",
        "items": {"coverage_total": len(items), "coverage_test": len(cov_rows),
                  "relevance_pairs": len(rel_rows), "relevance_test": len(test)},
        "thresholds": thresholds.__dict__ | {"lexical_dev_best": lex_dev_best, "fit_on": "dev only",
                                             "keep_rule": "highest score with dev recall >= 0.95",
                                             "high_rule": "lowest score with dev precision >= 0.90"},
        "relevance": relevance,
        "coverage_test": cov_report([r for r in cov_rows if r["split"] == "test"]),
        "coverage_dev": cov_report([r for r in cov_rows if r["split"] == "dev"]),
        "coverage_test_by_corpus": {c: cov_report([r for r in cov_rows if r["corpus"] == c and r["split"] == "test"])
                                    for c in ("s3", "mh")},
        "rule_selection": select_rule([r for r in cov_rows if r["split"] == "dev"], labels),
        "judge_cost": {
            "items": len(calls), "calls": sum(calls),
            "items_skipped_by_cascade": sum(1 for c in calls if c == 0),
            "median_ms_per_call": statistics.median(r["judge_ms"] for r in cov_rows if r.get("judge_calls")) if calls and sum(calls) else None,
            "mean_prompt_tokens_per_call": round(statistics.mean(r["judge_prompt_tokens"] for r in cov_rows if r.get("judge_calls")), 1) if calls and sum(calls) else None,
        },
        "rows": cov_rows,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("items", "thresholds", "relevance", "coverage_test", "judge_cost")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
