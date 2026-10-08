"""Hash-bound independent review packets and conservative statistical gates."""
import hashlib
import json
import math
import random


LABELS = {"relevance": {"relevant", "irrelevant", "unclear"},
          "coverage": {"supported", "partial", "unsupported", "contradicted", "unclear"},
          "claim": {"supported", "partial", "unsupported", "contradicted", "unclear"},
          "task": {"approved", "reject", "unclear"},
          "answer": {"correct", "incorrect", "unclear"},
          "disagreement": {"lexical_early_stop", "semantic_false_rejection", "splitter_error",
                           "judge_context_omission", "citation_mismatch", "missing_evidence", "unclear"}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def component_input_key(row):
    # Run IDs, row IDs and slot numbering are provenance, not independent facts.
    slot = {k: v for k, v in row["slot"].items() if k != "id"}
    return digest([row["kind"], row["corpus"], row["question"], slot, row["evidence"]])


def validate_component_partitions(rows):
    assigned = {}
    for row in rows:
        family, split = row.get("family_id"), row.get("split")
        if not family or split not in {"dev", "test"}:
            raise ValueError("component source family and dev/test partition required")
        docs = set(row.get("families", [])) | {p["document_id"] for p in row["evidence"] if p.get("document_id")}
        keys = [(row["corpus"], "family", family), (row["corpus"], "query", row["qid"])]
        keys += [(row["corpus"], "document", doc) for doc in docs]
        for key in keys:
            if key in assigned and assigned[key] != (family, split):
                raise ValueError("component query/document family crosses source families/splits")
            assigned[key] = family, split


def packet(items, purpose):
    if len({r["id"] for r in items}) != len(items):
        raise ValueError("duplicate review ids")
    return {"version": "independent-review-v2", "purpose": purpose, "source_sha256": digest(items),
            "items": items, "reviews": [], "status": "pending_independent_human_review"}


def validated_labels(source, reviewed, *, require_all=True):
    if source["items"] != reviewed.get("items") or source["source_sha256"] != digest(source["items"]):
        raise ValueError("review input changed")
    if reviewed.get("source_sha256") != source["source_sha256"]:
        raise ValueError("review is not bound to this input")
    ids = {r["id"]: r for r in source["items"]}
    grouped = {}
    for row in reviewed.get("reviews", []):
        rid = row.get("id")
        if rid not in ids or row.get("label") not in LABELS[ids[rid]["kind"]]:
            raise ValueError("unknown id or invalid label")
        if row.get("reviewer_type") != "human" or not row.get("reviewer", "").strip() or not row.get("reason", "").strip():
            raise ValueError("independent named human and reason required; assistant/model labels are not gold")
        if row.get("independent") is not True:
            raise ValueError("independence must be explicitly attested")
        grouped.setdefault(rid, []).append(row)
    labels = {}
    for rid, rows in grouped.items():
        if len({r["reviewer"] for r in rows}) != len(rows):
            raise ValueError("duplicate reviewer for item")
        if len({r["label"] for r in rows}) == 1:
            labels[rid] = rows[0]["label"]
        else:
            decisions = [r for r in reviewed.get("adjudications", []) if r.get("id") == rid]
            if len(decisions) != 1:
                raise ValueError("disputed item requires one independent adjudication")
            decision = decisions[0]
            if (decision.get("reviewer_type") != "human" or decision.get("independent") is not True
                    or not decision.get("reviewer") or decision["reviewer"] in {r["reviewer"] for r in rows}
                    or not decision.get("reason") or decision.get("label") not in LABELS[ids[rid]["kind"]]):
                raise ValueError("invalid independent dispute adjudication")
            labels[rid] = decision["label"]
    if require_all and set(labels) != set(ids):
        raise ValueError(f"independent review incomplete: {len(labels)}/{len(ids)}")
    return labels


def group_splits(rows):
    """Connected query/document/version families; include distractors to avoid leakage.

    Large components stay together even if this leaves no useful test split. Never
    split a family merely to manufacture a sufficient calibration sample.
    """
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)
    for row in rows:
        keys = ["q:" + row["corpus"] + ":" + row["qid"]] + ["d:" + doc for doc in row["families"]]
        for key in keys[1:]:
            union(keys[0], key)
    for row in rows:
        family = find("q:" + row["corpus"] + ":" + row["qid"])
        row["family_id"] = family
        row["split"] = "dev" if int(digest(family)[:8], 16) % 10 < 4 else "test"
    return rows


def wilson(success, total):
    if not total:
        return [None, None]
    z = 1.95996398454
    p = success / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0, center-half), min(1, center+half)]


def binary(rows):
    tp = sum(r[0] and r[1] for r in rows)
    fp = sum(not r[0] and r[1] for r in rows)
    fn = sum(r[0] and not r[1] for r in rows)
    return {"n": len(rows), "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "precision_ci": wilson(tp, tp+fp), "recall_ci": wilson(tp, tp+fn)}


def macro_f1(rows, labels=("supported", "partial", "unsupported", "contradicted")):
    scores = []
    for label in labels:
        tp = sum(g == p == label for g, p in rows)
        fp = sum(g != label and p == label for g, p in rows)
        fn = sum(g == label and p != label for g, p in rows)
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return sum(scores)/len(labels)


def paired_ci(differences, *, groups=None, seed=42, trials=4000):
    if not differences:
        return [None, None]
    # Bootstrap whole source families when evaluating component variants.
    blocks = {}
    for index, value in enumerate(differences):
        blocks.setdefault(groups[index] if groups else index, []).append(value)
    keys = list(blocks)
    if len(keys) < 2:
        return [None, None]
    rng = random.Random(seed)
    means = []
    for _ in range(trials):
        values = [v for key in rng.choices(keys, k=len(keys)) for v in blocks[key]]
        means.append(sum(values)/len(values))
    means.sort()
    return [means[int(trials*.025)], means[min(trials-1, int(trials*.975))]]
