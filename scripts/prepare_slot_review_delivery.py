"""Reduce duplicate review work before any labels exist; pilot uses dev only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_runtime import atomic_json  # noqa: E402
from research_review import component_input_key, digest, packet, validate_component_partitions  # noqa: E402


def delivery(source):
    if source["source_sha256"] != digest(source["items"]) or source.get("reviews") or source.get("adjudications"):
        raise ValueError("only immutable unreviewed inputs may be deduplicated")
    validate_component_partitions(source["items"])
    unique, mapping = {}, {}
    for row in source["items"]:
        if row["kind"] not in {"relevance", "coverage", "claim"}:
            raise ValueError("component review inputs only")
        key = component_input_key(row)
        prior = unique.get(key)
        if prior and (prior["family_id"], prior["split"]) != (row["family_id"], row["split"]):
            raise ValueError("identical component input crosses family/split")
        unique.setdefault(key, row)
        mapping[row["id"]] = unique[key]["id"]
    rows = list(unique.values())
    expanded = packet(rows, "distinct actual-input component review; no labels inferred")
    expanded["deduplication"] = {"parent_source_sha256": source["source_sha256"],
                                 "representative_by_id": mapping, "removed": len(source["items"])-len(rows)}
    pilot_rows = []
    for kind, target in (("relevance", 100), ("coverage", 80), ("claim", 80)):
        pools = [sorted((r for r in rows if r["kind"] == kind and r["corpus"] == corpus
                        and r["split"] == "dev"), key=lambda r: digest(r["id"])) for corpus in ("s3", "mh")]
        picked = []
        while len(picked) < target and any(pools):
            for pool in pools:
                if pool and len(picked) < target:
                    picked.append(pool.pop(0))
        pilot_rows.extend(picked)
    pilot = packet(pilot_rows, "development-only independent annotation pilot; no test inputs")
    pilot["parent_source_sha256"] = expanded["source_sha256"]
    return expanded, pilot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    paths = [args.output_dir / n for n in ("expanded-distinct.json", "pilot.json", "delivery-summary.json")]
    if any(path.exists() for path in paths):
        raise ValueError("delivery already exists; never overwrite independent review inputs")
    source = json.loads(args.source.read_text())
    expanded, pilot = delivery(source)
    summary = {"version": "slot-review-delivery-v1", "independent_human_labels": 0,
               "raw_rows": len(source["items"]), "distinct_rows": len(expanded["items"]),
               "duplicate_rows_removed": expanded["deduplication"]["removed"],
               "distinct_counts": dict(Counter(r["kind"] for r in expanded["items"])),
               "pilot_counts": dict(Counter(r["kind"] for r in pilot["items"])),
               "pilot_dev_only": all(r["split"] == "dev" for r in pilot["items"]),
               "strata": dict(Counter(f'{r["kind"]}:{r["corpus"]}:{r["split"]}' for r in expanded["items"])),
               "source_sha256": source["source_sha256"], "expanded_sha256": expanded["source_sha256"]}
    for path, value in zip(paths, (expanded, pilot, summary), strict=True):
        atomic_json(path, value)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
