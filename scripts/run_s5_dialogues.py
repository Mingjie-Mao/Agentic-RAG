"""S5-E1 comparison: does rewriting earlier questions into the current one help retrieval?

Retrieval only. Rewriting exists to change *what is retrieved*, so that is what is
measured first; generation is only worth running for a mode that already retrieves
better. Every turn is re-retrieved and re-authorised from scratch — nothing carries
over except the text the user typed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from app.clients import Models
from app.evaluation import (
    allowed_sources,
    canonical,
    identities,
    prepare_index,
    read_manifest,
    retrieval_scores,
)
from app.rewrite import rewrite_query

DATASET = "fixtures/s3-v2"
DEPTH = 50
GROUPS = {"A": "off", "B": "rule", "C": "llm", "D": "llm_history"}


def subject_carried(subject, query):
    return subject in query


def run_turn(dialogue, turn, history, mode, manifest, search, models, chunk_map, users):
    user = users[dialogue["user"]]
    started = time.monotonic()
    query, trace = rewrite_query(turn["question"], history, mode=mode, models=models)
    rewrite_seconds = time.monotonic() - started
    allowed = allowed_sources(manifest, user)
    version_ids = [source["sha256"] for source in allowed]
    readable = {source["id"] for source in allowed}
    vector = models.embed([query])[0]
    hits = search.retrieve_hybrid(query, vector, user.tenant_id, version_ids, DEPTH)
    ranked = list(dict.fromkeys(chunk_map[h["chunk_id"]]["document_id"] for h in hits))
    return {
        "dialogue": dialogue["id"],
        "turn": turn["turn"],
        "kind": turn["kind"],
        "subject": dialogue["subject"],
        "question": turn["question"],
        "query": query,
        "rewrite": trace,
        "rewrite_seconds": rewrite_seconds,
        "standalone_answerable": turn["standalone_answerable"],
        "retrieval": retrieval_scores(turn["source_ids"], ranked[:10]),
        "ranked_documents": ranked[:10],
        "permission_violations": sum(1 for key in ranked if key not in readable),
        # Did the rewrite recover the missing subject, keep the stated conditions,
        # and refrain from dragging the old subject into a new question?
        "subject_recovered": subject_carried(dialogue["subject"], query),
        "conditions_kept": all(value in query for value in turn.get("must_preserve", [])),
        "conditions_required": len(turn.get("must_preserve", [])),
        "contaminated": any(value in query for value in turn.get("must_not_mention", [])),
    }


def summarise(rows):
    follow = [r for r in rows if not r["standalone_answerable"]]
    shift = [r for r in rows if r["kind"] == "topic_shift"]
    carry = [r for r in rows if r["conditions_required"]]
    opening = [r for r in rows if r["kind"] == "opening"]
    recall = [r["retrieval"]["recall_at_5"] for r in rows if r["retrieval"]["recall_at_5"] is not None]
    follow_recall = [r["retrieval"]["recall_at_5"] for r in follow]
    return {
        "turns": len(rows),
        "recall_at_5": sum(recall) / len(recall),
        "follow_up_recall_at_5": sum(follow_recall) / len(follow_recall),
        "opening_recall_at_5": sum(r["retrieval"]["recall_at_5"] for r in opening) / len(opening),
        "subject_recovered": sum(r["subject_recovered"] for r in follow) / len(follow),
        "conditions_kept": (sum(r["conditions_kept"] for r in carry) / len(carry)) if carry else None,
        "topic_shift_contaminated": sum(r["contaminated"] for r in shift) / len(shift),
        "permission_violations": sum(r["permission_violations"] for r in rows),
        "rewrite_seconds_mean": sum(r["rewrite_seconds"] for r in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="A,B,C,D")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="artifacts/s5-rewrite")
    args = parser.parse_args()

    dialogues = json.loads(Path("fixtures/s5-dialogues/dialogues.json").read_text())[: args.limit]
    manifest = read_manifest(DATASET)
    snapshot, search, chunks, _ = prepare_index(DATASET)
    chunk_map = {chunk["id"]: chunk for chunk in chunks}
    users = identities()
    models = Models()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "snapshot": snapshot,
        "dataset": DATASET,
        "dialogues": len(dialogues),
        "retrieval_mode": "hybrid",
        "depth": DEPTH,
        "groups": {key: GROUPS[key] for key in args.groups.split(",")},
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path("app/rewrite.py"), Path(__file__).resolve().relative_to(Path.cwd())]
        },
        "scope": "retrieval only; rewriting drives retrieval, generation is not run here",
        "boundary": "only earlier question text crosses a turn; evidence never does",
    }
    summary = {"config": config, "results": {}}
    for group in args.groups.split(","):
        mode = GROUPS[group]
        rows = []
        path = out / f"group-{group}-{mode}.jsonl"
        with path.open("w") as handle:
            for number, dialogue in enumerate(dialogues, 1):
                history = []
                for turn in dialogue["turns"]:
                    row = run_turn(
                        dialogue, turn, history, mode, manifest, search, models, chunk_map, users
                    )
                    rows.append(row)
                    handle.write(canonical(row) + "\n")
                    history.append(turn["question"])
                if number % 10 == 0:
                    print(f"{group}({mode}): {number}/{len(dialogues)}", flush=True)
        summary["results"][group] = summarise(rows) | {"mode": mode, "per_turn": str(path)}
    Path(f"{args.out}-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    header = f"{'组':4s}{'模式':14s}{'整体R@5':>9s}{'追问R@5':>9s}{'开场R@5':>9s}{'主语恢复':>10s}{'条件保真':>10s}{'换题污染':>10s}{'耗时':>8s}"
    print(header)
    for group, value in summary["results"].items():
        kept = "—" if value["conditions_kept"] is None else f"{value['conditions_kept']:.3f}"
        print(
            f"{group:4s}{value['mode']:14s}{value['recall_at_5']:9.3f}{value['follow_up_recall_at_5']:9.3f}"
            f"{value['opening_recall_at_5']:9.3f}{value['subject_recovered']:10.3f}{kept:>10s}"
            f"{value['topic_shift_contaminated']:10.3f}{value['rewrite_seconds_mean']:8.2f}s"
        )


if __name__ == "__main__":
    main()
