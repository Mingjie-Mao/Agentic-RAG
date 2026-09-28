"""Were the frozen holdout's failures missing the fact from the model's context?

§11.1 attributed the 13 literal misses of the one-shot holdout to generation, because
12 of 13 had the gold *document* in Recall@5. That is a document-level reading. This
replays the run's own stored evidence handles: it resolves the four chunks that were
admitted to generation from the evaluation snapshot the run was indexed from, and asks
whether each gold fact string occurs in them. Nothing is re-run or re-scored; the
holdout stays one-shot.

As a check on the method itself, the same test is applied to the questions the run got
right — if they did not have their facts in context either, the method would be wrong.
"""

import collections
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/s7-holdout/holdout-hybrid-generated.jsonl"
SUMMARY = ROOT / "artifacts/s7-holdout-summary.json"
QUESTIONS = ROOT / "fixtures/s3-v2/questions.json"
OUT = ROOT / "artifacts/holdout-passage-attribution.json"
ADMITTED = 4  # the frozen holdout configuration admitted the top four chunks


def compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def main():
    snapshot = json.loads(SUMMARY.read_text())["config"]["snapshot"]
    chunks_file = ROOT / ".runtime/evaluation" / snapshot / "chunks.json"
    if not chunks_file.exists():
        raise SystemExit(f"evaluation snapshot {snapshot[:12]} is not on this machine; run make s3-freeze")
    chunks = {chunk["id"]: chunk for chunk in json.loads(chunks_file.read_text())}
    questions = {q["id"]: q for q in json.loads(QUESTIONS.read_text())}
    counts, failures = collections.Counter(), []
    for line in RUN.read_text().splitlines():
        row = json.loads(line)
        facts = questions[row["id"]]["facts"]
        if not facts:
            continue
        context = compact(" ".join(chunks[hit["chunk_id"]]["text"] for hit in row["hits"][:ADMITTED]))
        present = [compact(fact) in context for fact in facts]
        state = "all_facts_in_context" if all(present) else "fact_missing_from_context"
        counts[f"{row['failure_class']}|{state}"] += 1
        if row["failure_class"] != "ok":
            failures.append(
                {
                    "id": row["id"],
                    "failure_class": row["failure_class"],
                    "status": row["answer"]["status"],
                    "facts_in_context": f"{sum(present)}/{len(present)}",
                }
            )
    output = {
        "run": RUN.relative_to(ROOT).as_posix(),
        "snapshot": snapshot,
        "admitted_chunks": ADMITTED,
        "method": "exact gold fact string (whitespace removed) inside the admitted chunk text",
        "counts": dict(sorted(counts.items())),
        "failures": failures,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
