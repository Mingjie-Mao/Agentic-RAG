"""Does the subgoal splitter find the number of things a question asks?

Labels are in fixtures/splitter/labels.json (question text only; no answers are read).
Rules were changed against the dev items; the test items are reported, not tuned on.
Hard 30 is not part of this set.
"""

import json
from pathlib import Path

from agent.planner import subgoals

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "fixtures/splitter/labels.json").read_text())
    report, errors = {}, []
    for split in ("dev", "test"):
        rows = [r for r in data["items"] if r["split"] == split]
        got = {r["id"]: len(subgoals(r["question"])) for r in rows}
        multi = [r for r in rows if r["expected_asks"] > 1]
        single = [r for r in rows if r["expected_asks"] == 1]
        report[split] = {
            "questions": len(rows),
            "exact_count": sum(got[r["id"]] == r["expected_asks"] for r in rows),
            "multi_ask_found": f"{sum(got[r['id']] >= 2 for r in multi)}/{len(multi)}",
            "single_kept_single": f"{sum(got[r['id']] == 1 for r in single)}/{len(single)}",
        }
        errors += [{"split": split, "id": r["id"], "expected": r["expected_asks"], "got": got[r["id"]],
                    "question": r["question"]} for r in rows if got[r["id"]] != r["expected_asks"]]
    out = {"report": report, "errors": errors}
    (ROOT / "artifacts/splitter-eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
