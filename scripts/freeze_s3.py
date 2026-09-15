"""Freeze reproducible inputs; human review remains a separately recorded prerequisite."""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from run_s3_baseline import model_versions

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="fixtures/s3")
args = parser.parse_args()
root = Path(args.dataset)
files = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(root.glob("*.json"))
    if p.name != "freeze.json"
}
record = {
    "files": files,
    "models": model_versions(),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "frozen_for_development_evaluation",
    "human_review": "pending; freeze records immutable inputs, not approval of gold quality",
    "holdout": "not run; source-only audit permitted, no tuning on held-out answers",
}
target = root / "freeze.json"
if target.exists():
    previous = json.loads(target.read_text())
    assert previous["files"] == files and previous["models"] == record["models"], "Frozen inputs changed"
    print("Existing freeze verified")
else:
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print("Frozen:", target)
