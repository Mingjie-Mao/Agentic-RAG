"""Business coverage projection for audit/API; LangGraph owns execution recovery.

These lexical subgoal states describe evidence and answers, never the next graph node.
"""

from dataclasses import dataclass


@dataclass
class WorkflowState:
    items: list[dict]

    @classmethod
    def start(cls, labels: list[str]):
        return cls([{"item": label, "state": "pending", "evidence_refs": [], "claim_indices": []}
                    for label in labels])

    @classmethod
    def restore(cls, snapshot: dict, labels: list[str]):
        rows = snapshot.get("items", [])
        if [row.get("item") for row in rows] != labels:
            return cls.start(labels)
        return cls([{**row, "evidence_refs": list(row.get("evidence_refs", [])),
                     "claim_indices": list(row.get("claim_indices", []))} for row in rows])

    def observe(self, coverage: dict[str, list[str]], *, blocked: bool = False):
        for row in self.items:
            refs = coverage.get(row["item"], [])
            if row["state"] == "blocked" and not blocked and not refs:
                continue
            row.update(state="blocked" if blocked else "evidence_candidate" if refs else "evidence_missing",
                       evidence_refs=refs, claim_indices=[])

    def check_answers(self, missing: list[str], claims: list[dict], *, valid: bool):
        for row in self.items:
            if row["state"] == "blocked":
                continue
            row["claim_indices"] = [index for index, claim in enumerate(claims, 1)
                                    if set(row["evidence_refs"]) & set(claim["evidence_ids"])]
            row["state"] = "answer_covered" if valid and row["item"] not in missing and row["claim_indices"] else "answer_missing"

    @property
    def evidence_ready(self):
        return bool(self.items) and all(row["state"] == "evidence_candidate" for row in self.items)

    def snapshot(self):
        return {"items": self.items, "basis": "lexical_coverage_proxy; not semantic correctness"}
