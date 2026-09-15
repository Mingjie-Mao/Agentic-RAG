"""Author the multi-turn follow-up set for S5-E1.

A follow-up only tests rewriting if it cannot be answered on its own, so that is
asserted rather than assumed. Every turn is derived from s3-v2 development
documents; no held-out family is touched.
"""

import json
from pathlib import Path

from app.evaluation import fact_present, prepare_snapshot

DATASET = Path("fixtures/s3-v2")
ROOT = Path("fixtures/s5-dialogues")
TURNS_PER_DIALOGUE = 4


def openings(name, field, value):
    return f"星桥{name}的{field}是多少？"


def build():
    topics = [
        t
        for t in json.loads(Path("fixtures/s3_topics.json").read_text())["topics"]
        if t["split"] == "development"
    ]
    manifest = json.loads((DATASET / "manifest.json").read_text())
    by_id = {row["id"]: row for row in manifest}
    snapshot, _, blocks = prepare_snapshot(str(DATASET))
    text_of = {key: "\n".join(b["text"] for b in value) for key, value in blocks.items()}

    dialogues = []
    for topic_index, topic in enumerate(topics):
        group = topic["group"]
        user = "xq-engineer" if group == "engineering" else "xq-support"
        # Item 3 of each topic is group-private in s3; keep dialogues on public items.
        usable = [(i, item) for i, item in enumerate(topic["items"]) if i != 2]
        for order, (item_index, item) in enumerate(usable[:4]):
            name, field_a, value_a, field_b, value_b = item
            key = f"eval-{topic['id']}-{item_index + 1}"
            assert key in by_id and by_id[key]["split"] == "development", key
            # The shift turn must genuinely change subject, and stay readable by the
            # same user, or it tests nothing. Search for the next topic in this group.
            other = next(
                topics[(topic_index + step) % len(topics)]
                for step in range(1, len(topics))
                if topics[(topic_index + step) % len(topics)]["group"] == group
                and topics[(topic_index + step) % len(topics)]["id"] != topic["id"]
            )
            other_item = other["items"][0]
            other_key = f"eval-{other['id']}-1"
            assert by_id[other_key]["split"] == "development", other_key
            assert other_item[0] != name, "Topic shift must change the subject"

            turns = [
                {
                    "turn": 1,
                    "kind": "opening",
                    "question": openings(name, field_a, value_a),
                    "resolved_question": openings(name, field_a, value_a),
                    "standalone_answerable": True,
                    "expected": "answered",
                    "facts": [value_a],
                    "source_ids": [key],
                },
                {
                    "turn": 2,
                    "kind": "ellipsis",
                    "question": f"那{field_b}呢？",
                    "resolved_question": f"星桥{name}的{field_b}是多少？",
                    "standalone_answerable": False,
                    "expected": "answered",
                    "facts": [value_b],
                    "source_ids": [key],
                },
                {
                    "turn": 3,
                    "kind": "condition_carry" if order % 2 else "anaphora",
                    "question": (
                        f"这个 {value_a} 从什么时候开始执行？" if order % 2 else "超出这个限制要怎么办？"
                    ),
                    "resolved_question": (
                        f"星桥{name}的{field_a} {value_a} 从什么时候开始执行？"
                        if order % 2
                        else f"星桥{name}超出规定范围要怎么办？"
                    ),
                    "standalone_answerable": False,
                    "expected": "answered",
                    "facts": ["2026 年 9 月 1 日"] if order % 2 else ["例外申请"],
                    "source_ids": [key],
                    "must_preserve": [value_a] if order % 2 else [],
                },
                {
                    "turn": 4,
                    "kind": "topic_shift",
                    "question": openings(other_item[0], other_item[1], other_item[2]),
                    "resolved_question": openings(other_item[0], other_item[1], other_item[2]),
                    "standalone_answerable": True,
                    "expected": "answered",
                    "facts": [other_item[2]],
                    "source_ids": [other_key],
                    "must_not_mention": [name],
                },
            ]
            dialogues.append(
                {
                    "id": f"D{len(dialogues) + 1:03}",
                    "user": user,
                    "family": topic["id"],
                    "subject": name,
                    "turns": turns,
                    "review": {
                        "status": "requires_human_review",
                        "author": "Codex",
                        "source_check": "facts located in the cited s3-v2 development documents",
                    },
                }
            )

    audit(dialogues, text_of, by_id)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "dialogues.json").write_text(json.dumps(dialogues, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "meta.json").write_text(
        json.dumps(
            {
                "version": "s5-dialogues-v1",
                "derived_from": str(DATASET),
                "snapshot": snapshot,
                "dialogues": len(dialogues),
                "turns": sum(len(d["turns"]) for d in dialogues),
                "follow_up_turns": sum(
                    1 for d in dialogues for t in d["turns"] if not t["standalone_answerable"]
                ),
                "split": "development only; no held-out family is used",
                "purpose": "S5-E1: measure anaphora resolution, condition preservation and rewrite harm",
                "review_state": "source-audited draft; independent human review not yet performed",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        f"s5-dialogues: {len(dialogues)} 组会话 / {sum(len(d['turns']) for d in dialogues)} 轮，"
        f"其中 {sum(1 for d in dialogues for t in d['turns'] if not t['standalone_answerable'])} 轮无法独立回答"
    )


def audit(dialogues, text_of, by_id):
    """A follow-up that already names its subject would pass without any rewriting."""
    for dialogue in dialogues:
        subject = dialogue["subject"]
        for turn in dialogue["turns"]:
            for key in turn["source_ids"]:
                assert by_id[key]["split"] == "development", f"{dialogue['id']} uses a held-out source"
            combined = "\n".join(text_of[key] for key in turn["source_ids"])
            missing = [fact for fact in turn["facts"] if not fact_present(fact, combined)]
            assert not missing, f"{dialogue['id']} turn {turn['turn']} facts not in source: {missing}"
            if not turn["standalone_answerable"]:
                assert subject not in turn["question"], (
                    f"{dialogue['id']} turn {turn['turn']} names its own subject, "
                    "so it does not require rewriting and cannot measure one"
                )
                assert subject in turn["resolved_question"], (
                    f"{dialogue['id']} turn {turn['turn']} resolved form lost the subject"
                )
            for value in turn.get("must_preserve", []):
                assert value in turn["question"] and value in turn["resolved_question"], (
                    f"{dialogue['id']} turn {turn['turn']} must carry {value} through rewriting"
                )
            for value in turn.get("must_not_mention", []):
                assert value not in turn["resolved_question"], (
                    f"{dialogue['id']} turn {turn['turn']} is a topic shift; "
                    f"the previous subject {value} must not be carried over"
                )
    follow_ups = [t for d in dialogues for t in d["turns"] if not t["standalone_answerable"]]
    assert len(follow_ups) >= 2 * len(dialogues), "Each dialogue needs at least two follow-up turns"


if __name__ == "__main__":
    build()
