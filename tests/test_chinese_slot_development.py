import copy

import pytest

from scripts.prepare_chinese_slot_development import merge_development, recipe
from scripts.research_review import packet


def test_synthetic_source_is_predesignated_development_without_human_labels():
    source = recipe()
    assert source["origin"] == "assistant_authored_synthetic"
    assert source["partition"] == "development_only" and source["independent_human_labels"] == 0
    assert len(source["documents"]) == 6
    assert sum(len(d["questions"]) for d in source["documents"]) == 18
    assert all("仅用于合成开发数据" in d["text"] for d in source["documents"])


def sources():
    old = packet([{"id": "old", "question": "旧问题", "families": ["old-doc"], "split": "test"}], "old")
    new = packet([{"id": "new", "question": "新问题", "families": ["new-doc"], "split": "dev",
                   "partition": "development_only"}], "new")
    return old, new


def test_merge_preserves_existing_test_inputs_and_requires_disjoint_sources():
    old, new = sources()
    before = copy.deepcopy(old)
    merged = merge_development(old, new)
    assert merged["items"][0] == old["items"][0] and old == before
    assert merged["reviews"] == []
    new["items"][0]["families"] = ["old-doc"]
    new = packet(new["items"], "overlapping")
    with pytest.raises(ValueError, match="overlaps"):
        merge_development(old, new)


def test_reviewed_or_non_dev_inputs_cannot_be_merged_into_a_fresh_review_packet():
    old, new = sources()
    new["reviews"] = [{"id": "new", "label": "supported"}]
    with pytest.raises(ValueError, match="unreviewed"):
        merge_development(old, new)
    new["reviews"] = []
    new["items"][0]["split"] = "test"
    new = packet(new["items"], "incorrect partition")
    with pytest.raises(ValueError, match="wholly predesignated"):
        merge_development(old, new)
