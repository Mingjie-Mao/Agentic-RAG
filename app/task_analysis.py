"""Cheap task signals used before asking a model to plan or judge."""

import re


CONFLICT_MARKERS = ("冲突", "不一致", "是否一致", "矛盾", "到底", "哪个规定", "哪些时段")
COMPOSITE_MARKERS = ("分别", "以及", "同时", "并且", "和两项")
VERSION_MARKERS = ("版本", "变更", "变化", "新版", "旧版", "历史", "之前")


def conflict_intent(text: str) -> bool:
    return any(marker in text for marker in CONFLICT_MARKERS)


def needs_document_diversity(text: str) -> bool:
    return conflict_intent(text) or any(marker in text for marker in COMPOSITE_MARKERS)


def version_intent(text: str) -> bool:
    return any(marker in text for marker in VERSION_MARKERS)


def acceptance_items(text: str) -> list[str]:
    """Expose obvious subquestions to the generator without another model call."""
    clean = text.strip().rstrip("？?")
    if not any(marker in clean for marker in COMPOSITE_MARKERS):
        return [clean]
    parts = [part.strip(" ，,。") for part in re.split(r"、|以及|并且", clean) if part.strip()]
    if len(parts) == 1 and ("分别" in clean or "和两项" in clean):
        parts = [part.strip(" ，,。") for part in re.split(r"\s+和\s+|和", clean) if part.strip()]
    return parts[:8] if len(parts) > 1 else [clean]
