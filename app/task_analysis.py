"""Cheap task signals used before asking a model to plan or judge."""

import re


CONFLICT_MARKERS = ("冲突", "不一致", "是否一致", "矛盾", "到底", "哪个规定", "哪些时段")
COMPOSITE_MARKERS = ("分别", "以及", "同时", "并且", "和两项")
VERSION_MARKERS = ("版本", "变更", "变化", "新版", "旧版", "历史", "之前")
JUDGMENT_MARKERS = ("是否", "是不是", "有没有", "能否", "会不会", "对不对", "一致吗", "对吗")
# A question that opens with an auxiliary verb expects a verdict, not a description.
# The last clause matters too: "After …, was there a change in …?" asks for one.
_AUXILIARY = re.compile(
    r"^\s*(do|does|did|is|are|was|were|has|have|had|can|could|will|would|should)\b",
    re.IGNORECASE,
)
# "What institution, … , is the focal point …?" ends in an auxiliary clause but asks
# for a name. A leading wh-word settles that before the clause rule is consulted.
_WH_WORD = re.compile(r"^\s*(what|which|who|whom|whose|when|where|why|how)\b", re.IGNORECASE)
# English inverts the auxiliary for a yes/no question, and a long fronted adjunct
# pushes the inversion into the middle: "After …, did The Verge's report … align?"
_INVERSION = re.compile(
    r",\s*(do|does|did|is|are|was|were|has|have|had|can|could|will|would|should)\b",
    re.IGNORECASE,
)
# Naming more than one source, or asking to compare them, means the evidence has to
# come from more than one document — a budget question, not a phrasing question.
_SOURCE_NOUN = re.compile(r"\b(article|report|piece|story|post|source)s?\b", re.IGNORECASE)
_MULTI_SOURCE = re.compile(
    r"\b(both|respectively|compare[ds]?|comparison|difference|each of|either)\b", re.IGNORECASE
)
_ENGLISH_NEXT_QUESTION = re.compile(
    r"\s*(?:;|\?\s+|,\s+and\s+|\s+and\s+(?=what\b|which\b|who\b|when\b|where\b|why\b|how\b))(?=(?:and\s+)?(?:what|which|who|when|where|why|how|"
    r"do|does|did|is|are|was|were|can|could|has|have|had)\b)",
    re.IGNORECASE,
)
_EXPLICIT_CHOICE = re.compile(
    r"\b(?:larger\s+or\s+smaller|greater\s+or\s+less|more\s+or\s+less|"
    r"earlier\s+or\s+later|before\s+or\s+after|consistent\s+or\s+inconsistent)\b",
    re.IGNORECASE,
)


def conflict_intent(text: str) -> bool:
    return any(marker in text for marker in CONFLICT_MARKERS)


def needs_document_diversity(text: str) -> bool:
    return conflict_intent(text) or any(marker in text for marker in COMPOSITE_MARKERS)


def judgment_intent(text: str) -> bool:
    """True when the expected answer is a verdict (yes / no), not a description."""
    clean = (text or "").strip()
    if _EXPLICIT_CHOICE.search(clean):
        return False
    if any(marker in clean for marker in JUDGMENT_MARKERS):
        return True
    if not clean.endswith(("?", "？")) or _WH_WORD.match(clean):
        return False
    last = clean.rstrip("?？").rsplit(",", 1)[-1]
    return bool(
        _AUXILIARY.match(clean)
        or _AUXILIARY.match(last.strip())
        or _INVERSION.search(clean)
    )


def multi_source_intent(text: str) -> bool:
    """True when the question itself says the answer spans several documents."""
    clean = text or ""
    return bool(_MULTI_SOURCE.search(clean)) or len(_SOURCE_NOUN.findall(clean)) >= 2


def version_intent(text: str) -> bool:
    return any(marker in text for marker in VERSION_MARKERS) or bool(re.search(
        r"\b(?:versions?|revisions?|historical|previous|older|newer)\b", text or "", re.IGNORECASE
    )) or bool(
        len(re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)|(?:\d{4}\s*年\s*)?\d{1,2}\s*月\s*\d{1,2}\s*日", text or "")) >= 2
        and re.search(r"policy|regulation|政策|规定|制度|工单", text or "", re.IGNORECASE)
    )


def historical_route_intent(text: str) -> bool:
    """A current SDK version is not a request for a document's version history."""
    clean = text or ""
    return bool(re.search(
        r"历史|旧版|之前|变更|演变|(?:比较|对比).{0,20}版本|"
        r"\b(?:previous|older|historical)\b.{0,30}\b(?:versions?|revisions?|polic(?:y|ies)|rules?|standards?)\b|"
        r"\brevisions?\b|\bcompare\b.{0,40}\bversions?\b",
        clean, re.IGNORECASE,
    )) or bool(
        len(re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)|(?:\d{4}\s*年\s*)?\d{1,2}\s*月\s*\d{1,2}\s*日", clean)) >= 2
        and re.search(r"policy|regulation|政策|规定|制度|工单", clean, re.IGNORECASE)
    )


def acceptance_items(text: str) -> list[str]:
    """Expose obvious subquestions to the generator without another model call."""
    clean = text.strip().rstrip("？?")
    english = english_subquestions(clean)
    if len(english) > 1:
        return english
    if not any(marker in clean for marker in COMPOSITE_MARKERS):
        return [clean]
    parts = [part.strip(" ，,。") for part in re.split(r"、|以及|并且", clean) if part.strip()]
    if len(parts) == 1 and ("分别" in clean or "和两项" in clean):
        parts = [part.strip(" ，,。") for part in re.split(r"\s+和\s+|和", clean) if part.strip()]
    return parts[:8] if len(parts) > 1 else [clean]


def english_subquestions(text: str) -> list[str]:
    """Split only explicit independent English asks; never split 'A and B both…'."""
    if re.search(r"[一-鿿]", text or ""):
        return [text.strip().rstrip("?")]
    parts = [part.strip(" ,;?") for part in _ENGLISH_NEXT_QUESTION.split(text or "") if part.strip(" ,;?")]
    return parts[:6] if len(parts) > 1 else [text.strip().rstrip("?")]


def explicit_choice(text: str) -> str | None:
    """Return an explicit pair of answer options, without deciding which is true."""
    match = _EXPLICIT_CHOICE.search(text or "")
    return match.group() if match else None
