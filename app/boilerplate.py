"""Page furniture that web articles carry into their text.

Newsletter sign-up forms, share bars, app promotions and affiliate disclosures are not
content: they rank for nothing a user asks, but a 440-character sign-up form at the top
of an article fills its first chunk, and that chunk is then retrieved and cited as
evidence. This removes such paragraphs before chunking.

It is deliberately narrow. A paragraph is dropped only when it is short and consists of
page furniture — a marker alone is not enough in a long paragraph, because an article
*about* newsletters or subscriptions is content. The original file is untouched; only
what gets chunked changes, and the pipeline records that the filter ran.
"""

import re

# Words that only *suggest* furniture: articles about sign-up offers or subscriber
# counts use them too, so they never drop a paragraph on their own.
_WEAK = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (r"\bsign(?:ing)? up\b", r"\bsubscri(?:be|bers?|ption)\b", r"\binbox\b", r"\bfollowers\b")
]
_STRONG = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bnewsletters?\b",
        r"valid email",
        r"privacy notice",
        r"\{\{\s*[#^/]?\w+\s*\}\}",
        r"click here",
        r"^advertisement$|\badvertisement\b.{0,40}\bad\b",
        r"article continues below",
        r"facebook\s+twitter",
        r"affiliate links?",
        r"earn (?:a )?commission",
        r"get the .{0,30}\bapp\b",
        r"^©|\bcopyright of\b",
        r"skip past",
    )
]


def is_web_boilerplate(text: str) -> bool:
    clean = " ".join(text.split())
    if not clean or clean.startswith("#") or len(clean) > 700:
        return False
    strong = sum(bool(marker.search(clean)) for marker in _STRONG)
    weak = sum(bool(marker.search(clean)) for marker in _WEAK)
    # A disclosure tacked onto a real sentence ("Spider-Man 2 is out Oct. 20 on PS5.
    # We may earn a commission…") is kept: every sentence has to be furniture.
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", clean) if part]
    if not all(any(m.search(part) for m in _STRONG + _WEAK) for part in sentences):
        return False
    if len(clean) <= 120:
        return strong >= 1
    return strong >= 1 and strong + weak >= 2


def strip_web_boilerplate(passages):
    """Return the passages worth chunking and how many were dropped."""
    kept = [passage for passage in passages if not is_web_boilerplate(passage.text)]
    return kept, len(passages) - len(kept)
