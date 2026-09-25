"""Conservative exact-value checks for facts with an unambiguous source label.

These checks never invent a value: a slot exists only when an authorized evidence
line contains the requested label and exactly one matching value. Ambiguous lines
are left to the model and the ordinary citation validator.
"""

import re


_EVENT_REQUEST = re.compile(
    r"\bevent[_ -]?id\b|事件\s*(?:ID|标识符)|去重字段|deduplication\s+field",
    re.IGNORECASE,
)
_EVENT_LABEL = re.compile(r"\bevent[_ -]?id\b", re.IGNORECASE)
_EVENT_VALUE = re.compile(r"\bevent[_ -]?id\b\s*[:=：]\s*([A-Za-z0-9][A-Za-z0-9_.:-]{2,})", re.IGNORECASE)
_ROLLBACK_REQUEST = re.compile(r"回滚|rollback", re.IGNORECASE)
_THRESHOLD_REQUEST = re.compile(r"阈值|门槛|threshold|trigger", re.IGNORECASE)
_TARGET_REQUEST = re.compile(r"目标版本|回滚到.+版本|rollback target|target version", re.IGNORECASE)
_TARGET_LINE = re.compile(r"回滚(?:使用|到|至)\s*([^，。；;\n]{2,40})")
_PERCENT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*%(?![\w])")
_CHINESE_PERCENT = re.compile(r"百分之[零一二三四五六七八九十]+")
_CHINESE_NUMBERS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _percent_number(value: str) -> float | None:
    if value.startswith("百分之"):
        number = value.removeprefix("百分之")
        return float(_CHINESE_NUMBERS[number]) if number in _CHINESE_NUMBERS else None
    return float(value.replace(" ", "").removesuffix("%"))


def required_slots(question: str, evidence: list[dict]) -> list[dict]:
    """Extract only uniquely labelled values relevant to the user's request."""
    slots = []
    if _EVENT_REQUEST.search(question):
        for row in evidence:
            values = {value.rstrip(".,;:") for value in _EVENT_VALUE.findall(row["text"])}
            if len(values) == 1:
                slots.append({"field": "event_id", "value": values.pop(), "chunk_id": row["chunk_id"]})
        if not slots:
            source = next((row for row in evidence if _EVENT_LABEL.search(row["text"])), None)
            if source:
                slots.append({"field": "event_id_field", "value": "event_id", "chunk_id": source["chunk_id"]})
    if _ROLLBACK_REQUEST.search(question) and _THRESHOLD_REQUEST.search(question):
        for row in evidence:
            for line in row["text"].splitlines():
                if not _ROLLBACK_REQUEST.search(line):
                    continue
                values = {re.sub(r"\s+", "", item) for item in _PERCENT.findall(line)}
                values.update(_CHINESE_PERCENT.findall(line))
                if len(values) == 1:
                    slots.append({"field": "rollback_threshold", "value": values.pop(), "chunk_id": row["chunk_id"]})
    if _ROLLBACK_REQUEST.search(question) and _TARGET_REQUEST.search(question):
        for row in evidence:
            for line in row["text"].splitlines():
                match = _TARGET_LINE.search(line)
                if not match:
                    continue
                value = match.group(1).strip()
                if "版本" in value or "镜像" in value:
                    slots.append({"field": "rollback_target", "value": value, "chunk_id": row["chunk_id"]})
    # Different documents may have different values for the same label. In that
    # case there is no unconditional fact to force into the answer.
    by_field = {}
    for slot in slots:
        by_field.setdefault(slot["field"], set()).add(slot["value"])
    return [slot for slot in slots if len(by_field[slot["field"]]) == 1]


def missing_slots(slots: list[dict], claims: list[dict]) -> list[dict]:
    answer = " ".join(claim["text"] for claim in claims)
    missing = []
    for slot in slots:
        value = slot["value"]
        if slot["field"] == "rollback_threshold":
            wanted = _percent_number(value)
            candidates = _PERCENT.findall(answer) + _CHINESE_PERCENT.findall(answer)
            found = wanted is not None and any(_percent_number(item) == wanted for item in candidates)
        else:
            found = (value in answer) if re.search(r"[\u3400-\u9fff]", value) else bool(
                re.search(r"(?<![\w.])" + re.escape(value) + r"(?![\w.])", answer)
            )
        if not found:
            missing.append(slot)
    return missing
