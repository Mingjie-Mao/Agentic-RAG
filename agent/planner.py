"""Deterministic planning for the Agent workflow.

Every function here is a pure function of the goal text, or of evidence the caller has
already authorized. There is no model call, no stored state and no corpus-specific
vocabulary: this module decides *where to look next* and *what is still missing*,
never what is true. Anything it produces is either a retrieval query or a subgoal
label, both of which are re-checked downstream by ACL, citation validation and the
benchmark scorer.
"""

import re

from app.task_analysis import english_subquestions

_CJK = re.compile(r"[一-鿿]")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.@-]*|\d+(?:[.:]\d+)*%?")
# "并" starts a new subgoal only in front of a verb: 并说明 / 并给出 do, 并且 / 并替代 do not.
_ACTION_VERB = "查核检说给列报计确找对提算"

_CLAUSE = re.compile(
    r"[；;。？?！!\n]+"
    r"|，\s*(?=先|再|然后|接着|之后|最后|同时|以及|并)"
    r"|、"
    r"|以及"
    r"|并(?=[" + _ACTION_VERB + r"])"
)
# In "只有 X 才继续 Y" / "如果 X 就 Y", X is the previous hop's observation; the subgoal is Y.
_CONDITION = re.compile(r"^(只有|如果|若|当|假如)[^。；]{0,40}?[，,]?\s*(才|则|就)(继续)?")
_LEADING = re.compile(r"^(先|再|然后|接着|之后|最后|同时|并且|并|以及|又|还要|还需|请|则|才)+")
# Only one arm of a branch is ever taken, so requiring both to be answered is wrong.
_BRANCH = re.compile(r"^(否则|若|如果|假如|不一致|一致)")
_INTERROGATIVE = (
    "什么", "多少", "多久", "哪", "是否", "如何", "怎样", "怎么", "为何", "原因", "几", "?", "？",
)

# Question scaffolding. Removing it changes noise, not retrieval intent.
_LOW_INFORMATION = (
    "请问", "麻烦", "帮我", "我想知道", "想问一下", "告诉我", "是什么意思", "什么意思",
    "是什么", "有哪些", "哪一个", "哪两个", "哪些", "什么", "多久", "多少", "几次",
    "怎样", "怎么", "如何", "应该", "该按", "需要", "可以", "能否", "是否", "那个",
    "这个", "这种", "一下", "到底", "究竟", "突然", "真的", "以后", "的话", "而言",
    "给出", "说明", "客户说", "有人说", "请", "呢", "吗", "啊", "了", "的",
)
# Instruction verbs say what to do, not what is true; neither retrieval nor the
# coverage check needs them.
_INSTRUCTION_VERBS = (
    "核对", "查明", "查询", "查找", "检查", "列出", "计算", "比较", "找出", "确定",
    "继续", "同时", "分别", "然后", "先", "再", "查",
)
# Verbs that make a fragment a request. They mark an item as asked, but never
# become query terms.
_REQUEST_VERBS = _INSTRUCTION_VERBS + (
    "说明", "给出", "列举", "报告", "提供", "返回", "解释", "核实", "判断",
)
_STRIP = " ，,。；;：:？?！!、　"


def _content(text: str) -> str:
    remainder = text or ""
    for word in _LOW_INFORMATION + _INSTRUCTION_VERBS:
        remainder = remainder.replace(word, " ")
    return remainder


def _clean(text: str) -> str:
    return text.strip(_STRIP)


def subgoals(goal: str) -> list[str]:
    """Split a compound goal into checkable subgoals, newest scaffolding removed.

    A goal without an explicit sequence or enumeration marker stays one item, so short
    factual questions keep exactly the behaviour they had before.
    """
    if not _CJK.search(goal or ""):
        return english_subquestions(goal)
    fragments = []
    for raw in _CLAUSE.split(goal or ""):
        fragment = _clean(raw)
        if not fragment:
            continue
        fragment = _clean(_CONDITION.sub("", fragment))
        fragment = _clean(_LEADING.sub("", fragment))
        if len(fragment) < 3 or not (_CJK.search(fragment) or _TOKEN.search(fragment)):
            continue
        if _BRANCH.match(fragment):
            continue
        asked = any(word in fragment for word in _INTERROGATIVE) or any(
            verb in fragment for verb in _REQUEST_VERBS
        )
        fragments.append((fragment, asked))
    items = []
    for index, (fragment, asked) in enumerate(fragments):
        # In "A、B 分别是多少" or "给出 A、B、C" the request sits on one member of the
        # enumeration only. A short noun phrase next to a real question is a parallel
        # subgoal; a long background statement is not, and still reaches the model as
        # part of the whole goal.
        enumerated = len(fragment) <= 15 and any(
            row[1] for position, row in enumerate(fragments) if position != index
        )
        if asked or enumerated:
            items.append(fragment)
    if len(items) < 2:
        return [_clean(goal or "")]
    return items[:6]


def keywords(text: str) -> str:
    """Drop question scaffolding, keep entities, codes, metric names and numbers."""
    remainder = _content(text)
    parts = [part for part in re.split(r"[^0-9A-Za-z一-鿿%.:_@-]+", remainder) if part]
    return " ".join(dict.fromkeys(parts))


def recovery_query(goal: str, previous: str) -> str | None:
    """One controlled rewrite after a search came back empty.

    The rewrite is lexical: question scaffolding is removed and the remaining
    entities, error codes and metric names are kept. It is used at most once per task
    and must differ from the query that missed, otherwise there is nothing to retry.
    """
    candidate = keywords(goal)
    if not candidate or _normalized(candidate) == _normalized(previous):
        candidate = " ".join(_TOKEN.findall(goal or "")) or candidate
    if not candidate or _normalized(candidate) == _normalized(previous):
        return None
    return candidate[:500]


_EN_STOP = set(
    "a an the and or but if for of to in on at by with from as is are was were be been "
    "do does did has have had can could will would should what which who when where why how "
    "article report piece story source both same each other this that these those".split()
)


def retrieval_quality(goal: str, refs: list[str], matches: list[dict]) -> str:
    """Cheap, conservative retrieval grade for one controlled retry.

    A nonempty result with no visible snippets is *unknown*, not irrelevant. For
    visible snippets, an off-topic verdict requires at least four meaningful query
    terms and zero overlap with the best three passages (including their titles).
    This grade only decides whether to search again; it never proves an answer.
    """
    if not refs:
        return "empty"
    if not matches:
        return "unknown"
    wanted = {
        token.lower()
        for token in _TOKEN.findall(goal or "")
        if len(token) >= 4 and token.lower() not in _EN_STOP
    }
    wanted |= _grams(_content(goal)) - _tokens(goal)
    if len(wanted) < 4:
        return "unknown"
    for row in matches[:3]:
        material = f"{row.get('title', '')} {row.get('snippet', '')}"
        if wanted & _grams(material):
            return "candidate"
    return "irrelevant"


def carry_forward_query(items: list[str], evidence: list[dict], limit: int = 400) -> str:
    """Build the second-hop query from the missing subgoals plus what hop one returned.

    A latent link — the second hop's topic only appears inside the first hop's material
    — cannot be reached from the question alone. Carrying the authorized text of the
    first hop forward is the deterministic substitute for guessing the link. Only the
    best-ranked passage is carried: a second one pulls the query towards whatever else
    the first search happened to return.
    """
    head = " ".join(keywords(item) for item in items)
    tail = " ".join(str(row.get("text", ""))[:160] for row in evidence[:1])
    return (f"{head} {tail}".strip())[:limit]


def _normalized(value: str) -> str:
    return "".join(str(value).lower().split())


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text or "")}


def _grams(text: str) -> set[str]:
    joined = "".join(_CJK.findall(_normalized(text)))
    grams = {joined[index : index + 2] for index in range(max(0, len(joined) - 1))}
    return grams | _tokens(text)


NO_EVIDENCE_MARKERS = ("无直接提及", "未提及", "没有提及", "未找到", "没有找到", "无相关", "缺少依据", "无法确定")


def uncovered_items(items: list[str], claims: list[dict], goal: str = "") -> list[str]:
    """Return the subgoals the answer has not actually answered.

    Three things count as not answered: no claim shares enough wording with the
    subgoal, the closest claim only restates the request, and the closest claim says
    outright that it found nothing. The check is lexical on purpose — it decides
    whether to look again, never whether an answer is true.
    """
    if len(items) < 2:
        return []
    goal_text = goal or " ".join(items)
    goal_grams = _grams(goal_text)
    missing = []
    for item in items:
        wanted = _grams(_content(item))
        if not wanted:
            continue
        answered = False
        for claim in claims:
            text = claim["text"] if isinstance(claim, dict) else str(claim)
            found = _grams(text)
            if not found or len(wanted & found) / len(wanted) < 0.25:
                continue
            if any(marker in text for marker in NO_EVIDENCE_MARKERS):
                continue
            new_tokens = {token.lower() for token in _TOKEN.findall(text)} - _tokens(goal_text)
            if not new_tokens and len(found - goal_grams) / len(found) < 0.2:
                # The claim restates the request and carries no value of its own.
                continue
            answered = True
            break
        if not answered:
            missing.append(item)
    return missing


_ANAPHORA = ("该", "此", "其", "它", "上述", "前述", "这")
_AND = re.compile(r"[和及与](?=[^，,]{2,})")


def repair_question(items: list[str], missing: list[str]) -> str:
    """Ask the open subgoals on their own, keeping the antecedent of a reference.

    Asking "根据该特征核对当前限制" alone leaves the reference dangling, so the
    subgoals before it come along. What never comes along is the conditional wording
    of the original goal: repeating it is what made the first pass restate the request
    instead of executing it.
    """
    positions = [items.index(item) for item in missing if item in items]
    first = min(positions) if positions else 0
    context = any(word in item for item in missing for word in _ANAPHORA)
    wanted = set(missing)
    selected = [
        item
        for index, item in enumerate(items)
        if item in wanted or (context and index < first)
    ]
    return "；".join(selected or missing)


def checklist(missing: list[str]) -> list[str]:
    """Split "A 和 B" into two items the answer has to cover separately."""
    items = []
    for item in missing:
        items.extend(part for part in (_clean(part) for part in _AND.split(item)) if len(part) >= 2)
    return items or missing


_MONTH_DAY = re.compile(r"\d{1,2}\s*月\s*\d{1,2}\s*日")
_VERSION_LABEL = re.compile(r"v\s*\d+|第\s*\d+\s*版")
_CHAIN_MARKERS = ("按时间", "历史上", "各版本", "全部版本", "所有版本", "演变", "依次")


def version_depth(goal: str) -> int:
    """How many adjacent version pairs a temporal question needs.

    Two dates are one transition; three dates are two. A question that asks for the
    history rather than for one change gets at least two, so the oldest version is
    never silently dropped.
    """
    points = len(set(_MONTH_DAY.findall(goal or ""))) or len(set(_VERSION_LABEL.findall(goal or "")))
    depth = max(points - 1, 1)
    if any(marker in (goal or "") for marker in _CHAIN_MARKERS):
        depth = max(depth, 2)
    return min(depth, 3)


def version_pairs(versions: list[dict], goal: str, budget: int) -> list[tuple[str, str]]:
    """Adjacent (older, newer) version ids, newest transition first."""
    ordered = [row["version_id"] for row in versions if row.get("version_id")]
    depth = min(version_depth(goal), budget, max(len(ordered) - 1, 0))
    return [(ordered[index + 1], ordered[index]) for index in range(depth)]


def compact_observation(tool: str, summary: dict, data: dict, handles: list[str]) -> dict:
    """What the dynamic policy is allowed to see about one tool result.

    The policy needs to know what it got and what is still missing, not the full text
    of every match. Passing whole tool payloads back on every step is what made the
    dynamic prompt grow without bound.
    """
    matches = [
        {"title": row.get("title"), "chunk_id": row.get("chunk_id"), "snippet": str(row.get("snippet", ""))[:120]}
        for row in (data.get("matches") or [])[:3]
    ]
    observation = {
        "tool": tool,
        "status": summary.get("status"),
        "error_code": summary.get("error_code"),
        "evidence_count": summary.get("evidence_count"),
        "titles": summary.get("titles"),
        "handles": handles[:6],
    }
    if matches:
        observation["matches"] = matches
    for key in ("versions", "diff_lines", "checks"):
        if data.get(key):
            observation[key] = json_trim(data[key])
    if data.get("memories"):
        # Long-term memory is the user's own text, not enterprise material, and it is
        # not trusted. The policy is told that preferences exist and nothing else, so
        # a planted string cannot be copied into a query and end up in the visible
        # trace. Generation still receives it, clearly labelled as preferences.
        observation["memories"] = {"count": len(data["memories"]), "usable_as": "preferences_only"}
    return observation


def json_trim(value, limit: int = 6):
    if isinstance(value, list):
        return [json_trim(row, limit) for row in value[:limit]]
    if isinstance(value, dict):
        return {key: str(item)[:120] if isinstance(item, str) else item for key, item in value.items()}
    return str(value)[:120]
