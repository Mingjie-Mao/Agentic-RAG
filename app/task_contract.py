"""Opt-in answer contracts for explicit comparisons, predicates and version facts.

These are bounded literal checks, not a semantic judge. Unknown syntax, ambiguous
values, mismatched units and incomplete history remain unknown. No benchmark IDs,
expected answers or per-task wording are embedded in this module.
"""

from decimal import Decimal
import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.task_analysis import acceptance_items, historical_route_intent, version_intent


class SlotSpec(BaseModel):
    slot_id: str
    query: str
    subject: str = ""
    attribute: str = ""
    value_type: Literal["number", "date", "enum", "text"] = "text"
    required: bool = True
    applies_if: Literal["always", "true", "false"] = "always"
    history_kind: Literal["current", "first", "changes", "as_of"] | None = None


class ConditionSpec(BaseModel):
    condition_id: str = "condition_1"
    operand_slot_id: str
    operator: Literal["gt", "ge", "lt", "le", "eq", "unknown"]
    threshold: str | None = None
    unit: str | None = None


class ConditionState(BaseModel):
    condition_id: str
    result: Literal["true", "false", "unknown"]
    operands_with_refs: list[dict] = Field(default_factory=list)
    active_slot_ids: list[str] = Field(default_factory=list)
    reason: str


class TaskContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    intent: Literal["ordinary", "comparison", "conditional", "history"]
    slots: list[SlotSpec]
    conditions: list[ConditionSpec] = Field(default_factory=list)
    history_kind: Literal["current", "first", "changes", "as_of"] = "current"
    version_chain_complete: bool = False
    version_ids: list[str] = Field(default_factory=list)
    versions: list[dict] = Field(default_factory=list)
    reason: str = "explicit_structure"


class BoundValue(BaseModel):
    slot_id: str
    subject: str
    attribute: str
    value: str
    unit: str
    chunk_id: str
    version_id: str = ""
    quote: str


_UNIT = r"工作日|business days?|分钟|minutes?|小时|hours?|秒|seconds?|天|days?|年|years?|次|requests?|行|rows?|元|%|MB|GB"
_VALUE = re.compile(rf"(?<![0-9A-Za-z_.,])(?P<number>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?(?:\s*[万亿])?|[零一二三四五六七八九十百千万亿]+)\s*(?:个\s*)?(?P<unit>{_UNIT})(?![A-Za-z])", re.I)
_ENUM = re.compile(r"(?<![A-Za-z0-9])(?:已批准|已拒绝|已启用|已停用|启用|停用|生产|测试|approved|rejected|enabled|disabled|P[123])(?![A-Za-z0-9])", re.I)
_ROW_LIMIT_QUERY = r"(?:单次|单个|每次)[^，。；;\n?]{0,28}(?:最多|最大|上限|允许)[^，。；;\n?]{0,12}行|\b(?:maximum|max)\s+(?:number of\s+)?rows\b|\brow limit\b"
_ROW_LIMIT_SOURCE = r"(?:单次|单个|每次)[^，。；;\n]{0,20}(?:最多|最大|上限|不超过)|行数上限|最大行数|\b(?:maximum|max)\s+(?:number of\s+)?rows\b|\brow limit\b"
_METRICS = (
    (_ROW_LIMIT_QUERY, _ROW_LIMIT_SOURCE),
    (r"状态|等级|status|tier", r"状态|等级|status|tier"),
    (r"生效日期|生效时间|effective date", r"生效日期|生效时间|生效|effective"),
    (r"(?<![A-Za-z])RTO(?![A-Za-z])|恢复时间目标", r"(?<![A-Za-z])RTO(?![A-Za-z])|恢复时间目标"),
    (r"(?<![A-Za-z])RPO(?![A-Za-z])|恢复点目标", r"(?<![A-Za-z])RPO(?![A-Za-z])|恢复点目标"),
    (r"重试", r"重试|retries|retry"),
    (r"配额|额度|quota", r"配额|额度|调用上限|quota|rate limit"),
    (r"首次响应|响应时限|response time", r"首次响应|响应时限|响应时间|response time"),
    (r"超时(?:阈值|时限|时间)?|timeout", r"超时(?:阈值|时限|时间)?|timeout"),
    (r"保留", r"保留|retention"),
    (r"住宿|每晚", r"住宿|每晚|每夜"),
    (r"提交", r"提交|submit"),
    (r"补办|reissu", r"补办|reissu"),
    (r"有效期|validity", r"有效期|有效|validity"),
    (r"阈值|threshold", r"阈值|错误率|threshold"),
)


def number(raw):
    raw = re.sub(r"\s+", "", raw)
    if not raw:
        return None
    decimal = re.fullmatch(r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)([万亿]?)", raw)
    if decimal:
        scale = {"":1, "万":10000, "亿":100000000}[decimal[2]]
        return Decimal(decimal[1].replace(",", "")) * scale
    digits = dict(zip("零一二三四五六七八九", range(10), strict=True))
    total, section, digit = 0, 0, 0
    for char in raw:
        if char in digits:
            digit = digits[char]
        elif char in "十百千":
            section += (digit or 1) * {"十":10,"百":100,"千":1000}[char]
            digit = 0
        elif char == "万":
            total += (section + digit) * 10000
            section, digit = 0, 0
        elif char == "亿":
            total = (total + section + digit) * 100000000
            section, digit = 0, 0
        else:
            return None
    return Decimal(total + section + digit)


def unit_value(raw, unit):
    if unit == "enum":
        return "enum", raw.casefold()
    if unit == "date":
        return "date", Decimal(date.fromisoformat(raw).toordinal())
    value = number(raw)
    if value is None:
        return None
    groups = {
        "秒": ("seconds", 1), "second": ("seconds", 1), "seconds": ("seconds", 1),
        "分钟": ("seconds", 60), "minute": ("seconds", 60), "minutes": ("seconds", 60),
        "小时": ("seconds", 3600), "hour": ("seconds", 3600), "hours": ("seconds", 3600),
        "天": ("days", 1), "day": ("days", 1), "days": ("days", 1),
        "工作日": ("business_days", 1), "business day": ("business_days", 1), "business days": ("business_days", 1),
        "次": ("requests", 1), "request": ("requests", 1), "requests": ("requests", 1),
        "行": ("rows", 1), "row": ("rows", 1), "rows": ("rows", 1),
        "年": ("years", 1), "year": ("years", 1), "years": ("years", 1),
    }
    dimension, scale = groups.get(unit.lower(), (unit.lower(), 1))
    return dimension, value * scale


def _clean(text):
    return re.sub(r"^(?:先查|先查询|查询|给出|说明|比较|对比|对照|先核对|查一下)\s*", "", text.strip(" ：:，,；;。？?"))


def _metric(text):
    for query, source in _METRICS:
        match = re.search(query, text, re.I)
        if match:
            return match.group(), source
    return "", ""


def _slot(query, sid, applies="always", subject=""):
    query = _clean(query)
    attribute, _ = _metric(query)
    # A metric mentioned as background is not a request for its numerical value.
    # E.g. "during first response, notify which role" asks for a role, not a time.
    qualitative=bool(re.search(r'哪个(?:角色|人员|部门)|谁|\bwho\b',query,re.I))
    if not subject and "的" in query and not historical_route_intent(query):
        candidate = query.split("的", 1)[0].strip()
        candidate = re.split(r'期间|之后|之前', candidate)[-1].strip()
        log = re.search(r'(操作|审计|业务|访问|错误|应用|系统)日志$', candidate)
        if log:
            candidate = log.group()
        # Restrict the inferred subject to named local entities. Generic action
        # phrases and clause-level qualifiers are not entity scope.
        candidate = re.sub(r'(?:单个请求|默认请求|请求)$', '', candidate).strip()
        if re.search(r"数据库|工单|SDK|专业版|基础版|门禁卡|日志$|服务$|通知$|回滚$", candidate, re.I):
            subject = candidate
    if not subject:
        logs = re.findall(r'(?:操作|审计|业务|访问|错误|应用|系统)日志', query)
        if len(set(logs)) == 1:
            subject = logs[0]
        elif attribute:
            prefix = query[:query.find(attribute)].strip()
            prefix = re.sub(r'(?:最多|最大|默认|单个请求|请求|的|需要|单次)\s*', '', prefix).strip()
            prefix = re.sub(r'每\s*(?:分钟|小时|秒|天|日)', '', prefix).strip()
            if re.fullmatch(r'[\w\s\u4e00-\u9fff]{2,32}(?:服务|通知|SDK|版|数据库|工单|门禁卡)', prefix, re.I):
                subject = prefix
    return SlotSpec(slot_id=sid, query=_clean(query), subject=subject,
                    attribute=attribute, applies_if=applies, value_type="text" if qualitative else "enum" if re.search(r"状态|等级|status|tier", query, re.I) else "date" if re.search(r"生效|effective date", query, re.I) else "number" if attribute else "text")


def comparison_slots(question):
    """Resolve explicit A/B requests; arbitrary implicit referents stay unresolved."""
    text = _clean(question)
    # A 的 attribute 比 B 长/多多少.
    match = re.search(r"^(.+?)的(.+?)比(.+?)(?:长|短|多|少|高|低|大|小)(?:多少|几)", text)
    if match:
        a, attribute, b = match.groups()
        return [_slot(a + "的" + attribute, "left", subject=a),
                _slot(b + "的" + attribute, "right", subject=b)]
    # A and B share an attribute. The shared prefix may be supplied only on A.
    match = re.search(r"^(.+?)(?:和|与|及|\s+and\s+)(.+?)的(.+?)(?:相差|差多少|分别|是否相同|一样|一致)", text, re.I)
    if match and "的" not in match.group(1):
        a, b, attribute = match.groups()
        return [_slot(a + "的" + attribute, "left", subject=a),
                _slot(b + "的" + attribute, "right", subject=b)]
    # Two independent named attributes; keep the request for each side intact.
    prefix = re.split(r"[，,？?]", text)[0]
    pieces = re.split(r"和|与|以及|\s+and\s+", prefix, maxsplit=1, flags=re.I)
    if len(pieces) == 2 and all("的" in part for part in pieces):
        return [_slot(part, sid, subject=part.split("的", 1)[0].strip())
                for part, sid in zip(pieces, ("left", "right"), strict=True)]
    return []


def build_contract(question):
    history = historical_route_intent(question) or bool(version_intent(question) and re.search(
        r"各.*版本|最早|怎么变化|有没有变化|相比|"
        r"\b(?:first|earliest)\s+(?:versions?|revisions?)\b|"
        r"\b(?:versions?|revisions?)\b.{0,40}\b(?:changes?|as\s+of)\b", question, re.I))
    change_condition = bool(history and re.search(r"(?:如果|若)有(?:变化)?[，,？?]", question) and re.search(r"变化|变成|变为", question))
    if history and not change_condition and re.search(r"如果|只有|若|\bif\b", question, re.I):
        return TaskContract(question=question, intent="conditional", slots=[_slot(question, "operand")],
            conditions=[ConditionSpec(operand_slot_id="operand", operator="unknown")], reason="compound_history_condition_unknown")
    if history:
        kind = ("first" if re.search(r"最早|首个版本|首次版本|\bfirst\s+version\b|\bearliest\b", question, re.I)
                else "as_of" if re.search(r"\d{4}\s*[年-]|as of", question, re.I)
                else "changes")
        metric_count = sum(bool(re.search(pattern, question, re.I)) for pattern, _ in _METRICS)
        log_types = set(re.findall(r"(?:操作|审计|业务|访问|错误|应用|系统)日志", question))
        mixed = bool(kind == "first" and re.search(r"当前|最新|current|latest", question, re.I))
        pieces = [p.strip() for p in re.split(r'[？?；;]', question) if p.strip()]
        slots = [_slot(question, "history")]
        # Separate explicitly delimited asks, retaining the temporal scope of each.
        # Undelimited multi-attribute requests remain unknown rather than partial.
        if len(pieces) > 1 and not change_condition:
            slots = []
            for i, piece in enumerate(pieces):
                slot = _slot(piece, f"history_{i}")
                slot.history_kind = ("first" if re.search(r"最早|earliest", piece, re.I)
                    else "changes" if re.search(r"变化|相比|changes?|compared", piece, re.I)
                    else "as_of" if re.search(r"\d{4}\s*[年-]|as of", piece, re.I)
                    else "current" if re.search(r"当前|最新|现行|current|latest", piece, re.I) else kind)
                slots.append(slot)
        unresolved = (len(log_types) > 1 or mixed or
            (metric_count > 1 and len(slots) == 1) or
            any(not slot.attribute for slot in slots))
        return TaskContract(question=question, intent="history", slots=slots, history_kind=kind,
            reason="multiple_history_scopes_unresolved" if unresolved else "change_condition" if change_condition else "explicit_structure")
    match = re.search(r"(?:如果|只有|若|\bif\b)\s*(.+?)(?:[，,；;]|(?:时(?=[，,再才继则])，?|\bthen\b))", question, re.I)
    if match:
        base = _clean(question[:match.start()])
        predicate = match.group(1)
        bound = re.search(rf"(不超过|至少|至多|大于等于|小于等于|超过|大于|低于|小于|等于|>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)\s*(?:个\s*)?({_UNIT})", predicate, re.I)
        date_bound = re.search(r"(不晚于|不早于|早于|晚于|等于)\s*(\d{4}(?:年\d{1,2}月\d{1,2}日|-\d{2}-\d{2}))", predicate)
        enum_bound = re.search(r"等于\s*(已批准|已拒绝|已启用|已停用|启用|停用|生产|测试|approved|rejected|enabled|disabled|P[123])(?=$|[，,。\s])", predicate, re.I)
        if not base:
            marker = bound or date_bound or enum_bound
            base = _clean(predicate[:marker.start()]) if marker else ""
        operators = {"不晚于": "le", "不早于": "ge", "早于": "lt", "晚于": "gt","超过": "gt", "大于": "gt", ">": "gt", "至少": "ge", "大于等于": "ge", ">=": "ge",
                     "低于": "lt", "小于": "lt", "<": "lt", "不超过": "le", "至多": "le", "小于等于": "le", "<=": "le", "等于": "eq", "=": "eq"}
        tail = question[match.end():]
        branches = re.split(r"[；;，,]?\s*(?:否则|else)\s*", tail, maxsplit=1, flags=re.I)
        slots = [_slot(base, "operand")]
        for branch, applies in zip(branches, ("true", "false")):
            cleaned = re.sub(r"^(?:(?:才|再|继续|then)\s*)+", "", branch.strip(), flags=re.I)
            slots.extend(_slot(item, f"{applies}_{i}", applies) for i, item in enumerate(acceptance_items(cleaned), 1))
        # More than one predicate or no unambiguous operand is deliberately unknown.
        marker = bound or date_bound or enum_bound
        simple = bool(marker and base and not re.search(r"并且|或者|且|或|\band\b|\bor\b|不低于|不少于|不大于|未超过", predicate, re.I))
        threshold, unit = (bound[2], bound[3]) if bound else (None, None)
        if date_bound:
            from app.source_facts import dates
            extracted = dates(date_bound[2])
            if len(extracted) == 1:
                threshold, unit = extracted[0][1].isoformat(), "date"
            else:
                simple = False
        if enum_bound:
            threshold, unit = enum_bound[1], "enum"
        condition = ConditionSpec(operand_slot_id="operand", operator=("eq" if enum_bound else operators[marker[1]]) if simple else "unknown",
                                  threshold=threshold if simple else None, unit=unit if simple else None)
        return TaskContract(question=question, intent="conditional", slots=slots, conditions=[condition])
    if re.search(r"如果|只有|若(?!干)|\bif\b", question, re.I):
        return TaskContract(question=question, intent="conditional", slots=[_slot(question, "operand")],
            conditions=[ConditionSpec(operand_slot_id="operand", operator="unknown")], reason="unresolved_condition")
    if re.search(r"相差|差多少|长多少|短多少|哪个更多|哪个更少|比较|对比|分别|difference|compare", question, re.I):
        slots = comparison_slots(question)
        numeric_request = bool(re.search(r'相差|差多少|长多少|短多少|哪个更多|哪个更少', question))
        if (len(slots) == 2 and all(s.value_type == 'number' for s in slots)) or numeric_request:
            return TaskContract(question=question, intent="comparison", slots=slots,
                                reason="explicit_structure" if slots else "unresolved_comparison_slots")
        # Qualitative and time-window comparisons still use grounded generation.
        # The numeric candidate must not intercept unsupported prose comparisons.
    from app.verdict import question_slots
    return TaskContract(question=question, intent="ordinary", slots=[_slot(item, f"slot_{i}")
                        for i, item in enumerate(question_slots(question), 1)])


def _subject_matches(subject, title, sentence):
    def clean(text):
        return "".join(text.casefold().split())
    subject, title, sentence = map(clean, (subject, title, sentence))
    subject = re.sub(r"^([sp]\d+)工单$", r"\1", subject)
    if not subject or subject in sentence or (subject in title and "|" not in sentence):
        return True
    # A document title can supply a shared product prefix, but not an unrelated
    # subject from a different row of the same table.
    return any(subject[:i] in title and subject[i:] in sentence
               for i in range(2, len(subject) - 1))


def bind_values(slot, evidence):
    """Only uniquely local numeric attributes. A missing binding is not an answer."""
    _, pattern = _metric(slot.query)
    if not pattern:
        return []
    found = []
    for row in evidence:
        title, text = row.get("title", ""), row["text"]
        # Thousands separators belong to the original numeric span; sentence
        # commas outside a number still delimit independently labelled values.
        parts = re.split(r"([；;，。\n]|(?<!\d),|,(?!\d))", text)
        tables = None
        for position in range(0, len(parts), 2):
            sentence = parts[position]
            anchor = re.search(pattern, sentence, re.I)
            headings = list(re.finditer(r'(?m)^#{1,6}\s+([^\n]+)', ''.join(parts[:position])))
            heading = headings[-1].group(1) if headings else ''
            heading_subject = bool(slot.subject and slot.subject in heading)
            same_sentence_subject = bool(position >= 2 and parts[position - 1] in {"，", ","}
                and slot.subject and slot.subject in parts[position - 2]
                and anchor and not sentence[:anchor.start()].strip())
            if not _subject_matches(slot.subject, title, sentence) and not same_sentence_subject and not heading_subject:
                continue
            table = "|" in sentence and bool(re.search(pattern, text, re.I))
            if not anchor and not table:
                continue
            bound_text, bound_quote = sentence, sentence.strip()
            if heading_subject and not _subject_matches(slot.subject, title, sentence):
                # Keep the contiguous original section with its scope heading and
                # complete predicate/action sentence, rather than citing a fragment.
                bound_quote = text[headings[-1].start():].strip()
            if table:
                from app.clients import evidence_spans
                if tables is None:
                    sources, _ = evidence_spans([row])
                    tables = [s['quote'] for s in sources.values() if '|' in s['quote']]
                containing = [t for t in tables if sentence.strip() in t.splitlines()]
                if len(containing) != 1:
                    continue
                def cells(line):
                    return [cell.strip() for cell in line.strip().strip('|').split('|')]
                header = cells(containing[0].splitlines()[0])
                columns = [i for i, cell in enumerate(header) if re.search(pattern, cell, re.I)]
                values = cells(sentence)
                if len(columns) != 1 or len(values) != len(header):
                    continue
                bound_text, bound_quote = values[columns[0]], containing[0]
                anchor = None  # The header binds the attribute to this exact cell.
            if slot.value_type == "enum":
                candidates = list(_ENUM.finditer(bound_text, anchor.end() if anchor else 0))
                if (anchor or table) and len(candidates) == 1 and not re.search(r"不是|并非|不为|not", bound_text[:candidates[0].start()], re.I):
                    found.append(BoundValue(slot_id=slot.slot_id, subject=slot.subject, attribute=slot.attribute,
                        value=candidates[0].group(), unit="enum", chunk_id=row["chunk_id"],
                        version_id=row.get("version_id", ""), quote=bound_quote))
                continue
            if slot.value_type == "date":
                from app.source_facts import dates
                date_values = dates(bound_text)
                if (anchor or table) and len(date_values) == 1:
                    found.append(BoundValue(slot_id=slot.slot_id, subject=slot.subject, attribute=slot.attribute,
                        value=date_values[0][1].isoformat(), unit="date", chunk_id=row["chunk_id"],
                        version_id=row.get("version_id", ""), quote=bound_quote))
                continue
            candidates = list(_VALUE.finditer(bound_text, anchor.end() if anchor else 0))
            if not candidates and anchor:
                # Deadline phrasing commonly places its value before the verb.
                before = list(_VALUE.finditer(sentence[:anchor.start()]))
                if len(before) == 1 and not any(re.search(other, sentence[:anchor.start()], re.I)
                        for _, other in _METRICS if other != pattern):
                    candidates = before
            if not candidates:
                continue
            candidate = candidates[0]
            if pattern == _ROW_LIMIT_SOURCE and unit_value(candidate["number"], candidate["unit"])[0] != "rows":
                continue
            if len(candidates) > 1:
                gap = sentence[candidate.end():candidates[1].start()]
                if not any(re.search(other, gap, re.I) for _, other in _METRICS if other != pattern):
                    continue
            # A new labelled attribute between anchor and value is not the value of
            # this attribute. Multiple table values require a column-aware parser.
            between = sentence[anchor.end():candidate.start()] if anchor and candidate.start() >= anchor.end() else ""
            if re.search(r"不是|不为|并非|不得|不超过|至少|最多|最少|不小于|不大于|not|at least|up to", between, re.I):
                continue
            if len(between) > 28 or (table and len(candidates) != 1):
                continue
            if any(re.search(other, between, re.I) for _, other in _METRICS if other != pattern):
                continue
            # A query such as 操作日志/审计日志 must bind its requested log type.
            requested_logs = re.findall(r"(?:操作|审计|业务|访问|错误|应用|系统)日志", slot.query)
            if requested_logs and not any(item in sentence for item in requested_logs):
                continue
            value = number(candidate["number"])
            if value is None:
                continue
            found.append(BoundValue(slot_id=slot.slot_id, subject=slot.subject, attribute=slot.attribute,
                                    value=str(value), unit=candidate["unit"], chunk_id=row["chunk_id"],
                                    version_id=row.get("version_id", ""), quote=bound_quote))
    # Keep version identity: distinct versions are never merged into a current fact.
    grouped = {}
    for value in found:
        grouped.setdefault(value.version_id, []).append(value)
    return [values[0] for values in grouped.values()
            if len({unit_value(v.value, v.unit) for v in values}) == 1]


def evaluate_condition(contract, evidence):
    condition = contract.conditions[0]
    slot = next(s for s in contract.slots if s.slot_id == condition.operand_slot_id)
    values = bind_values(slot, evidence)
    result, reason = "unknown", "unbound_or_ambiguous_operand"
    if len(values) == 1 and condition.operator != "unknown":
        left = unit_value(values[0].value, values[0].unit)
        right = unit_value(condition.threshold, condition.unit)
        if left and right and left[0] == right[0]:
            a, b = left[1], right[1]
            truth = {"gt": a > b, "ge": a >= b, "lt": a < b, "le": a <= b, "eq": a == b}[condition.operator]
            result, reason = ("true" if truth else "false"), "explicit_numeric_predicate"
        else:
            reason = "unit_mismatch"
    return ConditionState(condition_id=condition.condition_id, result=result,
                          operands_with_refs=[v.model_dump() for v in values],
                          active_slot_ids=[s.slot_id for s in contract.slots
                                           if s.applies_if == "always" or s.applies_if == result], reason=reason)


def value_claim(value, evidence):
    from app.clients import Claim
    source = next(row for row in evidence if row["chunk_id"] == value.chunk_id)
    return Claim(text=f"{value.subject + '的' if value.subject else ''}{value.attribute}为 {value.value} {value.unit if value.unit not in {'date', 'enum'} else ''}。",
                 evidence_ids=[source["id"]], quotes=[value.quote])


def contract_generate(models, question, evidence, *, contract=None, options=None):
    """Shared generation entry. Deterministic answers require literal local bindings."""
    from app.clients import GeneratedAnswer

    contract = contract or build_contract(question)
    trace = {"version": "task-contract-v1", "contract": contract.model_dump(), "condition": None,
             "bound_values": [], "status": "ordinary",
             "slot_states": {slot.slot_id: "unknown" for slot in contract.slots}}
    options = dict(options or {})
    if contract.intent == "conditional":
        state = evaluate_condition(contract, evidence)
        trace["condition"] = state.model_dump()
        trace["slot_states"] = {slot.slot_id: ("unknown" if state.result == "unknown"
                                              or slot.slot_id in state.active_slot_ids else "inactive")
                                for slot in contract.slots}
        values = [BoundValue.model_validate(v) for v in state.operands_with_refs]
        active = [s for s in contract.slots if s.slot_id in state.active_slot_ids]
        if state.result == "unknown":
            return GeneratedAnswer(answerable=False, claims=[]), {"task_contract": trace | {"status": "unknown_condition"}}
        trace["slot_states"]["operand"] = "supported"
        branches = [s for s in active if s.applies_if != "always"]
        # A false predicate with no else branch is fully answered by its operand.
        # The unrequested branch is never generated or allowed into final claims.
        if not branches or all(s.query == active[0].query or re.search(r"只回答|只给出", s.query) for s in branches):
            return GeneratedAnswer(answerable=True, claims=[value_claim(v, evidence) for v in values]), {
                "task_contract": trace | {"status": "condition_gated", "bound_values": [v.model_dump() for v in values]}}
        # Generate branch facts separately. Numeric branches are bound in code;
        # prose branches must satisfy the existing checklist and cite original text.
        claims = [value_claim(v, evidence) for v in values]
        combined_usage = {}
        for slot in branches:
            bound = bind_values(slot, evidence)
            if len(bound) == 1:
                trace["slot_states"][slot.slot_id] = "supported"
                claims.append(value_claim(bound[0], evidence))
                continue
            if slot.value_type in {"number", "date", "enum"}:
                return GeneratedAnswer(answerable=False, claims=[]), {
                    "task_contract": trace | {"status": "active_slot_incomplete"}}
            branch_options=options | {'acceptance_items_override':[slot.query],'check_conflict':False}
            from app.config import settings
            if settings().answer_quality_enabled:
                from app.answer_quality import recover_answer
                generated,extra=recover_answer(models,slot.query,evidence,options=branch_options)
            else:
                generated,extra=models.generate(slot.query,evidence,**branch_options)
            from agent.planner import evidence_coverage
            if extra.get('quality_validation_failed') or not generated.answerable or not generated.claims or not evidence_coverage(
                    [slot.query], [{"chunk_id": str(i), "text": c.text} for i, c in enumerate(generated.claims)], slot.query)[slot.query]:
                return GeneratedAnswer(answerable=False, claims=[]), {
                    "task_contract": trace | {"status": "active_slot_incomplete"}}
            # Inactive numeric attributes must never leak into prose branches.
            inactive = [s for s in contract.slots if s.applies_if not in {"always", state.result}]
            inactive_values = [v for s in inactive for v in bind_values(s, evidence)]
            active_keys = {(v.attribute.casefold(), unit_value(v.value, v.unit))
                           for s in active for v in bind_values(s, evidence)}
            for claim in generated.claims:
                for value in inactive_values:
                    if (value.attribute.casefold(), unit_value(value.value, value.unit)) in active_keys:
                        continue
                    if re.search(re.escape(value.attribute), claim.text, re.I) and any(
                            unit_value(m["number"], m["unit"]) == unit_value(value.value, value.unit)
                            for m in _VALUE.finditer(claim.text)):
                        return GeneratedAnswer(answerable=False, claims=[]), {
                            "task_contract": trace | {"status": "inactive_branch_output"}}
            trace["slot_states"][slot.slot_id] = "lexical_coverage_only"
            claims.extend(generated.claims)
            for key in ("prompt_tokens", "completion_tokens"):
                combined_usage[key] = combined_usage.get(key, 0) + (extra.get(key) or 0)
        combined_usage["task_contract"] = trace | {"status": "condition_gated"}
        return GeneratedAnswer(answerable=True, claims=claims), combined_usage
    if contract.intent == "comparison" and len(contract.slots) == 2:
        pairs = [bind_values(slot, evidence) for slot in contract.slots]
        if all(len(values) == 1 for values in pairs):
            left, right = pairs[0][0], pairs[1][0]
            lv, rv = unit_value(left.value, left.unit), unit_value(right.value, right.unit)
            if lv and rv and lv[0] == rv[0] and lv[0] not in {"date", "enum"}:
                from app.clients import Claim

                trace["slot_states"] = {slot.slot_id: "supported" for slot in contract.slots}
                claims = [value_claim(value, evidence) for value in (left, right)]
                # Express the difference in the left source's unit, even if the
                # right source uses hours and the left source uses minutes.
                scale = unit_value("1", left.unit)[1]
                base_delta = abs(lv[1] - rv[1])
                delta, delta_unit = base_delta / scale, left.unit
                if delta * scale != base_delta and lv[0] == 'seconds':
                    delta_unit = '分钟' if base_delta % 60 == 0 else '秒'
                    delta = base_delta / unit_value('1', delta_unit)[1]
                direction = "相同" if lv[1] == rv[1] else (f"{left.subject}更大" if lv[1] > rv[1] else f"{right.subject}更大")
                claims.append(Claim(text=f"两者相差 {delta} {delta_unit}；{direction}。",
                                    evidence_ids=[claims[0].evidence_ids[0], claims[1].evidence_ids[0]],
                                    quotes=[left.quote, right.quote]))
                return GeneratedAnswer(answerable=True, claims=claims), {
                    "task_contract": trace | {"status": "explicit_comparison",
                                              "bound_values": [left.model_dump(), right.model_dump()]}}
        # Unknown inputs cannot produce a complete numerical comparison.
        return GeneratedAnswer(answerable=False, claims=[]), {"task_contract": trace | {"status": "comparison_inputs_unknown"}}
    if contract.intent == "comparison":
        return GeneratedAnswer(answerable=False, claims=[]), {"task_contract": trace | {"status": "unresolved_comparison_slots"}}
    if contract.intent == "history":
        return history_generate(contract, evidence, trace)
    from app.config import settings
    if settings().answer_quality_enabled:
        from app.evidence_scope import scoped_windows
        scoped, scope_trace = scoped_windows(question, evidence)
        if scoped is not None:
            return scoped, {'task_contract': trace, 'evidence_scope': scope_trace}
        from app.answer_quality import recover_answer
        generated, usage = recover_answer(models, question, evidence, options=options)
    else:
        generated, usage = models.generate(question, evidence, **options)
    usage["task_contract"] = trace
    return generated, usage


def history_generate(contract, evidence, trace):
    """Answer each explicit history slot; never return partial scope as success."""
    from app.clients import GeneratedAnswer
    if contract.reason == "multiple_history_scopes_unresolved":
        return GeneratedAnswer(answerable=False, claims=[]), {"task_contract": trace | {"status": "history_scope_unknown"}}
    if len(contract.slots) > 1:
        claims, bindings = [], []
        for slot in contract.slots:
            scoped = contract.model_copy(update={"question": slot.query, "slots": [slot],
                "history_kind": slot.history_kind or contract.history_kind})
            answer, usage = _history_slot_generate(scoped, evidence, trace)
            if not answer.answerable:
                return answer, usage
            claims += answer.claims
            bindings += usage["task_contract"]["bound_values"]
        if len(claims) > 8:
            return GeneratedAnswer(answerable=False, claims=[]), {"task_contract": trace | {"status": "history_scope_budget"}}
        return GeneratedAnswer(answerable=True, claims=claims), {"task_contract": trace | {
            "status": "version_values_verified", "bound_values": bindings,
            'slot_states': {s.slot_id:'supported' for s in contract.slots}}}
    return _history_slot_generate(contract, evidence, trace)


def _history_slot_generate(contract, evidence, trace):
    from app.clients import GeneratedAnswer
    if contract.reason == "multiple_history_scopes_unresolved":
        return GeneratedAnswer(answerable=False, claims=[]), {"task_contract": trace | {"status": "history_scope_unknown"}}
    values = bind_values(contract.slots[0], evidence)
    by_version = {v.version_id: v for v in values}
    if (not contract.version_chain_complete or not contract.version_ids
            or any(vid not in by_version for vid in contract.version_ids)):
        return GeneratedAnswer(answerable=False, claims=[]), {"task_contract": trace | {"status": "history_incomplete"}}
    ordered = [by_version[vid] for vid in contract.version_ids]
    if contract.history_kind == "first":
        ordered = ordered[:1]
    elif contract.history_kind == "as_of":
        from app.temporal import select_effective_versions
        versions = contract.versions or [{"version_id": row["version_id"], "effective_interval": row.get("effective_interval", {})}
                                        for row in evidence]
        unique = {v["version_id"]: v for v in versions}
        selection = select_effective_versions(contract.question, list(unique.values()))
        if selection["status"] != "selected":
            return GeneratedAnswer(answerable=False, claims=[]), {"task_contract": trace | {"status": "effective_date_unknown"}}
        wanted = {v["version_id"] for v in selection["selections"]}
        if len(wanted) != 1 and re.search(r'\d{4}\s*年\s*\d{1,2}\s*月(?!\s*\d)', contract.question):
            return GeneratedAnswer(answerable=False, claims=[]), {"task_contract": trace | {"status": "effective_date_unknown"}}
        ordered = [value for value in ordered if value.version_id in wanted]
    elif contract.history_kind == "current":
        active = [v['version_id'] for v in contract.versions if v.get('is_active')]
        if len(active) != 1:
            return GeneratedAnswer(answerable=False, claims=[]), {"task_contract": trace | {"status": "active_version_unknown"}}
        ordered = [v for v in ordered if v.version_id == active[0]]
    claims = []
    for value in ordered:
        claim = value_claim(value, evidence)
        source = next(r for r in evidence if r["chunk_id"] == value.chunk_id)
        record = next((v for v in contract.versions if v['version_id'] == value.version_id), {})
        label = record.get('filename', source.get("version_label", value.version_id))
        claim.text = f"{label}：{claim.text}"
        claims.append(claim)
    if contract.history_kind == "changes" and len(ordered) > 1:
        from app.clients import Claim
        unchanged = len({unit_value(v.value, v.unit) for v in ordered}) == 1
        if contract.reason == "change_condition":
            trace["condition"] = {"result": "false" if unchanged else "true", "reason": "compared_complete_version_values",
                                  "operands_with_refs": [v.model_dump() for v in ordered]}
        change_text = ("已读取版本中的该属性保持不变。" if unchanged else
                       f"已读取版本中的该属性发生变化：从 {ordered[0].value} {ordered[0].unit} 变为 {ordered[-1].value} {ordered[-1].unit}。")
        claims.append(Claim(text=change_text,
                            evidence_ids=[next(r["id"] for r in evidence if r["chunk_id"] == v.chunk_id) for v in ordered],
                            quotes=[v.quote for v in ordered]))
    return GeneratedAnswer(answerable=True, claims=claims), {
        "task_contract": trace | {"status": "version_values_verified",
                                  'slot_states': {contract.slots[0].slot_id:'supported'},
                                  "bound_values": [v.model_dump() for v in ordered]}}
