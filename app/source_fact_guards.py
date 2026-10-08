"""Literal field boundaries and local date roles, with conservative ambiguity handling."""
import re


_UNITS = re.compile(r'^\s*(?:%|％|minutes?\b|hours?\b|seconds?\b|days?\b|ms\b|bps\b|秒|分钟|小时|天|次)',re.I)
BINDING_VERSION = 'literal-local-binding-v3'
_ROLE = re.compile(r'(?P<publication>published|publication|posted|发布日期|刊登|发布)'
                   r'|(?P<effective>effective|takes? effect|in force|生效|实施)'
                   r'|(?P<event>occurred|happened|event date|发生|举行)'
                   r'|(?P<opening>\b(?:open(?:ed|ing|s)?|launch(?:ed|es)?|released?)\b|开业|开幕|上线)',re.I)


def literal_field(field, quote, *, complete_number=False):
    if not field:
        return True
    if not field.strip():
        return False
    before = r'(?<![A-Za-z0-9_.+−-])' if re.match(r'[A-Za-z0-9+−-]',field) else ''
    after = r'(?![A-Za-z0-9_])' if re.search(r'[A-Za-z0-9%％]$',field) else ''
    if re.search(r'\d$',field):
        after += r'(?!\.\d)'
    matches=list(re.finditer(before+re.escape(field)+after,quote,re.I))
    if not matches:
        return False
    if complete_number and re.fullmatch(r'[+−-]?\d+(?:\.\d+)?',field.strip()):
        return any(not _UNITS.match(quote[match.end():]) for match in matches)
    return True


def date_role(raw, quote, *, position=None):
    """Find the nearest local role cue; a distant effective cue cannot label publication."""
    index=quote.casefold().find(raw.casefold()) if position is None else position
    if index<0:
        return None
    previous=[match for match in _ROLE.finditer(quote[:index]) if index-match.end()<=60]
    tail=quote[index+len(raw):]
    following=[match for match in _ROLE.finditer(tail) if match.start()<=12
               and not re.search(r'[,，;；。]',tail[:match.start()])]
    if previous:
        nearest=previous[-1]
        if not following or index-nearest.end()<=following[0].start():
            return nearest.lastgroup
    if following:
        return following[0].lastgroup
    return previous[-1].lastgroup if previous else None


def inclusive_end(raw, quote):
    index=quote.casefold().find(raw.casefold())
    if index<0:
        return False
    prefix=quote[:index][-32:]
    if re.search(r'\b(?:through|including)\s*$|至\s*$',prefix,re.I):
        return True
    return bool(re.search(r'\buntil\s*$',prefix,re.I)
                and re.search(r'inclusive|including that day|含当日',quote,re.I))
