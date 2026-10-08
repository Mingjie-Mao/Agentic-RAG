"""Small, explicit English search facets for multi-document questions.

These are search queries, not gold facts or required answer slots. Ambiguous wording
falls back to the original question. All results are still scoped and ACL-checked by
the ordinary retrieval path.
"""
import re


def retrieval_facets(question: str) -> list[str]:
    text = (question or '').strip().rstrip('?').strip()
    if len(text) < 40 or re.search(r'[\u4e00-\u9fff]', text):
        return []
    # Explicit comparative constructions enumerate source-side descriptions.
    between = re.search(r'\bbetween\s+(.+?)\s+and\s+(.+?)(?:,\s*|$)', text, re.I)
    if between:
        parts = [between.group(1), between.group(2)]
    else:
        from_to = re.search(r'\bfrom\s+(.+?)(?:\?|$)', text, re.I)
        parts = re.split(r'\s+to\s+', from_to.group(1), maxsplit=3, flags=re.I) if from_to else []
    parts = [part.strip(' ,;:.') for part in parts]
    if 2 <= len(parts) <= 3 and all(len(part.split()) >= 4 for part in parts):
        return parts
    # A list of at least three separate action clauses can describe different
    # articles. Keep the first clause's entity context on later searches.
    clauses = [part.strip(' ,;:.') for part in re.split(r',\s*(?:and\s+)?', text)]
    if len(clauses) >= 3 and all(len(part.split()) >= 4 for part in clauses[:3]):
        anchor = ' '.join(clauses[0].split()[:10])
        return [clauses[0], *(f'{anchor} {part}' for part in clauses[1:3])]
    return []
