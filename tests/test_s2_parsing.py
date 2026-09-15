import json
import os
from pathlib import Path

import pytest

from app.chunking import chunk_blocks
from app.parsing import ParseError, parse_document

TYPES = {
    "md": "text/markdown",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
CASES = (
    json.loads(Path("fixtures/s2/manifest.json").read_text())
    if Path("fixtures/s2/manifest.json").exists()
    else []
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: Path(c["path"]).name)
def test_format_fixture(case):
    if case["format"] == "pdf" and not case["expect_error"] and os.getenv("RAG_RUN_LAYOUT_TESTS") != "1":
        pytest.skip("requires downloaded layout models")
    data = Path(case["path"]).read_bytes()
    if case["expect_error"]:
        with pytest.raises(ParseError):
            parse_document(data, TYPES[case["format"]], structured=True)
        return
    blocks = parse_document(data, TYPES[case["format"]], structured=True)
    joined = "\n".join(b.text for b in blocks)
    assert all(fact in joined for fact in case["facts"])
    for strategy in ["fixed", "structure"]:
        chunks = chunk_blocks(blocks, strategy=strategy)
        assert chunks and all(
            c.locator["sources"] and c.locator["token_count_estimate"] > 0 for c in chunks
        )
        assert all(
            any(b.text[start:end] in c.text for c in chunks)
            for b in blocks
            for start, end in [(0, min(20, len(b.text)))]
        )
    if case["format"] == "docx":
        assert all("page" not in b.locator for b in blocks)
    if case["format"] == "xlsx":
        assert all(b.locator["sheet"] and b.locator["range"] for b in blocks)
    if case["format"] == "pdf":
        assert all(b.locator["page"] >= 1 and b.locator["provenance"] for b in blocks)


def test_formula_cache_is_not_invented():
    missing = parse_document(Path("fixtures/s2/formula-missing.xlsx").read_bytes(), TYPES["xlsx"])
    cached = parse_document(Path("fixtures/s2/formula-cached.xlsx").read_bytes(), TYPES["xlsx"])
    a = next(c for b in missing for c in b.locator["cells"] if c["formula"])
    b = next(c for block in cached for c in block.locator["cells"] if c["formula"])
    assert a["cached_value"] is None and a["cache_status"] == "missing"
    assert b["cached_value"] == "157" and b["formula"] == "=B2+C2"


def test_structural_boundaries_change_chunking_without_losing_text():
    blocks = parse_document(
        b"# One\n\nfirst statement.\n\n# Two\n\nsecond statement.", TYPES["md"], structured=True
    )
    fixed = chunk_blocks(blocks, strategy="fixed")
    structured = chunk_blocks(blocks, strategy="structure")
    assert len(structured) > len(fixed)
    assert all(
        len({tuple(s["locator"]["heading_path"]) for s in chunk.locator["sources"]}) == 1
        for chunk in structured
    )
