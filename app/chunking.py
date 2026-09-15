"""Two boundary policies with complete source maps and explicit token estimates."""

import os
from pathlib import Path
from functools import lru_cache

from app.parsing import Passage, split_passages


@lru_cache(maxsize=1)
def tokenizer():
    import tiktoken

    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".runtime/tiktoken").resolve()))
    return tiktoken.get_encoding("cl100k_base")


def token_count(text):
    return len(tokenizer().encode(text, disallowed_special=()))


def chunk_blocks(blocks, size=700, overlap=100, strategy="structure"):
    if strategy not in {"fixed", "structure"}:
        raise ValueError("未知分块方式")
    groups, current = [], []
    for block in blocks:
        key = (block.locator.get("page"), block.locator.get("sheet"))
        previous = current[-1] if current else None
        boundary = previous is not None and key != (
            previous.locator.get("page"),
            previous.locator.get("sheet"),
        )
        if strategy == "structure" and previous is not None:
            boundary = (
                boundary
                or block.locator.get("heading_path") != previous.locator.get("heading_path")
                or "table" in block.locator.get("block_type", "")
                or "table" in previous.locator.get("block_type", "")
                or sum(len(b.text) + 1 for b in current) + len(block.text) > size
            )
        if boundary:
            groups.append(current)
            current = []
        current.append(block)
    if current:
        groups.append(current)
    output = []
    for group in groups:
        text, offsets = "", []
        for block in group:
            if text:
                text += "\n"
            start = len(text)
            text += block.text
            offsets.append({"start": start, "end": len(text), "locator": block.locator})
        base = dict(group[0].locator)
        # Split concatenated text in its own coordinates; original positions are
        # always kept in sources, including blocks spanning a fixed window.
        base["kind_for_source"] = base["kind"]
        temp = Passage(text, {**base, "kind": "combined"})
        for piece in split_passages([temp], size, overlap, strategy):
            a, b = piece.locator["char_start"], piece.locator["char_end"]
            sources = [
                {
                    **span,
                    "overlap_start": max(a, span["start"]) - span["start"],
                    "overlap_end": min(b, span["end"]) - span["start"],
                }
                for span in offsets
                if span["start"] < b and span["end"] > a
            ]
            locator = {
                **base,
                "sources": sources,
                "char_start": a,
                "char_end": b,
                "chunk_strategy": strategy,
                "token_count_estimate": token_count(piece.text),
                "token_counter": "cl100k_base",
            }
            output.append(Passage(piece.text, locator))
    return output
