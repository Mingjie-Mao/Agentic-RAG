"""Cross-encoder reranking (S5).

Fusion orders candidates without ever reading the query and a passage together;
a cross-encoder does, which is why it can reorder what fusion cannot. It is off by
default and only becomes the default if the frozen set says it earns its latency.

The score is a relevance ordering signal. It is not a probability that the answer
is correct, and it is never shown to a reader as one.
"""

import os
from pathlib import Path
import threading

os.environ.setdefault("HF_HOME", str(Path(".runtime/huggingface").resolve()))

from app.clients import DependencyError  # noqa: E402
from app.config import settings  # noqa: E402

_lock = threading.Lock()
_loaded = {}


def _model():
    cfg = settings()
    key = cfg.rerank_model
    with _lock:
        if key not in _loaded:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - dependency is installed
                raise DependencyError("重排模型依赖未安装") from exc
            try:
                tokenizer = AutoTokenizer.from_pretrained(key, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(key, local_files_only=True)
            except OSError as exc:
                raise DependencyError(f"重排模型 {key} 尚未下载，请先运行 make rerank-model") from exc
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(device).eval()
            _loaded[key] = (tokenizer, model, device, torch)
        return _loaded[key]


def rerank(query, candidates, top_n=None, text_of=None):
    """Reorder `candidates` by cross-encoder relevance, best first.

    `text_of(candidate) -> str` supplies the passage text. Candidates beyond the
    configured window keep their fused order and are appended unchanged, so the
    reranker can only reorder what it actually scored.
    """
    cfg = settings()
    window = cfg.rerank_candidates
    if not candidates:
        return candidates
    head, tail = candidates[:window], candidates[window:]
    tokenizer, model, device, torch = _model()
    pairs = [(query, (text_of(item) if text_of else item["text"])) for item in head]
    scores = []
    for start in range(0, len(pairs), cfg.rerank_batch):
        batch = pairs[start : start + cfg.rerank_batch]
        encoded = tokenizer(
            [p[0] for p in batch],
            [p[1] for p in batch],
            padding=True,
            truncation=True,
            max_length=cfg.rerank_max_tokens,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits.view(-1).float()
        scores.extend(logits.tolist())
    ranked = sorted(
        (
            {**item, "rerank_score": score, "fused_rank": index + 1}
            for index, (item, score) in enumerate(zip(head, scores, strict=True))
        ),
        key=lambda item: -item["rerank_score"],
    )
    result = ranked + tail
    return result[:top_n] if top_n else result
