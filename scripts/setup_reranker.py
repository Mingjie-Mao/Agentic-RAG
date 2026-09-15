"""Download the cross-encoder reranker into the project runtime directory."""

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".runtime/huggingface").resolve()))

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from app.config import settings  # noqa: E402

name = settings().rerank_model
AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)
print(f"Reranker ready: {name} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters)")
