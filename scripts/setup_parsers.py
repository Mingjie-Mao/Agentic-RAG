from pathlib import Path
import os

os.environ.setdefault("HF_HOME", str(Path(".runtime/huggingface").resolve()))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".runtime/tiktoken").resolve()))

from docling.utils.model_downloader import download_models
import tiktoken

download_models(
    output_dir=Path(".runtime/docling-models").resolve(),
    with_layout=True,
    with_tableformer=True,
    with_code_formula=False,
    with_picture_classifier=False,
    with_rapidocr=False,
    progress=False,
)
tiktoken.get_encoding("cl100k_base")
print("PDF layout/table models and token counter ready; OCR and remote document services disabled")
