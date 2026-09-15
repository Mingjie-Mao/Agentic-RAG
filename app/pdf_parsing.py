"""Docling layout extraction; every block keeps its real file page and bounding box."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from app.parsing import ParseError, Passage


@lru_cache(maxsize=1)
def converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        document_timeout=180,
        artifacts_path=Path(".runtime/docling-models").resolve(),
        enable_remote_services=False,
        allow_external_plugins=False,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=4),
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


def parse_layout_pdf(data):
    from docling.datamodel.base_models import DocumentStream

    try:
        conversion = converter().convert(
            DocumentStream(name="source.pdf", stream=BytesIO(data)),
            max_num_pages=100,
            max_file_size=10 * 1024 * 1024,
        )
        if str(conversion.status).split(".")[-1].lower() != "success":
            raise ParseError("PDF 版面解析未完整成功，未发布部分结果")
        document, result, headings = conversion.document, [], []
        for item, _ in document.iterate_items():
            label = str(getattr(item, "label", "")).split(".")[-1].lower()
            text = getattr(item, "text", "")
            if label == "table":
                text = item.export_to_markdown(doc=document)
            if not text.strip() or not getattr(item, "prov", None):
                continue
            if label in {"section_header", "title"}:
                headings = [text.strip()]
            provenance = [
                {"page": p.page_no, "bbox": p.bbox.model_dump(mode="json"), "charspan": list(p.charspan)}
                for p in item.prov
            ]
            page = provenance[0]["page"]
            result.append(
                Passage(
                    text,
                    {
                        "kind": "pdf",
                        "page": page,
                        "block_type": label,
                        "heading_path": list(headings),
                        "provenance": provenance,
                        "item_ref": item.self_ref,
                        "label": f"第 {page} 页",
                    },
                )
            )
        if not result:
            raise ParseError("PDF 没有可检索的版面文字")
        return result
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError("PDF 版面解析失败，请确认解析模型已下载或检查文件") from exc
