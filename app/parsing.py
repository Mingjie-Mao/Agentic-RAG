from dataclasses import dataclass
from io import BytesIO
import re

from pypdf import PdfReader


class ParseError(ValueError):
    pass


@dataclass
class Passage:
    text: str
    locator: dict


def parse_document(data: bytes, media_type: str, *, structured=False) -> list[Passage]:
    if media_type.endswith("wordprocessingml.document"):
        from app.office_parsing import parse_docx

        return parse_docx(data)
    if media_type.endswith("spreadsheetml.sheet"):
        from app.office_parsing import parse_xlsx

        return parse_xlsx(data)
    if media_type == "text/markdown":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ParseError("Markdown 必须使用 UTF-8 编码") from exc
        if "\x00" in text:
            raise ParseError("文件包含无效二进制内容")
        if not text.strip():
            raise ParseError("文件没有可检索的文字")
        text = text.replace("\r\n", "\n")
        if structured:
            return markdown_blocks(text)
        return [Passage(text, {"kind": "markdown", "label": "原文"})]
    if media_type != "application/pdf":
        raise ParseError("支持 Markdown、文本型 PDF、DOCX 和 XLSX")
    if not data.startswith(b"%PDF-"):
        raise ParseError("不是有效的 PDF 文件")
    try:
        pdf = PdfReader(BytesIO(data), strict=False)
        if pdf.is_encrypted:
            raise ParseError("暂不支持加密 PDF，请提供未加密版本")
        if len(pdf.pages) > 100:
            raise ParseError("S1 单份 PDF 最多支持 100 页")
        pages = []
        for number, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(Passage(text, {"kind": "pdf", "page": number, "label": f"第 {number} 页"}))
        if not pages or sum(len(p.text) for p in pages) < 10:
            raise ParseError("未提取到足够文字；扫描 PDF 需要 OCR，S1 暂未支持")
        if structured:
            from app.pdf_parsing import parse_layout_pdf

            return parse_layout_pdf(data)
        return pages
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError("PDF 无法解析，请检查文件是否损坏") from exc


def split_passages(passages: list[Passage], size=700, overlap=100, strategy="fixed") -> list[Passage]:
    if size <= overlap or overlap < 0:
        raise ValueError("Invalid chunk size/overlap")
    chunks = []
    for passage in passages:
        text = passage.text
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            # Prefer complete lines/sentences without producing very small fragments.
            if end < len(text):
                boundaries = [m.end() for m in re.finditer(r"[。！？\n]", text[start:end])]
                if boundaries and boundaries[-1] > size // 2:
                    end = start + boundaries[-1]
            part = text[start:end]
            if part.strip():
                locator = {**passage.locator, "char_start": start, "char_end": end}
                if locator["kind"] == "markdown":
                    locator.update(
                        line_start=text[:start].count("\n") + passage.locator.get("line_start", 1),
                        line_end=text[:end].count("\n") + passage.locator.get("line_start", 1),
                    )
                    locator["label"] = f"第 {locator['line_start']}–{locator['line_end']} 行"
                locator["chunk_strategy"] = strategy
                chunks.append(Passage(part, locator))
            if end == len(text):
                break
            start = end - overlap
    return chunks


def markdown_blocks(text):
    lines = text.splitlines(keepends=True)
    result, headings, buffer, start, offset, fenced = [], [], [], 0, 0, False

    def flush():
        nonlocal buffer
        content = "".join(buffer)
        if content.strip():
            kind = (
                "code"
                if content.lstrip().startswith("```")
                else "table"
                if any(line.lstrip().startswith("|") for line in buffer)
                else "paragraph"
            )
            line_start = text[:start].count("\n") + 1
            result.append(
                Passage(
                    content,
                    {
                        "kind": "markdown",
                        "block_type": kind,
                        "heading_path": list(headings),
                        "source_char_start": start,
                        "line_start": line_start,
                        "label": f"第 {line_start} 行起",
                    },
                )
            )
        buffer = []

    for line in lines:
        if not fenced and re.match(r"^#{1,6}\s", line):
            flush()
            level = len(line) - len(line.lstrip("#"))
            headings = headings[: level - 1] + [line.lstrip("#").strip()]
        if not buffer:
            start = offset
        buffer.append(line)
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and not line.strip():
            flush()
        offset += len(line)
    flush()
    return result
