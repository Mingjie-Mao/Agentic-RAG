"""Generate source fixtures and their provenance manifest; no model-generated gold labels."""

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

ROOT = Path(__file__).resolve().parents[1]


def main():
    catalog = json.loads((ROOT / "fixtures/catalog.json").read_text())
    directory = ROOT / "fixtures/documents"
    directory.mkdir(exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    style = ParagraphStyle("source", fontName="STSong-Light", fontSize=11, leading=19, wordWrap="CJK")
    manifest = []
    for doc in catalog["documents"]:
        suffix = doc["format"]
        path = directory / f"{doc['id']}.{suffix}"
        body = f"# {doc['title']}\n\n{catalog['provenance']}\n\n{doc['body']}\n"
        if suffix == "md":
            path.write_text(body, encoding="utf-8")
        else:
            story = []
            for line in body.splitlines():
                story.extend([Paragraph(escape(line.lstrip("# ")) or " ", style), Spacer(1, 5)])
            SimpleDocTemplate(
                str(path), title=doc["title"], author="Enterprise-RAG synthetic fixture", invariant=1
            ).build(story)
        tenant = doc.get("tenant", "xingqiao")
        manifest.append(
            {
                "id": "seed-" + doc["id"],
                "source_key": doc["id"],
                "tenant_id": tenant,
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "title": doc["title"],
                "family": doc["family"],
                "split": "reserved" if doc["family"] == "billing" else "development",
                "tenant_public": doc.get("public", False),
                "groups": doc.get("groups", []),
                "effective_from": doc.get("effective_from"),
                "effective_to": doc.get("effective_to"),
                "provenance": catalog["provenance"],
                "version": catalog["dataset_version"],
            }
        )
    (ROOT / "fixtures/manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"Generated {len(manifest)} fixtures ({sum(x['path'].endswith('.pdf') for x in manifest)} PDF)."
    )


if __name__ == "__main__":
    main()
