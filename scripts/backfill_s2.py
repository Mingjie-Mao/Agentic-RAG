"""Enrich existing immutable versions without replacing source IDs or embeddings."""

import hashlib

from sqlalchemy import select

from app.clients import Search
from app.config import settings
from app.db import SessionLocal
from app.models import Chunk, Document, DocumentVersion
from app.parsing import parse_document

search = Search()
search.ensure_index()
count = 0
with SessionLocal() as db:
    rows = db.execute(
        select(Document, DocumentVersion)
        .join(DocumentVersion, Document.active_version_id == DocumentVersion.id)
        .where(Document.deleted.is_(False))
    ).all()
    for document, version in rows:
        if not version.parsed_blocks:
            data = (settings().storage_dir / version.storage_key).read_bytes()
            assert hashlib.sha256(data).hexdigest() == version.content_hash
            blocks = parse_document(
                data, version.media_type, structured=version.pipeline["parser"] != "pypdf6+utf8-v1"
            )
            version.parsed_blocks = [{"text": b.text, "locator": b.locator} for b in blocks]
        for chunk in db.scalars(select(Chunk).where(Chunk.version_id == version.id)):
            result = search.request(
                "POST",
                f"/{search.index}/_update/{chunk.id}",
                json={"doc": {"text": chunk.text, "title": document.title}},
            )
            assert result["result"] in {"updated", "noop"}
        db.commit()
        count += 1
search.request("POST", f"/{search.index}/_refresh")
print(
    f"Backfilled parsed views and BM25 fields for {count} documents; source/version/chunk identities preserved"
)
