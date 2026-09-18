"""Finite, read-only knowledge tools. Caller identity is always server supplied."""

import difflib
from dataclasses import asdict, dataclass

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.clients import Models, Search
from app.config import settings
from app.lifecycle import readable_chunk
from app.models import Chunk, DocumentVersion
from app.retrieval import retrieve_authorized
from app.security import require_chunk, require_document
from agent.memory_adapter import LongTermMemoryAdapter, MemoryUnavailable


@dataclass
class ToolResult:
    status: str
    data: dict
    evidence_refs: list[str]
    freshness: dict
    error_code: str | None = None
    retryable: bool = False
    usage: dict | None = None

    def dump(self):
        return asdict(self)


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=4, ge=1, le=8)


class EvidenceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_ids: list[str] = Field(min_length=1, max_length=8)


class DocumentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1, max_length=64)
    version_id: str | None = Field(default=None, max_length=64)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4, ge=1, le=8)


class VersionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1, max_length=64)
    limit: int = Field(default=10, ge=1, le=20)


class CompareArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1, max_length=64)
    from_version_id: str | None = Field(default=None, max_length=64)
    to_version_id: str | None = Field(default=None, max_length=64)
    max_diff_lines: int = Field(default=120, ge=20, le=300)


class VerifyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_ids: list[str] = Field(min_length=1, max_length=20)


class MemorySearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=5, ge=1, le=8)


ARGUMENTS = {
    "search_documents": SearchArgs,
    "retrieve_evidence": EvidenceArgs,
    "open_document": DocumentArgs,
    "get_document_version": VersionArgs,
    "compare_versions": CompareArgs,
    "verify_chunk_access": VerifyArgs,
    "search_memory": MemorySearchArgs,
}


class KnowledgeTools:
    def __init__(self, db, user, *, models=None, search=None, memory=None):
        self.db = db
        self.user = user
        self.models = models or Models()
        self.search = search or Search()
        self.memory = memory or LongTermMemoryAdapter(user)

    def call(self, name, arguments):
        schema = ARGUMENTS.get(name)
        if schema is None:
            return ToolResult("error", {}, [], {}, "unknown_tool", False)
        try:
            parsed = schema.model_validate(arguments)
            return getattr(self, name)(parsed)
        except HTTPException as exc:
            return ToolResult("error", {}, [], {}, f"http_{exc.status_code}", False)
        except ValueError:
            return ToolResult("error", {}, [], {}, "invalid_arguments", False)
        except MemoryUnavailable:
            return ToolResult("error", {}, [], {}, "memory_unavailable", True)

    def search_memory(self, args: MemorySearchArgs):
        if not self.memory.enabled:
            return ToolResult("error", {}, [], {}, "memory_not_configured", False)
        result = self.memory.search(args.query, args.limit)
        return ToolResult(
            "ok",
            result,
            [],
            {"scope": "trusted_user_namespace", "checked_now": True},
        )

    def search_documents(self, args: SearchArgs):
        found = retrieve_authorized(
            self.db,
            self.user,
            args.query,
            cfg=settings(),
            models=self.models,
            search=self.search,
            top_k=args.top_k,
        )
        rows = []
        for evidence in found.evidence:
            rows.append(
                {
                    "document_id": evidence["document_id"],
                    "version_id": evidence["version_id"],
                    "chunk_id": evidence["chunk_id"],
                    "title": evidence["title"],
                    "locator": evidence["locator"],
                    "snippet": evidence["text"][:500],
                }
            )
        refs = [row["chunk_id"] for row in rows]
        return ToolResult(
            "ok",
            {"query": args.query, "matches": rows, "candidate_count": len(found.candidates)},
            refs,
            {"scope": "active_versions", "checked_now": True},
            usage={"embed_ms": round(found.embed_ms, 1), "retrieval_ms": round(found.retrieval_ms, 1)},
        )

    def retrieve_evidence(self, args: EvidenceArgs):
        rows = []
        for chunk_id in dict.fromkeys(args.chunk_ids):
            chunk, version, document = require_chunk(
                self.db, self.user, chunk_id, active_only=True
            )
            rows.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "version_id": version.id,
                    "title": document.title,
                    "text": chunk.text,
                    "locator": chunk.locator,
                    "metadata": document.metadata_json,
                }
            )
        return ToolResult(
            "ok", {"evidence": rows}, [row["chunk_id"] for row in rows], {"checked_now": True}
        )

    def open_document(self, args: DocumentArgs):
        document = require_document(self.db, self.user, args.document_id)
        version_id = args.version_id or document.active_version_id
        version = self.db.get(DocumentVersion, version_id) if version_id else None
        if version is None or version.document_id != document.id:
            raise HTTPException(404, "版本不存在或当前账号无权访问")
        chunks = self.db.scalars(
            select(Chunk)
            .where(Chunk.version_id == version.id)
            .order_by(Chunk.ordinal)
            .offset(args.offset)
            .limit(args.limit)
        ).all()
        rows = [
            {"chunk_id": row.id, "ordinal": row.ordinal, "text": row.text, "locator": row.locator}
            for row in chunks
        ]
        return ToolResult(
            "ok",
            {
                "document_id": document.id,
                "title": document.title,
                "version_id": version.id,
                "is_active": version.id == document.active_version_id,
                "chunks": rows,
                "next_offset": args.offset + len(rows) if len(rows) == args.limit else None,
            },
            [row["chunk_id"] for row in rows],
            {"document_revision": document.revision, "checked_now": True},
        )

    def get_document_version(self, args: VersionArgs):
        document = require_document(self.db, self.user, args.document_id)
        versions = self.db.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document.id)
            .order_by(DocumentVersion.created_at.desc())
            .limit(args.limit)
        ).all()
        return ToolResult(
            "ok",
            {
                "document_id": document.id,
                "title": document.title,
                "active_version_id": document.active_version_id,
                "versions": [
                    {
                        "version_id": row.id,
                        "filename": row.filename,
                        "content_hash": row.content_hash,
                        "status": row.status,
                        "created_at": row.created_at.isoformat(),
                        "is_active": row.id == document.active_version_id,
                    }
                    for row in versions
                ],
            },
            [],
            {"document_revision": document.revision, "checked_now": True},
        )

    def compare_versions(self, args: CompareArgs):
        document = require_document(self.db, self.user, args.document_id)
        versions = self.db.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document.id, DocumentVersion.status == "ready")
            .order_by(DocumentVersion.created_at.desc())
        ).all()
        by_id = {row.id: row for row in versions}
        to_version = by_id.get(args.to_version_id or document.active_version_id)
        from_version = by_id.get(args.from_version_id) if args.from_version_id else None
        if from_version is None and to_version:
            from_version = next((row for row in versions if row.id != to_version.id), None)
        if from_version is None or to_version is None or from_version.id == to_version.id:
            return ToolResult("error", {}, [], {}, "two_versions_required", False)

        def version_chunks(version_id):
            return self.db.scalars(
                select(Chunk).where(Chunk.version_id == version_id).order_by(Chunk.ordinal)
            ).all()

        left, right = version_chunks(from_version.id), version_chunks(to_version.id)
        left_lines = "\n".join(row.text for row in left).splitlines()
        right_lines = "\n".join(row.text for row in right).splitlines()
        diff = list(
            difflib.unified_diff(
                left_lines,
                right_lines,
                fromfile=from_version.filename,
                tofile=to_version.filename,
                lineterm="",
                n=2,
            )
        )
        truncated = len(diff) > args.max_diff_lines
        refs = []
        for index in range(10):
            if index < len(left):
                refs.append(left[index].id)
            if index < len(right):
                refs.append(right[index].id)
        return ToolResult(
            "ok",
            {
                "document_id": document.id,
                "title": document.title,
                "from_version_id": from_version.id,
                "to_version_id": to_version.id,
                "diff_lines": diff[: args.max_diff_lines],
                "truncated": truncated,
                "note": "这是确定性文本差异，不等同于语义政策变化。",
            },
            refs,
            {"document_revision": document.revision, "checked_now": True},
        )

    def verify_chunk_access(self, args: VerifyArgs):
        checks = [readable_chunk(self.db, self.user, chunk_id) for chunk_id in args.chunk_ids]
        return ToolResult(
            "ok",
            {"checks": checks, "all_readable": all(row["readable"] for row in checks)},
            [row["chunk_id"] for row in checks if row["readable"]],
            {"checked_now": True},
        )
