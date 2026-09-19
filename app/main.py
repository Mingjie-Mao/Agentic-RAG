from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
import json
import re

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.clients import DependencyError
from app.config import settings
from app.db import get_db
from app.lifecycle import add_version, purge, readable_chunk, set_access, soft_delete
from app.ingestion import queue_document
from app.models import Answer, Chunk, DocumentVersion, Job, LoginSession, Tenant, User
from app.qa import answer_question, visible_answer
from app.observability import Timer, answer_record, log_request, request_id
from app.security import (
    COOKIE,
    authenticate,
    current_user,
    hash_token,
    readable_documents,
    require_chunk,
    require_document,
    validate_grant,
)
from agent.controller import cancel_task, create_task, require_task, resume_task, task_payload
from agent.memory_adapter import LongTermMemoryAdapter, MemoryUnavailable
from app.models import AgentTask
from app.trial import (
    is_trial_user,
    require_trial_read_only,
    reserve_trial_request,
    trial_status,
    trial_usernames,
)


@asynccontextmanager
async def lifespan(_):
    settings().storage_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Agentic-RAG", version="0.1.0", lifespan=lifespan)
DB = Annotated[Session, Depends(get_db)]
Identity = Annotated[User, Depends(current_user)]


@app.middleware("http")
async def trace_request(request: Request, call_next):
    """Stamp every request so one answer can be followed through the logs."""
    identifier = request.headers.get("X-Request-Id") or request_id()
    request.state.request_id = identifier
    with Timer() as timer:
        response = await call_next(request)
    response.headers["X-Request-Id"] = identifier
    log_request(
        request_id=identifier,
        route=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration_ms=timer.ms,
    )
    return response


@app.middleware("http")
async def browser_boundary(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        if request.headers.get("X-Requested-With") != "AgenticRAG":
            return JSONResponse({"detail": "缺少请求校验头"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin not in settings().allowed_origins:
            return JSONResponse({"detail": "不允许的请求来源"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(DependencyError)
async def dependency_error(_, exc):
    # The stage decides what the reader may safely be told about retrying.
    stage = getattr(exc, "stage", "unknown")
    return JSONResponse(
        {"detail": str(exc), "stage": stage, "model_reached": stage == "generation"},
        status_code=503,
    )


class LoginBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class AccessBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    groups: list[str] = Field(default_factory=list, max_length=10)
    tenant_public: bool = False


class QuestionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=2, max_length=1000)
    # The client may replay the user's own earlier questions so a follow-up can be
    # completed into a searchable query. It only steers retrieval; every document is
    # authorised again from scratch, so replayed text grants nothing.
    history: list[str] = Field(default_factory=list, max_length=10)


class AgentTaskBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = Field(min_length=4, max_length=1500)
    mode: Literal["auto", "workflow", "dynamic"] = "auto"
    max_steps: int = Field(default=6, ge=2, le=8)
    document_id: str | None = Field(default=None, max_length=64)
    from_version_id: str | None = Field(default=None, max_length=64)
    to_version_id: str | None = Field(default=None, max_length=64)
    # Long-term memory is an optional personalization layer. The paired A/B
    # experiment found no factual-accuracy gain and measurable latency/token
    # overhead, so callers must opt in explicitly.
    use_memory: bool = False


class MemoryWriteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=4, max_length=2000)
    session_id: str | None = Field(default=None, max_length=200)


def user_info(user, db):
    tenant = db.get(Tenant, user.tenant_id)
    return {
        "id": user.id,
        "name": user.display_name,
        "username": user.username,
        "tenant": tenant.name,
        "tenant_id": tenant.id,
        "role": user.role,
        "groups": user.groups,
        "trial": trial_status(db, user),
    }


@app.get("/api/health")
def health(db: DB):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "release": "agent-v1", "agent": "v1"}
    except Exception:
        return JSONResponse({"status": "unavailable"}, status_code=503)


@app.get("/api/demo/accounts")
def demo_accounts(db: DB):
    if not settings().demo_mode:
        raise HTTPException(404)
    users = list(db.scalars(select(User).where(User.active.is_(True))))
    if settings().trial_mode:
        allowed = trial_usernames()
        users = [user for user in users if user.username in allowed]
    return {
        "accounts": [user_info(u, db) for u in users],
        "password": settings().demo_password,
        "notice": "仅含虚构资料的临时试用环境；请勿上传或输入真实敏感数据。",
    }


@app.post("/api/auth/login")
def login(body: LoginBody, response: Response, db: DB):
    user, token = authenticate(db, body.username, body.password)
    response.set_cookie(
        COOKIE,
        token,
        httponly=True,
        samesite="strict",
        secure=settings().cookie_secure,
        max_age=settings().session_hours * 3600,
        path="/",
    )
    return user_info(user, db)


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, db: DB):
    token = request.cookies.get(COOKIE)
    row = db.get(LoginSession, hash_token(token)) if token else None
    if row:
        db.delete(row)
        db.commit()
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: Identity, db: DB):
    return user_info(user, db)


@app.get("/api/system")
def system_info(user: Identity):
    return {
        "stage": "Phase B · Agent MVP",
        "retrieval": settings().retrieval_mode,
        "embedding_model": settings().embed_model,
        "generation_model": settings().chat_model,
        "formats": ["md", "pdf", "docx", "xlsx"],
        "api_cost": "0 AUD（本地模型）",
        "max_upload_mb": settings().max_upload_bytes // (1024 * 1024),
    }


def document_info(db, doc):
    version = db.scalar(
        select(DocumentVersion)
        .where(DocumentVersion.document_id == doc.id)
        .order_by(DocumentVersion.created_at.desc())
        .limit(1)
    )
    count = (
        db.scalar(select(func.count()).select_from(Chunk).where(Chunk.version_id == version.id))
        if version
        else 0
    )
    return {
        "id": doc.id,
        "title": doc.title,
        "status": version.status if version else "queued",
        "version_id": version.id if version else None,
        "filename": version.filename if version else "",
        "media_type": version.media_type if version else "",
        "error": version.error if version else None,
        "chunks": count,
        "groups": doc.read_groups,
        "tenant_public": doc.tenant_public,
        "metadata": doc.metadata_json,
        "timings": version.timings if version else {},
        "deduplicated": getattr(doc, "_deduplicated", False),
    }


@app.get("/api/documents")
def list_documents(user: Identity, db: DB):
    return [document_info(db, doc) for doc in readable_documents(db, user)]


@app.post("/api/documents", status_code=202)
def upload_document(
    user: Identity,
    db: DB,
    file: UploadFile = File(...),
    title: str = Form(""),
    groups: str = Form("[]"),
    tenant_public: bool = Form(False),
):
    require_trial_read_only(user)
    try:
        group_list = json.loads(groups)
        if not isinstance(group_list, list) or any(not isinstance(g, str) for g in group_list):
            raise ValueError
    except ValueError:
        raise HTTPException(422, "groups 必须是字符串数组")
    group_list = validate_grant(user, group_list, tenant_public)
    filename = Path(file.filename or "file.md").name
    if len(filename) > 200 or re.search(r"[\x00-\x1f]", filename):
        raise HTTPException(422, "文件名不合法")
    title = title.strip() or filename
    if len(title) > 200:
        raise HTTPException(422, "标题最多 200 字")
    data = file.file.read(settings().max_upload_bytes + 1)
    try:
        document, _ = queue_document(db, user, filename, data, title, group_list, tenant_public)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return document_info(db, document)


@app.get("/api/documents/{document_id}")
def get_document(document_id: str, user: Identity, db: DB):
    return document_info(db, require_document(db, user, document_id))


@app.get("/api/documents/{document_id}/processing")
def processing_preview(document_id: str, user: Identity, db: DB):
    doc = require_document(db, user, document_id)
    history = db.scalars(
        select(DocumentVersion)
        .where(DocumentVersion.document_id == doc.id)
        .order_by(DocumentVersion.created_at.desc())
    ).all()
    if not history:
        raise HTTPException(404, "资料尚未生成任何版本")
    version = history[0]
    chunks = db.scalars(
        select(Chunk).where(Chunk.version_id == version.id).order_by(Chunk.ordinal)
    ).all()
    return {
        "document_id": doc.id,
        "title": doc.title,
        "version_id": version.id,
        "active_version_id": doc.active_version_id,
        "status": version.status,
        "pipeline": version.pipeline,
        # Which exact file and which processing configuration produced these chunks.
        # One entry today; version replacement lands in S6 and reuses this shape.
        "versions": [
            {
                "version_id": row.id,
                "filename": row.filename,
                "media_type": row.media_type,
                "content_hash": row.content_hash,
                "status": row.status,
                "created_at": row.created_at.isoformat(),
                "active": row.id == doc.active_version_id,
                "chunks": db.scalar(
                    select(func.count()).select_from(Chunk).where(Chunk.version_id == row.id)
                ),
                "parser": row.pipeline.get("parser"),
                "chunking": f"{row.pipeline.get('chunking')} · {row.pipeline.get('chunk_chars')}字 / 重叠{row.pipeline.get('overlap')}字",
                "embedding": row.pipeline.get("embedding_model"),
                "timings": row.timings,
            }
            for row in history
        ],
        "blocks": version.parsed_blocks,
        "chunks": [{"id": c.id, "text": c.text, "locator": c.locator} for c in chunks],
        "original_url": f"/api/versions/{version.id}/original",
    }


@app.post("/api/documents/{document_id}/versions", status_code=202)
def replace_version(
    document_id: str, user: Identity, db: DB, file: UploadFile = File(...), title: str = Form("")
):
    """Queue a replacement. The active version keeps serving until this one publishes."""
    require_trial_read_only(user)
    data = file.file.read()
    try:
        document, version, _ = add_version(db, user, document_id, file.filename or "", data)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if title.strip():
        document.title = title.strip()
        db.commit()
    return {
        "id": document.id,
        "version_id": version.id,
        "active_version_id": document.active_version_id,
        "status": version.status,
        "note": "新版本索引完成并确认可搜索后才会生效；在此之前旧版本继续可用",
    }


@app.patch("/api/documents/{document_id}/access")
def change_access(document_id: str, body: AccessBody, user: Identity, db: DB):
    require_trial_read_only(user)
    groups = validate_grant(user, body.groups, body.tenant_public)
    document = set_access(db, user, document_id, groups, body.tenant_public)
    return document_info(db, document)


@app.delete("/api/documents/{document_id}")
def remove_document(document_id: str, user: Identity, db: DB):
    """Commit the deletion first; index and file cleanup follow and may fail safely."""
    require_trial_read_only(user)
    _, versions, storage_keys = soft_delete(db, user, document_id)
    cleanup = purge(versions, storage_keys)
    return {"id": document_id, "deleted": True, "versions": len(versions), "cleanup": cleanup}


@app.get("/api/access/chunks/{chunk_id}")
def chunk_readable(chunk_id: str, user: Identity, db: DB):
    """Whether this user may read this chunk right now.

    Exists so that anything derived from this corpus elsewhere can be revoked when a
    reader loses access, without that system needing to model permissions itself.
    """
    return readable_chunk(db, user, chunk_id)


@app.post("/api/documents/{document_id}/retry")
def retry_document(document_id: str, user: Identity, db: DB):
    require_trial_read_only(user)
    doc = require_document(db, user, document_id, write=True)
    version = db.scalar(
        select(DocumentVersion)
        .where(DocumentVersion.document_id == doc.id)
        .order_by(DocumentVersion.created_at.desc())
        .limit(1)
    )
    if not version or version.status != "failed":
        raise HTTPException(409, "只有处理失败的文件可以重试")
    job = db.scalar(select(Job).where(Job.version_id == version.id).with_for_update())
    job.state, job.attempts, job.lease_token = "queued", 0, None
    version.status, version.error = "queued", None
    db.commit()
    return {"status": "queued"}


@app.get("/api/evidence/{chunk_id}")
def get_evidence(chunk_id: str, user: Identity, db: DB):
    chunk, version, document = require_chunk(db, user, chunk_id)
    return {
        "chunk_id": chunk.id,
        "title": document.title,
        "text": chunk.text,
        "locator": chunk.locator,
        "version_id": version.id,
        "filename": version.filename,
        "media_type": version.media_type,
        "content_hash": version.content_hash,
        "original_url": f"/api/versions/{version.id}/original",
    }


@app.get("/api/versions/{version_id}/original")
def original(version_id: str, user: Identity, db: DB):
    version = db.get(DocumentVersion, version_id)
    if not version:
        raise HTTPException(404, "资料不存在或当前账号无权访问")
    require_document(db, user, version.document_id)
    path = settings().storage_dir.resolve() / version.storage_key
    if not path.is_file():
        raise HTTPException(404, "原文已不可用")
    media = "text/plain; charset=utf-8" if version.media_type == "text/markdown" else version.media_type
    return FileResponse(
        path,
        media_type=media,
        filename=version.filename,
        content_disposition_type="attachment"
        if version.media_type.endswith(("wordprocessingml.document", "spreadsheetml.sheet"))
        else "inline",
    )


@app.post("/api/chat")
def chat(body: QuestionBody, user: Identity, db: DB, request: Request):
    question = body.question.strip()
    if len(question) < 2:
        raise HTTPException(422, "请输入至少两个字符的问题")
    history = [line.strip()[:1000] for line in body.history if line.strip()]
    reserve_trial_request(db, user, "chat")
    payload = answer_question(db, user, question, history)
    # Costs and stage timings are recorded; the question and the evidence are not.
    log_request(
        request_id=getattr(request.state, "request_id", None),
        route="/api/chat",
        tenant_id=user.tenant_id,
        user_id=user.id,
        answer_id=payload.get("id"),
        **answer_record(payload),
    )
    return payload


@app.post("/api/agent/tasks", status_code=202)
def start_agent_task(body: AgentTaskBody, user: Identity, db: DB):
    """Run a bounded read-only knowledge task and retain its auditable trajectory."""
    if is_trial_user(user) and body.mode == "dynamic":
        raise HTTPException(403, "访客试用不开放高延迟动态模式，请使用自动路由")
    reserve_trial_request(db, user, "agent")
    task_input = {
        key: value
        for key, value in {
            "document_id": body.document_id,
            "from_version_id": body.from_version_id,
            "to_version_id": body.to_version_id,
            "use_memory": body.use_memory,
        }.items()
        if value is not None
    }
    requested_mode = body.mode
    # Current benchmark shows no quality gain from free-form planning. Auto therefore
    # selects the deterministic workflow; dynamic remains available as an experiment.
    mode = "workflow" if requested_mode == "auto" else requested_mode
    task_input["requested_mode"] = requested_mode
    task = create_task(db, user, body.goal.strip(), mode, body.max_steps, task_input)
    return task_payload(db, user, task)


@app.get("/api/agent/tasks")
def list_agent_tasks(user: Identity, db: DB):
    rows = db.scalars(
        select(AgentTask)
        .where(AgentTask.user_id == user.id, AgentTask.tenant_id == user.tenant_id)
        .order_by(AgentTask.created_at.desc())
        .limit(30)
    ).all()
    return [task_payload(db, user, row) for row in rows]


@app.get("/api/agent/tasks/{task_id}")
def get_agent_task(task_id: str, user: Identity, db: DB):
    return task_payload(db, user, require_task(db, user, task_id))


@app.post("/api/agent/tasks/{task_id}/resume")
def resume_agent_task(task_id: str, user: Identity, db: DB):
    reserve_trial_request(db, user, "agent")
    task = resume_task(db, user, task_id)
    return task_payload(db, user, task)


@app.post("/api/agent/memories")
def write_agent_memory(body: MemoryWriteBody, request: Request, user: Identity):
    """Explicit memory write; Agent execution never writes memories implicitly."""
    require_trial_read_only(user)
    try:
        adapter = LongTermMemoryAdapter(user)
        result = adapter.remember(
            body.content.strip(),
            body.session_id,
            request.headers.get("Idempotency-Key") or request.state.request_id,
        )
        return {"status": "accepted", "memory": result}
    except MemoryUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/agent/tasks/{task_id}/cancel")
def cancel_agent_task(task_id: str, user: Identity, db: DB):
    task = cancel_task(db, user, task_id)
    return task_payload(db, user, task)


@app.get("/api/trial/status")
def get_trial_status(user: Identity, db: DB):
    return trial_status(db, user)


@app.get("/api/history")
def history(user: Identity, db: DB):
    rows = db.scalars(
        select(Answer)
        .where(Answer.user_id == user.id, Answer.tenant_id == user.tenant_id)
        .order_by(Answer.created_at.desc())
        .limit(30)
    ).all()
    return [visible_answer(db, user, row) for row in rows]


@app.get("/api/history/{answer_id}")
def get_history(answer_id: str, user: Identity, db: DB):
    row = db.get(Answer, answer_id)
    if not row:
        raise HTTPException(404, "记录不存在")
    return visible_answer(db, user, row)


dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
