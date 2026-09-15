import hashlib
from datetime import timedelta
from secrets import token_urlsafe

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Chunk, Document, DocumentVersion, LoginSession, User, now

hasher = PasswordHasher()
COOKIE = "rag_session"


def hash_token(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def authenticate(db: Session, username: str, password: str) -> tuple[User, str]:
    user = db.scalar(select(User).where(User.username == username, User.active.is_(True)))
    try:
        valid = user is not None and hasher.verify(user.password_hash, password)
    except (VerificationError, InvalidHashError):
        valid = False
    if not valid:
        raise HTTPException(401, "账号或密码不正确")
    token = token_urlsafe(40)
    db.add(
        LoginSession(
            token_hash=hash_token(token),
            user_id=user.id,
            expires_at=now() + timedelta(hours=settings().session_hours),
        )
    )
    db.commit()
    return user, token


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(COOKIE, "")
    session = db.get(LoginSession, hash_token(token)) if token else None
    if session is None or session.expires_at <= now():
        raise HTTPException(401, "请先登录")
    user = db.get(User, session.user_id, populate_existing=True)
    if user is None or not user.active:
        raise HTTPException(401, "账号已停用")
    return user


def can_read(user: User, document: Document) -> bool:
    return bool(
        user.active
        and document.tenant_id == user.tenant_id
        and not document.deleted
        and (
            user.role == "admin"
            or document.owner_id == user.id
            or document.tenant_public
            or set(user.groups).intersection(document.read_groups)
        )
    )


def can_write(user: User, document: Document) -> bool:
    return bool(
        user.active
        and document.tenant_id == user.tenant_id
        and not document.deleted
        and (user.role == "admin" or document.owner_id == user.id)
    )


def readable_documents(db: Session, user: User) -> list[Document]:
    db.refresh(user)
    documents = db.scalars(
        select(Document)
        .where(Document.tenant_id == user.tenant_id, Document.deleted.is_(False))
        .execution_options(populate_existing=True)
    ).all()
    return [d for d in documents if can_read(user, d)]


def require_document(db: Session, user: User, document_id: str, write=False) -> Document:
    db.refresh(user)
    document = db.get(Document, document_id, populate_existing=True)
    if document is None or not (can_write(user, document) if write else can_read(user, document)):
        # Do not distinguish an existing forbidden document from an unknown ID.
        raise HTTPException(404, "资料不存在或当前账号无权访问")
    return document


def require_chunk(
    db: Session, user: User, chunk_id: str, active_only=False
) -> tuple[Chunk, DocumentVersion, Document]:
    chunk = db.get(Chunk, chunk_id)
    version = db.get(DocumentVersion, chunk.version_id) if chunk else None
    if not chunk or not version:
        raise HTTPException(404, "证据不存在或当前账号无权访问")
    document = require_document(db, user, version.document_id)
    if active_only and document.active_version_id != version.id:
        raise HTTPException(409, "资料版本发生变化，请重新提问")
    return chunk, version, document
