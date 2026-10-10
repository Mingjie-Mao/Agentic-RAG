"""Server-side limits for an ephemeral public interview trial."""

from datetime import timezone
from secrets import token_urlsafe
from uuid import NAMESPACE_URL, uuid5

from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.models import (
    AgentEvent,
    AgentTask,
    Answer,
    LoginSession,
    ToolExecution,
    TrialRequest,
    User,
    now,
)


def trial_usernames() -> set[str]:
    return {item.strip() for item in settings().trial_usernames.split(",") if item.strip()} | {VISITOR_USERNAME}


VISITOR_USERNAME = "visitor@xingqiao.demo"


def visitor_user(db) -> User:
    """A stable member account has its own quota and owns no test documents."""
    cfg = settings()
    if not (cfg.demo_mode and cfg.trial_mode):
        raise HTTPException(404)
    configured = trial_usernames() - {VISITOR_USERNAME}
    source = db.scalar(select(User).where(
        User.username.in_(configured), User.active.is_(True), User.role == "member"
    ).order_by(User.id))
    if source is None:
        raise HTTPException(503, "只读演示身份尚未配置")
    identity = str(uuid5(NAMESPACE_URL, f"agentic-rag:visitor:{source.tenant_id}"))
    user = db.get(User, identity)
    if user is None:
        user = User(id=identity, tenant_id=source.tenant_id, username=VISITOR_USERNAME,
                    display_name="演示访客", role="member", groups=list(source.groups),
                    password_hash=token_urlsafe(64), active=True)
        try:
            with db.begin_nested():
                db.add(user)
                db.flush()
            db.commit()
        except IntegrityError:
            db.rollback()
            user = db.get(User, identity)
    if (user is None or not user.active or user.role != "member"
            or user.tenant_id != source.tenant_id or not set(user.groups) <= set(source.groups)):
        raise HTTPException(503, "只读演示身份需要检查")
    return user


def is_trial_user(user: User) -> bool:
    return settings().trial_mode and user.username in trial_usernames()


def require_trial_login(username: str) -> None:
    # Every demo account shares the published demo password, so while the trial is
    # public only the listed visitor accounts may sign in; the same message as a wrong
    # password avoids revealing which accounts exist.
    if settings().trial_mode and username not in trial_usernames():
        raise HTTPException(401, "账号或密码不正确")


def require_trial_read_only(user: User) -> None:
    if is_trial_user(user):
        raise HTTPException(403, "访客试用为只读模式，不能修改资料或长期记忆")


def reserve_trial_request(db, user: User, kind: str) -> None:
    if not is_trial_user(user):
        return
    if kind == "agent":
        active = db.scalar(
            select(func.count())
            .select_from(AgentTask)
            .where(AgentTask.user_id == user.id, AgentTask.status.in_(("queued", "running")))
        )
        if active:
            raise HTTPException(429, "访客同一时间只能运行一个 Agent 任务")
    today = now().astimezone(timezone.utc).date()
    limit = max(1, settings().trial_daily_limit)
    for slot in range(1, limit + 1):
        try:
            with db.begin_nested():
                db.add(TrialRequest(user_id=user.id, usage_date=today, slot=slot, kind=kind))
                db.flush()
            db.commit()
            return
        except IntegrityError:
            continue
    raise HTTPException(429, f"今日访客试用额度已用完（每天 {limit} 次）")


def trial_status(db, user: User) -> dict:
    if not is_trial_user(user):
        return {"enabled": False}
    today = now().astimezone(timezone.utc).date()
    used = db.scalar(
        select(func.count())
        .select_from(TrialRequest)
        .where(TrialRequest.user_id == user.id, TrialRequest.usage_date == today)
    ) or 0
    limit = max(1, settings().trial_daily_limit)
    return {"enabled": True, "read_only": True, "used": used, "limit": limit, "remaining": max(0, limit - used)}


def reset_trial(db, user: User) -> dict:
    """Remove visitor-created history while preserving seeded documents."""
    task_ids = list(db.scalars(select(AgentTask.id).where(AgentTask.user_id == user.id)))
    if task_ids:
        from agent.graph_runtime import graph_checkpointer
        with graph_checkpointer(db) as saver:
            for task_id in task_ids:
                saver.delete_thread(f"{user.tenant_id}:{user.id}:{task_id}")
        db.execute(delete(AgentEvent).where(AgentEvent.task_id.in_(task_ids)))
        db.execute(delete(ToolExecution).where(ToolExecution.task_id.in_(task_ids)))
    deleted_tasks = db.query(AgentTask).filter(AgentTask.user_id == user.id).delete(synchronize_session=False)
    deleted_answers = db.query(Answer).filter(Answer.user_id == user.id).delete(synchronize_session=False)
    deleted_sessions = db.query(LoginSession).filter(LoginSession.user_id == user.id).delete(synchronize_session=False)
    deleted_usage = db.execute(delete(TrialRequest).where(TrialRequest.user_id == user.id)).rowcount
    db.commit()
    return {
        "task_ids": len(task_ids),
        "tasks": deleted_tasks,
        "answers": deleted_answers,
        "sessions": deleted_sessions,
        "usage": deleted_usage,
    }
