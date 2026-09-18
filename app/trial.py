"""Server-side limits for an ephemeral public interview trial."""

from datetime import timezone

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
    return {item.strip() for item in settings().trial_usernames.split(",") if item.strip()}


def is_trial_user(user: User) -> bool:
    return settings().trial_mode and user.username in trial_usernames()


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
