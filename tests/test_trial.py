from fastapi import HTTPException
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from agent.controller import create_task
from app.config import settings
from app.models import Base, Tenant, TrialRequest, User
from app.trial import (
    is_trial_user,
    require_trial_read_only,
    reserve_trial_request,
    reset_trial,
    trial_status,
)


def trial_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = Session(engine, expire_on_commit=False)
    db.add(Tenant(id="tenant-a", name="A"))
    user = User(
        id="trial-user",
        tenant_id="tenant-a",
        username="visitor@example.test",
        display_name="Visitor",
        password_hash="unused",
        role="member",
        groups=["support"],
        active=True,
    )
    db.add(user)
    db.commit()
    return db, user


def enable_trial(monkeypatch, limit=2):
    cfg = settings()
    monkeypatch.setattr(cfg, "trial_mode", True)
    monkeypatch.setattr(cfg, "trial_usernames", "visitor@example.test")
    monkeypatch.setattr(cfg, "trial_daily_limit", limit)


def test_trial_is_read_only_and_reserves_a_hard_daily_budget(monkeypatch):
    db, user = trial_db()
    enable_trial(monkeypatch)

    assert is_trial_user(user)
    try:
        require_trial_read_only(user)
    except HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("trial mutation must be rejected")

    reserve_trial_request(db, user, "chat")
    reserve_trial_request(db, user, "agent")
    assert trial_status(db, user)["remaining"] == 0
    try:
        reserve_trial_request(db, user, "chat")
    except HTTPException as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("daily limit must be enforced")


def test_trial_allows_only_one_active_agent_and_can_be_reset(monkeypatch):
    db, user = trial_db()
    enable_trial(monkeypatch, limit=10)
    reserve_trial_request(db, user, "agent")
    create_task(db, user, "检查支持政策", "workflow", 4)

    try:
        reserve_trial_request(db, user, "agent")
    except HTTPException as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("second active Agent task must be rejected")

    result = reset_trial(db, user)
    assert result["tasks"] == 1
    assert db.scalar(select(func.count()).select_from(TrialRequest)) == 0
    assert trial_status(db, user)["remaining"] == 10
