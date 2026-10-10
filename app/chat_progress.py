"""Ephemeral UI telemetry, not an executor or a checkpoint store.

The synchronous QA request still owns execution and persists answers as before.
Only stage names and an answer handle live here, scoped to a tenant and user.
This is deliberately for the current single-process demo; multi-worker deployment
must use a shared telemetry store before exposing this polling endpoint.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from time import monotonic

from fastapi import HTTPException

_current = ContextVar("chat_progress", default=None)
_rows = {}
_lock = Lock()
TTL_SECONDS = 900
MAX_ROWS = 256


def _prune():
    expired = [key for key, row in _rows.items() if monotonic() - row["touched"] > TTL_SECONDS]
    for key in expired:
        del _rows[key]


def report_stage(stage):
    key = _current.get()
    if key is not None:
        with _lock:
            if key in _rows:
                _rows[key].update(stage=stage, touched=monotonic())


@contextmanager
def track_chat(user, identifier):
    key = (user.tenant_id, user.id, identifier)
    with _lock:
        _prune()
        if key in _rows:
            raise HTTPException(409, "该问题已经提交，请查看原请求状态")
        if len(_rows) >= MAX_ROWS:
            raise HTTPException(503, "问答服务繁忙，请稍后再试")
        _rows[key] = {
            "stage": "accepted",
            "status": "running",
            "started": monotonic(),
            "touched": monotonic(),
        }
    token = _current.set(key)
    try:
        yield _rows[key]
    except Exception:
        with _lock:
            _rows[key].update(status="failed", touched=monotonic())
        raise
    finally:
        _current.reset(token)


def complete_chat(row, answer_id):
    with _lock:
        row.update(
            status="completed",
            stage="completed",
            answer_id=answer_id,
            touched=monotonic(),
            elapsed_seconds=round(monotonic() - row["started"], 1),
        )


def chat_progress(user, identifier):
    with _lock:
        _prune()
        row = _rows.get((user.tenant_id, user.id, identifier))
        if row is None:
            raise HTTPException(404, "请求状态不可用；可在问答记录中查看已完成结果")
        return {key: value for key, value in row.items() if key not in {"started", "touched"}} | {
            "elapsed_seconds": row.get("elapsed_seconds", round(monotonic() - row["started"], 1))
        }
