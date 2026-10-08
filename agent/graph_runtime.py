"""Official LangGraph persistence with server-owned task identity.

The framework owns execution snapshots. ACL validation, cumulative model budgets and
an auditable tool ledger remain application rules, including on graph recovery.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
from weakref import WeakKeyDictionary, finalize

from fastapi import HTTPException
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from psycopg import Connection
from psycopg.rows import dict_row

from app.config import settings

_SQLITE_SAVERS = WeakKeyDictionary()
GRAPH_VERSION = "enterprise-agent-langgraph-v1"


def graph_identity(task, user):
    if task.user_id != user.id or task.tenant_id != user.tenant_id or not user.active:
        raise HTTPException(404, "任务不存在")
    return {"configurable": {"thread_id": f"{task.tenant_id}:{task.user_id}:{task.id}"},
            "recursion_limit": max(64, task.max_steps * 12 + 32)}


def graph_signature(task):
    cfg = settings().model_dump(mode="json")
    operational = {"database_url", "demo_password", "memory_tokens_json", "storage_dir", "allowed_origins",
                   "cookie_secure", "session_hours", "max_upload_bytes", "demo_mode", "trial_mode",
                   "trial_usernames", "trial_daily_limit", "agent_lease_seconds"}
    behavior = {key: val for key, val in cfg.items() if key not in operational}
    for key in ("semantic_slot_gate_path", "semantic_slot_calibration_path", "answer_semantic_audit_gate_path"):
        if cfg[key]:
            path = Path(cfg[key])
            behavior[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if key=='answer_semantic_audit_gate_path' and path.is_file():
                gate=json.loads(path.read_text())
                for entry in ('cases_path','result_path'):
                    linked=Path(gate[entry])
                    behavior[entry+'_sha256']=hashlib.sha256(linked.read_bytes()).hexdigest() if linked.is_file() else None
    root = Path(__file__).resolve().parents[1]
    code = hashlib.sha256()
    for folder in ["agent", "app"]:
        for path in sorted((root / folder).glob("*.py")):
            code.update(path.name.encode())
            code.update(path.read_bytes())
    return hashlib.sha256(json.dumps({"version": GRAPH_VERSION, "code": code.hexdigest(),
        "goal": task.goal, "mode": task.mode, "max_steps": task.max_steps,
        "input": {k: v for k, v in task.input.items() if not k.startswith("_")},
        "behavior": behavior}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@contextmanager
def graph_checkpointer(db):
    # Only primitive state is stored; never deserialize model/tool/session objects.
    serde = JsonPlusSerializer(allowed_json_modules=[], allowed_msgpack_modules=[])
    engine = db.get_bind()
    if engine.dialect.name == "sqlite":
        saver = _SQLITE_SAVERS.get(engine)
        if saver is None:
            database = engine.url.database
            path = database + ".langgraph.sqlite" if database and database != ":memory:" else ":memory:"
            saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=serde)
            _SQLITE_SAVERS[engine] = saver
            finalize(engine, saver.conn.close)
        yield saver
    elif engine.dialect.name == "postgresql":
        # Separate checkpoint commits must never commit an application's DB session.
        url = engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
        with Connection.connect(url, autocommit=True, prepare_threshold=0,
                                row_factory=dict_row, connect_timeout=5) as conn:
            saver = PostgresSaver(conn, serde=serde)
            saver.setup()  # Official, idempotent checkpoint-schema migrations.
            yield saver
    else:
        raise RuntimeError("LangGraph checkpoints require PostgreSQL or SQLite")
