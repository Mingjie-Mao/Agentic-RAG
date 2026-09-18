"""Add durable Agent tasks, events and idempotent tool executions."""

from alembic import op
import sqlalchemy as sa

revision = "0003_agent"
down_revision = "0002_processing"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "agent_tasks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("user_id", sa.String(64), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("input", sa.JSON(), nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("step_no", sa.Integer(), nullable=False),
        sa.Column("max_steps", sa.Integer(), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("evidence_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agent_tasks_tenant_id", "agent_tasks", ["tenant_id"])
    op.create_index("ix_agent_tasks_user_id", "agent_tasks", ["user_id"])
    op.create_index("ix_agent_tasks_status", "agent_tasks", ["status"])
    op.create_table(
        "agent_events",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(64), sa.ForeignKey("agent_tasks.id"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(30), nullable=False),
        sa.Column("tool_name", sa.String(50)),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("evidence_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("task_id", "sequence"),
    )
    op.create_index("ix_agent_events_task_id", "agent_events", ["task_id"])
    op.create_table(
        "tool_executions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(64), sa.ForeignKey("agent_tasks.id"), nullable=False),
        sa.Column("tool_name", sa.String(50), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("evidence_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("task_id", "request_hash"),
    )
    op.create_index("ix_tool_executions_task_id", "tool_executions", ["task_id"])


def downgrade():
    op.drop_table("tool_executions")
    op.drop_table("agent_events")
    op.drop_table("agent_tasks")
