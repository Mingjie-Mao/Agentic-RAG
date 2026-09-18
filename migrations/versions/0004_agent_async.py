"""Add worker lease state to Agent tasks."""

from alembic import op
import sqlalchemy as sa

revision = "0004_agent_async"
down_revision = "0003_agent"
branch_labels = depends_on = None


def upgrade():
    op.add_column(
        "agent_tasks", sa.Column("attempts", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column("agent_tasks", sa.Column("lease_until", sa.DateTime(timezone=True)))
    op.add_column("agent_tasks", sa.Column("lease_token", sa.String(64)))


def downgrade():
    op.drop_column("agent_tasks", "lease_token")
    op.drop_column("agent_tasks", "lease_until")
    op.drop_column("agent_tasks", "attempts")
