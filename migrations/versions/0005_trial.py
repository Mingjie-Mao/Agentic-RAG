"""Add concurrency-safe public trial request slots."""

from alembic import op
import sqlalchemy as sa

revision = "0005_trial"
down_revision = "0004_agent_async"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "trial_requests",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("usage_date", sa.Date(), nullable=False),
        sa.Column("slot", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "usage_date", "slot"),
    )
    op.create_index("ix_trial_requests_user_id", "trial_requests", ["user_id"])
    op.create_index("ix_trial_requests_usage_date", "trial_requests", ["usage_date"])


def downgrade():
    op.drop_index("ix_trial_requests_usage_date", table_name="trial_requests")
    op.drop_index("ix_trial_requests_user_id", table_name="trial_requests")
    op.drop_table("trial_requests")
