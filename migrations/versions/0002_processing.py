"""Retain source structure and serialize duplicate ingestion."""

from alembic import op
import sqlalchemy as sa

revision = "0002_processing"
down_revision = "0001"
branch_labels = depends_on = None


def upgrade():
    op.add_column("documents", sa.Column("ingest_key", sa.String(64), nullable=True))
    op.create_unique_constraint("uq_documents_ingest_key", "documents", ["ingest_key"])
    op.add_column(
        "document_versions", sa.Column("parsed_blocks", sa.JSON(), nullable=False, server_default="[]")
    )


def downgrade():
    op.drop_column("document_versions", "parsed_blocks")
    op.drop_constraint("uq_documents_ingest_key", "documents", type_="unique")
    op.drop_column("documents", "ingest_key")
