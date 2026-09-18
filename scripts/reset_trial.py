"""Reset one configured public-trial account without touching seeded documents."""

import argparse

from sqlalchemy import select

from app.db import SessionLocal
from app.models import User
from app.trial import reset_trial, trial_usernames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", help="Configured trial username; defaults to the only one")
    args = parser.parse_args()
    configured = trial_usernames()
    username = args.username or (next(iter(configured)) if len(configured) == 1 else "")
    if not username or username not in configured:
        raise SystemExit("Choose one username from RAG_TRIAL_USERNAMES")
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.username == username))
        if not user:
            raise SystemExit("Configured trial user does not exist")
        print(reset_trial(db, user))


if __name__ == "__main__":
    main()
