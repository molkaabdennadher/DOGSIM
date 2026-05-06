"""
Session persistence (sqlite).

Sessions hold the *live* conversation state (messages, in-flight rejections,
negative signals collected this turn). Long-term, *cross-session* memory lives
in feedback.py — keep the two concerns separate.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from backend.maya.schemas import UserSession

DB_PATH = Path(__file__).parent.parent.parent / "artifacts" / "aiAdvisor" / "sessions.db"


def _init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id            TEXT PRIMARY KEY,
                user_name     TEXT NOT NULL,
                user_email    TEXT NOT NULL DEFAULT '',
                messages      TEXT NOT NULL DEFAULT '[]',
                excluded_ids  TEXT NOT NULL DEFAULT '[]',
                neg_signals   TEXT NOT NULL DEFAULT '[]',
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Forward-compatible migration for older DBs that pre-date user_email.
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN user_email TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass                                  # Column already exists.
        conn.commit()


@contextmanager
def _conn():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        yield conn


def create_session(user_name: str, user_email: str = "") -> UserSession:
    session_id = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_name, user_email) VALUES (?, ?, ?)",
            (session_id, user_name, user_email),
        )
        conn.commit()
    return UserSession(session_id=session_id, user_name=user_name, user_email=user_email)


def get_session(session_id: str) -> UserSession | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    if not row:
        return None
    s = UserSession(
        session_id=row["id"],
        user_name=row["user_name"],
        user_email=row["user_email"] or "",
    )
    s.messages         = json.loads(row["messages"])
    s.excluded_ids     = json.loads(row["excluded_ids"])
    s.negative_signals = json.loads(row["neg_signals"])
    return s


def save_session(session: UserSession):
    with _conn() as conn:
        conn.execute(
            """UPDATE sessions
               SET messages     = ?,
                   excluded_ids = ?,
                   neg_signals  = ?,
                   user_email   = ?,
                   updated_at   = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (
                json.dumps(session.messages),
                json.dumps(session.excluded_ids),
                json.dumps(session.negative_signals),
                session.user_email,
                session.session_id,
            ),
        )
        conn.commit()


_init_db()
