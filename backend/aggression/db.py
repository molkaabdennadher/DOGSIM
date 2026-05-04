"""
SQLite persistence for aggression incidents.

The schema closely mirrors `aggression_events` from the research notebook so
that operations staff can run the same SQL on either store. Two extra columns
(`severity_rationale`, `contact_decision`, `report_markdown`) capture the
backend agent's reasoning that the notebook didn't have.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "artifacts" / "agressionDetection" / "aggression_incidents.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS aggression_incidents (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id         TEXT    NOT NULL UNIQUE,
    timestamp           TEXT    NOT NULL,
    video_source        TEXT,
    frame_number        INTEGER,

    incident_type       TEXT,
    person_track_id     INTEGER,
    dog_track_id        INTEGER,
    distance_px         REAL,

    ema_score           REAL,
    sustained_frames    INTEGER,
    evidence_list       TEXT,           -- JSON array

    llm_confirmed       INTEGER DEFAULT 0,
    llm_confidence      REAL,
    llm_severity_hint   TEXT,
    llm_reason          TEXT,

    severity            TEXT,           -- minor | serious | critical (agent verdict)
    severity_rationale  TEXT,           -- short justification from the agent
    contact_decision    TEXT,           -- JSON array of channels
    report_markdown     TEXT,           -- the full incident report

    location_label      TEXT,
    latitude            REAL,
    longitude           REAL,

    notifications_sent  TEXT,           -- JSON array
    actions_taken       TEXT,           -- JSON array

    keyframes_path      TEXT DEFAULT '', -- path to the JPEG mosaic (empty if none)

    created_at          TEXT    DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_agg_severity   ON aggression_incidents(severity);
CREATE INDEX IF NOT EXISTS idx_agg_created_at ON aggression_incidents(created_at);
"""

# Idempotent migration: add keyframes_path column to existing DBs.
_MIGRATE_SQL = """
ALTER TABLE aggression_incidents ADD COLUMN keyframes_path TEXT DEFAULT '';
"""


def init_db() -> None:
    """Idempotent — safe to call at startup. Also migrates older DBs."""
    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA_SQL)
        # Add keyframes_path to existing tables that pre-date this column.
        try:
            conn.execute(_MIGRATE_SQL)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists — safe to ignore


@contextmanager
def _conn():
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        yield c


# ── Insert ────────────────────────────────────────────────────────────────────

def store_incident(state: dict) -> int:
    """Insert one incident. Returns the new auto-increment id.

    Idempotent on `incident_id`: if an incident with the same id already exists
    we update it in place rather than duplicate.
    """
    incident = state.get("incident") or {}
    payload = (
        incident.get("incident_id", ""),
        incident.get("timestamp", ""),
        incident.get("video_source", ""),
        int(incident.get("frame_number", -1)),
        incident.get("incident_type", "unknown"),
        int(incident.get("person_track_id", -1)),
        int(incident.get("dog_track_id", -1)),
        float(incident.get("distance_px", 0.0)),
        float(incident.get("ema_score", 0.0)),
        int(incident.get("sustained_frames", 0)),
        json.dumps(incident.get("evidence_list", [])),
        int(bool(incident.get("llm_confirmed", False))),
        float(incident.get("llm_confidence", 0.0)),
        incident.get("llm_severity_hint", ""),
        incident.get("llm_reason", ""),
        state.get("severity", ""),
        state.get("severity_rationale", ""),
        json.dumps(state.get("contact_decision", [])),
        state.get("report_markdown", ""),
        incident.get("location_label", ""),
        incident.get("latitude"),
        incident.get("longitude"),
        json.dumps(state.get("notifications_sent", [])),
        json.dumps(state.get("actions_taken", [])),
        state.get("keyframes_path", ""),
    )
    cols = (
        "incident_id, timestamp, video_source, frame_number, "
        "incident_type, person_track_id, dog_track_id, distance_px, "
        "ema_score, sustained_frames, evidence_list, "
        "llm_confirmed, llm_confidence, llm_severity_hint, llm_reason, "
        "severity, severity_rationale, contact_decision, report_markdown, "
        "location_label, latitude, longitude, "
        "notifications_sent, actions_taken, keyframes_path"
    )
    placeholders = ",".join(["?"] * 25)

    with _conn() as c:
        # Upsert by incident_id.
        existing = c.execute(
            "SELECT id FROM aggression_incidents WHERE incident_id = ?",
            (payload[0],),
        ).fetchone()
        if existing:
            update_set = ", ".join(f"{name.strip()}=?" for name in cols.split(","))
            c.execute(
                f"UPDATE aggression_incidents SET {update_set} WHERE id = ?",
                (*payload, existing["id"]),
            )
            c.commit()
            return int(existing["id"])
        cur = c.execute(
            f"INSERT INTO aggression_incidents ({cols}) VALUES ({placeholders})",
            payload,
        )
        c.commit()
        return int(cur.lastrowid)


# ── Read ──────────────────────────────────────────────────────────────────────

def list_incidents(limit: int = 50, severity: str | None = None) -> list[dict]:
    sql  = "SELECT * FROM aggression_incidents"
    args: list = []
    if severity:
        sql += " WHERE severity = ?"
        args.append(severity)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(int(limit))

    with _conn() as c:
        rows = c.execute(sql, args).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_incident(incident_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM aggression_incidents WHERE incident_id = ?",
            (incident_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def _row_to_dict(row: sqlite3.Row) -> dict:
    out = dict(row)
    # Decode JSON-serialised columns so callers get usable Python types back.
    for col in ("evidence_list", "contact_decision", "notifications_sent", "actions_taken"):
        try:
            out[col] = json.loads(out.get(col) or "[]")
        except (TypeError, json.JSONDecodeError):
            out[col] = []
    return out


# Bootstrap on import so the module is always usable.
init_db()
