"""
SQLite persistence for shelters and dogs.

Design principles
-----------------
- Every public function opens a fresh connection — no module-level connection
  is kept open. This guarantees that the ReAct agent always reads the latest
  committed state, even if another request modified the DB milliseconds before.
- All writes are committed immediately so the next query sees them.
- JSON columns (none here) are avoided — all fields are scalar for simplicity.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.ali.schemas import Dog, Shelter

DB_PATH = Path(__file__).parent.parent.parent / "artifacts" / "aliAdvisor" / "shelter_assignment.db"
log = logging.getLogger("ali.db")

# ── Monthly cost estimates per dog attribute (configurable) ───────────────────
# All values in Tunisian Dinars (DT) per month.
COST_CONFIG = {
    "base":             80.0,
    "minor_injury":     40.0,
    "serious_injury":  120.0,
    "skin_disease":     50.0,
    "pregnant":         60.0,
    "abnormal":         40.0,
    "not_vaccinated":   20.0,
    "not_dewormed":     15.0,
}

_TRUE_VALUES = {"1", "true", "yes", "y", "oui", "o"}
_FALSE_VALUES = {"0", "false", "no", "n", "non"}


def estimate_monthly_cost(dog: Dog) -> float:
    cost = COST_CONFIG["base"]
    if dog.health == "Minor Injury":
        cost += COST_CONFIG["minor_injury"]
    elif dog.health == "Serious Injury":
        cost += COST_CONFIG["serious_injury"]
    if dog.has_skin_disease:
        cost += COST_CONFIG["skin_disease"]
    if dog.is_pregnant:
        cost += COST_CONFIG["pregnant"]
    if dog.behaviour == "abnormal":
        cost += COST_CONFIG["abnormal"]
    if not _coerce_bool(dog.vaccinated):
        cost += COST_CONFIG["not_vaccinated"]
    if not _coerce_bool(dog.dewormed):
        cost += COST_CONFIG["not_dewormed"]
    return round(cost, 2)


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return bool(value)


def _coerce_bool_int(value) -> int:
    return int(_coerce_bool(value))


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shelters (
    id                   TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    location             TEXT NOT NULL DEFAULT '',
    capacity             INTEGER NOT NULL DEFAULT 0,
    free_spots           INTEGER NOT NULL DEFAULT 0,
    reserved_spots       INTEGER NOT NULL DEFAULT 0,
    monthly_budget       REAL    NOT NULL DEFAULT 0.0,
    current_monthly_cost REAL    NOT NULL DEFAULT 0.0,
    expertise_level      TEXT    NOT NULL DEFAULT 'beginner',
    staff_count          INTEGER NOT NULL DEFAULT 1,
    shelter_type         TEXT    NOT NULL DEFAULT 'public',
    has_vet              INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT    DEFAULT CURRENT_TIMESTAMP,
    updated_at           TEXT    DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dogs (
    pet_id               TEXT PRIMARY KEY,
    name                 TEXT NOT NULL DEFAULT '',
    age                  REAL NOT NULL DEFAULT 0.0,
    breed                TEXT NOT NULL DEFAULT 'mixed',
    gender               TEXT NOT NULL DEFAULT 'Male',
    vaccinated           TEXT NOT NULL DEFAULT 'No',
    dewormed             TEXT NOT NULL DEFAULT 'No',
    sterilized           TEXT NOT NULL DEFAULT 'No',
    health               TEXT NOT NULL DEFAULT 'Healthy',
    shelter_id           TEXT NOT NULL DEFAULT '-1',
    behaviour            TEXT NOT NULL DEFAULT 'normal',
    is_pregnant          INTEGER NOT NULL DEFAULT 0,
    expected_litter_size INTEGER NOT NULL DEFAULT 0,
    has_skin_disease     INTEGER NOT NULL DEFAULT 0,
    priority_score       INTEGER NOT NULL DEFAULT 0,
    status               TEXT    NOT NULL DEFAULT 'waiting',
    monthly_cost         REAL    NOT NULL DEFAULT 0.0,
    assignment_reason    TEXT    NOT NULL DEFAULT '',
    assigned_at          TEXT    DEFAULT NULL,
    created_at           TEXT    DEFAULT CURRENT_TIMESTAMP,
    updated_at           TEXT    DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dogs_shelter   ON dogs(shelter_id);
CREATE INDEX IF NOT EXISTS idx_dogs_status    ON dogs(status);
CREATE INDEX IF NOT EXISTS idx_dogs_priority  ON dogs(priority_score DESC);
"""


def _is_corrupt_db_error(exc: sqlite3.DatabaseError) -> bool:
    msg = str(exc).lower()
    return "malformed" in msg or "not a database" in msg


def _quarantine_corrupt_db(exc: sqlite3.DatabaseError) -> None:
    """Move a corrupted SQLite DB aside so startup can rebuild a clean one."""
    if not DB_PATH.exists():
        return

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    backup = DB_PATH.with_name(f"{DB_PATH.stem}.corrupt-{stamp}{DB_PATH.suffix}")
    log.warning("Ali database is corrupt (%s). Moving it to %s", exc, backup)
    DB_PATH.replace(backup)

    for suffix in ("-wal", "-shm"):
        sidecar = DB_PATH.with_name(DB_PATH.name + suffix)
        if sidecar.exists():
            sidecar.replace(backup.with_name(backup.name + suffix))


def _open_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    c = sqlite3.connect(DB_PATH)
    try:
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")   # allow concurrent reads
        return c
    except sqlite3.DatabaseError:
        c.close()
        raise


@contextmanager
def _conn():
    """Fresh connection per call — guarantees live data for every query."""
    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    try:
        with _open_connection() as c:
            yield c
    except sqlite3.DatabaseError as exc:
        if _is_corrupt_db_error(exc):
            _quarantine_corrupt_db(exc)
            init_db()
        raise


def init_db() -> None:
    try:
        with _open_connection() as c:
            c.executescript(_SCHEMA)
            c.commit()
    except sqlite3.DatabaseError as exc:
        if not _is_corrupt_db_error(exc):
            raise
        _quarantine_corrupt_db(exc)
        with _open_connection() as c:
            c.executescript(_SCHEMA)
            c.commit()


# ── Shelter CRUD ──────────────────────────────────────────────────────────────

def add_shelter(s: dict) -> str:
    shelter_id = s.get("id") or str(uuid.uuid4())[:8].upper()
    with _conn() as c:
        c.execute("""
            INSERT INTO shelters
              (id, name, location, capacity, free_spots, reserved_spots,
               monthly_budget, current_monthly_cost,
               expertise_level, staff_count, shelter_type, has_vet)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            shelter_id,
            s["name"], s.get("location", ""),
            int(s["capacity"]), int(s["capacity"]), 0,
            float(s["monthly_budget"]), 0.0,
            s.get("expertise_level", "beginner"),
            int(s.get("staff_count", 1)),
            s.get("shelter_type", "public"),
            _coerce_bool_int(s.get("has_vet", False)),
        ))
        c.commit()
    return shelter_id


def get_shelter(shelter_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM shelters WHERE id = ?", (shelter_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_shelters() -> list[dict]:
    """Always queries live DB — no cache."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM shelters ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def update_shelter_field(shelter_id: str, field: str, value) -> bool:
    allowed = {"name", "location", "capacity", "free_spots", "monthly_budget",
               "expertise_level", "staff_count", "shelter_type", "has_vet",
               "current_monthly_cost", "reserved_spots"}
    if field not in allowed:
        return False
    with _conn() as c:
        c.execute(
            f"UPDATE shelters SET {field}=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (value, shelter_id)
        )
        c.commit()
    return True


def _adjust_shelter_after_assign(c, shelter_id: str, dog: Dog, direction: int) -> None:
    """direction=+1 → adding dog, direction=-1 → removing dog."""
    cost  = dog.monthly_cost * direction
    spots = 1 * direction
    # For a pregnant dog, also adjust reserved_spots for the litter
    reserved = 0
    if dog.is_pregnant and dog.expected_litter_size > 0:
        reserved = dog.expected_litter_size * direction

    c.execute("""
        UPDATE shelters
        SET free_spots           = free_spots           - ?,
            reserved_spots       = MAX(0, reserved_spots + ?),
            current_monthly_cost = current_monthly_cost + ?,
            updated_at           = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (spots, reserved, cost, shelter_id))


# ── Dog CRUD ──────────────────────────────────────────────────────────────────

def upsert_dog(dog: Dog) -> None:
    with _conn() as c:
        c.execute("""
            INSERT INTO dogs
              (pet_id, name, age, breed, gender,
               vaccinated, dewormed, sterilized, health,
               shelter_id, behaviour, is_pregnant, expected_litter_size,
               has_skin_disease, priority_score, status,
               monthly_cost, assignment_reason, assigned_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(pet_id) DO UPDATE SET
              name=excluded.name, age=excluded.age, breed=excluded.breed,
              gender=excluded.gender, vaccinated=excluded.vaccinated,
              dewormed=excluded.dewormed, sterilized=excluded.sterilized,
              health=excluded.health, shelter_id=excluded.shelter_id,
              behaviour=excluded.behaviour, is_pregnant=excluded.is_pregnant,
              expected_litter_size=excluded.expected_litter_size,
              has_skin_disease=excluded.has_skin_disease,
              priority_score=excluded.priority_score, status=excluded.status,
              monthly_cost=excluded.monthly_cost,
              assignment_reason=excluded.assignment_reason,
              assigned_at=excluded.assigned_at,
              updated_at=CURRENT_TIMESTAMP
        """, (
            dog.pet_id, dog.name, dog.age, dog.breed, dog.gender,
            dog.vaccinated, dog.dewormed, dog.sterilized, dog.health,
            dog.shelter_id, dog.behaviour,
            int(dog.is_pregnant), dog.expected_litter_size,
            int(dog.has_skin_disease),
            dog.priority_score, dog.status,
            dog.monthly_cost, dog.assignment_reason,
            "CURRENT_TIMESTAMP" if dog.status == "assigned" else None,
        ))
        c.commit()


def assign_dog_to_shelter(pet_id: str, shelter_id: str,
                           reason: str, monthly_cost: float) -> bool:
    """Atomically assign a dog and update shelter counters."""
    with _conn() as c:
        dog_row = c.execute(
            "SELECT * FROM dogs WHERE pet_id=?", (pet_id,)
        ).fetchone()
        shelter_row = c.execute(
            "SELECT * FROM shelters WHERE id=?", (shelter_id,)
        ).fetchone()
        if not dog_row or not shelter_row:
            return False

        dog = _row_to_dog(dog_row)
        dog.monthly_cost = monthly_cost

        # Update dog record
        c.execute("""
            UPDATE dogs SET shelter_id=?, status='assigned',
              assignment_reason=?, assigned_at=CURRENT_TIMESTAMP,
              monthly_cost=?, updated_at=CURRENT_TIMESTAMP
            WHERE pet_id=?
        """, (shelter_id, reason, monthly_cost, pet_id))

        # Update shelter counters
        _adjust_shelter_after_assign(c, shelter_id, dog, direction=+1)
        c.commit()
    return True


def release_dog(pet_id: str, reason: str = "adopted") -> Optional[dict]:
    """Remove a dog from its shelter and free the spot.

    Returns the freed shelter's id and the dog's monthly_cost so the
    waiting-list logic can find the next eligible candidate.
    """
    with _conn() as c:
        dog_row = c.execute(
            "SELECT * FROM dogs WHERE pet_id=?", (pet_id,)
        ).fetchone()
        if not dog_row:
            return None
        dog = _row_to_dog(dog_row)
        if dog.shelter_id == "-1":
            return None   # already in waiting list

        freed_shelter_id = dog.shelter_id

        # Mark dog as released
        c.execute("""
            UPDATE dogs SET shelter_id='-1', status='released',
              assignment_reason=?, updated_at=CURRENT_TIMESTAMP
            WHERE pet_id=?
        """, (f"Released: {reason}", pet_id))

        # Return spot to shelter
        _adjust_shelter_after_assign(c, freed_shelter_id, dog, direction=-1)
        c.commit()

    return {"freed_shelter_id": freed_shelter_id, "monthly_cost": dog.monthly_cost}


def get_dog(pet_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM dogs WHERE pet_id=?", (pet_id,)).fetchone()
    return dict(row) if row else None


def get_dogs_in_shelter(shelter_id: str) -> list[dict]:
    """Always queries live DB."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM dogs WHERE shelter_id=? AND status='assigned' ORDER BY priority_score DESC",
            (shelter_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_waiting_list() -> list[dict]:
    """Dogs not yet assigned, ordered by priority (highest first)."""
    with _conn() as c:
        rows = c.execute("""
            SELECT * FROM dogs
            WHERE shelter_id='-1' AND status='waiting'
            ORDER BY priority_score DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_dogs_by_criteria(health: str = None, behaviour: str = None,
                          status: str = None, has_skin_disease: bool = None,
                          is_pregnant: bool = None) -> list[dict]:
    """Flexible filter — all parameters optional. Always live DB."""
    clauses, params = [], []
    if health:
        clauses.append("health=?"); params.append(health)
    if behaviour:
        clauses.append("behaviour=?"); params.append(behaviour)
    if status:
        clauses.append("status=?"); params.append(status)
    if has_skin_disease is not None:
        clauses.append("has_skin_disease=?"); params.append(_coerce_bool_int(has_skin_disease))
    if is_pregnant is not None:
        clauses.append("is_pregnant=?"); params.append(_coerce_bool_int(is_pregnant))
    sql = "SELECT * FROM dogs"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY priority_score DESC"
    with _conn() as c:
        rows = c.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_all_dogs() -> list[dict]:
    """Return every dog in the database (all statuses)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM dogs ORDER BY priority_score DESC, name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_capacity_risks() -> list[dict]:
    """Return shelters whose effective free spots are ≤ 20% of total capacity,
    along with the count of pregnant dogs currently inside each."""
    with _conn() as c:
        rows = c.execute("""
            SELECT
              s.id,
              s.name,
              s.capacity,
              s.free_spots,
              s.reserved_spots,
              COALESCE(
                (SELECT COUNT(*) FROM dogs d
                 WHERE d.shelter_id = s.id AND d.is_pregnant = 1 AND d.status = 'assigned'),
                0
              ) AS pregnant_count
            FROM shelters s
            WHERE (s.free_spots - s.reserved_spots) * 1.0 / MAX(s.capacity, 1) <= 0.20
            ORDER BY (s.free_spots - s.reserved_spots) ASC
        """).fetchall()
    return [dict(r) for r in rows]


def get_pregnant_dog_summary() -> list[dict]:
    """Return pregnant dogs grouped with their assigned shelter metadata."""
    with _conn() as c:
        rows = c.execute("""
            SELECT
              d.pet_id,
              d.name AS dog_name,
              d.expected_litter_size,
              d.status,
              d.shelter_id,
              COALESCE(s.name, 'Waiting list') AS shelter_name,
              COALESCE(s.location, '') AS shelter_location
            FROM dogs d
            LEFT JOIN shelters s ON s.id = d.shelter_id
            WHERE d.is_pregnant = 1
            ORDER BY s.name, d.priority_score DESC, d.name
        """).fetchall()
    return [dict(r) for r in rows]


def get_statistics() -> dict:
    """Live aggregate statistics across all shelters and dogs."""
    with _conn() as c:
        s = c.execute("""
            SELECT
              COUNT(*)                                        AS total_shelters,
              SUM(capacity)                                   AS total_capacity,
              SUM(free_spots)                                 AS total_free_spots,
              SUM(reserved_spots)                             AS total_reserved,
              SUM(CASE WHEN has_vet=1 THEN 1 ELSE 0 END)     AS shelters_with_vet,
              SUM(monthly_budget)                             AS total_budget,
              SUM(current_monthly_cost)                       AS total_cost
            FROM shelters
        """).fetchone()
        d = c.execute("""
            SELECT
              COUNT(*)                                                    AS total_dogs,
              SUM(CASE WHEN status='assigned'  THEN 1 ELSE 0 END)        AS assigned,
              SUM(CASE WHEN status='waiting'   THEN 1 ELSE 0 END)        AS waiting,
              SUM(CASE WHEN status='released'  THEN 1 ELSE 0 END)        AS released,
              SUM(CASE WHEN health='Serious Injury' THEN 1 ELSE 0 END)   AS serious_injury,
              SUM(CASE WHEN is_pregnant=1      THEN 1 ELSE 0 END)        AS pregnant,
              SUM(CASE WHEN has_skin_disease=1 THEN 1 ELSE 0 END)        AS skin_disease,
              SUM(CASE WHEN behaviour='abnormal' THEN 1 ELSE 0 END)      AS abnormal
            FROM dogs
        """).fetchone()
    return {
        "total_shelters":   s["total_shelters"]   or 0,
        "total_capacity":   s["total_capacity"]   or 0,
        "total_free_spots": s["total_free_spots"] or 0,
        "total_reserved":   s["total_reserved"]   or 0,
        "shelters_with_vet": s["shelters_with_vet"] or 0,
        "total_budget":     s["total_budget"]     or 0.0,
        "total_cost":       s["total_cost"]       or 0.0,
        "total_dogs":       d["total_dogs"]       or 0,
        "assigned":         d["assigned"]         or 0,
        "waiting":          d["waiting"]          or 0,
        "released":         d["released"]         or 0,
        "serious_injury":   d["serious_injury"]   or 0,
        "pregnant":         d["pregnant"]         or 0,
        "skin_disease":     d["skin_disease"]     or 0,
        "abnormal":         d["abnormal"]         or 0,
    }
