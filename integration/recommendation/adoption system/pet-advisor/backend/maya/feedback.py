"""
Feedback persistence (Tâche 2).

Goal
----
Turn user rejections into a *durable, reusable* signal so that:

1. Pets the user has already explicitly rejected never reappear in their next session.
2. Aggregate negative preferences (avoided breeds, sizes, animal types) bias the
   ranker toward alternatives the user is more likely to accept.
3. The LLM sees a short, human-readable summary of "what this user has historically
   disliked" so it can phrase its questions/suggestions accordingly.

Storage
-------
A single sqlite table `rejections` keyed by `user_email`. We deliberately do NOT
use the session-id as the key — the whole point is for the signal to outlive a
single chat session.

Why not vector-based personalization?
-------------------------------------
A learned user-embedding would be more powerful but needs hundreds of interactions
per user to converge. With a few rejections per session, a transparent rule-based
profile (boost down breeds/sizes the user dislikes) is a much better cost/value
trade-off and stays fully explainable.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from typing import Iterable

from backend.maya.schemas import RejectionRecord, UserPreferenceProfile

DB_PATH = Path(__file__).parent.parent.parent / "artifacts" / "aiAdvisor" / "feedback.db"

# How many rejections of a given attribute are needed before we treat it as
# a *systematic* avoidance signal (vs. a one-off bad fit).
SYSTEMATIC_THRESHOLD = 2


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def _init_db() -> None:
    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rejections (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email    TEXT NOT NULL,
                pet_id        TEXT NOT NULL,
                reason        TEXT NOT NULL DEFAULT '',
                breed         TEXT NOT NULL DEFAULT '',
                size          TEXT NOT NULL DEFAULT '',
                age_months    INTEGER NOT NULL DEFAULT 0,
                animal_type   TEXT NOT NULL DEFAULT '',
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rejections_user ON rejections(user_email)"
        )
        conn.commit()


@contextmanager
def _conn():
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        yield c


# ── Public API ────────────────────────────────────────────────────────────────

def record_rejection(rec: RejectionRecord) -> None:
    """Insert one rejection event. Idempotent only on (email, pet_id) — we keep
    duplicates because the *count* of rejections informs systematic-avoidance."""
    if not rec.user_email or not rec.pet_id:
        return                                   # Nothing useful to store

    with _conn() as c:
        c.execute(
            """INSERT INTO rejections
               (user_email, pet_id, reason, breed, size, age_months, animal_type)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.user_email,
                rec.pet_id,
                rec.reason,
                rec.breed,
                rec.size,
                rec.age_months,
                rec.animal_type,
            ),
        )
        c.commit()


def fetch_rejections(user_email: str) -> list[RejectionRecord]:
    if not user_email:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM rejections WHERE user_email = ? ORDER BY created_at DESC",
            (user_email,),
        ).fetchall()
    return [
        RejectionRecord(
            user_email=r["user_email"],
            pet_id=r["pet_id"],
            reason=r["reason"],
            breed=r["breed"],
            size=r["size"],
            age_months=r["age_months"],
            animal_type=r["animal_type"],
        )
        for r in rows
    ]


def build_profile(user_email: str) -> UserPreferenceProfile:
    """Aggregate a user's rejection history into a usable preference profile."""
    rejections = fetch_rejections(user_email)
    profile    = UserPreferenceProfile(user_email=user_email)
    profile.total_rejections = len(rejections)
    if not rejections:
        return profile

    # Always exclude every individual pet the user has ever rejected.
    profile.excluded_pet_ids = sorted({r.pet_id for r in rejections if r.pet_id})

    # Surface attributes that have been rejected at least SYSTEMATIC_THRESHOLD times
    # — those are reliable "avoid" signals, not noise from a single bad match.
    profile.avoided_breeds       = _systematic(r.breed       for r in rejections)
    profile.avoided_sizes        = _systematic(r.size        for r in rejections)
    profile.avoided_animal_types = _systematic(r.animal_type for r in rejections)

    profile.rejection_reasons = [r.reason for r in rejections if r.reason][:8]
    return profile


def summarize_for_llm(profile: UserPreferenceProfile) -> str:
    """
    Render the profile as a natural-language memory block injected into the
    system prompt. Written in the first person as Maya's memory, not as a
    list of rules.
    """
    if profile.total_rejections == 0:
        return ""

    parts = []

    if profile.avoided_breeds:
        breeds = ", ".join(profile.avoided_breeds)
        parts.append(f"Cette personne a déjà rejeté des animaux de race {breeds} — mieux vaut éviter.")

    if profile.avoided_sizes:
        sizes = ", ".join(profile.avoided_sizes)
        parts.append(f"Elle n'a pas semblé à l'aise avec les gabarits : {sizes}.")

    if profile.avoided_animal_types:
        types = ", ".join(profile.avoided_animal_types)
        parts.append(f"Elle a tendance à rejeter les {types} — à garder en tête.")

    if profile.rejection_reasons:
        reasons = " ; ".join(profile.rejection_reasons[:4])
        parts.append(f"Les raisons qu'elle a données : « {reasons} ».")

    parts.append(f"Au total, {profile.total_rejections} suggestion{'s' if profile.total_rejections > 1 else ''} n'ont pas convenu.")

    return "\n".join(parts)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _systematic(values: Iterable[str]) -> list[str]:
    """Return values that appear >= SYSTEMATIC_THRESHOLD times (case-insensitive)."""
    counts = Counter(v.strip().lower() for v in values if v and v.strip())
    return sorted(v for v, n in counts.items() if n >= SYSTEMATIC_THRESHOLD)


# Bootstrap the schema once at import time.
_init_db()
