"""
excel_parser.py — Parse an uploaded .xlsx file of dogs.

Expected columns (case-insensitive, extra columns ignored):
    PetID, Name, Age, Breed, Gender, Vaccinated, Dewormed, Sterilized,
    Health, ShelterID, Behaviour, IsPregnant, ExpectedLitterSize, HasSkinDisease
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import IO, Union

import pandas as pd

from backend.ali.schemas import Dog
import backend.ali.db as db

SUPPORTED_EXTENSIONS = {".xlsx"}

_COL_MAP = {
    "petid": "pet_id", "name": "name", "age": "age", "breed": "breed",
    "gender": "gender", "vaccinated": "vaccinated", "dewormed": "dewormed",
    "sterilized": "sterilized", "health": "health", "shelterid": "shelter_id",
    "behaviour": "behaviour", "ispregnant": "is_pregnant",
    "expectedlittersize": "expected_litter_size", "hasskindisease": "has_skin_disease",
}

_VALID_HEALTH    = {"healthy", "minor injury", "serious injury"}
_VALID_BEHAVIOUR = {"normal", "abnormal"}


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() in ("yes", "true", "1")


def _norm_health(val: str) -> str:
    v = str(val).strip().lower()
    return {"healthy": "Healthy",
            "minor injury": "Minor Injury",
            "serious injury": "Serious Injury"}.get(v, "Healthy")


def _norm_behaviour(val: str) -> str:
    v = str(val).strip().lower()
    return v if v in _VALID_BEHAVIOUR else "normal"


def parse_file(source, suffix: str = ".xlsx"):
    """Parse an .xlsx file into (dogs, warnings).

    Args:
        source:  File path (str/Path) or file-like object.
        suffix:  Must be '.xlsx'. Kept as parameter for backward compat.

    Returns:
        (list[Dog], list[str]) — dogs ready for assignment, parsing warnings.

    Raises:
        ValueError: If the file is not a .xlsx file.
    """
    if suffix.lower() != ".xlsx":
        raise ValueError(
            f"Only .xlsx files are supported. Got: '{suffix}'. "
            "Please export your spreadsheet as Excel (.xlsx) before uploading."
        )

    try:
        df = pd.read_excel(source, engine="openpyxl", dtype=str)
    except Exception as exc:
        raise ValueError(f"Could not read Excel file: {exc}") from exc

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "").replace("_", "") for c in df.columns]
    df = df.rename(columns=_COL_MAP)
    for col in _COL_MAP.values():
        if col not in df.columns:
            df[col] = None

    warnings: list[str] = []
    dogs:     list[Dog]  = []

    for i, row in df.iterrows():
        row_num = i + 2   # 1-based + header row

        # ShelterID filter — skip dogs already placed
        try:
            s_id = int(row.get("shelter_id", -1) or -1)
        except (ValueError, TypeError):
            s_id = -1
        if s_id != -1:
            warnings.append(f"Row {row_num}: ShelterID={s_id} — skipped (already placed).")
            continue

        # PetID
        pet_id = str(row.get("pet_id", "") or "").strip()
        if not pet_id or pet_id.lower() in ("nan", "none", ""):
            pet_id = "DOG-" + uuid.uuid4().hex[:8].upper()
            warnings.append(f"Row {row_num}: Missing PetID — auto-assigned {pet_id}.")

        # Name (required)
        name = str(row.get("name", "") or "").strip()
        if not name or name.lower() in ("nan", "none"):
            warnings.append(f"Row {row_num}: Missing Name — row skipped.")
            continue

        # Age
        try:
            age = float(row.get("age", 0) or 0)
        except (ValueError, TypeError):
            age = 0.0

        # Breed
        breed = str(row.get("breed", "") or "Mixed").strip()
        if not breed or breed.lower() in ("nan", "none"):
            breed = "Mixed"

        # Gender
        gender_raw = str(row.get("gender", "") or "").strip().capitalize()
        gender = gender_raw if gender_raw in ("Male", "Female") else "Male"

        # Booleans
        vaccinated       = "Yes" if _to_bool(row.get("vaccinated",       False)) else "No"
        dewormed         = "Yes" if _to_bool(row.get("dewormed",         False)) else "No"
        sterilized       = "Yes" if _to_bool(row.get("sterilized",       False)) else "No"
        is_pregnant      = _to_bool(row.get("is_pregnant",      False))
        has_skin_disease = _to_bool(row.get("has_skin_disease", False))

        # Health
        health_raw = str(row.get("health", "") or "Healthy").strip()
        health     = _norm_health(health_raw)
        if health == "Healthy" and health_raw.lower() not in ("healthy", "nan", "none", ""):
            warnings.append(f"Row {row_num}: Unknown health '{health_raw}' — defaulted to Healthy.")

        # Behaviour
        behaviour = _norm_behaviour(str(row.get("behaviour", "") or "normal"))

        # Litter size
        try:
            expected_litter_size = int(float(row.get("expected_litter_size", 0) or 0))
        except (ValueError, TypeError):
            expected_litter_size = 0
        if is_pregnant and expected_litter_size < 1:
            warnings.append(f"Row {row_num}: {name} is pregnant but litter size is 0 — defaulting to 1.")
            expected_litter_size = 1

        dogs.append(Dog(
            pet_id=pet_id, name=name, age=age, breed=breed, gender=gender,
            vaccinated=vaccinated, dewormed=dewormed, sterilized=sterilized,
            health=health, shelter_id="-1", behaviour=behaviour,
            is_pregnant=is_pregnant, expected_litter_size=expected_litter_size,
            has_skin_disease=has_skin_disease,
        ))

    return dogs, warnings


def parse_excel(source, suffix: str = ".xlsx"):
    """Backward-compatible alias for parse_file."""
    return parse_file(source, suffix)
