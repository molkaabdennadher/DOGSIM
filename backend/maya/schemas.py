"""
Pydantic & dataclass schemas shared across the backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

from pydantic import BaseModel


# ── REST / WebSocket payloads ──────────────────────────────────────────────────

class LoginRequest(BaseModel):
    name: str
    email: str           # Plain str (not EmailStr) — keep zero-extra-deps. Validated as non-empty in main.py.


class LoginResponse(BaseModel):
    session_id: str
    message: str


class UploadResponse(BaseModel):
    image_id: str


# ── Aggression incident submission ────────────────────────────────────────────

class AggressionIncidentRequest(BaseModel):
    """Payload posted by the detector (Colab notebook) when an incident is confirmed.

    Mirrors backend.aggression.schemas.AggressionIncident but is a Pydantic
    model so FastAPI can validate the request automatically.
    """
    incident_id:    str
    timestamp:      str
    video_source:   str = ""
    frame_number:   int = -1

    incident_type:  str = "unknown"      # human_to_dog | dog_to_human | unknown
    person_track_id: int = -1
    dog_track_id:    int = -1
    distance_px:     float = 0.0

    ema_score:        float = 0.0
    sustained_frames: int = 0
    evidence_list:    list[str] = []

    llm_confirmed:           bool = False
    llm_confidence:          float = 0.0
    llm_severity_hint:       str = ""
    llm_reason:              str = ""
    llm_recommended_action:  str = ""

    location_label:  str = ""
    latitude:        Optional[float] = None
    longitude:       Optional[float] = None

    image_jpeg_b64:  str = ""            # optional annotated frame


class WSIncoming(BaseModel):
    message: str
    image_id: Optional[str] = None


# ── Domain objects ─────────────────────────────────────────────────────────────

class Pet(BaseModel):
    pet_id: str
    name: str
    animal_type: str
    breed: str
    age_months: int
    gender: str
    fee: float
    description: str
    similarity_score: float


class DogMatch(BaseModel):
    pet_id: str
    name: str
    breed: str
    similarity_score: float
    confidence: Literal["high", "medium", "low"]


# ── Feedback ───────────────────────────────────────────────────────────────────

@dataclass
class RejectionRecord:
    user_email: str
    pet_id: str
    reason: str
    breed: str = ""
    size: str = ""
    age_months: int = 0
    animal_type: str = ""


@dataclass
class UserPreferenceProfile:
    user_email: str
    excluded_pet_ids: list[str] = field(default_factory=list)
    avoided_breeds: list[str] = field(default_factory=list)
    avoided_sizes: list[str] = field(default_factory=list)
    avoided_animal_types: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    total_rejections: int = 0


# ── Agent response ─────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    text: str
    pets: list[dict] = field(default_factory=list)
    found_dog: Optional[dict] = None
    unavailable: bool = False


# ── Session state ──────────────────────────────────────────────────────────────

@dataclass
class UserSession:
    session_id: str
    user_name: str
    user_email: str = ""
    messages: list[dict] = field(default_factory=list)
    excluded_ids: list[str] = field(default_factory=list)
    negative_signals: list[str] = field(default_factory=list)
