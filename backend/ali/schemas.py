"""
Pydantic models and dataclasses shared across the shelter-assignment module.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel


# ── REST payloads ─────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    name: str


class LoginResponse(BaseModel):
    session_id: str
    message: str


class UploadResponse(BaseModel):
    file_id: str


class ReleaseRequest(BaseModel):
    pet_id: str
    reason: str = "adopted"


class AddShelterRequest(BaseModel):
    id: Optional[str] = None      # auto-generated if omitted
    name: str
    location: str
    capacity: int
    monthly_budget: float
    expertise_level: str          # beginner | intermediate | expert
    staff_count: int
    shelter_type: str             # private | public
    has_vet: bool


class UpdateShelterRequest(BaseModel):
    field: str
    value: str | int | float | bool


# ── Domain dataclasses (used internally by optimizer + tools) ─────────────────

@dataclass
class Dog:
    pet_id: str
    name: str
    age: float
    breed: str
    gender: str                   # Male | Female
    vaccinated: str               # Yes | No
    dewormed: str                 # Yes | No
    sterilized: str               # Yes | No
    health: str                   # Healthy | Minor Injury | Serious Injury
    behaviour: str                # normal | abnormal
    is_pregnant: bool
    expected_litter_size: int     # 0 if not pregnant
    has_skin_disease: bool
    shelter_id: str = "-1"
    priority_score: int = 0
    status: str = "waiting"       # assigned | waiting | released
    monthly_cost: float = 0.0
    assignment_reason: str = ""


@dataclass
class Shelter:
    id: str
    name: str
    location: str
    capacity: int
    free_spots: int
    reserved_spots: int           # spots reserved for incoming litters
    monthly_budget: float
    current_monthly_cost: float
    expertise_level: str          # beginner | intermediate | expert
    staff_count: int
    shelter_type: str             # private | public
    has_vet: bool

    @property
    def effective_free_spots(self) -> int:
        """Spots actually available — free_spots minus litter reservations."""
        return self.free_spots - self.reserved_spots

    @property
    def budget_remaining(self) -> float:
        return self.monthly_budget - self.current_monthly_cost

    @property
    def fill_rate(self) -> float:
        if self.capacity == 0:
            return 1.0
        assigned = self.capacity - self.free_spots
        return assigned / self.capacity


@dataclass
class AssignmentResult:
    assigned: list[dict] = field(default_factory=list)   # [{pet_id, shelter_id, reason}]
    waiting:  list[dict] = field(default_factory=list)   # [{pet_id, reason}]
    total: int = 0
    assigned_count: int = 0
    waiting_count: int = 0


# ── Agent response ─────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    text: str
    data: Optional[dict] = None   # structured payload (assignment report, etc.)
