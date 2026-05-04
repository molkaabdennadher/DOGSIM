"""
optimizer.py — Priority scoring and hybrid assignment algorithm.

Assignment pipeline
-------------------
1. calculate_priority_score(dog):  purely rule-based urgency score.
2. is_complex(dog):                True if dog needs an LLM judgement call.
3. get_eligible_shelters(dog, shelters): hard constraint filtering.
4. compatibility_score(dog, shelter):   soft numeric score for simple dogs.
5. llm_assign(dog, shelters, client):   single Groq call for complex dogs.
6. run_assignment(dogs, db_shelters, groq_client):
   - Sort dogs by priority DESC.
   - For each dog: get eligible shelters (live state).
   - complex → llm_assign;  simple → pick max compatibility_score.
   - Persist via db.assign_dog_to_shelter(), refresh shelter state.
   - Dogs with no eligible shelter → waiting list.
   - Return AssignmentResult.

"Simple" dogs: Healthy + normal behaviour + not pregnant + no skin disease.
All others are "complex" — a single LLM call inspects current shelter state
and returns (shelter_id, reason).  This limits LLM calls to ≈20–30 % of
the batch while still bringing AI judgement where it matters.

Constraint summary
------------------
HARD (must be satisfied, else skip shelter):
  - capacity:     shelter.effective_free_spots >= 1  (or >=2 for pregnant)
  - vet:          dog.health == "Serious Injury"  →  shelter.has_vet == True
  - budget:       shelter.monthly_budget - shelter.current_monthly_cost
                  >= dog.monthly_cost
  - expertise:    dog.behaviour == "abnormal"
                  →  shelter.expertise_level in ("intermediate", "expert")

SOFT (prefer, handled via compatibility score or LLM prompt):
  - has_vet preferred for skin disease
  - less crowded shelters preferred for skin disease
  - expert shelter preferred for abnormal behaviour
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from groq import Groq

from backend.ali.schemas import Dog, Shelter, AssignmentResult
import backend.ali.db as db

# ── Priority score weights ────────────────────────────────────────────────────
PRIORITY_WEIGHTS = {
    "serious_injury": 100,
    "pregnant":        85,
    "skin_disease":    60,
    "abnormal":        50,
    "minor_injury":    40,
    "not_vaccinated":  15,
    "not_dewormed":    10,
}

# Same default as ali/agent.py — overridable via GROQ_MODEL env var.
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


# ── Priority scoring ──────────────────────────────────────────────────────────

def calculate_priority_score(dog: Dog) -> int:
    """Return an urgency score; higher = assigned first."""
    score = 0
    if dog.health == "Serious Injury":
        score += PRIORITY_WEIGHTS["serious_injury"]
    elif dog.health == "Minor Injury":
        score += PRIORITY_WEIGHTS["minor_injury"]
    if dog.is_pregnant:
        score += PRIORITY_WEIGHTS["pregnant"]
    if dog.has_skin_disease:
        score += PRIORITY_WEIGHTS["skin_disease"]
    if dog.behaviour == "abnormal":
        score += PRIORITY_WEIGHTS["abnormal"]
    # dog.vaccinated / dog.dewormed are stored as the string "Yes" or "No"
    # so we must compare explicitly — `not "No"` is False in Python.
    if str(dog.vaccinated).strip().lower() not in ("yes", "true", "1"):
        score += PRIORITY_WEIGHTS["not_vaccinated"]
    if str(dog.dewormed).strip().lower() not in ("yes", "true", "1"):
        score += PRIORITY_WEIGHTS["not_dewormed"]
    return score


def is_complex(dog: Dog) -> bool:
    """Return True if the dog requires an LLM assignment decision."""
    return (
        dog.health in ("Serious Injury", "Minor Injury")
        or dog.is_pregnant
        or dog.has_skin_disease
        or dog.behaviour == "abnormal"
    )


# ── Hard constraint filtering ─────────────────────────────────────────────────

def get_eligible_shelters(dog: Dog, shelters: list[Shelter]) -> list[Shelter]:
    """Filter shelters by hard constraints, returning only valid candidates."""
    eligible = []
    spots_needed = 1 + (dog.expected_litter_size if dog.is_pregnant else 0)
    dog_cost = dog.monthly_cost or db.estimate_monthly_cost(dog)

    for s in shelters:
        # Capacity
        if s.effective_free_spots < spots_needed:
            continue
        # Vet required for serious injury
        if dog.health == "Serious Injury" and not s.has_vet:
            continue
        # Budget
        budget_remaining = s.monthly_budget - s.current_monthly_cost
        if budget_remaining < dog_cost:
            continue
        # Expertise for abnormal behaviour
        if dog.behaviour == "abnormal" and s.expertise_level == "beginner":
            continue
        eligible.append(s)

    return eligible


# ── Compatibility score (simple dogs, algorithmic path) ───────────────────────

def compatibility_score(dog: Dog, shelter: Shelter) -> float:
    """
    Compute a [0, 1] compatibility score used for simple dogs.
    Higher score = better fit.
    """
    score = 0.0

    # Capacity headroom (0–0.4): prefer shelters with more room
    max_spots = max(shelter.capacity, 1)
    score += 0.4 * (shelter.effective_free_spots / max_spots)

    # Budget headroom (0–0.3)
    budget_remaining = shelter.monthly_budget - shelter.current_monthly_cost
    budget_ratio = min(budget_remaining / max(shelter.monthly_budget, 1), 1.0)
    score += 0.3 * budget_ratio

    # Expertise bonus (0–0.2)
    expertise_bonus = {"beginner": 0.0, "intermediate": 0.1, "expert": 0.2}
    score += expertise_bonus.get(shelter.expertise_level, 0.0)

    # Vet bonus for skin disease (0–0.1)
    if dog.has_skin_disease and shelter.has_vet:
        score += 0.1

    return score


# ── LLM assignment (complex dogs) ─────────────────────────────────────────────

def llm_assign(
    dog: Dog,
    eligible_shelters: list[Shelter],
    groq_client: Groq,
) -> tuple[Optional[str], str]:
    """
    Ask the LLM to choose the best shelter for a complex dog.

    Returns
    -------
    (shelter_id, reason)  — shelter_id is None if the LLM cannot place the dog.
    """
    shelter_summaries = "\n".join(
        f"- ID {s.id} | {s.name} | location: {s.location} | "
        f"effective_free_spots: {s.effective_free_spots} | "
        f"budget_remaining: {s.monthly_budget - s.current_monthly_cost:.0f} DT | "
        f"expertise: {s.expertise_level} | has_vet: {s.has_vet} | "
        f"type: {s.shelter_type} | staff: {s.staff_count}"
        for s in eligible_shelters
    )

    dog_profile = (
        f"Name: {dog.name}, Age: {dog.age} months, Breed: {dog.breed}, "
        f"Gender: {dog.gender}, Health: {dog.health}, "
        f"Behaviour: {dog.behaviour}, Vaccinated: {dog.vaccinated}, "
        f"Dewormed: {dog.dewormed}, Sterilized: {dog.sterilized}, "
        f"Pregnant: {dog.is_pregnant}"
        + (f" (litter: {dog.expected_litter_size})" if dog.is_pregnant else "")
        + f", Skin disease: {dog.has_skin_disease}, "
        f"Estimated monthly cost: {dog.monthly_cost:.0f} DT"
    )

    prompt = f"""You are an expert animal welfare coordinator assigning dogs to shelters.

DOG PROFILE:
{dog_profile}

ELIGIBLE SHELTERS (hard constraints already satisfied):
{shelter_summaries}

Your task: choose the BEST shelter for this dog considering:
1. Medical needs (vet access, expertise level)
2. Capacity headroom (pregnancy needs extra space for puppies)
3. Budget sustainability
4. Welfare quality (staff count, expertise)

Respond in JSON only, no other text:
{{"shelter_id": "<chosen_id_or_null>", "reason": "<one sentence justification>"}}"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON even if the model wraps it in markdown
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None, "LLM returned unparseable response"
        data = json.loads(match.group())
        shelter_id = data.get("shelter_id") or None
        reason = data.get("reason", "LLM assignment")
        return shelter_id, reason
    except Exception as exc:
        return None, f"LLM error: {exc}"


# ── Main assignment entry point ───────────────────────────────────────────────

def run_assignment(
    dogs: list[Dog],
    groq_client: Groq,
) -> AssignmentResult:
    """
    Assign all dogs to shelters, updating the DB after each assignment so
    that subsequent dogs see the latest capacity/budget state.

    Args
    ----
    dogs:         List of Dog objects with shelter_id == -1 (unassigned).
                  monthly_cost is computed here if not already set.
    groq_client:  Initialized Groq client for complex dog decisions.

    Returns
    -------
    AssignmentResult with .assigned and .waiting lists.
    """
    # Attach cost estimates
    for dog in dogs:
        if not dog.monthly_cost:
            dog.monthly_cost = db.estimate_monthly_cost(dog)
        if not dog.priority_score:
            dog.priority_score = calculate_priority_score(dog)

    # Sort by priority descending — most urgent dogs get first pick
    dogs_sorted = sorted(dogs, key=lambda d: d.priority_score, reverse=True)

    assigned: list[dict] = []
    waiting:  list[dict] = []

    for dog in dogs_sorted:
        # Always fetch live shelter state so previous assignments are reflected
        shelter_rows = db.get_all_shelters()
        shelters = [
            Shelter(
                id=r["id"],
                name=r["name"],
                location=r["location"],
                capacity=r["capacity"],
                free_spots=r["free_spots"],
                reserved_spots=r["reserved_spots"],
                monthly_budget=r["monthly_budget"],
                current_monthly_cost=r["current_monthly_cost"],
                expertise_level=r["expertise_level"],
                staff_count=r["staff_count"],
                shelter_type=r["shelter_type"],
                has_vet=bool(r["has_vet"]),
            )
            for r in shelter_rows
        ]

        eligible = get_eligible_shelters(dog, shelters)

        if not eligible:
            dog.status = "waiting"
            db.upsert_dog(dog)
            waiting.append({
                "pet_id": dog.pet_id,
                "name":   dog.name,
                "reason": "No eligible shelter found (capacity/budget/constraints)",
                "priority_score": dog.priority_score,
            })
            continue

        # ── Choose shelter ──────────────────────────────────────────────────
        if is_complex(dog):
            chosen_id, reason = llm_assign(dog, eligible, groq_client)
            # Validate LLM returned a real eligible shelter
            valid_ids = {s.id for s in eligible}
            if chosen_id not in valid_ids:
                # Fallback: pick highest compatibility score
                best = max(eligible, key=lambda s: compatibility_score(dog, s))
                chosen_id = best.id
                reason = f"LLM fallback (algorithmic) — {reason}"
        else:
            best = max(eligible, key=lambda s: compatibility_score(dog, s))
            chosen_id = best.id
            reason = "Algorithmic assignment (simple/healthy dog)"

        # ── Persist ─────────────────────────────────────────────────────────
        dog.shelter_id = chosen_id
        dog.status = "assigned"
        dog.assignment_reason = reason
        # First insert the dog row (status=assigned already set above), then
        # call assign_dog_to_shelter which atomically updates shelter counters.
        db.upsert_dog(dog)
        db.assign_dog_to_shelter(
            pet_id=dog.pet_id,
            shelter_id=chosen_id,
            reason=reason,
            monthly_cost=dog.monthly_cost or db.estimate_monthly_cost(dog),
        )

        assigned.append({
            "pet_id":         dog.pet_id,
            "name":           dog.name,
            "shelter_id":     chosen_id,
            "reason":         reason,
            "priority_score": dog.priority_score,
        })

    return AssignmentResult(
        assigned=assigned,
        waiting=waiting,
        total=len(dogs_sorted),
        assigned_count=len(assigned),
        waiting_count=len(waiting),
    )
