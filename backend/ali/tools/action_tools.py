"""
action_tools.py — Mutation tools for the Ali ReAct agent.

Every write is immediately committed so the next query sees the updated state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from groq import Groq
from langchain_core.tools import tool

import backend.ali.db as db
from backend.ali.schemas import Dog
from backend.ali.optimizer import run_assignment
from backend.ali.tools.excel_parser import parse_file as parse_spreadsheet, SUPPORTED_EXTENSIONS

_groq_client: Optional[Groq] = None


def set_groq_client(client: Groq) -> None:
    global _groq_client
    _groq_client = client


@tool
def release_dog_from_shelter(pet_id: str) -> str:
    """Remove a dog from its current shelter and place it back on the waiting
    list. The shelter's free spot is immediately restored. If there are dogs
    on the waiting list, the next highest-priority dog is automatically
    reassigned."""
    dog = db.get_dog(pet_id)
    if not dog:
        return f"Dog '{pet_id}' not found."
    if dog["status"] != "assigned":
        return f"Dog '{pet_id}' is not currently assigned (status: {dog['status']})."

    result = db.release_dog(pet_id)
    if not result:
        return f"Failed to release dog '{pet_id}'."

    prev_shelter_id = result["freed_shelter_id"]
    result_lines = [
        f"Dog '{pet_id}' ({dog['name']}) released from shelter {prev_shelter_id}.",
        "Dog is now on the waiting list.",
    ]

    # Try to fill the vacated spot from waiting list
    waiting = db.get_waiting_list()
    if not waiting:
        result_lines.append("Waiting list is empty — no reassignment needed.")
        return "\n".join(result_lines)

    next_row = waiting[0]
    next_dog = Dog(
        pet_id=next_row["pet_id"],
        name=next_row["name"],
        age=next_row["age"],
        breed=next_row["breed"],
        gender=next_row["gender"],
        vaccinated=next_row["vaccinated"],
        dewormed=next_row["dewormed"],
        sterilized=next_row["sterilized"],
        health=next_row["health"],
        shelter_id="-1",
        behaviour=next_row["behaviour"],
        is_pregnant=bool(next_row["is_pregnant"]),
        expected_litter_size=next_row["expected_litter_size"],
        has_skin_disease=bool(next_row["has_skin_disease"]),
        priority_score=next_row["priority_score"],
        monthly_cost=next_row["monthly_cost"],
    )

    if _groq_client is None:
        result_lines.append("Groq client not initialised — cannot reassign waiting dog.")
        return "\n".join(result_lines)

    assignment = run_assignment([next_dog], _groq_client)
    if assignment.assigned:
        a = assignment.assigned[0]
        result_lines.append(
            f"Waiting dog '{next_dog.pet_id}' ({next_dog.name}) "
            f"→ assigned to shelter {a['shelter_id']}. Reason: {a['reason']}"
        )
    else:
        result_lines.append(
            f"Waiting dog '{next_dog.pet_id}' ({next_dog.name}) "
            "could not be placed — no eligible shelter available."
        )

    return "\n".join(result_lines)


@tool
def process_excel_upload(file_path: str) -> str:
    """Process an uploaded .xlsx file of dogs and assign them to shelters.
    Argument: absolute path to the .xlsx file on disk.
    Only .xlsx format is supported.
    Returns a summary of assignments made."""
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return f"Unsupported file format '{path.suffix}'. Only .xlsx is accepted."

    dogs, warnings = parse_spreadsheet(path)
    if not dogs:
        warn_text = "\n".join(warnings) if warnings else "No valid rows found."
        return f"No dogs to assign.\n{warn_text}"

    if _groq_client is None:
        return "Groq client not initialised — cannot run LLM assignment."

    result = run_assignment(dogs, _groq_client)

    lines = [
        f"Assignment complete: {result.assigned_count}/{result.total} dogs placed, "
        f"{result.waiting_count} on waiting list.",
    ]
    if warnings:
        lines.append("\nParsing warnings:")
        lines.extend(f"  • {w}" for w in warnings)
    if result.assigned:
        lines.append("\nAssigned:")
        for a in result.assigned:
            lines.append(
                f"  [{a['pet_id']}] {a['name']} → shelter {a['shelter_id']} | {a['reason']}"
            )
    if result.waiting:
        lines.append("\nWaiting list:")
        for w in result.waiting:
            lines.append(f"  [{w['pet_id']}] {w['name']} | {w['reason']}")

    return "\n".join(lines)


@tool
def add_shelter(shelter_json: str) -> str:
    """Add a new shelter to the system. Pass a JSON string with fields:
    name, location, capacity (int), monthly_budget (float),
    expertise_level (beginner/intermediate/expert), staff_count (int),
    shelter_type (public/private), has_vet (true/false).
    Optionally include id (string) — auto-generated if omitted.
    Example: {"name":"Safe Haven","location":"Tunis","capacity":20,
    "monthly_budget":5000,"expertise_level":"intermediate","staff_count":5,
    "shelter_type":"private","has_vet":true}"""
    try:
        data = json.loads(shelter_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    required = ["name", "location", "capacity", "monthly_budget",
                "expertise_level", "staff_count", "shelter_type", "has_vet"]
    missing = [f for f in required if f not in data]
    if missing:
        return f"Missing required fields: {missing}"

    shelter_id = db.add_shelter(data)
    return f"Shelter '{data['name']}' (ID: {shelter_id}) added successfully."


@tool
def update_shelter_field(update_json: str) -> str:
    """Update a single field on an existing shelter.
    Pass JSON with: shelter_id, field, value.
    Updatable fields: name, location, monthly_budget, expertise_level,
    staff_count, shelter_type, has_vet.
    Example: {"shelter_id":"S3","field":"has_vet","value":true}"""
    try:
        data = json.loads(update_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    shelter_id = data.get("shelter_id")
    field      = data.get("field")
    value      = data.get("value")

    if not shelter_id or not field or value is None:
        return "Required: shelter_id, field, value."

    allowed = {"name", "location", "monthly_budget", "expertise_level",
               "staff_count", "shelter_type", "has_vet"}
    if field not in allowed:
        return f"Field '{field}' cannot be updated. Allowed: {sorted(allowed)}"

    ok = db.update_shelter_field(shelter_id, field, value)
    if not ok:
        return f"Update failed — shelter '{shelter_id}' not found or field invalid."
    return f"Updated shelter {shelter_id}: {field} = {value}."


# -- Tool list exported to __init__.py
ACTION_TOOLS = [
    release_dog_from_shelter,
    process_excel_upload,
    add_shelter,
    update_shelter_field,
]
