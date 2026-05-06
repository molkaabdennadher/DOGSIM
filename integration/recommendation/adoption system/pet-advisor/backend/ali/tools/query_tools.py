"""
query_tools.py — ReAct query tools for the Ali agent.

Every tool opens a fresh DB connection via db module (no caching).
This guarantees the agent always reads the latest committed state.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

import backend.ali.db as db


def _normalize_boolish(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1", "oui"}:
            return True
        if v in {"false", "no", "0", "non"}:
            return False
    return value


@tool
def get_all_shelters(_: str = "") -> str:
    """List every shelter with its current capacity, free spots, budget, and
    expertise. Use this to get an overview of all shelters."""
    shelters = db.get_all_shelters()
    if not shelters:
        return "No shelters registered yet."
    lines = []
    for s in shelters:
        eff = s["free_spots"] - s["reserved_spots"]
        budget_left = s["monthly_budget"] - s["current_monthly_cost"]
        lines.append(
            f"[{s['id']}] {s['name']} | {s['location']} | "
            f"capacity={s['capacity']}, free={s['free_spots']}, "
            f"reserved={s['reserved_spots']}, effective_free={eff} | "
            f"budget={s['monthly_budget']} DT, spent={s['current_monthly_cost']:.0f} DT, "
            f"remaining={budget_left:.0f} DT | "
            f"expertise={s['expertise_level']}, staff={s['staff_count']}, "
            f"vet={'Yes' if s['has_vet'] else 'No'}, type={s['shelter_type']}"
        )
    return "\n".join(lines)


@tool
def get_shelter_details(shelter_id: str) -> str:
    """Get full details for a single shelter by its ID."""
    s = db.get_shelter(shelter_id)
    if not s:
        return f"Shelter '{shelter_id}' not found."
    eff = s["free_spots"] - s["reserved_spots"]
    budget_left = s["monthly_budget"] - s["current_monthly_cost"]
    return (
        f"Shelter: {s['name']} (ID: {s['id']})\n"
        f"Location: {s['location']}\n"
        f"Capacity: {s['capacity']} | Free: {s['free_spots']} | "
        f"Reserved: {s['reserved_spots']} | Effective free: {eff}\n"
        f"Monthly budget: {s['monthly_budget']} DT | "
        f"Spent: {s['current_monthly_cost']:.0f} DT | "
        f"Remaining: {budget_left:.0f} DT\n"
        f"Expertise: {s['expertise_level']} | Staff: {s['staff_count']}\n"
        f"Type: {s['shelter_type']} | Has vet: {'Yes' if s['has_vet'] else 'No'}"
    )


@tool
def get_capacity_risks(_: str = "") -> str:
    """List shelters whose effective free spots are ≤ 20% of total capacity.
    Use this to identify shelters at risk of overcrowding."""
    risks = db.get_capacity_risks()
    if not risks:
        return "No shelters are currently at capacity risk (all have > 20% effective free spots)."
    lines = ["Shelters at capacity risk (≤ 20% effective free spots):"]
    for r in risks:
        eff = r["free_spots"] - r["reserved_spots"]
        pct = round(100 * eff / max(r["capacity"], 1), 1)
        lines.append(
            f"  [{r['id']}] {r['name']}: {eff}/{r['capacity']} effective free ({pct}%) "
            f"| pregnant dogs inside: {r.get('pregnant_count', 0)}"
        )
    return "\n".join(lines)


@tool
def get_dogs_in_shelter(shelter_id: str) -> str:
    """List all dogs currently assigned to a specific shelter."""
    dogs = db.get_dogs_in_shelter(shelter_id)
    if not dogs:
        return f"No dogs currently in shelter '{shelter_id}'."
    lines = [f"Dogs in shelter {shelter_id}:"]
    for d in dogs:
        lines.append(
            f"  [{d['pet_id']}] {d['name']} | {d['breed']} | age {d['age']}m | "
            f"health={d['health']} | behaviour={d['behaviour']} | "
            f"pregnant={'Yes' if d['is_pregnant'] else 'No'} | "
            f"skin_disease={'Yes' if d['has_skin_disease'] else 'No'} | "
            f"cost={d['monthly_cost']:.0f} DT/mo"
        )
    return "\n".join(lines)


@tool
def get_waiting_list(_: str = "") -> str:
    """List all dogs on the waiting list (not yet assigned to any shelter),
    ordered by priority score descending."""
    waiting = db.get_waiting_list()
    if not waiting:
        return "The waiting list is empty — all dogs have been placed."
    lines = ["Waiting list (highest priority first):"]
    for d in waiting:
        lines.append(
            f"  [{d['pet_id']}] {d['name']} | priority={d['priority_score']} | "
            f"health={d['health']} | behaviour={d['behaviour']} | "
            f"pregnant={'Yes' if d['is_pregnant'] else 'No'} | "
            f"cost={d['monthly_cost']:.0f} DT/mo"
        )
    return "\n".join(lines)


@tool
def get_pregnant_dogs(_: str = "") -> str:
    """List pregnant dogs, their assigned shelters, and total expected puppies.
    Use this for questions about pregnant dogs or expected puppies."""
    rows = db.get_pregnant_dog_summary()
    if not rows:
        return "No pregnant dogs are currently registered."

    total_puppies = sum(int(r.get("expected_litter_size") or 0) for r in rows)
    lines = [
        f"Pregnant dogs: {len(rows)}",
        f"Expected puppies: {total_puppies}",
        "",
        "By shelter:",
    ]
    for r in rows:
        shelter = r["shelter_name"]
        if r.get("shelter_location"):
            shelter += f" ({r['shelter_location']})"
        lines.append(
            f"  - {shelter}: {r['dog_name']} [{r['pet_id']}], "
            f"{r['expected_litter_size']} expected puppies, status={r['status']}"
        )
    return "\n".join(lines)


@tool
def get_statistics(_: str = "") -> str:
    """Return aggregate statistics: total dogs, assigned, waiting,
    total monthly cost, shelter utilisation."""
    # db.get_statistics() returns a flat dict — access fields directly
    st = db.get_statistics()
    return (
        f"=== Shelter Statistics ===\n"
        f"Total shelters: {st.get('total_shelters', 0)}\n"
        f"Total capacity: {st.get('total_capacity', 0)}\n"
        f"Total free spots: {st.get('total_free_spots', 0)}\n"
        f"Total reserved spots: {st.get('total_reserved', 0)}\n"
        f"Shelters with vet: {st.get('shelters_with_vet', 0)}\n"
        f"Total monthly budget: {st.get('total_budget', 0):.0f} DT\n"
        f"Total monthly cost: {st.get('total_cost', 0):.0f} DT\n\n"
        f"=== Dog Statistics ===\n"
        f"Total dogs: {st.get('total_dogs', 0)}\n"
        f"Assigned: {st.get('assigned', 0)}\n"
        f"Waiting: {st.get('waiting', 0)}\n"
        f"Released: {st.get('released', 0)}\n"
        f"Serious injury: {st.get('serious_injury', 0)}\n"
        f"Pregnant: {st.get('pregnant', 0)}\n"
        f"Skin disease: {st.get('skin_disease', 0)}\n"
        f"Abnormal behaviour: {st.get('abnormal', 0)}"
    )


@tool
def get_dogs_by_criteria(criteria_json: str) -> str:
    """Search dogs by criteria. Pass a JSON string with any of these optional
    keys: status (assigned/waiting/released), health, behaviour,
    is_pregnant (true/false), has_skin_disease (true/false).
    To find all waiting dogs: {"status": "waiting"}
    To find pregnant dogs: {"is_pregnant": true}
    Returns a formatted list of matching dogs."""
    import json as _json
    try:
        criteria = _json.loads(criteria_json) if criteria_json.strip() else {}
    except _json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    dogs = db.get_all_dogs()

    # Apply filters
    status       = criteria.get("status")
    health       = criteria.get("health")
    behaviour    = criteria.get("behaviour")
    is_pregnant  = criteria.get("is_pregnant")
    skin_disease = criteria.get("has_skin_disease")

    results = []
    for d in dogs:
        if status       and d.get("status")           != status:                  continue
        if health       and d.get("health","").lower() != health.lower():          continue
        if behaviour    and d.get("behaviour","").lower() != behaviour.lower():    continue
        if is_pregnant  is not None and bool(d.get("is_pregnant")) != bool(is_pregnant): continue
        if skin_disease is not None and bool(d.get("has_skin_disease")) != bool(skin_disease): continue
        results.append(d)

    if not results:
        return "No dogs match the given criteria."

    lines = [f"Found {len(results)} dog(s):"]
    for d in results:
        lines.append(
            f"  [{d['pet_id']}] {d['name']} | "
            f"Status: {d['status']} | Health: {d['health']} | "
            f"Shelter: {d.get('shelter_id', '-')} | "
            f"Pregnant: {'Yes' if d.get('is_pregnant') else 'No'} | "
            f"Priority: {d.get('priority_score', 0):.1f}"
        )
    return "\n".join(lines)


# -- Tool list exported to __init__.py
QUERY_TOOLS = [
    get_all_shelters,
    get_shelter_details,
    get_capacity_risks,
    get_dogs_in_shelter,
    get_waiting_list,
    get_pregnant_dogs,
    get_statistics,
    get_dogs_by_criteria,
]
