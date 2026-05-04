"""Aggression detection module — LangGraph sub-agent for post-detection handling.

Public surface
--------------
process_incident(incident_dict, image_bytes=None) -> dict
    Run the full agent pipeline (severity → decide → notify → report → store)
    and return the final state.

list_incidents(limit=50)            -> list[dict]
get_incident(incident_id)           -> dict | None
"""
from backend.aggression.agent import process_incident
from backend.aggression.db import (
    init_db,
    list_incidents,
    get_incident,
)

__all__ = [
    "process_incident",
    "init_db",
    "list_incidents",
    "get_incident",
]
