"""
Schemas for the aggression module.

`AggressionIncident` is the immutable input shape — what the Colab pipeline
(or any external detector) submits to the backend. It must contain enough
context for the backend agent to reason about severity and choose actions.

`AggressionState` is the mutable, agent-internal shape that flows through
the LangGraph nodes — a TypedDict consumed by `agent.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypedDict, Literal


# ── REST-facing input ─────────────────────────────────────────────────────────

@dataclass
class AggressionIncident:
    """Payload sent by the detector (Colab notebook) when an incident is confirmed.

    Field meanings mirror the notebook IncidentState — keep them in sync.
    """
    incident_id:    str                    # uuid4 string assigned by the detector
    timestamp:      str                    # ISO-8601, when the incident was confirmed
    video_source:   str                    # filename / URL / camera id
    frame_number:   int

    incident_type:  Literal["human_to_dog", "dog_to_human", "unknown"]
    person_track_id: int
    dog_track_id:    int
    distance_px:     float

    ema_score:        float
    sustained_frames: int
    evidence_list:    list[str]            # signal names that fired

    # Detector-side LLM confirmation (Gemini in the notebook).
    llm_confirmed:           bool
    llm_confidence:          float
    llm_severity_hint:       str           # the detector's severity guess (the agent re-evaluates)
    llm_reason:              str
    llm_recommended_action:  str

    # Optional location metadata (used to pick the right hospital / NGO).
    location_label:  str = ""              # human-readable, e.g. "Tunis, Lac 2"
    latitude:        Optional[float] = None
    longitude:       Optional[float] = None


# ── Agent state ───────────────────────────────────────────────────────────────

class AggressionState(TypedDict, total=False):
    """LangGraph state for the post-detection agent.

    Total=False so nodes can return partial dicts and the reducer (LangGraph
    default = dict-merge) keeps everything else.
    """
    # ── Inputs (frozen after invocation) ──
    incident:        dict                  # AggressionIncident as dict
    image_jpeg_b64:  str                   # optional annotated frame, base64

    # ── Agent decisions ──
    severity:                 Literal["minor", "serious", "critical"]
    severity_rationale:       str          # short LLM justification
    contact_decision:         list[Literal["hospital", "animal_rights", "shelter_log"]]

    # ── Side effects ──
    notifications_sent:       list[dict]    # each: {target, channel, status, payload}
    report_markdown:          str
    db_event_id:              int
    actions_taken:            list[str]
