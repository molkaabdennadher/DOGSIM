"""
Aggression post-detection agent — LangGraph state machine.

Changes from v1
---------------
LLM backend   Switched from Ollama (llama3.2 local) to Groq API
               (configurable via GROQ_API_KEY + GROQ_MODEL env vars).
               Falls back gracefully if the env var is missing.

Vision support If image_jpeg_b64 is present (key-frame mosaic from the
               video analyser), the severity node uses a vision-capable
               Groq model (llama-4-scout-17b-16e-instruct) and a temporal
               reasoning prompt instead of the static single-frame prompt.

JSON parsing   Replaced the fragile ``re.search(r'\\{.*\\}', raw, re.DOTALL)``
               with a layered strict parser that strips markdown fences,
               tries json.loads, then falls back to a bounded regex extraction.

Temporal prompt New SEVERITY_PROMPT_VISION is explicitly designed for
               multi-frame mosaic analysis (chronological grid of key frames).

Pipeline (unchanged topology)
-------------------------------
        evaluate_severity → decide_measure → notify_contacts → generate_report → store_incident → END
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langgraph.graph import StateGraph, END
from openai import OpenAI

from backend.aggression import db as aggression_db
from backend.aggression import tools as aggression_tools
from backend.aggression.schemas import AggressionState

log = logging.getLogger("aggression.agent")

# ── LLM config (Groq via OpenAI-compatible endpoint) ─────────────────────────

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Default models — overridable via environment variables.
# Text-only (severity triage when no mosaic is available):
GROQ_TEXT_MODEL   = os.environ.get("GROQ_TEXT_MODEL",   "llama-3.3-70b-versatile")
# Vision-capable (severity triage with key-frame mosaic):
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

_LLM_CLIENT: OpenAI | None = None


def _client() -> OpenAI:
    """Return the shared OpenAI-compatible Groq client, creating it on first call.

    The API key is read from the environment at first call — NOT at module
    import time — so that load_dotenv() in main.py always takes effect
    regardless of import order.
    """
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            log.warning(
                "GROQ_API_KEY not set — LLM calls will fail. "
                "Severity evaluation will use the heuristic fallback."
            )
        _LLM_CLIENT = OpenAI(base_url=_GROQ_BASE_URL, api_key=api_key or "dummy")
    return _LLM_CLIENT


# ── JSON parsing helpers ───────────────────────────────────────────────────────

def _strict_json_parse(text: str) -> dict | None:
    """Parse JSON from LLM output reliably.

    Strategy (most to least strict):
      1. Strip markdown code fences and call json.loads directly.
      2. Find the first balanced {...} block and parse it.
      3. Use a bounded regex to extract a simple flat object.
    """
    # Strip common markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$",          "",  text.strip())

    # Pass 1: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Pass 2: extract first balanced { … } block
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i+1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                start = -1

    # Pass 3: bounded flat-object regex (last resort)
    m = re.search(r'\{[^{}]{1,800}\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


# ── Severity prompts ───────────────────────────────────────────────────────────

SEVERITY_PROMPT_TEXT = """You are a triage assistant for an animal-aggression monitoring system.
Read the incident summary below and decide its severity. Return STRICT JSON only:

  {"severity": "minor" | "serious" | "critical",
   "rationale": "<one sentence, max 25 words>"}

Severity scale:
  "critical": clear physical attack with injury risk imminent (or confirmed fall).
  "serious":  likely aggressive contact — intervention needed soon.
  "minor":    tense or borderline situation — log but no urgent action.

Bias toward the higher class when uncertain — under-triaging is more harmful than over-triaging.
"""

SEVERITY_PROMPT_VISION = """You are a triage assistant for an animal-aggression monitoring system.
You are looking at a mosaic of video key frames arranged in CHRONOLOGICAL ORDER
(left-to-right, top-to-bottom). Each frame shows bounding boxes and, where available,
skeleton keypoints overlaid on the scene.

Analyse the TEMPORAL SEQUENCE across all frames:
  - How does the body language and proximity evolve frame-to-frame?
  - Is aggression escalating, de-escalating, or sustained?
  - Are there signals of physical contact, lunging, falling, or retreating?

Return STRICT JSON only:
  {"severity": "minor" | "serious" | "critical",
   "rationale": "<one sentence describing temporal observations, max 30 words>"}

Severity scale:
  "critical": physical attack or injury visible / progressing across frames.
  "serious":  aggressive contact or sustained approach across multiple frames.
  "minor":    single frame or brief tension — not escalating.

Bias toward the higher class when uncertain.
"""


def _format_incident_text(inc: dict) -> str:
    return (
        f"Incident type: {inc.get('incident_type', 'unknown')}\n"
        f"EMA score: {inc.get('ema_score', 0):.3f}\n"
        f"Sustained frames above threshold: {inc.get('sustained_frames', 0)}\n"
        f"Edge-to-edge distance (px): {inc.get('distance_px', 0):.1f}\n"
        f"Detector confidence: {inc.get('llm_confidence', 0):.2f}\n"
        f"Detector reason: {inc.get('llm_reason', '')}\n"
        f"Severity hint from detector: {inc.get('llm_severity_hint', '')}\n"
        f"Evidence signals: {', '.join(inc.get('evidence_list', [])) or '(none)'}\n"
    )


# ── Node 1: evaluate severity ─────────────────────────────────────────────────

def _node_evaluate_severity(state: AggressionState) -> dict:
    inc       = state.get("incident", {}) or {}
    image_b64 = state.get("image_jpeg_b64", "") or ""

    # Defaults from detector in case LLM call fails
    severity  = inc.get("llm_severity_hint") or "minor"
    rationale = ""

    try:
        text_block = _format_incident_text(inc)

        if image_b64:
            # Vision path: send mosaic as base64 image with temporal prompt
            messages = [
                {"role": "system", "content": SEVERITY_PROMPT_VISION},
                {"role": "user", "content": [
                    {"type": "text", "text": text_block},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ]
            model = GROQ_VISION_MODEL
        else:
            # Text-only path
            messages = [
                {"role": "system", "content": SEVERITY_PROMPT_TEXT},
                {"role": "user",   "content": text_block},
            ]
            model = GROQ_TEXT_MODEL

        resp = _client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        raw    = (resp.choices[0].message.content or "").strip()
        parsed = _strict_json_parse(raw)

        if isinstance(parsed, dict):
            sev = str(parsed.get("severity", "")).lower().strip()
            if sev in {"minor", "serious", "critical"}:
                severity  = sev
                rationale = str(parsed.get("rationale", ""))[:300]

    except Exception as exc:
        log.warning("Severity LLM failed (%s) — heuristic fallback", exc)
        severity  = _heuristic_severity(inc)
        rationale = "LLM unavailable — derived from EMA score and sustained frames."

    return {
        "severity":           severity,
        "severity_rationale": rationale,
        "actions_taken":      [*(state.get("actions_taken") or []), "severity_evaluated"],
    }


def _heuristic_severity(inc: dict) -> str:
    """Deterministic fallback — reads thresholds from aggression_config.json."""
    from backend.aggression.config import load_aggression_config as _lcfg
    try:
        thr = _lcfg().get("severity_thresholds", {})
    except Exception:
        thr = {}
    critical_t = float(thr.get("critical", 0.82))
    serious_t  = float(thr.get("serious",  0.65))

    ema       = float(inc.get("ema_score", 0.0))
    sustained = int(inc.get("sustained_frames", 0))
    evidence  = set(inc.get("evidence_list", []))

    # Falls always escalate severity
    if "person_fall_detected" in evidence:
        return "critical" if ema >= serious_t else "serious"

    if ema >= critical_t or sustained >= 20:
        return "critical"
    if ema >= serious_t or sustained >= 10:
        return "serious"
    return "minor"


# ── Node 2: decide measure ────────────────────────────────────────────────────

def _node_decide_measure(state: AggressionState) -> dict:
    """Pure routing — no LLM, fully explainable.

    Decision matrix (severity × incident_type):
                        human_to_dog           dog_to_human        unknown
      critical          hospital + rights       hospital + rights   hospital + rights
      serious           rights                  hospital            rights
      minor             shelter_log             shelter_log         shelter_log
    """
    severity = state.get("severity", "minor")
    inc_type = (state.get("incident") or {}).get("incident_type", "unknown")

    if severity == "critical":
        decision = ["hospital", "animal_rights"]
    elif severity == "serious":
        if inc_type == "dog_to_human":
            decision = ["hospital"]
        else:   # human_to_dog or unknown
            decision = ["animal_rights"]
    else:
        decision = ["shelter_log"]

    return {
        "contact_decision": decision,
        "actions_taken":    [*(state.get("actions_taken") or []),
                             f"decided:{','.join(decision)}"],
    }


# ── Node 3: notify contacts ───────────────────────────────────────────────────

_TOOL_MAP = {
    "hospital":      aggression_tools.notify_hospital,
    "animal_rights": aggression_tools.notify_animal_rights,
    "shelter_log":   aggression_tools.shelter_log,
}


def _node_notify_contacts(state: AggressionState) -> dict:
    decisions = state.get("contact_decision") or []
    sent      = list(state.get("notifications_sent") or [])
    actions   = list(state.get("actions_taken") or [])

    for channel in decisions:
        tool = _TOOL_MAP.get(channel)
        if not tool:
            sent.append({"channel": channel, "status": "unknown_tool"})
            continue
        try:
            sent.append(tool(state))
        except Exception as exc:
            log.exception("Notification %s crashed", channel)
            sent.append({"channel": channel, "status": "failed", "detail": str(exc)})
        actions.append(f"notified:{channel}")

    return {"notifications_sent": sent, "actions_taken": actions}


# ── Node 4: generate report ───────────────────────────────────────────────────

def _node_generate_report(state: AggressionState) -> dict:
    key_frame = state.get("image_jpeg_b64", "") or ""
    md   = aggression_tools.generate_report_markdown(dict(state), key_frame_b64=key_frame)
    inc  = state.get("incident", {}) or {}
    path = aggression_tools.write_report(md, inc.get("incident_id", "unknown"))
    return {
        "report_markdown": md,
        "actions_taken":   [*(state.get("actions_taken") or []),
                            f"report_written:{path}"],
    }


# ── Node 5: store ─────────────────────────────────────────────────────────────

def _node_store_incident(state: AggressionState) -> dict:
    db_id = aggression_db.store_incident(dict(state))
    return {
        "db_event_id":   db_id,
        "actions_taken": [*(state.get("actions_taken") or []),
                         f"stored:db#{db_id}"],
    }


# ── Build & cache the compiled graph ─────────────────────────────────────────

def _build_graph():
    g = StateGraph(AggressionState)
    g.add_node("evaluate_severity", _node_evaluate_severity)
    g.add_node("decide_measure",    _node_decide_measure)
    g.add_node("notify_contacts",   _node_notify_contacts)
    g.add_node("generate_report",   _node_generate_report)
    g.add_node("store_incident",    _node_store_incident)

    g.set_entry_point("evaluate_severity")
    g.add_edge("evaluate_severity", "decide_measure")
    g.add_edge("decide_measure",    "notify_contacts")
    g.add_edge("notify_contacts",   "generate_report")
    g.add_edge("generate_report",   "store_incident")
    g.add_edge("store_incident",    END)
    return g.compile()


_GRAPH: Any | None = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ── Public entry-point ────────────────────────────────────────────────────────

def process_incident(incident_dict: dict, image_jpeg_b64: str = "") -> dict:
    """Run the full LangGraph agent pipeline on a confirmed aggression incident.

    Args:
        incident_dict:   Fields from AggressionIncident (as dict).
        image_jpeg_b64:  Base-64 JPEG mosaic from the video analyser.
                         When present, the severity node uses vision-capable
                         model for multi-frame temporal reasoning.

    Returns:
        Final agent state dict (severity, decision, notifications, report, db id).
    """
    initial_state: AggressionState = {
        "incident":       incident_dict,
        "image_jpeg_b64": image_jpeg_b64,
        "actions_taken":  [],
    }
    final_state = _graph().invoke(initial_state)
    return dict(final_state)
