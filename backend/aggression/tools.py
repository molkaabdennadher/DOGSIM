"""
Side-effect tools for the aggression agent.

Each tool returns a structured `{status, target, channel, payload, ...}` dict
that the agent appends into `state["notifications_sent"]`. We never raise
inside a tool — failures are recorded but must not break the report+store
chain that follows.

Why no Twilio / WhatsApp / real SMTP wired by default
-----------------------------------------------------
Hard-coding a particular notification provider would force every deploy to
configure that provider just to run. Instead we:
  - Always write the notification payload to disk under
    `artifacts/aggression_outbox/` so a human (or a CRON job) can pick it up.
  - Send via SMTP only if `SMTP_HOST` / `SMTP_USER` / `SMTP_PASS` env vars are
    set — otherwise we log the intent and return `status="queued"`.

Configuring real recipients is documented in AGGRESSION_README.md.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

log = logging.getLogger("aggression.tools")

OUTBOX_DIR = Path(__file__).parent.parent.parent / "artifacts" / "agressionDetection" / "aggression_outbox"
OUTBOX_DIR.mkdir(exist_ok=True, parents=True)


# ── Default contacts (overridable via env or aggression_config.json) ─────────

def _load_contacts() -> dict:
    """Resolve recipients from (in order):
       1. aggression_config.json in artifacts/agressionDetection/  (preferred)
       2. environment variables                                      (fallback)
       3. hardcoded "demo" defaults                                  (last resort)
    """
    from backend.aggression.config import load_aggression_config
    cfg: dict = load_aggression_config()

    contacts = (cfg or {}).get("contacts", {}) or {}
    return {
        "hospital_email":       contacts.get("hospital_email")
                                 or os.environ.get("AGGRESSION_HOSPITAL_EMAIL", "demo-hospital@example.org"),
        "hospital_label":       contacts.get("hospital_label")
                                 or os.environ.get("AGGRESSION_HOSPITAL_LABEL", "Local emergency department"),
        "animal_rights_email":  contacts.get("animal_rights_email")
                                 or os.environ.get("AGGRESSION_ANIMAL_RIGHTS_EMAIL", "demo-spa@example.org"),
        "animal_rights_label":  contacts.get("animal_rights_label")
                                 or os.environ.get("AGGRESSION_ANIMAL_RIGHTS_LABEL", "Animal welfare association"),
    }


def _smtp_enabled() -> bool:
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS"))


def _send_email(to_addr: str, subject: str, body: str) -> tuple[str, str]:
    """Send via SMTP if fully configured, else queue to disk.

    Returns (status, detail) where status ∈ {"sent", "queued", "failed"}.
    """
    if not _smtp_enabled():
        path = _queue_to_outbox(to_addr, subject, body)
        return "queued", str(path)

    try:
        msg = MIMEMultipart()
        msg["From"]    = os.environ["SMTP_USER"]
        msg["To"]      = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        host = os.environ["SMTP_HOST"]
        port = int(os.environ.get("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=10) as srv:
            srv.starttls()
            srv.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            srv.sendmail(os.environ["SMTP_USER"], [to_addr], msg.as_string())
        return "sent", to_addr
    except Exception as exc:
        log.exception("SMTP send failed; falling back to outbox.")
        path = _queue_to_outbox(to_addr, subject, body, error=str(exc))
        return "failed", str(path)


def _send_email_with_frame(to_addr: str, subject: str, body: str,
                            frame_b64: str = "") -> tuple[str, str]:
    """Send email with optional key frame attachment."""
    if not _smtp_enabled():
        # Include frame reference in the queued file
        path = _queue_to_outbox(to_addr, subject, body,
                                extra_note=f"[Key frame attached: {len(frame_b64)} bytes base64]" if frame_b64 else "")
        return "queued", str(path)
    try:
        msg = MIMEMultipart()
        msg["From"] = os.environ["SMTP_USER"]
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Attach key frame as inline image if provided
        if frame_b64:
            import base64
            frame_bytes = base64.b64decode(frame_b64)
            from email.mime.image import MIMEImage
            img_part = MIMEImage(frame_bytes, _subtype="jpeg")
            img_part.add_header("Content-Disposition", "attachment",
                               filename="aggression_keyframe.jpg")
            img_part.add_header("Content-ID", "<keyframe>")
            msg.attach(img_part)

        host = os.environ["SMTP_HOST"]
        port = int(os.environ.get("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=10) as srv:
            srv.starttls()
            srv.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            srv.sendmail(os.environ["SMTP_USER"], [to_addr], msg.as_string())
        return "sent", to_addr
    except Exception as exc:
        log.exception("SMTP send failed; falling back to outbox.")
        path = _queue_to_outbox(to_addr, subject, body, error=str(exc))
        return "failed", str(path)


def _queue_to_outbox(to_addr: str, subject: str, body: str, error: str = "", extra_note: str = "") -> Path:
    """Write the message to disk so an operator can pick it up later."""
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_to = to_addr.replace("@", "_at_").replace("/", "_")
    path = OUTBOX_DIR / f"{stamp}_{safe_to}.eml"
    payload = (
        f"To: {to_addr}\nSubject: {subject}\n"
        f"X-Smtp-Status: {'pending' if not error else f'failed ({error})'}\n"
        f"{f'X-Note: {extra_note}' + chr(10) if extra_note else ''}"
        f"\n{body}\n"
    )
    path.write_text(payload, encoding="utf-8")
    return path


# ── Public tools used by the LangGraph agent ──────────────────────────────────

def notify_hospital(state: dict) -> dict:
    """Notify the local hospital — used when a human was bitten / attacked."""
    contacts = _load_contacts()
    incident = state.get("incident", {})
    key_frame_b64 = state.get("image_jpeg_b64", "") or ""
    subject  = f"[URGENT] Suspected dog-attack on human — incident {incident.get('incident_id','?')}"
    body     = _compose_body(state, channel="hospital")
    status, detail = _send_email_with_frame(contacts["hospital_email"], subject, body, key_frame_b64)
    return {
        "target":  contacts["hospital_label"],
        "channel": "hospital",
        "status":  status,
        "detail":  detail,
    }


def notify_animal_rights(state: dict) -> dict:
    """Notify the local animal-welfare NGO — used when a human attacks a dog."""
    contacts = _load_contacts()
    incident = state.get("incident", {})
    key_frame_b64 = state.get("image_jpeg_b64", "") or ""
    subject  = f"[ALERT] Suspected animal abuse — incident {incident.get('incident_id','?')}"
    body     = _compose_body(state, channel="animal_rights")
    status, detail = _send_email_with_frame(contacts["animal_rights_email"], subject, body, key_frame_b64)
    return {
        "target":  contacts["animal_rights_label"],
        "channel": "animal_rights",
        "status":  status,
        "detail":  detail,
    }


def shelter_log(state: dict) -> dict:
    """For minor incidents we just log internally — no external contact."""
    return {
        "target":  "Shelter operations log",
        "channel": "shelter_log",
        "status":  "logged",
        "detail":  "internal",
    }


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report_markdown(state: dict, key_frame_b64: str = "") -> str:
    """Compose a deterministic, structured incident report in Markdown.

    Kept template-based on purpose: the agent's *judgment* lives in the severity
    + decide nodes; the report is just an audit trail. A consistent format also
    makes it grep-friendly for downstream pipelines.
    """
    inc = state.get("incident", {})
    sev = state.get("severity", "minor")
    rationale = state.get("severity_rationale", "")
    decisions = state.get("contact_decision", [])
    notifs    = state.get("notifications_sent", [])

    evidence_lines = [f"  - {e}" for e in inc.get("evidence_list", [])] or ["  - (none recorded)"]
    notif_lines    = [
        f"  - **{n.get('target','?')}** ({n.get('channel','?')}) — {n.get('status','?')}"
        for n in notifs
    ] or ["  - (none)"]

    md = f"""# Incident report — {inc.get('incident_id','?')}

**Generated**: {datetime.utcnow().isoformat()}Z
**Severity**: `{sev.upper()}`
**Type**: `{inc.get('incident_type','unknown')}`
**Source**: {inc.get('video_source','?')} (frame {inc.get('frame_number','?')})

## Detection summary
- EMA score: **{inc.get('ema_score', 0):.3f} / 1.0**
- Sustained frames: {inc.get('sustained_frames', 0)}
- Edge-to-edge distance: {inc.get('distance_px', 0):.1f}px
- Person track id: {inc.get('person_track_id', '?')}
- Dog track id: {inc.get('dog_track_id', '?')}
- Detector LLM confidence: {inc.get('llm_confidence', 0):.2f}
- Detector LLM reason: {inc.get('llm_reason', '')!s}

### Evidence signals
{chr(10).join(evidence_lines)}

## Agent verdict
- **Severity**: `{sev}`
- **Rationale**: {rationale or '_n/a_'}
- **Decided contacts**: {", ".join(decisions) or "_none_"}

## Notifications dispatched
{chr(10).join(notif_lines)}
"""
    # Add key frame if provided
    if key_frame_b64:
        md += f"""
## Frame clé
![Frame d'agression](data:image/jpeg;base64,{key_frame_b64})
"""

    md += f"""
## Location
- Label: {inc.get('location_label','') or '_unknown_'}
- Coordinates: {inc.get('latitude','?')}, {inc.get('longitude','?')}
"""
    return md


def write_report(report_md: str, incident_id: str) -> str:
    """Persist the report next to the outbox so operators can grep through it."""
    reports_dir = OUTBOX_DIR.parent / "aggression_reports"
    reports_dir.mkdir(exist_ok=True, parents=True)
    path = reports_dir / f"{incident_id}.md"
    path.write_text(report_md, encoding="utf-8")
    return str(path)


# ── Internal: shared body for emails ──────────────────────────────────────────

def _compose_body(state: dict, channel: str) -> str:
    inc = state.get("incident", {})
    severity = state.get("severity", "minor")
    intro = (
        "Our automated monitoring system detected a likely incident requiring your attention."
        if channel == "hospital"
        else "Our automated monitoring system detected a likely act of animal mistreatment."
    )
    return (
        f"{intro}\n\n"
        f"Severity: {severity.upper()}\n"
        f"Incident id: {inc.get('incident_id','?')}\n"
        f"Time: {inc.get('timestamp','?')}\n"
        f"Source: {inc.get('video_source','?')} (frame {inc.get('frame_number','?')})\n"
        f"Location: {inc.get('location_label','') or 'unknown'}\n"
        f"\nDetector reason: {inc.get('llm_reason','')}\n"
        f"Agent rationale: {state.get('severity_rationale','')}\n\n"
        "A full report is attached in the operator's incident dashboard.\n"
    )
