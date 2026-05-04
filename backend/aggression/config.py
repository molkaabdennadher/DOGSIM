"""Centralized config loader for the aggression module."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("aggression.config")

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts" / "agressionDetection"


def load_aggression_config() -> dict:
    """Load aggression_config.json from artifacts/agressionDetection/."""
    cfg_path = ARTIFACTS_DIR / "aggression_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.warning("aggression_config.json present but unparseable: %s", exc)
    return {}
