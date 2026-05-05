# ============================================================
#  INFRACTION REPORTER
#  Génère 3 frames annotées + JSON pour chaque infraction
#
#  frame_1_approach  → humain s'approche (visage visible, debout)
#  frame_2_deposit   → humain pose le plat (action prouvée)
#  frame_3_departure → humain repart (avant l'arrivée du chien)
# ============================================================

import cv2
import json
import os
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from ..core.backtracker import BacktrackResult
from ..core.detector import Detection


@dataclass
class InfractionReport:
    infraction_id:       str
    timestamp:           str
    camera_id:           str
    in_feeding_zone:     bool
    dog_track_id:        int
    dog_position:        tuple
    stagnation_seconds:  float
    human_found:         bool
    deposit_position:    tuple | None
    distance_at_deposit: float | None
    seconds_before_dog:  float | None
    frames_searched:     int
    image_approach_path: str | None = None
    image_deposit_path:  str | None = None
    image_departure_path: str | None = None
    notes:               str = ""


class InfractionReporter:

    def __init__(self, output_dir: str = "cv_module/output",
                 camera_id: str = "CAM_001"):
        self.output_dir = output_dir
        self.camera_id  = camera_id
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

    # ── Annotation générique ───────────────────────────────────────────
    def _draw_person(self, frame: np.ndarray, det: Detection,
                     color: tuple, label: str) -> np.ndarray:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def _draw_anchor(self, frame: np.ndarray, anchor: tuple) -> np.ndarray:
        cv2.circle(frame, anchor, 50, (0, 165, 255), 2)
        cv2.drawMarker(frame, anchor, (0, 165, 255),
                       cv2.MARKER_CROSS, 30, 2)
        cv2.putText(frame, "Emplacement plat",
                    (anchor[0] - 70, anchor[1] + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        return frame

    def _header_banner(self, frame: np.ndarray,
                       text: str, color: tuple) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, text, (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    # ── Frame 1 : APPROCHE ─────────────────────────────────────────────
    def _annotate_approach(self, frame: np.ndarray,
                           result: BacktrackResult) -> np.ndarray:
        f = frame.copy()
        anchor = result.stationarity_event.anchor_position
        f = self._draw_anchor(f, anchor)

        if result.detection_approach:
            f = self._draw_person(f, result.detection_approach,
                                  (0, 200, 255),
                                  f"SUSPECT - approche")
            # Flèche vers l'ancre
            cv2.arrowedLine(f,
                            result.detection_approach.centroid,
                            anchor,
                            (0, 200, 255), 2, tipLength=0.2)

        f = self._header_banner(
            f,
            f"[1/3] APPROCHE  |  Cam: {self.camera_id}",
            (80, 120, 0)
        )
        return f

    # ── Frame 2 : DÉPÔT ────────────────────────────────────────────────
    def _annotate_deposit(self, frame: np.ndarray,
                          result: BacktrackResult) -> np.ndarray:
        f = frame.copy()
        anchor = result.stationarity_event.anchor_position
        f = self._draw_anchor(f, anchor)

        if result.detection_deposit:
            det = result.detection_deposit
            f   = self._draw_person(f, det, (0, 0, 220),
                                    f"SUSPECT - DEPOT DU PLAT")
            # Ligne suspect → emplacement
            cv2.line(f, det.centroid, anchor, (0, 0, 220), 2, cv2.LINE_AA)
            cv2.putText(f,
                        f"{result.distance_at_deposit:.0f}px",
                        ((det.centroid[0]+anchor[0])//2,
                         (det.centroid[1]+anchor[1])//2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

            # Info temporelle
            if result.seconds_before_dog is not None:
                cv2.putText(f,
                            f"{result.seconds_before_dog:.0f}s avant arrivee chien",
                            (det.bbox[0], det.bbox[3] + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        f = self._header_banner(
            f,
            f"[2/3] DEPOT DU PLAT  |  INFRACTION CONSTATEE",
            (0, 0, 180)
        )
        return f

    # ── Frame 3 : DÉPART ───────────────────────────────────────────────
    def _annotate_departure(self, frame: np.ndarray,
                            result: BacktrackResult) -> np.ndarray:
        f = frame.copy()
        anchor = result.stationarity_event.anchor_position
        f = self._draw_anchor(f, anchor)

        if result.detection_departure:
            f = self._draw_person(f, result.detection_departure,
                                  (180, 0, 200),
                                  "SUSPECT - quitte les lieux")
            # Flèche qui s'éloigne
            cx, cy = result.detection_departure.centroid
            dx = cx - anchor[0]
            dy = cy - anchor[1]
            norm = max(1, np.sqrt(dx**2 + dy**2))
            end = (int(cx + 60 * dx / norm), int(cy + 60 * dy / norm))
            cv2.arrowedLine(f, (cx, cy), end,
                            (180, 0, 200), 2, tipLength=0.25)

        f = self._header_banner(
            f,
            f"[3/3] DEPART  |  Chien arrive {result.seconds_before_dog:.0f}s apres"
            if result.seconds_before_dog else "[3/3] DEPART",
            (100, 0, 150)
        )
        return f

    # ── Sauvegarde ─────────────────────────────────────────────────────
    def _save_frame(self, frame: np.ndarray, inf_id: str, suffix: str) -> str:
        path = os.path.join(self.output_dir, "frames", f"{inf_id}_{suffix}.jpg")
        cv2.imwrite(path, frame)
        return path

    # ── Génération complète ────────────────────────────────────────────
    def generate(self, result: BacktrackResult,
                 infraction_idx: int = 0) -> InfractionReport:

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        inf_id = f"INF_{ts_str}_{infraction_idx:04d}"

        path_approach = path_deposit = path_departure = None

        if result.found:
            # Frame 1 — approche
            if result.frame_approach:
                annotated = self._annotate_approach(result.frame_approach.frame, result)
                path_approach = self._save_frame(annotated, inf_id, "1_approche")
                print(f"[Reporter] Approche  → {path_approach}")

            # Frame 2 — dépôt (la plus importante)
            if result.frame_deposit:
                annotated = self._annotate_deposit(result.frame_deposit.frame, result)
                path_deposit = self._save_frame(annotated, inf_id, "2_depot")
                print(f"[Reporter] Depot     → {path_deposit}")

            # Frame 3 — départ
            if result.frame_departure:
                annotated = self._annotate_departure(result.frame_departure.frame, result)
                path_departure = self._save_frame(annotated, inf_id, "3_depart")
                print(f"[Reporter] Depart    → {path_departure}")

        # ── Rapport JSON ────────────────────────────────────────────────
        report = InfractionReport(
            infraction_id        = inf_id,
            timestamp            = datetime.now().isoformat(),
            camera_id            = self.camera_id,
            in_feeding_zone      = False,
            dog_track_id         = result.stationarity_event.track_id,
            dog_position         = result.stationarity_event.anchor_position,
            stagnation_seconds   = result.stationarity_event.duration_sec,
            human_found          = result.found,
            deposit_position     = (result.detection_deposit.centroid
                                    if result.detection_deposit else None),
            distance_at_deposit  = result.distance_at_deposit,
            seconds_before_dog   = result.seconds_before_dog,
            frames_searched      = result.all_searched_frames,
            image_approach_path  = path_approach,
            image_deposit_path   = path_deposit,
            image_departure_path = path_departure,
            notes                = (
                f"Visite humaine détectée : frames "
                f"{result.visit.start_frame_idx}→{result.visit.end_frame_idx}"
                if result.found and result.visit
                else "Aucun humain trouvé"
            )
        )

        json_path = os.path.join(self.output_dir, "reports", f"{inf_id}.json")
        report_dict = asdict(report)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        print(f"[Reporter] Rapport   → {json_path}")

        return report
