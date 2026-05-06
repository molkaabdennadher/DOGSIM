# ============================================================
#  PIPELINE PRINCIPAL — Orchestration du module CV
#
#  Flux complet :
#    Frame → Buffer → Detector → Tracker
#         → EatingDetector.update_frame() (accumulation)
#         → StationarityDetector
#         ↓ (si stagnation détectée)
#         → EatingDetector.confirm()  ← NOUVEAU FILTRE
#              ├─ Signal 1 : bol/assiette détecté près du chien ?
#              └─ Signal 2 : mouvement de tête (optical flow) ?
#         ↓ (si au moins un signal positif)
#         → Backtracker → Reporter → Infraction
#         ↓ (si aucun signal)
#         → Ignoré (chien au repos, pas d'infraction)
# ============================================================

import cv2
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cv_module.config import (
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IOU,
    BUFFER_SECONDS,
    STATIONARITY_THRESHOLD_PX, STATIONARITY_MIN_SECONDS,
    BACKTRACK_WINDOW_SECONDS,
    MAX_LOST_FRAMES,
    OUTPUT_DIR,
)
from cv_module.core.video_buffer         import VideoBuffer
from cv_module.core.detector             import YOLODetector
from cv_module.core.tracker              import CentroidTracker
from cv_module.core.stationarity         import StationarityDetector
from cv_module.core.eating_detector      import EatingDetector
from cv_module.core.backtracker          import Backtracker
from cv_module.utils.infraction_reporter import InfractionReporter


class FeedingInfractionPipeline:

    def __init__(self,
                 source,
                 camera_id:  str  = "CAM_001",
                 display:    bool = True,
                 max_frames: int | None = None):
        self.source     = source
        self.camera_id  = camera_id
        self.display    = display
        self.max_frames = max_frames

        self.infraction_count  = 0
        self.false_pos_skipped = 0   # compteur de chiens ignorés (pas en train de manger)
        self.reports           = []

    def _init_components(self, fps: float):
        print(f"[Pipeline] Initialisation (fps={fps:.1f})")

        self.detector     = YOLODetector(YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IOU)
        self.buffer       = VideoBuffer(BUFFER_SECONDS, fps)
        self.tracker      = CentroidTracker(MAX_LOST_FRAMES)
        self.stationarity = StationarityDetector(
            radius_px   = STATIONARITY_THRESHOLD_PX,
            min_seconds = STATIONARITY_MIN_SECONDS,
            fps         = fps
        )
        self.eating = EatingDetector(
            bowl_radius_px     = 120,
            flow_threshold     = 1.2,
            flow_window_frames = 12
        )
        self.backtracker = Backtracker(
            detector                 = self.detector,
            buffer                   = self.buffer,
            backtrack_window_seconds = BACKTRACK_WINDOW_SECONDS,
            sample_every_n_frames    = max(1, int(fps // 5))
        )
        self.reporter = InfractionReporter(
            output_dir = OUTPUT_DIR,
            camera_id  = self.camera_id
        )

    def _draw_live_overlay(self, frame, tracks, stationarity_events,
                           eating_confirmations: dict):
        h, w = frame.shape[:2]

        for t in tracks:
            if t.label == "dog":
                eating = eating_confirmations.get(t.track_id)
                if eating and eating.is_eating:
                    color = (0, 0, 255)    # rouge = mange
                    suffix = " [MANGE]"
                else:
                    color = (0, 200, 100)  # vert = stationnaire / inactif
                    suffix = ""
            else:
                color  = (200, 200, 0)
                suffix = ""

            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{t.label} #{t.track_id}{suffix}",
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for ev in stationarity_events:
            cv2.circle(frame, ev.anchor_position, 70, (0, 69, 255), 3)
            cv2.putText(frame,
                        f"STAGNATION {ev.duration_sec:.1f}s",
                        (ev.anchor_position[0] - 80, ev.anchor_position[1] - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 69, 255), 2)

        cv2.putText(frame,
                    f"Infractions: {self.infraction_count}  |  "
                    f"Ignores (repos): {self.false_pos_skipped}",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return frame

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir : {self.source}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Pipeline] Source={self.source} | FPS={fps:.1f} | Frames={total_frames}")

        self._init_components(fps)

        frame_idx           = 0
        t_start             = time.time()
        eating_confirmations = {}   # track_id → EatingConfirmation (pour overlay)

        print("[Pipeline] Démarrage... (appuyer sur 'q' pour arrêter)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.max_frames and frame_idx >= self.max_frames:
                break

            # ── 1. Buffer ─────────────────────────────────────────────
            self.buffer.push(frame, frame_idx)

            # ── 2. Détection ──────────────────────────────────────────
            detections    = self.detector.detect(frame, detect_dogs=True,
                                                  detect_persons=True)

            # ── 3. Tracking ───────────────────────────────────────────
            active_tracks = self.tracker.update(detections, frame_idx)
            dog_tracks    = [t for t in active_tracks if t.label == "dog"]

            # ── 4. Accumulation optical flow (à chaque frame) ─────────
            for dog in dog_tracks:
                self.eating.update_frame(frame, dog)

            # ── 5. Stationnarité ──────────────────────────────────────
            stationarity_events = self.stationarity.check_all(dog_tracks, frame_idx)

            # ── 6. Filtre "le chien mange-t-il vraiment ?" ────────────
            for event in stationarity_events:
                dog_track = next(
                    (t for t in dog_tracks if t.track_id == event.track_id), None
                )
                if dog_track is None:
                    continue

                confirmation = self.eating.confirm(
                    self.detector, frame, dog_track
                )
                eating_confirmations[event.track_id] = confirmation

                print(f"\n[Pipeline] Stagnation #{event.track_id} — "
                      f"Mange ? {'OUI' if confirmation.is_eating else 'NON'} "
                      f"({confirmation.reason})")

                if not confirmation.is_eating:
                    # ── Chien au repos → on ignore ─────────────────────
                    print(f"[Pipeline] ↳ Ignoré (pas de signal d'alimentation)")
                    self.false_pos_skipped += 1
                    continue

                # ── Chien qui mange → Backtrack ────────────────────────
                print(f"[Pipeline] ↳ Alimentation confirmée → backtrack...")
                bt_result = self.backtracker.search(event, fps)

                if bt_result.found:
                    print(f"[Pipeline] ✓ Suspect trouvé | "
                          f"dépôt {bt_result.seconds_before_dog:.0f}s avant le chien")
                else:
                    print(f"[Pipeline] ✗ Aucun suspect trouvé "
                          f"({bt_result.all_searched_frames} frames analysées)")

                report = self.reporter.generate(bt_result, self.infraction_count)
                self.reports.append(report)
                self.infraction_count += 1

            # ── 7. Affichage live ─────────────────────────────────────
            if self.display:
                display_frame = self._draw_live_overlay(
                    frame.copy(), active_tracks,
                    stationarity_events, eating_confirmations
                )
                cv2.imshow("Feeding Infraction Detector", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Pipeline] Arrêt.")
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                print(f"[Pipeline] Frame {frame_idx}/{total_frames} | "
                      f"{elapsed:.1f}s | "
                      f"Infractions={self.infraction_count} "
                      f"Ignores={self.false_pos_skipped}")

        cap.release()
        if self.display:
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        print(f"\n[Pipeline] ══════════════════════════════════")
        print(f"[Pipeline] Terminé en {elapsed:.1f}s")
        print(f"[Pipeline] Frames traitées    : {frame_idx}")
        print(f"[Pipeline] Infractions        : {self.infraction_count}")
        print(f"[Pipeline] Ignorés (au repos) : {self.false_pos_skipped}")
        print(f"[Pipeline] Rapports dans      : {OUTPUT_DIR}/")
        print(f"[Pipeline] ══════════════════════════════════\n")

        return self.reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feeding Infraction Detector")
    parser.add_argument("--source",     default=0)
    parser.add_argument("--camera-id",  default="CAM_001")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    FeedingInfractionPipeline(
        source     = source,
        camera_id  = args.camera_id,
        display    = not args.no_display,
        max_frames = args.max_frames
    ).run()
