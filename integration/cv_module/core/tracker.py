# ============================================================
#  TRACKER — Suivi multi-objets par IoU + centroïdes
#  Léger, sans dépendance externe (pas besoin de DeepSORT)
# ============================================================

import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from .detector import Detection


@dataclass
class Track:
    """
    Un objet suivi dans la vidéo.
    Maintient l'historique des positions pour la stationnarité.
    """
    track_id:      int
    label:         str                     # "dog" ou "person"
    centroid:      tuple                   # position courante (cx, cy)
    bbox:          tuple                   # bbox courante
    confidence:    float
    age:           int = 0                 # nb de frames depuis création
    lost_frames:   int = 0                 # nb de frames sans détection
    is_active:     bool = True
    position_history: list = field(default_factory=list)  # [(cx, cy, frame_idx), ...]

    def update(self, detection: Detection, frame_idx: int):
        self.centroid   = detection.centroid
        self.bbox       = detection.bbox
        self.confidence = detection.confidence
        self.lost_frames = 0
        self.age        += 1
        self.position_history.append((detection.centroid[0],
                                      detection.centroid[1],
                                      frame_idx))

    def mark_lost(self):
        self.lost_frames += 1

    def last_known_position(self) -> tuple | None:
        if self.position_history:
            return (self.position_history[-1][0], self.position_history[-1][1])
        return None


class CentroidTracker:
    """
    Tracker léger basé sur la distance euclidienne entre centroïdes.
    Associe les nouvelles détections aux tracks existants.

    Avantage : aucune dépendance externe, rapide, suffisant pour un prototype.
    """

    def __init__(self, max_lost_frames: int = 15,
                 max_distance_px: int = 80):
        self.max_lost_frames = max_lost_frames
        self.max_distance_px = max_distance_px
        self._next_id        = 0
        self.tracks: dict[int, Track] = {}

    def _euclidean(self, p1: tuple, p2: tuple) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def update(self, detections: list[Detection],
               frame_idx: int) -> list[Track]:
        """
        Mettre à jour les tracks avec les nouvelles détections.

        Returns:
            Liste des tracks actifs après mise à jour.
        """
        # --- Séparer les tracks actifs par label ---
        active_tracks = {tid: t for tid, t in self.tracks.items()
                         if t.is_active}

        # --- Associer détections → tracks (greedy nearest-neighbor) ---
        matched_track_ids   = set()
        matched_detect_idxs = set()

        # Construire matrice de distances
        track_ids  = list(active_tracks.keys())
        track_list = [active_tracks[tid] for tid in track_ids]

        if track_list and detections:
            for det_idx, det in enumerate(detections):
                best_dist  = float('inf')
                best_tid   = None

                for t in track_list:
                    if t.label != det.label:
                        continue
                    if t.track_id in matched_track_ids:
                        continue

                    dist = self._euclidean(t.centroid, det.centroid)
                    if dist < best_dist and dist < self.max_distance_px:
                        best_dist = dist
                        best_tid  = t.track_id

                if best_tid is not None:
                    active_tracks[best_tid].update(det, frame_idx)
                    matched_track_ids.add(best_tid)
                    matched_detect_idxs.add(det_idx)

        # --- Créer de nouveaux tracks pour les détections non associées ---
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detect_idxs:
                new_track = Track(
                    track_id         = self._next_id,
                    label            = det.label,
                    centroid         = det.centroid,
                    bbox             = det.bbox,
                    confidence       = det.confidence,
                    position_history = [(det.centroid[0], det.centroid[1], frame_idx)]
                )
                self.tracks[self._next_id] = new_track
                self._next_id += 1

        # --- Marquer les tracks non associés comme perdus ---
        for tid in track_ids:
            if tid not in matched_track_ids:
                active_tracks[tid].mark_lost()
                if active_tracks[tid].lost_frames > self.max_lost_frames:
                    active_tracks[tid].is_active = False

        return [t for t in self.tracks.values() if t.is_active]

    def get_dogs(self) -> list[Track]:
        return [t for t in self.tracks.values()
                if t.is_active and t.label == "dog"]

    def get_persons(self) -> list[Track]:
        return [t for t in self.tracks.values()
                if t.is_active and t.label == "person"]
