# ============================================================
#  STATIONARITY — Détection de stagnation d'un chien
#
#  Logique : un chien est considéré "en train de manger" si
#  son centroïde reste dans un rayon R pendant au moins T secondes.
# ============================================================

import numpy as np
from dataclasses import dataclass
from .tracker import Track


@dataclass
class StationarityEvent:
    """
    Décrit un épisode de stagnation détecté.
    """
    track_id:       int
    start_frame:    int
    end_frame:      int
    duration_sec:   float
    anchor_position: tuple     # (cx, cy) position centrale de stagnation
    confirmed:      bool = False  # True quand le seuil de durée est atteint


class StationarityDetector:
    """
    Surveille les tracks de chiens et détecte quand l'un d'eux
    reste immobile suffisamment longtemps pour déclencher un backtrack.

    Algorithme :
        1. Pour chaque chien tracké, calculer l'écart-type de ses positions
           sur les N dernières secondes.
        2. Si l'écart-type (x et y) < seuil_pixels ET durée > seuil_secondes
           → stagnation confirmée → retourner un StationarityEvent.
    """

    def __init__(self,
                 radius_px: int   = 60,
                 min_seconds: float = 8.0,
                 fps: float = 25.0):
        """
        Args:
            radius_px    : rayon maximum de déplacement autorisé (pixels)
            min_seconds  : durée minimale de stagnation pour confirmation
            fps          : frames par seconde de la vidéo
        """
        self.radius_px   = radius_px
        self.min_seconds = min_seconds
        self.fps         = fps
        self.min_frames  = int(min_seconds * fps)

        # track_id → StationarityEvent en cours (non confirmé)
        self._pending: dict[int, StationarityEvent] = {}
        # track_id → StationarityEvent confirmé (déjà signalé)
        self._confirmed: dict[int, StationarityEvent] = {}

    def _compute_anchor(self, positions: list) -> tuple:
        """Calculer le centroïde moyen d'une séquence de positions."""
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def _max_deviation(self, positions: list, anchor: tuple) -> float:
        """Distance maximale depuis l'ancre parmi toutes les positions."""
        return max(
            np.sqrt((p[0] - anchor[0])**2 + (p[1] - anchor[1])**2)
            for p in positions
        )

    def check(self, dog_track: Track,
              current_frame: int) -> StationarityEvent | None:
        """
        Vérifier si un chien est en train de stagner.

        Args:
            dog_track     : track d'un chien (doit être label == "dog")
            current_frame : index de la frame courante

        Returns:
            StationarityEvent si stagnation confirmée (première fois seulement),
            None sinon.
        """
        tid = dog_track.track_id
        history = dog_track.position_history  # [(cx, cy, frame_idx), ...]

        # Pas assez d'historique
        if len(history) < self.min_frames:
            return None

        # Déjà confirmé → ne pas re-signaler
        if tid in self._confirmed:
            return None

        # Fenêtre glissante = les min_frames derniers points
        window = history[-self.min_frames:]
        positions = [(p[0], p[1]) for p in window]

        anchor    = self._compute_anchor(positions)
        max_dev   = self._max_deviation(positions, anchor)

        if max_dev <= self.radius_px:
            # Stagnation confirmée !
            start_frame = window[0][2]
            duration    = (current_frame - start_frame) / self.fps

            event = StationarityEvent(
                track_id        = tid,
                start_frame     = start_frame,
                end_frame       = current_frame,
                duration_sec    = duration,
                anchor_position = anchor,
                confirmed       = True
            )
            self._confirmed[tid] = event
            return event

        else:
            # Pas encore stable — réinitialiser si le chien a bougé
            if tid in self._pending:
                del self._pending[tid]

        return None

    def reset_track(self, track_id: int):
        """Réinitialiser le suivi d'un track (quand il disparaît)."""
        self._pending.pop(track_id, None)
        self._confirmed.pop(track_id, None)

    def check_all(self, dog_tracks: list[Track],
                  current_frame: int) -> list[StationarityEvent]:
        """
        Vérifier tous les tracks de chiens et retourner les événements confirmés.
        """
        events = []
        for track in dog_tracks:
            event = self.check(track, current_frame)
            if event:
                events.append(event)
        return events
