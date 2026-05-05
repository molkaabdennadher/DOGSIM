# ============================================================
#  VIDEO BUFFER — Buffer circulaire de frames
#  Stocke les N dernières secondes de vidéo pour le backtrack
# ============================================================

from collections import deque
import cv2
import numpy as np
import time


class FrameRecord:
    """Un enregistrement de frame avec ses métadonnées."""

    def __init__(self, frame: np.ndarray, timestamp: float, frame_idx: int):
        self.frame      = frame.copy()   # copie de la frame brute
        self.timestamp  = timestamp      # temps réel (secondes)
        self.frame_idx  = frame_idx      # index dans la vidéo


class VideoBuffer:
    """
    Buffer circulaire qui garde en mémoire les `buffer_seconds` dernières secondes.

    Usage :
        buffer = VideoBuffer(buffer_seconds=60, fps=25)
        buffer.push(frame, frame_idx)

        # Récupérer les frames des 45 dernières secondes
        past_frames = buffer.get_last_n_seconds(45)
    """

    def __init__(self, buffer_seconds: int, fps: float):
        self.buffer_seconds = buffer_seconds
        self.fps            = fps
        self.max_frames     = int(buffer_seconds * fps)
        self._buffer        = deque(maxlen=self.max_frames)
        self._start_time    = None

    def push(self, frame: np.ndarray, frame_idx: int):
        """Ajouter une frame au buffer."""
        if self._start_time is None:
            self._start_time = time.time()

        timestamp = frame_idx / self.fps
        record = FrameRecord(frame, timestamp, frame_idx)
        self._buffer.append(record)

    def get_last_n_seconds(self, n_seconds: float) -> list[FrameRecord]:
        """
        Retourner les frames des `n_seconds` dernières secondes.
        La liste est ordonnée du plus ancien au plus récent.
        """
        if not self._buffer:
            return []

        latest_ts = self._buffer[-1].timestamp
        cutoff_ts = latest_ts - n_seconds

        return [r for r in self._buffer if r.timestamp >= cutoff_ts]

    def get_frame_at_second(self, seconds_ago: float) -> FrameRecord | None:
        """
        Retourner la frame la plus proche de `seconds_ago` secondes dans le passé.
        """
        if not self._buffer:
            return None

        latest_ts  = self._buffer[-1].timestamp
        target_ts  = latest_ts - seconds_ago

        closest = min(self._buffer, key=lambda r: abs(r.timestamp - target_ts))
        return closest

    def get_all(self) -> list[FrameRecord]:
        """Retourner toutes les frames du buffer."""
        return list(self._buffer)

    def size(self) -> int:
        return len(self._buffer)

    def duration_seconds(self) -> float:
        """Durée couverte par le buffer en secondes."""
        if len(self._buffer) < 2:
            return 0.0
        return self._buffer[-1].timestamp - self._buffer[0].timestamp

    def __len__(self):
        return len(self._buffer)
