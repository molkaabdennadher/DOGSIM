# ============================================================
#  EATING DETECTOR — Confirmation que le chien mange vraiment
#
#  Problème : un chien immobile peut juste se reposer, renifler,
#  ou attendre — sans qu'il y ait de plat ni d'infraction.
#
#  Solution : deux signaux complémentaires
#
#  Signal 1 — Détection de bol/assiette (YOLO classe 45)
#    Si un bol est détecté près du chien → confirmation forte
#
#  Signal 2 — Optical flow dans la bounding box du chien
#    Quand un chien mange, sa tête bouge (mâche, lèche)
#    pendant que son corps reste stationnaire.
#    On mesure le mouvement interne dans la zone "tête" (moitié
#    supérieure de la bbox) : mouvement > seuil = mange.
#
#  Déclenchement final :
#    Stagnation confirmée + (Signal1 OU Signal2) → Backtrack
#    Stagnation confirmée + aucun signal         → Ignorer
# ============================================================

import cv2
import numpy as np
from dataclasses import dataclass
from .tracker import Track


# Classe COCO pour les contenants (bol, assiette)
BOWL_CLASS = 45   # "bowl" dans COCO


@dataclass
class EatingConfirmation:
    """Résultat de la vérification "le chien mange-t-il ?"."""
    is_eating:          bool

    # Détail des signaux
    bowl_detected:      bool
    bowl_distance_px:   float | None   # distance bol ↔ centroïde chien
    flow_score:         float          # magnitude moyenne du flux optique
    flow_triggered:     bool           # True si flow > seuil

    reason: str                        # explication lisible


class EatingDetector:
    """
    Vérifie qu'un chien stagnant est bien en train de manger.
    À appeler juste après qu'une stagnation a été confirmée.
    """

    def __init__(self,
                 bowl_radius_px:     int   = 120,
                 flow_threshold:     float = 1.2,
                 flow_window_frames: int   = 10):
        """
        Args:
            bowl_radius_px     : distance max entre le bol et le chien (pixels)
            flow_threshold     : magnitude min du flux optique pour confirmer
                                 le mouvement de tête (pixels/frame)
            flow_window_frames : nombre de frames récentes à analyser pour le flux
        """
        self.bowl_radius_px     = bowl_radius_px
        self.flow_threshold     = flow_threshold
        self.flow_window_frames = flow_window_frames

        # Stocke les dernières frames grises dans la zone tête de chaque track
        # track_id → deque de (frame_gray, bbox)
        self._head_history: dict[int, list] = {}

    # ── Signal 1 : détection de bol ───────────────────────────────────

    def _detect_bowls(self, detector, frame: np.ndarray) -> list[tuple]:
        """
        Détecter les bols/assiettes dans la frame.
        Returns: liste de centroïdes [(cx, cy), ...]
        """
        from ultralytics import YOLO
        results = detector.model(
            frame,
            conf=0.35,
            classes=[BOWL_CLASS],
            verbose=False
        )
        bowls = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                bowls.append((cx, cy))
        return bowls

    def _check_bowl_signal(self, detector, frame: np.ndarray,
                            dog_centroid: tuple) -> tuple[bool, float | None]:
        """
        Vérifier si un bol est présent près du chien.
        Returns: (bowl_found, distance_px)
        """
        bowls = self._detect_bowls(detector, frame)
        if not bowls:
            return False, None

        # Distance au bol le plus proche
        best_dist = min(
            np.sqrt((b[0]-dog_centroid[0])**2 + (b[1]-dog_centroid[1])**2)
            for b in bowls
        )
        return best_dist <= self.bowl_radius_px, float(best_dist)

    # ── Signal 2 : optical flow dans la zone tête ─────────────────────

    def _extract_head_region(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extraire la zone "tête" du chien = moitié supérieure de la bbox.
        C'est là que se produit le mouvement de mâchonnement/léchage.
        """
        x1, y1, x2, y2 = bbox
        head_y2 = y1 + (y2 - y1) // 2   # moitié haute
        region  = frame[y1:head_y2, x1:x2]
        if region.size == 0:
            return None
        return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    def update_frame(self, frame: np.ndarray, dog_track: Track):
        """
        Enregistrer la frame courante pour le track donné.
        À appeler à chaque frame pendant le tracking (pas seulement à la stagnation).
        """
        tid  = dog_track.track_id
        head = self._extract_head_region(frame, dog_track.bbox)
        if head is None:
            return

        if tid not in self._head_history:
            self._head_history[tid] = []
        self._head_history[tid].append(head)

        # Garder seulement les N dernières frames
        if len(self._head_history[tid]) > self.flow_window_frames + 1:
            self._head_history[tid].pop(0)

    def _check_flow_signal(self, track_id: int) -> tuple[bool, float]:
        """
        Calculer le flux optique sur la zone tête pour les dernières frames.

        Principe : si le chien mange, la tête bouge (chewing, licking).
        Le flux optique mesure le déplacement de pixels entre deux frames.
        Magnitude moyenne > seuil → mouvement → mange.

        Returns: (flow_triggered, flow_score)
        """
        history = self._head_history.get(track_id, [])
        if len(history) < 3:
            return False, 0.0

        flow_magnitudes = []

        # Calculer le flux entre chaque paire de frames consécutives
        for i in range(len(history) - 1):
            prev = history[i]
            curr = history[i + 1]

            # Redimensionner si les tailles ne correspondent pas (bbox qui change)
            if prev.shape != curr.shape:
                try:
                    curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))
                except:
                    continue

            # Flux optique dense de Farneback
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=10,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0
            )

            # Magnitude du flux (déplacement moyen en pixels/frame)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(float(np.mean(mag)))

        if not flow_magnitudes:
            return False, 0.0

        avg_flow = float(np.mean(flow_magnitudes))
        return avg_flow >= self.flow_threshold, avg_flow

    # ── Confirmation finale ────────────────────────────────────────────

    def confirm(self, detector, frame: np.ndarray,
                dog_track: Track) -> EatingConfirmation:
        """
        Vérifier si le chien est en train de manger.
        Combine les deux signaux.

        Args:
            detector  : YOLODetector (pour la détection de bol)
            frame     : frame courante
            dog_track : track du chien à vérifier

        Returns:
            EatingConfirmation avec is_eating = True/False
        """
        # Signal 1 — bol/assiette
        bowl_found, bowl_dist = self._check_bowl_signal(
            detector, frame, dog_track.centroid
        )

        # Signal 2 — optical flow
        flow_triggered, flow_score = self._check_flow_signal(dog_track.track_id)

        is_eating = bowl_found or flow_triggered

        # Construire le message explicatif
        parts = []
        if bowl_found:
            parts.append(f"bol détecté à {bowl_dist:.0f}px")
        if flow_triggered:
            parts.append(f"mouvement tête détecté (flow={flow_score:.2f})")
        if not is_eating:
            parts.append("aucun signal d'alimentation (chien au repos ?)")

        reason = " | ".join(parts) if parts else "inconnu"

        return EatingConfirmation(
            is_eating        = is_eating,
            bowl_detected    = bowl_found,
            bowl_distance_px = bowl_dist,
            flow_score       = flow_score,
            flow_triggered   = flow_triggered,
            reason           = reason
        )

    def reset_track(self, track_id: int):
        """Nettoyer l'historique d'un track supprimé."""
        self._head_history.pop(track_id, None)
