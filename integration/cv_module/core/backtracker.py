# ============================================================
#  BACKTRACKER — Remontée dans le buffer pour trouver l'humain
#
#  Scénario réel :
#    t=0s  → Humain entre dans la zone, pose le plat, repart
#    t=20s → Chien arrive et commence à manger
#    t=28s → Stagnation détectée → backtrack déclenché
#
#  Stratégie :
#    1. Remonter dans le buffer
#    2. Identifier la dernière "visite" d'un humain dans la zone de l'ancre
#       Une visite = séquence de frames où l'humain est dans la zone
#    3. À l'intérieur de cette visite, trouver le frame de DÉPÔT :
#       → moment où l'humain est à distance minimale de l'ancre
#         (il est au plus proche du sol = il pose quelque chose)
#    4. Sauvegarder aussi le frame d'APPROCHE (visage visible, debout)
#       et le frame de DÉPART (après avoir posé)
# ============================================================

import numpy as np
from dataclasses import dataclass, field
from .video_buffer import VideoBuffer, FrameRecord
from .detector import YOLODetector, Detection
from .stationarity import StationarityEvent


@dataclass
class Visit:
    """
    Une visite d'un humain dans la zone de l'ancre.
    Séquence continue de frames où l'humain était présent dans le rayon.
    """
    frames:     list          # list[(FrameRecord, Detection, float distance)]
    min_dist:   float         # distance minimale atteinte pendant la visite
    start_frame_idx: int
    end_frame_idx:   int

    @property
    def deposit_frame(self):
        """Frame où l'humain était le plus près de l'ancre = moment du dépôt."""
        return min(self.frames, key=lambda x: x[2])  # x[2] = distance

    @property
    def approach_frame(self):
        """Première frame de la visite = humain qui s'approche (visage visible)."""
        return self.frames[0]

    @property
    def departure_frame(self):
        """Dernière frame de la visite = humain qui repart."""
        return self.frames[-1]

    @property
    def duration_frames(self):
        return self.end_frame_idx - self.start_frame_idx


@dataclass
class BacktrackResult:
    """Résultat complet du backtrack."""
    found:               bool
    stationarity_event:  StationarityEvent

    # La visite identifiée (séquence complète)
    visit:               Visit | None

    # Les 3 frames clés extraites de la visite
    frame_approach:      FrameRecord | None   # Humain s'approche → visage visible
    detection_approach:  Detection  | None

    frame_deposit:       FrameRecord | None   # Humain pose le plat → action prouvée
    detection_deposit:   Detection  | None
    distance_at_deposit: float      | None    # Distance à l'ancre au moment du dépôt

    frame_departure:     FrameRecord | None   # Humain repart → confirme le départ avant le chien
    detection_departure: Detection  | None

    # Contexte temporel
    seconds_before_dog:  float | None    # Combien de temps avant l'arrivée du chien
    all_searched_frames: int = 0


class Backtracker:
    """
    Remonte dans le buffer pour identifier la visite du suspect
    et extraire les 3 frames clés de l'infraction.
    """

    def __init__(self,
                 detector: YOLODetector,
                 buffer: VideoBuffer,
                 backtrack_window_seconds: float = 45.0,
                 anchor_radius_px: int = 150,
                 sample_every_n_frames: int = 3,
                 visit_gap_frames: int = 15):
        """
        Args:
            detector                 : instance YOLO partagée
            buffer                   : buffer circulaire
            backtrack_window_seconds : jusqu'où remonter
            anchor_radius_px         : rayon autour de l'ancre = zone du plat
            sample_every_n_frames    : sous-échantillonnage pour la perf
            visit_gap_frames         : écart max entre 2 frames pour qu'elles
                                       appartiennent à la même visite
                                       (si l'humain disparaît N frames puis revient
                                       → nouvelle visite)
        """
        self.detector                 = detector
        self.buffer                   = buffer
        self.backtrack_window_seconds = backtrack_window_seconds
        self.anchor_radius_px         = anchor_radius_px
        self.sample_every_n_frames    = sample_every_n_frames
        self.visit_gap_frames         = visit_gap_frames

    def _dist(self, p1: tuple, p2: tuple) -> float:
        return float(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

    def _scan_frames(self, frames: list, anchor: tuple) -> list:
        """
        Scanner toutes les frames et retourner celles où un humain
        est présent dans le rayon de l'ancre.

        Returns:
            list of (FrameRecord, Detection, distance)
            Trié du plus ancien au plus récent.
        """
        hits = []
        for i, record in enumerate(frames):
            if i % self.sample_every_n_frames != 0:
                continue

            persons = self.detector.detect_persons(record.frame)
            if not persons:
                continue

            # Prendre l'humain le plus proche de l'ancre dans cette frame
            closest, closest_dist = None, float('inf')
            for p in persons:
                d = self._dist(p.centroid, anchor)
                if d < closest_dist:
                    closest_dist, closest = d, p

            if closest_dist <= self.anchor_radius_px:
                hits.append((record, closest, closest_dist))

        return hits  # du plus ancien au plus récent

    def _group_into_visits(self, hits: list) -> list[Visit]:
        """
        Regrouper les frames détectées en visites continues.
        Si deux frames consécutives sont séparées de plus de
        `visit_gap_frames` frames vidéo, c'est une nouvelle visite.
        """
        if not hits:
            return []

        visits   = []
        current  = [hits[0]]

        for prev, curr in zip(hits, hits[1:]):
            gap = curr[0].frame_idx - prev[0].frame_idx
            if gap <= self.visit_gap_frames * self.sample_every_n_frames:
                current.append(curr)
            else:
                visits.append(Visit(
                    frames          = current,
                    min_dist        = min(h[2] for h in current),
                    start_frame_idx = current[0][0].frame_idx,
                    end_frame_idx   = current[-1][0].frame_idx
                ))
                current = [curr]

        # Dernière visite
        visits.append(Visit(
            frames          = current,
            min_dist        = min(h[2] for h in current),
            start_frame_idx = current[0][0].frame_idx,
            end_frame_idx   = current[-1][0].frame_idx
        ))

        return visits

    def search(self, event: StationarityEvent, fps: float = 25.0) -> BacktrackResult:
        """
        Lancer le backtrack pour un événement de stagnation.

        Étapes :
            1. Récupérer les frames de la fenêtre de backtrack
            2. Scanner : trouver toutes les frames où un humain est dans la zone
            3. Regrouper en visites
            4. Prendre la visite la plus récente (avant la stagnation du chien)
            5. Extraire les 3 frames clés de cette visite
        """
        anchor = event.anchor_position
        stagnation_start_ts = event.start_frame / fps

        past_frames = self.buffer.get_last_n_seconds(self.backtrack_window_seconds)

        no_result = BacktrackResult(
            found=False, stationarity_event=event,
            visit=None,
            frame_approach=None, detection_approach=None,
            frame_deposit=None,  detection_deposit=None,
            distance_at_deposit=None,
            frame_departure=None, detection_departure=None,
            seconds_before_dog=None, all_searched_frames=0
        )

        if not past_frames:
            return no_result

        # ── 1. Scanner toutes les frames ──────────────────────────────
        hits = self._scan_frames(past_frames, anchor)
        searched_count = len(past_frames) // self.sample_every_n_frames

        if not hits:
            print(f"[Backtracker] Aucun humain détecté dans la zone "
                  f"(rayon={self.anchor_radius_px}px, "
                  f"{searched_count} frames analysées)")
            no_result.all_searched_frames = searched_count
            return no_result

        # ── 2. Regrouper en visites ───────────────────────────────────
        visits = self._group_into_visits(hits)
        print(f"[Backtracker] {len(visits)} visite(s) détectée(s) dans la zone")

        # ── 3. Prendre la visite la plus récente avant la stagnation ──
        # (éliminer les visites qui chevauchent la stagnation = c'est le chien)
        valid_visits = [v for v in visits
                        if v.end_frame_idx / fps < stagnation_start_ts + 2.0]

        if not valid_visits:
            print("[Backtracker] Aucune visite humaine avant l'arrivée du chien")
            no_result.all_searched_frames = searched_count
            return no_result

        # La visite la plus récente = le dernier suspect
        target_visit = valid_visits[-1]

        # ── 4. Extraire les 3 frames clés ─────────────────────────────
        approach_rec,  approach_det,  _                = target_visit.approach_frame
        deposit_rec,   deposit_det,   deposit_dist     = target_visit.deposit_frame
        departure_rec, departure_det, _                = target_visit.departure_frame

        # ── 5. Delta temporel ─────────────────────────────────────────
        deposit_ts         = deposit_rec.frame_idx / fps
        seconds_before_dog = max(0.0, stagnation_start_ts - deposit_ts)

        print(f"[Backtracker] ✓ Visite suspecte : "
              f"frames {target_visit.start_frame_idx}→{target_visit.end_frame_idx} "
              f"({target_visit.duration_frames} frames)")
        print(f"[Backtracker]   Frame approche  : {approach_rec.frame_idx}")
        print(f"[Backtracker]   Frame dépôt     : {deposit_rec.frame_idx} "
              f"(distance ancre={deposit_dist:.0f}px)")
        print(f"[Backtracker]   Frame départ    : {departure_rec.frame_idx}")
        print(f"[Backtracker]   Dépôt {seconds_before_dog:.1f}s avant arrivée du chien")

        return BacktrackResult(
            found               = True,
            stationarity_event  = event,
            visit               = target_visit,
            frame_approach      = approach_rec,
            detection_approach  = approach_det,
            frame_deposit       = deposit_rec,
            detection_deposit   = deposit_det,
            distance_at_deposit = deposit_dist,
            frame_departure     = departure_rec,
            detection_departure = departure_det,
            seconds_before_dog  = seconds_before_dog,
            all_searched_frames = searched_count
        )
