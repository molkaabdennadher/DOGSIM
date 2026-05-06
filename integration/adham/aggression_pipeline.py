# =============================================================
# aggression_pipeline.py — Détection d'agression animale
# Adham Module | DOGSIM-TN | ESPRIT 3A IA
#
# Adapté du notebook animal_aggression_detection.ipynb
# Pipeline : YOLOv8n detect + ByteTrack + YOLOv8n-pose + RiskScorer
#
# Fonction principale :
#   run_aggression_pipeline(video_source, camera_id, max_frames)
#   → liste d'événements détectés [{event_id, timestamp, ...}]
#
# Pas de dépendance Gemini ni email (désactivés en mode serveur)
# =============================================================

import os
import sys
import time
import base64
import sqlite3
import warnings
import traceback
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ── Chemins de sortie ─────────────────────────────────────────
_HERE        = Path(__file__).parent
CAPTURES_DIR = _HERE / "output" / "captures"
DB_PATH      = str(_HERE / "output" / "aggression_events.db")
os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR.parent, exist_ok=True)

# ── Modèles YOLO ──────────────────────────────────────────────
YOLO_DETECT_MODEL = "yolov8n.pt"      # nano détection (léger)
YOLO_POSE_MODEL   = "yolov8n-pose.pt" # nano pose

# ── Classes COCO ──────────────────────────────────────────────
PERSON_CLASS_ID = 0   # 'person'
DOG_CLASS_ID    = 16  # 'dog'

# ── Seuils ────────────────────────────────────────────────────
DETECT_CONF             = 0.40
POSE_CONF               = 0.40
PROXIMITY_THRESHOLD_PX  = 100
RISK_THRESHOLD          = 0.45
ALERT_FRAME_WINDOW      = 2

# ── Keypoints COCO 17 points ──────────────────────────────────
KP_SHOULDER_L, KP_SHOULDER_R = 5, 6
KP_ELBOW_L,    KP_ELBOW_R    = 7, 8
KP_WRIST_L,    KP_WRIST_R    = 9, 10
KP_HIP_L,      KP_HIP_R      = 11, 12
KP_KNEE_L,     KP_KNEE_R     = 13, 14
KP_ANKLE_L,    KP_ANKLE_R    = 15, 16

# ── Couleurs BGR (annotation) ─────────────────────────────────
COLOR_PERSON_OK    = (100, 200, 100)
COLOR_PERSON_WARN  = (0,   165, 255)
COLOR_PERSON_ALERT = (0,   0,   255)
COLOR_DOG          = (255, 180,  50)
COLOR_TEXT_BG      = (20,  20,  20)
FONT               = cv2.FONT_HERSHEY_SIMPLEX

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ─────────────────────────────────────────────────────────────
# Base de données SQLite
# ─────────────────────────────────────────────────────────────

def _init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS aggression_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            video_source    TEXT,
            frame_number    INTEGER,
            person_track_id INTEGER,
            dog_track_id    INTEGER,
            risk_score      REAL,
            gesture_score   REAL,
            proximity_score REAL,
            aggression_type TEXT,
            capture_path    TEXT
        )
    """)
    conn.commit()
    conn.close()


def _save_event(event: dict) -> int:
    """Persiste un événement en base. Tous les types numpy sont convertis."""
    import numpy as np

    def _to_py(v):
        """Convertit tout type numpy en type Python natif avant INSERT."""
        if isinstance(v, (np.integer,)):   return int(v)
        if isinstance(v, (np.floating,)):  return float(v)
        if isinstance(v, (np.bool_,)):     return bool(v)
        if isinstance(v, (np.ndarray,)):   return v.tolist()
        return v

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO aggression_events
        (timestamp, video_source, frame_number, person_track_id,
         dog_track_id, risk_score, gesture_score, proximity_score,
         aggression_type, capture_path)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        str(event['timestamp']),
        str(event.get('video_source', 'unknown')),
        int(_to_py(event.get('frame_number', -1))),
        int(_to_py(event.get('person_track_id', -1))),
        int(_to_py(event.get('dog_track_id', -1))),
        float(_to_py(event.get('risk_score', 0.0))),
        float(_to_py(event.get('gesture_score', 0.0))),
        float(_to_py(event.get('proximity_score', 0.0))),
        str(event.get('aggression_type', 'unknown')),
        str(event.get('capture_path', '')),
    ))
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return eid


# ─────────────────────────────────────────────────────────────
# RiskScorer — extrait du notebook Adham (inchangé)
# ─────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Calcule un score de risque d'agression à partir de :
    - Les keypoints d'une personne (gestes agressifs / fuite)
    - La distance entre les entités
    - Le mouvement relatif des entités
    """

    def __init__(self, history_len: int = 8):
        self._person_kp_history: dict = defaultdict(list)
        self._bbox_history: dict      = defaultdict(list)
        self.history_len              = history_len

    def update_entity_history(self, track_id: int, entity_type: str,
                               bbox: np.ndarray,
                               keypoints: Optional[np.ndarray] = None) -> None:
        if entity_type == 'person' and keypoints is not None:
            hist_kp = self._person_kp_history[track_id]
            hist_kp.append(keypoints.copy())
            if len(hist_kp) > self.history_len:
                hist_kp.pop(0)
        hist_bbox = self._bbox_history[track_id]
        hist_bbox.append(bbox.copy())
        if len(hist_bbox) > self.history_len:
            hist_bbox.pop(0)

    def _get_entity_velocity(self, track_id: int) -> float:
        hist = self._bbox_history.get(track_id, [])
        if len(hist) < 2:
            return 0.0
        velocities = []
        for i in range(1, len(hist)):
            prev, curr = hist[i-1], hist[i]
            pc = np.array([(prev[0]+prev[2])/2, (prev[1]+prev[3])/2])
            cc = np.array([(curr[0]+curr[2])/2, (curr[1]+curr[3])/2])
            velocities.append(np.linalg.norm(cc - pc))
        return min(1.0, np.mean(velocities) / 60.0) if velocities else 0.0

    def _get_limb_velocity(self, track_id: int) -> float:
        hist = self._person_kp_history.get(track_id, [])
        if len(hist) < 2:
            return 0.0
        arm_indices = [KP_ELBOW_L, KP_ELBOW_R, KP_WRIST_L, KP_WRIST_R]
        velocities  = []
        for i in range(1, len(hist)):
            prev, curr = hist[i-1], hist[i]
            for idx in arm_indices:
                if (idx < len(prev) and idx < len(curr)
                        and prev[idx][2] > 0.3 and curr[idx][2] > 0.3):
                    velocities.append(np.linalg.norm(curr[idx][:2] - prev[idx][:2]))
        return min(1.0, np.mean(velocities) / 60.0) if velocities else 0.0

    def _get_human_to_dog_gesture_score(self, track_id: int,
                                         keypoints: np.ndarray) -> float:
        scores = []

        def valid(idx):
            return idx < len(keypoints) and keypoints[idx][2] > 0.3

        # Poignets au-dessus des épaules
        wrist_above = []
        for w, s in [(KP_WRIST_L, KP_SHOULDER_L), (KP_WRIST_R, KP_SHOULDER_R)]:
            if valid(w) and valid(s):
                dy = keypoints[s][1] - keypoints[w][1]
                wrist_above.append(max(0.0, dy / 80.0))
        if wrist_above:
            scores.append(min(1.0, max(wrist_above)))

        # Extension du bras
        arm_ext = []
        for si, ei, wi in [(KP_SHOULDER_L, KP_ELBOW_L, KP_WRIST_L),
                            (KP_SHOULDER_R, KP_ELBOW_R, KP_WRIST_R)]:
            if valid(si) and valid(ei) and valid(wi):
                total = (np.linalg.norm(keypoints[ei][:2] - keypoints[si][:2])
                         + np.linalg.norm(keypoints[wi][:2] - keypoints[ei][:2]))
                direct = np.linalg.norm(keypoints[wi][:2] - keypoints[si][:2])
                if total > 5:
                    arm_ext.append(direct / total)
        if arm_ext:
            scores.append(max(arm_ext))

        # Coup de pied (genou levé)
        knee_lift = []
        for hi, ki in [(KP_HIP_L, KP_KNEE_L), (KP_HIP_R, KP_KNEE_R)]:
            if valid(hi) and valid(ki):
                dy = keypoints[hi][1] - keypoints[ki][1]
                knee_lift.append(max(0.0, dy / 60.0))
        if knee_lift:
            scores.append(min(1.0, max(knee_lift)))

        # Vélocité membres
        scores.append(self._get_limb_velocity(track_id))

        if not scores:
            return 0.0
        weights = [1.0] * (len(scores) - 1) + [2.0]
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    def _get_human_fleeing_score(self, person_track_id: int,
                                  person_bbox: np.ndarray,
                                  dog_track_id: int,
                                  dog_bbox: np.ndarray) -> float:
        person_vel = self._get_entity_velocity(person_track_id)
        if person_vel <= 0.1:
            return 0.0

        p_hist = self._bbox_history.get(person_track_id, [])
        d_hist = self._bbox_history.get(dog_track_id, [])
        if len(p_hist) < 2 or len(d_hist) < 2:
            return 0.0

        pp = np.array([(p_hist[-2][0]+p_hist[-2][2])/2,
                        (p_hist[-2][1]+p_hist[-2][3])/2])
        pc = np.array([(person_bbox[0]+person_bbox[2])/2,
                        (person_bbox[1]+person_bbox[3])/2])
        dc = np.array([(dog_bbox[0]+dog_bbox[2])/2,
                        (dog_bbox[1]+dog_bbox[3])/2])

        mv = pc - pp
        dv = pc - dc
        if np.linalg.norm(mv) == 0 or np.linalg.norm(dv) == 0:
            return 0.0

        cos_a = np.dot(mv, -dv) / (np.linalg.norm(mv) * np.linalg.norm(dv))
        return max(0.0, (cos_a + 1) / 2) * person_vel

    def _get_dog_aggressive_movement_score(self, dog_track_id: int,
                                            dog_bbox: np.ndarray,
                                            person_track_id: int,
                                            person_bbox: np.ndarray) -> float:
        dog_vel = self._get_entity_velocity(dog_track_id)
        if dog_vel <= 0.1:
            return 0.0

        d_hist = self._bbox_history.get(dog_track_id, [])
        p_hist = self._bbox_history.get(person_track_id, [])
        if len(d_hist) < 2 or len(p_hist) < 2:
            return 0.0

        dp = np.array([(d_hist[-2][0]+d_hist[-2][2])/2,
                        (d_hist[-2][1]+d_hist[-2][3])/2])
        dc = np.array([(dog_bbox[0]+dog_bbox[2])/2,
                        (dog_bbox[1]+dog_bbox[3])/2])
        pc = np.array([(person_bbox[0]+person_bbox[2])/2,
                        (person_bbox[1]+person_bbox[3])/2])

        mv = dc - dp
        pv = dc - pc
        if np.linalg.norm(mv) == 0 or np.linalg.norm(pv) == 0:
            return 0.0

        cos_a = np.dot(mv, -pv) / (np.linalg.norm(mv) * np.linalg.norm(pv))
        return max(0.0, (cos_a + 1) / 2) * dog_vel

    def _get_proximity_score_pairwise(self, bbox1: np.ndarray,
                                       bbox2: np.ndarray) -> tuple:
        c1x = (bbox1[0] + bbox1[2]) / 2
        c1y = (bbox1[1] + bbox1[3]) / 2
        c2x = (bbox2[0] + bbox2[2]) / 2
        c2y = (bbox2[1] + bbox2[3]) / 2
        dist  = np.sqrt((c1x-c2x)**2 + (c1y-c2y)**2)
        score = max(0.0, 1.0 - (dist / (PROXIMITY_THRESHOLD_PX * 2)))
        return score, dist

    def compute_pair(self,
                     person_track_id: int,
                     person_keypoints: Optional[np.ndarray],
                     person_bbox: np.ndarray,
                     dog_track_id: int,
                     dog_bbox: np.ndarray) -> dict:
        """Point d'entrée principal — calcule tous les scores pour une paire."""
        self.update_entity_history(person_track_id, 'person',
                                   person_bbox, person_keypoints)
        self.update_entity_history(dog_track_id, 'dog', dog_bbox)

        gesture_score = 0.0
        if person_keypoints is not None:
            gesture_score = self._get_human_to_dog_gesture_score(
                person_track_id, person_keypoints)

        prox_h2d, dist_h2d = self._get_proximity_score_pairwise(
            person_bbox, dog_bbox)
        human_aggresses_dog = gesture_score * 0.6 + prox_h2d * 0.4

        dog_move  = self._get_dog_aggressive_movement_score(
            dog_track_id, dog_bbox, person_track_id, person_bbox)
        flee      = self._get_human_fleeing_score(
            person_track_id, person_bbox, dog_track_id, dog_bbox)
        prox_d2h, dist_d2h = self._get_proximity_score_pairwise(
            dog_bbox, person_bbox)
        dog_aggresses_human = dog_move * 0.5 + flee * 0.3 + prox_d2h * 0.2

        return {
            'human_aggresses_dog_score':     round(human_aggresses_dog, 3),
            'human_gesture_score':           round(gesture_score, 3),
            'person_to_dog_proximity_score': round(prox_h2d, 3),
            'person_to_dog_distance_px':     round(dist_h2d, 1),
            'dog_aggresses_human_score':     round(dog_aggresses_human, 3),
            'dog_aggressive_movement_score': round(dog_move, 3),
            'human_fleeing_score':           round(flee, 3),
            'dog_to_person_proximity_score': round(prox_d2h, 3),
            'dog_to_person_distance_px':     round(dist_d2h, 1),
        }


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

def _draw_text_bg(frame, text, pos, color=(255,255,255), scale=0.45):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    x, y = pos
    cv2.rectangle(frame, (x-2, y-th-4), (x+tw+2, y+bl), COLOR_TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def annotate_frame(frame, person_detections, dog_detections,
                   risk_lines, frame_number, alert_active=False):
    out = frame.copy()

    for dog in dog_detections:
        x1, y1, x2, y2 = [int(v) for v in dog['bbox']]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_DOG, 2)
        _draw_text_bg(out, f"Dog #{dog['track_id']}", (x1, y1-5), COLOR_DOG)

    for person in person_detections:
        x1, y1, x2, y2 = [int(v) for v in person['bbox']]
        score = person.get('risk', 0.0)
        if score >= RISK_THRESHOLD:
            color = COLOR_PERSON_ALERT
        elif score >= RISK_THRESHOLD * 0.6:
            color = COLOR_PERSON_WARN
        else:
            color = COLOR_PERSON_OK
        thick = 3 if score >= RISK_THRESHOLD else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        _draw_text_bg(out, f"P#{person['track_id']} risk:{score:.2f}",
                       (x1, y1-5), color)

        kps = person.get('keypoints', [])
        if len(kps) >= 17:
            for c1, c2 in SKELETON_CONNECTIONS:
                if (c1 < len(kps) and c2 < len(kps)
                        and kps[c1][2] > 0.3 and kps[c2][2] > 0.3):
                    cv2.line(out,
                             (int(kps[c1][0]), int(kps[c1][1])),
                             (int(kps[c2][0]), int(kps[c2][1])),
                             color, 1, cv2.LINE_AA)
            for kp in kps:
                if kp[2] > 0.3:
                    cv2.circle(out, (int(kp[0]), int(kp[1])), 3, color, -1)

    for (pc, dc, score) in risk_lines:
        alpha = int(min(255, score * 300))
        cv2.line(out, pc, dc, (0, 0, alpha), 2, cv2.LINE_AA)
        mid = ((pc[0]+dc[0])//2, (pc[1]+dc[1])//2)
        _draw_text_bg(out, f"{score:.2f}", mid, (0, 0, 255))

    hc = (0, 0, 220) if alert_active else (200, 200, 200)
    ht = "AGRESSION DETECTEE" if alert_active else "Surveillance active"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (10, 10, 10), -1)
    cv2.putText(out, f"Frame {frame_number}  |  {ht}",
                (8, 20), FONT, 0.6, hc, 1, cv2.LINE_AA)
    return out


def _frame_to_b64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode un frame OpenCV en JPEG base64 pour l'API."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def run_aggression_pipeline(video_source,
                             camera_id:  str = "CAM_001",
                             max_frames: int = None,
                             progress_cb=None) -> list:
    """
    Détecte les agressions dans une vidéo.

    Args:
        video_source : chemin fichier ou entier (webcam)
        camera_id    : identifiant caméra pour les logs
        max_frames   : limite de frames à traiter (None = tout)
        progress_cb  : callable(msg: str) pour les mises à jour de progression

    Returns:
        Liste de dicts d'événements :
        {
          event_id, timestamp, camera_id, video_source,
          frame_number, person_track_id, dog_track_id,
          risk_score, gesture_score, proximity_score,
          aggression_type,  # 'human_to_dog' | 'dog_to_human'
          frame_b64,        # JPEG base64 du frame annoté
          capture_path,     # chemin JPG sauvegardé sur disque
        }
    """
    def _log(msg):
        print(f"[Adham] {msg}")
        if progress_cb:
            progress_cb(msg)

    _init_db()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics non installé — pip install ultralytics")

    device = 'cuda' if _has_cuda() else 'cpu'
    _log(f"Device : {device}")

    _log("Chargement YOLOv8n détection...")
    model_detect = YOLO(YOLO_DETECT_MODEL)
    model_detect.to(device)

    _log("Chargement YOLOv8n-pose...")
    model_pose = YOLO(YOLO_POSE_MODEL)
    model_pose.to(device)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_source}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _log(f"Vidéo ouverte — {total if total > 0 else '∞'} frames @ {fps:.1f} FPS")

    scorer           = RiskScorer(history_len=8)
    consecutive_risk = defaultdict(int)
    last_alert_frame = defaultdict(lambda: -9999)
    MIN_FRAMES_BTW   = int(fps * 10)

    events     = []
    frame_idx  = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break
            frame_idx += 1

            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                pfps    = frame_idx / elapsed if elapsed > 0 else 0
                pct     = f"{frame_idx}/{total}" if total > 0 else str(frame_idx)
                _log(f"Frame {pct} — {pfps:.1f} FPS — {len(events)} événement(s)")

            # ── Détection + tracking ──────────────────────────
            det = model_detect.track(
                frame, persist=True,
                classes=[PERSON_CLASS_ID, DOG_CLASS_ID],
                conf=DETECT_CONF, verbose=False,
                tracker='bytetrack.yaml'
            )

            persons = []
            dogs    = []

            if det and det[0].boxes is not None:
                boxes = det[0].boxes
                for i in range(len(boxes)):
                    cls_id   = int(boxes.cls[i].item())
                    track_id = int(boxes.id[i].item()) if boxes.id is not None else i
                    bbox     = boxes.xyxy[i].cpu().numpy()
                    if cls_id == PERSON_CLASS_ID:
                        persons.append({'bbox': bbox, 'track_id': track_id})
                    elif cls_id == DOG_CLASS_ID:
                        dogs.append({'bbox': bbox, 'track_id': track_id})

            if not persons or not dogs:
                for pid in list(consecutive_risk.keys()):
                    if pid not in {p['track_id'] for p in persons}:
                        consecutive_risk[pid] = 0
                continue

            # ── Pose estimation ───────────────────────────────
            pose_res = model_pose(frame, conf=POSE_CONF, verbose=False)
            kp_map   = {}

            if pose_res and pose_res[0].keypoints is not None:
                kps_data   = pose_res[0].keypoints.data.cpu().numpy()
                pose_boxes = pose_res[0].boxes.xyxy.cpu().numpy()

                for p in persons:
                    pb       = p['bbox']
                    best_iou = 0.2
                    best_kp  = None
                    for j, qb in enumerate(pose_boxes):
                        ix1  = max(pb[0], qb[0]); iy1 = max(pb[1], qb[1])
                        ix2  = min(pb[2], qb[2]); iy2 = min(pb[3], qb[3])
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        a1    = (pb[2]-pb[0]) * (pb[3]-pb[1])
                        a2    = (qb[2]-qb[0]) * (qb[3]-qb[1])
                        union = a1 + a2 - inter
                        iou   = inter / union if union > 0 else 0
                        if iou > best_iou:
                            best_iou = iou
                            best_kp  = kps_data[j] if j < len(kps_data) else None
                    if best_kp is not None:
                        kp_map[p['track_id']] = best_kp

            # ── Calcul du risque ──────────────────────────────
            risk_lines  = []
            alert_frame = False

            person_max = defaultdict(lambda: {
                'score': 0.0, 'type': 'none',
                'dog_id': -1, 'risk_data': {}
            })

            for person in persons:
                pid    = person['track_id']
                p_bbox = person['bbox']
                kps    = kp_map.get(pid)

                for dog in dogs:
                    did    = dog['track_id']
                    d_bbox = dog['bbox']
                    data   = scorer.compute_pair(pid, kps, p_bbox, did, d_bbox)

                    best  = 0.0
                    btype = 'none'
                    if data['human_aggresses_dog_score'] > best:
                        best  = data['human_aggresses_dog_score']
                        btype = 'human_to_dog'
                    if data['dog_aggresses_human_score'] > best:
                        best  = data['dog_aggresses_human_score']
                        btype = 'dog_to_human'

                    if best > person_max[pid]['score']:
                        person_max[pid] = {
                            'score': best, 'type': btype,
                            'dog_id': did, 'risk_data': data
                        }

                person['risk']          = person_max[pid]['score']
                person['agg_type']      = person_max[pid]['type']
                person['dog_id']        = person_max[pid]['dog_id']
                person['risk_data']     = person_max[pid]['risk_data']
                person['keypoints']     = kps if kps is not None else []

                if person['dog_id'] != -1 and person['risk'] > 0.3:
                    d_bbox = next(
                        (d['bbox'] for d in dogs if d['track_id'] == person['dog_id']),
                        None
                    )
                    if d_bbox is not None:
                        pc = (int((p_bbox[0]+p_bbox[2])/2),
                               int((p_bbox[1]+p_bbox[3])/2))
                        dc = (int((d_bbox[0]+d_bbox[2])/2),
                               int((d_bbox[1]+d_bbox[3])/2))
                        risk_lines.append((pc, dc, person['risk']))

                # Fenêtre de risque consécutif
                if person['risk'] >= RISK_THRESHOLD:
                    consecutive_risk[pid] += 1
                else:
                    consecutive_risk[pid] = max(0, consecutive_risk[pid] - 1)

                # ── Déclenchement de l'alerte ──────────────────
                frames_since = frame_idx - last_alert_frame[pid]

                if (consecutive_risk[pid] >= ALERT_FRAME_WINDOW
                        and frames_since > MIN_FRAMES_BTW
                        and person['risk'] > 0.0):

                    alert_frame = True
                    last_alert_frame[pid] = frame_idx
                    consecutive_risk[pid] = 0

                    # Annoter + capturer le frame
                    annotated    = annotate_frame(
                        frame, persons, dogs, risk_lines,
                        frame_idx, alert_active=True
                    )
                    frame_b64    = _frame_to_b64(annotated)
                    ts           = datetime.now().isoformat()

                    # Sauvegarder sur disque
                    cap_name     = f"aggression_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_f{frame_idx}.jpg"
                    cap_path     = str(CAPTURES_DIR / cap_name)
                    cv2.imwrite(cap_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

                    rd = person['risk_data']
                    event = {
                        'timestamp':        ts,
                        'camera_id':        camera_id,
                        'video_source':     str(video_source),
                        'frame_number':     frame_idx,
                        'person_track_id':  pid,
                        'dog_track_id':     person['dog_id'],
                        'risk_score':       round(person['risk'], 3),
                        'gesture_score':    round(rd.get('human_gesture_score', 0.0), 3),
                        'proximity_score':  round(rd.get('person_to_dog_proximity_score',
                                                          rd.get('dog_to_person_proximity_score', 0.0)), 3),
                        'aggression_type':  person['agg_type'],
                        'capture_path':     cap_path,
                        'frame_b64':        frame_b64,
                    }

                    eid = _save_event(event)
                    event['event_id'] = f"AGR_{eid:04d}_{frame_idx}"
                    events.append(event)

                    _log(f"🚨 Événement #{eid} | P#{pid} → D#{person['dog_id']} "
                         f"| score={person['risk']:.3f} | {person['agg_type']}")

    except KeyboardInterrupt:
        _log("Pipeline interrompu")
    finally:
        cap.release()

    elapsed = time.time() - start_time
    _log(f"Terminé — {frame_idx} frames en {elapsed:.1f}s "
         f"| {len(events)} événement(s) détecté(s)")

    return events


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Test rapide ───────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    src = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() \
          else (sys.argv[1] if len(sys.argv) > 1 else 0)
    evts = run_aggression_pipeline(src, max_frames=200,
                                    progress_cb=print)
    print(f"\n{len(evts)} événement(s) détecté(s)")
    for e in evts:
        print(f"  #{e['event_id']} frame={e['frame_number']} "
              f"score={e['risk_score']} type={e['aggression_type']}")
