# =============================================================
# aggression_api.py — Blueprint Flask pour le module Adham
# DOGSIM-TN | ESPRIT 3A IA
#
# Endpoints :
#   GET  /aggression/health            : état YOLO + dépendances
#   POST /aggression/upload            : vidéo locale → job_id
#   POST /aggression/youtube           : URL YouTube → job_id
#   GET  /aggression/job/<job_id>      : statut + résultats
#   GET  /aggression/jobs              : liste de tous les jobs
#   GET  /aggression/history           : événements SQLite persistés
# =============================================================

import sys
import os
import threading
import uuid
import shutil
import time
import base64
from pathlib import Path

from flask import Blueprint, request, jsonify

# ── Sérialisation numpy → types Python natifs ─────────────────
def _sanitize(obj):
    """
    Convertit récursivement tous les types numpy (float32, int64, ndarray…)
    en types Python natifs compatibles JSON.
    Appelé avant tout stockage dans JOBS pour éviter les TypeError au jsonify.
    """
    import numpy as np
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# ── Chemins ───────────────────────────────────────────────────
_HERE       = Path(__file__).parent
_UPLOAD_DIR = _HERE / "output" / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Blueprint ─────────────────────────────────────────────────
aggression_bp = Blueprint('aggression', __name__,
                           url_prefix='/aggression')

# ── Jobs en mémoire ───────────────────────────────────────────
# { job_id → { status, progress, events, event_count, error } }
JOBS: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────

def _run_job(job_id: str, video_path: str, camera_id: str,
             max_frames: int | None, cleanup: bool = False):
    """Exécute le pipeline dans un thread séparé."""
    try:
        JOBS[job_id]['status']   = 'running'
        JOBS[job_id]['progress'] = 'Chargement des modèles YOLO...'

        # Import local pour éviter un import au démarrage du serveur
        sys.path.insert(0, str(_HERE))
        from aggression_pipeline import run_aggression_pipeline

        def _cb(msg):
            JOBS[job_id]['progress'] = msg

        events = run_aggression_pipeline(
            video_source=video_path,
            camera_id=camera_id,
            max_frames=max_frames,
            progress_cb=_cb,
        )

        # Sérialiser + convertir tous les numpy types → Python natifs
        serialized = []
        for e in events:
            entry = {k: v for k, v in e.items() if k != 'frame_b64'}
            entry['frame'] = e.get('frame_b64')   # frame base64 (str pure)
            serialized.append(_sanitize(entry))   # ← conversion numpy ici

        JOBS[job_id].update({
            'status':      'done',
            'progress':    'Terminé',
            'event_count': len(events),
            'events':      serialized,
        })

    except Exception as exc:
        import traceback
        JOBS[job_id].update({
            'status':    'error',
            'progress':  'Erreur',
            'error':     str(exc),
            'traceback': traceback.format_exc(),
        })
    finally:
        if cleanup and os.path.exists(video_path):
            os.remove(video_path)


# ─────────────────────────────────────────────────────────────
# GET /aggression/health
# ─────────────────────────────────────────────────────────────

@aggression_bp.route('/health')
def health():
    try:
        from ultralytics import YOLO
        yolo_ok = True
    except ImportError:
        yolo_ok = False

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    yt_ok = shutil.which('yt-dlp') is not None

    return jsonify({
        'status':  'ok',
        'module':  'adham_aggression',
        'yolo':    yolo_ok,
        'cuda':    cuda_ok,
        'yt_dlp':  yt_ok,
        'device':  'cuda' if cuda_ok else 'cpu',
    })


# ─────────────────────────────────────────────────────────────
# POST /aggression/upload  — vidéo locale
# ─────────────────────────────────────────────────────────────

@aggression_bp.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Aucun fichier vidéo reçu'}), 400

    f          = request.files['video']
    camera_id  = request.form.get('camera_id', 'CAM_LOCAL')
    max_frames = request.form.get('max_frames')
    if max_frames:
        max_frames = int(max_frames)

    ext      = Path(f.filename).suffix or '.mp4'
    tmp_path = str(_UPLOAD_DIR / f"{uuid.uuid4()}{ext}")
    f.save(tmp_path)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        'status':      'queued',
        'progress':    'En attente...',
        'source':      f.filename,
        'event_count': 0,
        'events':      [],
    }

    threading.Thread(
        target=_run_job,
        args=(job_id, tmp_path, camera_id, max_frames, True),
        daemon=True,
    ).start()

    return jsonify({'job_id': job_id})


# ─────────────────────────────────────────────────────────────
# POST /aggression/youtube  — URL YouTube
# ─────────────────────────────────────────────────────────────

@aggression_bp.route('/youtube', methods=['POST'])
def youtube_video():
    data       = request.get_json(silent=True) or {}
    url        = data.get('url', '').strip()
    camera_id  = data.get('camera_id', 'CAM_YT')
    max_frames = data.get('max_frames', 500)

    if not url:
        return jsonify({'error': 'URL manquante'}), 400
    if shutil.which('yt-dlp') is None:
        return jsonify({
            'error': 'yt-dlp non installé',
            'fix':   'pip install yt-dlp',
        }), 500

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        'status':      'queued',
        'progress':    'Téléchargement YouTube...',
        'source':      url,
        'event_count': 0,
        'events':      [],
    }

    def _dl_and_run():
        try:
            tmp = str(_UPLOAD_DIR / f"{job_id}.mp4")
            JOBS[job_id]['progress'] = 'Téléchargement en cours...'
            ret = os.system(
                f'yt-dlp -f "best[ext=mp4][height<=480]" '
                f'-o "{tmp}" --no-playlist "{url}"'
            )
            if ret != 0 or not os.path.exists(tmp):
                JOBS[job_id].update({
                    'status': 'error',
                    'error':  'Échec du téléchargement YouTube',
                })
                return
            JOBS[job_id]['progress'] = 'Vidéo téléchargée, analyse en cours...'
            _run_job(job_id, tmp, camera_id, max_frames, cleanup=True)
        except Exception as e:
            JOBS[job_id].update({'status': 'error', 'error': str(e)})

    threading.Thread(target=_dl_and_run, daemon=True).start()
    return jsonify({'job_id': job_id})


# ─────────────────────────────────────────────────────────────
# GET /aggression/job/<job_id>
# ─────────────────────────────────────────────────────────────

@aggression_bp.route('/job/<job_id>')
def get_job(job_id: str):
    if job_id not in JOBS:
        return jsonify({'error': 'Job introuvable'}), 404
    return jsonify(JOBS[job_id])


# ─────────────────────────────────────────────────────────────
# GET /aggression/jobs  — liste résumée
# ─────────────────────────────────────────────────────────────

@aggression_bp.route('/jobs')
def list_jobs():
    out = {}
    for jid, info in JOBS.items():
        out[jid] = {k: v for k, v in info.items() if k != 'events'}
        out[jid]['event_count'] = info.get('event_count', 0)
    return jsonify(out)


# ─────────────────────────────────────────────────────────────
# GET /aggression/history  — événements SQLite + JOBS mémoire
# ─────────────────────────────────────────────────────────────

@aggression_bp.route('/history')
def history():
    """
    Retourne les 100 derniers événements détectés.
    Sources combinées :
      1. JOBS en mémoire  (session courante, status='done')
      2. SQLite           (sessions précédentes)
    Les doublons sont éliminés par (video_source, frame_number, person_track_id, dog_track_id).
    """
    import sqlite3

    seen   = set()   # clé de dédup
    result = []

    # ── 1. Événements en mémoire (JOBS terminés) ─────────────
    for job_info in JOBS.values():
        if job_info.get('status') != 'done':
            continue
        source = job_info.get('source', '')
        for e in job_info.get('events', []):
            key = (
                source,
                e.get('frame_number', 0),
                e.get('person_track_id', -1),
                e.get('dog_track_id', -1),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append({
                'id':               None,
                'event_id':         e.get('event_id'),   # ex. "AGR_0001_123"
                'timestamp':        e.get('timestamp', ''),
                'video_source':     e.get('video_source', source),
                'camera_id':        e.get('camera_id', ''),
                'frame_number':     e.get('frame_number'),
                'person_track_id':  e.get('person_track_id'),
                'dog_track_id':     e.get('dog_track_id'),
                'risk_score':       e.get('risk_score'),
                'gesture_score':    e.get('gesture_score'),
                'proximity_score':  e.get('proximity_score'),
                'aggression_type':  e.get('aggression_type', ''),
                'capture_path':     None,
                'frame':            e.get('frame'),   # base64 déjà encodé
            })

    # ── 2. Événements SQLite (sessions précédentes) ───────────
    db_path = str(_HERE / 'output' / 'aggression_events.db')
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT id, timestamp, video_source, frame_number,
                       person_track_id, dog_track_id, risk_score,
                       gesture_score, proximity_score, aggression_type,
                       capture_path
                FROM aggression_events
                ORDER BY id DESC LIMIT 100
            """).fetchall()
            conn.close()
            import struct

            def _fix_score(v):
                """Convertit les anciens scores stockés en bytes (float32 LE) en float Python."""
                if isinstance(v, (bytes, bytearray)) and len(v) == 4:
                    try:
                        return round(struct.unpack('<f', v)[0], 4)
                    except Exception:
                        return None
                if isinstance(v, (int, float)):
                    return float(v)
                return v

            for r in rows:
                entry = dict(r)
                # Corriger les scores potentiellement stockés comme BLOB
                for col in ('risk_score', 'gesture_score', 'proximity_score'):
                    entry[col] = _fix_score(entry.get(col))
                key = (
                    entry.get('video_source', ''),
                    entry.get('frame_number', 0),
                    entry.get('person_track_id', -1),
                    entry.get('dog_track_id', -1),
                )
                if key in seen:
                    continue
                seen.add(key)
                cp = entry.get('capture_path', '')
                if cp and os.path.exists(cp):
                    with open(cp, 'rb') as f:
                        entry['frame'] = ("data:image/jpeg;base64,"
                                          + base64.b64encode(f.read()).decode())
                else:
                    entry['frame'] = None
                result.append(entry)
        except Exception as e:
            pass  # SQLite indisponible → on retourne au moins les événements mémoire

    # ── Tri par timestamp décroissant ─────────────────────────
    result.sort(key=lambda x: x.get('timestamp') or '', reverse=True)

    return jsonify(result[:100])


# ── App standalone ────────────────────────────────────────────
if __name__ == '__main__':
    from flask import Flask
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(aggression_bp)
    print("[Adham] Démarrage sur http://localhost:5003")
    app.run(host='0.0.0.0', port=5003, debug=False)
