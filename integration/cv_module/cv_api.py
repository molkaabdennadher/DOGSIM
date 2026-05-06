# ============================================================
#  CV API — Serveur Flask pour tester le module CV
#  Lancer : python cv_module/cv_api.py
#  Accès  : http://localhost:5001
# ============================================================

import sys, os, threading, uuid, json, base64, tempfile, shutil, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Stockage des jobs en mémoire ──────────────────────────────
# job_id → { status, progress, reports, error }
JOBS: dict[str, dict] = {}

UPLOAD_DIR  = os.path.join(os.path.dirname(__file__), "output", "uploads")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "output", "reports")
FRAMES_DIR  = os.path.join(os.path.dirname(__file__), "output", "frames")
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def image_to_base64(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def run_pipeline_job(job_id: str, video_path: str, camera_id: str,
                     max_frames: int | None, cleanup: bool = False):
    """
    Lancer le pipeline dans un thread séparé.
    Met à jour JOBS[job_id] au fil de l'avancement.
    """
    try:
        JOBS[job_id]["status"]   = "running"
        JOBS[job_id]["progress"] = "Chargement du modèle YOLO..."

        from cv_module.pipeline import FeedingInfractionPipeline

        pipeline = FeedingInfractionPipeline(
            source     = video_path,
            camera_id  = camera_id,
            display    = False,
            max_frames = max_frames
        )

        JOBS[job_id]["progress"] = "Analyse en cours..."
        reports = pipeline.run()

        # Convertir les rapports en JSON sérialisable avec images base64
        results = []
        for r in reports:
            entry = {
                "infraction_id":       r.infraction_id,
                "timestamp":           r.timestamp,
                "camera_id":           r.camera_id,
                "dog_track_id":        r.dog_track_id,
                "stagnation_seconds":  r.stagnation_seconds,
                "human_found":         r.human_found,
                "seconds_before_dog":  r.seconds_before_dog,
                "distance_at_deposit": r.distance_at_deposit,
                "notes":               r.notes,
                "frames": {
                    "approach":  image_to_base64(r.image_approach_path),
                    "deposit":   image_to_base64(r.image_deposit_path),
                    "departure": image_to_base64(r.image_departure_path),
                }
            }
            results.append(entry)

        JOBS[job_id].update({
            "status":             "done",
            "progress":           "Terminé",
            "infraction_count":   len(reports),
            "results":            results,
        })

    except Exception as e:
        import traceback
        JOBS[job_id].update({
            "status":   "error",
            "progress": "Erreur",
            "error":    str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        if cleanup and os.path.exists(video_path):
            os.remove(video_path)


# ─────────────────────────────────────────────────────────────
# GET /cv/health
# ─────────────────────────────────────────────────────────────
@app.route("/cv/health", methods=["GET"])
def health():
    yt_dlp_ok = shutil.which("yt-dlp") is not None
    try:
        from ultralytics import YOLO
        yolo_ok = True
    except ImportError:
        yolo_ok = False

    return jsonify({
        "status":       "ok",
        "yolo":         yolo_ok,
        "yt_dlp":       yt_dlp_ok,
        "yt_dlp_hint":  "pip install yt-dlp" if not yt_dlp_ok else None,
    })


# ─────────────────────────────────────────────────────────────
# POST /cv/upload  — Vidéo locale
# ─────────────────────────────────────────────────────────────
@app.route("/cv/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "Aucun fichier vidéo reçu"}), 400

    file      = request.files["video"]
    camera_id = request.form.get("camera_id", "CAM_LOCAL")
    max_frames = request.form.get("max_frames", None)
    if max_frames:
        max_frames = int(max_frames)

    # Sauvegarder la vidéo uploadée
    ext      = os.path.splitext(file.filename)[1] or ".mp4"
    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    file.save(tmp_path)

    # Créer le job
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status":    "queued",
        "progress":  "En attente...",
        "source":    file.filename,
        "results":   [],
    }

    # Lancer en arrière-plan
    t = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, tmp_path, camera_id, max_frames, True),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id})


# ─────────────────────────────────────────────────────────────
# POST /cv/youtube  — URL YouTube
# ─────────────────────────────────────────────────────────────
@app.route("/cv/youtube", methods=["POST"])
def youtube_video():
    data      = request.get_json()
    url       = data.get("url", "").strip()
    camera_id = data.get("camera_id", "CAM_YT")
    max_frames = data.get("max_frames", 500)  # limiter par défaut pour YT

    if not url:
        return jsonify({"error": "URL manquante"}), 400

    if shutil.which("yt-dlp") is None:
        return jsonify({
            "error": "yt-dlp non installé",
            "fix":   "pip install yt-dlp"
        }), 500

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status":   "queued",
        "progress": "Téléchargement YouTube...",
        "source":   url,
        "results":  [],
    }

    def download_and_run():
        try:
            # Télécharger la vidéo avec yt-dlp
            tmp_path = os.path.join(UPLOAD_DIR, f"{job_id}.mp4")
            JOBS[job_id]["progress"] = "Téléchargement en cours..."

            ret = os.system(
                f'yt-dlp -f "best[ext=mp4][height<=480]" '
                f'-o "{tmp_path}" '
                f'--no-playlist '
                f'"{url}"'
            )
            if ret != 0 or not os.path.exists(tmp_path):
                JOBS[job_id].update({
                    "status": "error",
                    "error":  "Échec du téléchargement YouTube. "
                              "Vérifiez l'URL ou essayez une autre vidéo."
                })
                return

            JOBS[job_id]["progress"] = "Vidéo téléchargée, analyse en cours..."
            run_pipeline_job(job_id, tmp_path, camera_id, max_frames, cleanup=True)

        except Exception as e:
            JOBS[job_id].update({"status": "error", "error": str(e)})

    t = threading.Thread(target=download_and_run, daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


# ─────────────────────────────────────────────────────────────
# GET /cv/job/<job_id>  — Statut + résultats d'un job
# ─────────────────────────────────────────────────────────────
@app.route("/cv/job/<job_id>", methods=["GET"])
def get_job(job_id: str):
    if job_id not in JOBS:
        return jsonify({"error": "Job introuvable"}), 404
    return jsonify(JOBS[job_id])


# ─────────────────────────────────────────────────────────────
# GET /cv/jobs  — Liste de tous les jobs
# ─────────────────────────────────────────────────────────────
@app.route("/cv/jobs", methods=["GET"])
def list_jobs():
    summary = {}
    for jid, info in JOBS.items():
        # Copier toutes les clés sauf "results" (on la retraite ci-dessous)
        job_data = {k: v for k, v in info.items() if k != "results"}
        # Inclure results SANS les frames base64 (trop lourdes pour une liste)
        # L'onglet Infractions n'affiche que les métadonnées, pas les images
        if "results" in info:
            job_data["results"] = [
                {k: v for k, v in r.items() if k != "frames"}
                for r in info["results"]
            ]
        summary[jid] = job_data
    return jsonify(summary)


# ─────────────────────────────────────────────────────────────
# GET /cv/history  — Historique persisté sur disque
# ─────────────────────────────────────────────────────────────
@app.route("/cv/history", methods=["GET"])
def get_history():
    """
    Lit tous les rapports JSON sauvegardés dans output/reports/
    et reconstruit les URLs des images pour l'affichage frontend.
    Contrairement à /cv/jobs (mémoire vive, perdue au redémarrage),
    cet endpoint persiste entre les sessions.
    """
    infractions = []

    if not os.path.exists(REPORTS_DIR):
        return jsonify([])

    for fname in sorted(os.listdir(REPORTS_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(REPORTS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convertir les chemins d'images en URLs HTTP servables
            # Les JSON stockent des chemins relatifs depuis integration/
            # ex: "cv_module/output\frames\INF_..._1_approche.jpg"
            def path_to_url(p):
                if not p:
                    return None
                # Normaliser les séparateurs Windows → Unix
                p = p.replace("\\", "/")
                # Retirer un éventuel préfixe "./" ou absolu hors integration/
                if p.startswith("./"):
                    p = p[2:]
                return f"http://localhost:8080/{p}"

            data["frames"] = {
                "approach":  path_to_url(data.get("image_approach_path")),
                "deposit":   path_to_url(data.get("image_deposit_path")),
                "departure": path_to_url(data.get("image_departure_path")),
            }
            infractions.append(data)

        except Exception as e:
            print(f"[CV History] Erreur lecture {fname}: {e}")

    # Trier par timestamp décroissant (plus récent en premier)
    infractions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(infractions)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  CV API — Feeding Infraction Detector")
    print("  http://localhost:5001")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
