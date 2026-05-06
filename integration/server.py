"""
DOGSIM-TN — Serveur unifié
===========================
Lance UN SEUL serveur sur http://localhost:8080
qui sert :
  • Le site vitrine + dashboard  →  GET  /
  • Les fichiers statiques        →  GET  /<fichier>
  • API Pipeline S1/S2/S3        →  /api/*         (port 5000 fusionné)
  • API CV Module                 →  /cv/*          (port 5001 fusionné)
  • API Molka (SARIMAX + RL)      →  /molka/*       (module Molka)
  • API Agression (Adham)         →  /aggression/*  (module Adham)

Usage :
    python server.py
"""

import sys, os
from pathlib import Path
from flask import Flask, send_from_directory
from flask_cors import CORS

# ── Chemins ─────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent           # integration/
PIPELINE_DIR = HERE / "stray_dogs_v2"                   # integration/stray_dogs_v2/
CV_PARENT    = HERE                                      # integration/  (contient cv_module/)
MOLKA_DIR    = HERE / "molka"                            # integration/molka/
ADHAM_DIR    = HERE / "adham"                            # integration/adham/

# ── sys.path ────────────────────────────────────────────────
# Pipeline : api.py a besoin de pipeline/, utils/, config, etc.
sys.path.insert(0, str(PIPELINE_DIR))
# CV : cv_api.py a besoin de cv_module/ en tant que package
sys.path.insert(0, str(CV_PARENT))
# Molka : molka_api.py et ses modules sont dans integration/molka/
sys.path.insert(0, str(MOLKA_DIR))
# Adham : aggression_api.py et aggression_pipeline.py
sys.path.insert(0, str(ADHAM_DIR))

# ── Importer les deux apps Flask ────────────────────────────
print("🔄  Chargement du module Pipeline (S1/S2/S3)...")
from api import app as _pipeline_app          # stray_dogs_v2/api.py

print("🔄  Chargement du module CV...")
from cv_module.cv_api import app as _cv_app   # cv_module/cv_api.py

print("🔄  Chargement du module Molka (SARIMAX + RL)...")
try:
    from molka_api import molka_bp, init_molka
    _MOLKA_OK = True
except ImportError as e:
    print(f"⚠   Module Molka non disponible : {e}")
    _MOLKA_OK = False

print("🔄  Chargement du module Adham (Agression)...")
try:
    from aggression_api import aggression_bp
    _ADHAM_OK = True
except ImportError as e:
    print(f"⚠   Module Adham non disponible : {e}")
    _ADHAM_OK = False

# ── App maître ───────────────────────────────────────────────
master = Flask(__name__)
CORS(master)

# ── Copier toutes les routes de pipeline_app ─────────────────
for rule in _pipeline_app.url_map.iter_rules():
    if rule.endpoint == "static":
        continue
    view_func = _pipeline_app.view_functions[rule.endpoint]
    master.add_url_rule(
        rule.rule,
        endpoint=f"pipeline__{rule.endpoint}",
        view_func=view_func,
        methods=rule.methods,
    )

# ── Copier toutes les routes de cv_app ───────────────────────
for rule in _cv_app.url_map.iter_rules():
    if rule.endpoint == "static":
        continue
    view_func = _cv_app.view_functions[rule.endpoint]
    master.add_url_rule(
        rule.rule,
        endpoint=f"cv__{rule.endpoint}",
        view_func=view_func,
        methods=rule.methods,
    )

# ── Enregistrer le blueprint Molka ───────────────────────────
if _MOLKA_OK:
    master.register_blueprint(molka_bp)
    # Démarrage asynchrone : dataset en arrière-plan (modèles à la demande)
    with master.app_context():
        init_molka(eager=False)

# ── Enregistrer le blueprint Adham (Agression) ───────────────
if _ADHAM_OK:
    master.register_blueprint(aggression_bp)

# ── Servir les fichiers statiques de integration/ ────────────
@master.route("/")
def index():
    return send_from_directory(str(HERE), "DOGSIM-TN Final.html")

@master.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(HERE), filename)

# ── Lancement ────────────────────────────────────────────────
if __name__ == "__main__":
    port = 8080
    print()
    print("=" * 55)
    print("  🐕  DOGSIM-TN — Serveur unifié")
    print("=" * 55)
    print(f"  Site     →  http://localhost:{port}")
    print(f"  Pipeline →  http://localhost:{port}/api/health")
    print(f"  CV       →  http://localhost:{port}/cv/health")
    if _MOLKA_OK:
        print(f"  Molka    →  http://localhost:{port}/molka/health")
    if _ADHAM_OK:
        print(f"  Adham    →  http://localhost:{port}/aggression/health")
    print("=" * 55)
    print()
    master.run(host="0.0.0.0", port=port, debug=False, threaded=True)
