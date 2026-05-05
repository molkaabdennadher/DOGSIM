# =============================================================
# api.py — Serveur Flask pour tester le pipeline Stray Dogs
# Lancer : python api.py
# =============================================================

import sys, os, tempfile, json as _json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json

from pipeline.pipeline import (
    load_models,
    solution1_dog_density,
    solution2_bins_optimization,
    solution3_feeding_zones,
    run_pipeline,
)
from utils.osm_fetcher import get_district_data, extract_elements_from_png
from config import ZONE_COEFF, ROOT

# ── Chargement RGPH 2014 ──────────────────────────────────────
_RGPH_PATH = ROOT / "data" / "raw" / "rgph2014_secteurs.json"
_RGPH_DATA = []
if _RGPH_PATH.exists():
    with open(_RGPH_PATH, encoding="utf-8") as f:
        _RGPH_DATA = _json.load(f)
    print(f"[RGPH] {len(_RGPH_DATA)} secteurs chargés ✓")
else:
    print("[RGPH] Fichier rgph2014_secteurs.json introuvable")

app = Flask(__name__)
CORS(app)

# Charger les modèles au démarrage
MODELS = load_models()

def df_to_list(df):
    if df is None or (hasattr(df, 'empty') and df.empty):
        return []
    return json.loads(df.to_json(orient="records"))

# ─────────────────────────────────────────────────────────────
# GET /api/health
# ─────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    models_ok = all(v is not None for v in MODELS.values())
    return jsonify({
        "status": "ok",
        "models_loaded": models_ok,
        "zone_types": list(ZONE_COEFF.keys()),
    })


# ─────────────────────────────────────────────────────────────
# POST /api/solution1
# Body JSON :
#   { population, nb_menages, surface_km2, nb_food_poi,
#     nb_bins_org, zone_type, dist_centre }
# ─────────────────────────────────────────────────────────────
@app.route("/api/solution1", methods=["POST"])
def solution1():
    d = request.get_json(force=True)
    try:
        result = solution1_dog_density(
            population   = float(d.get("population", 10000)),
            nb_menages   = float(d.get("nb_menages", 2500)),
            surface_km2  = float(d.get("surface_km2", 2.5)),
            nb_food_poi  = int(d.get("nb_food_poi", 15)),
            nb_bins_org  = int(d.get("nb_bins_org", 8)),
            zone_type    = d.get("zone_type", "residential"),
            dist_centre  = float(d.get("dist_centre", 0.5)),
            models       = MODELS,
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ─────────────────────────────────────────────────────────────
# POST /api/pipeline
# Body JSON :
#   { district_name, population, nb_menages, surface_km2,
#     bbox: {lat_min,lat_max,lon_min,lon_max},
#     zone_type, n_zones,
#     bins: [{lat,lon,type}],
#     food_poi: [{lat,lon}],
#     schools: [{lat,lon}],
#     parks: [{lat,lon}] }
# ─────────────────────────────────────────────────────────────
@app.route("/api/pipeline", methods=["POST"])
def pipeline():
    d = request.get_json(force=True)
    try:
        def to_df(lst, cols):
            if lst:
                df = pd.DataFrame(lst)
                for c in cols:
                    if c not in df.columns:
                        df[c] = None
                return df
            return pd.DataFrame(columns=cols)

        bins_df     = to_df(d.get("bins"),     ["lat","lon","type"])
        food_df     = to_df(d.get("food_poi"), ["lat","lon"])
        schools_df  = to_df(d.get("schools"),  ["lat","lon"])
        parks_df    = to_df(d.get("parks"),    ["lat","lon"])

        bbox = d.get("bbox", {
            "lat_min": 36.85, "lat_max": 36.90,
            "lon_min": 10.15, "lon_max": 10.22,
        })

        result = run_pipeline(
            district_name = d.get("district_name", "Test"),
            population    = float(d.get("population", 10000)),
            nb_menages    = float(d.get("nb_menages", 2500)),
            surface_km2   = float(d.get("surface_km2", 2.5)),
            bbox          = bbox,
            bins_df       = bins_df,
            food_poi_df   = food_df,
            schools_df    = schools_df,
            parks_df      = parks_df,
            zone_type     = d.get("zone_type", "residential"),
            n_zones       = int(d.get("n_zones", 7)),
            models        = MODELS,
            verbose       = True,
        )

        # Sérialiser les DataFrames en JSON
        s2 = result["s2"]
        s3 = result["s3"]

        return jsonify({
            "ok": True,
            "s1": result["s1"],
            "s2": {
                "stats":            s2.get("stats", {}),
                "black_spots":      df_to_list(s2.get("black_spots")),
                "recommended_bins": df_to_list(s2.get("recommended_bins")),
                "bin_count_info":   s2.get("bin_count_info", {}),
            },
            "s3": {
                "feeding_zones": df_to_list(s3.get("feeding_zones")),
            },
            "meta": result["meta"],
        })
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 400


# ─────────────────────────────────────────────────────────────
# GET /api/rgph?q=Chotrana
# Recherche floue dans le RGPH 2014 (population, ménages...)
# ─────────────────────────────────────────────────────────────
@app.route("/api/rgph", methods=["GET"])
def rgph():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify({"ok": False, "error": "Paramètre 'q' manquant"}), 400

    if not _RGPH_DATA:
        return jsonify({"ok": False, "error": "Données RGPH non chargées"}), 500

    # 1. Correspondance exacte
    exact = [r for r in _RGPH_DATA if r["nom_lower"] == q]
    if exact:
        return jsonify({"ok": True, "match": "exact", "results": exact[:5]})

    # 2. Correspondance partielle (contient)
    partial = [r for r in _RGPH_DATA if q in r["nom_lower"] or r["nom_lower"] in q]
    if partial:
        return jsonify({"ok": True, "match": "partial", "results": partial[:10]})

    # 3. Recherche mot par mot
    words = q.split()
    scored = []
    for r in _RGPH_DATA:
        score = sum(1 for w in words if w in r["nom_lower"])
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    results = [r for _, r in scored[:10]]

    if results:
        return jsonify({"ok": True, "match": "fuzzy", "results": results})

    return jsonify({"ok": False, "error": f"Aucun secteur trouvé pour '{q}'", "results": []})


# ─────────────────────────────────────────────────────────────
# GET /api/osm_data?district=Chotrana
# Récupère bbox + POI réels depuis OSM pour un quartier
# ─────────────────────────────────────────────────────────────
@app.route("/api/osm_data", methods=["GET"])
def osm_data():
    district = request.args.get("district", "").strip()
    if not district:
        return jsonify({"ok": False, "error": "Paramètre 'district' manquant"}), 400
    try:
        data = get_district_data(district_name=district, country="Tunisia")
        bbox = data["bbox"]

        # Surface approx depuis bbox (en km²)
        import math
        dlat = (bbox["lat_max"] - bbox["lat_min"]) * 111.0
        dlon = (bbox["lon_max"] - bbox["lon_min"]) * 111.0 * math.cos(math.radians(bbox["center_lat"]))
        surface_km2 = round(dlat * dlon, 2)

        food_list    = df_to_list(data["food_poi"])
        schools_list = df_to_list(data["schools"])
        parks_list   = df_to_list(data["parks"])

        # Détecter si les données sont réelles OSM ou simulées (fallback)
        all_poi = df_to_list(data.get("poi", pd.DataFrame()))
        source_used = "simulated" if (
            all_poi and all(p.get("source") == "simulé" for p in all_poi)
        ) else "osm"

        return jsonify({
            "ok":           True,
            "district":     district,
            "bbox":         bbox,
            "surface_km2":  surface_km2,
            "nb_food_poi":  len(food_list),
            "nb_schools":   len(schools_list),
            "nb_parks":     len(parks_list),
            "food_poi":     food_list,
            "schools":      schools_list,
            "parks":        parks_list,
            "source":       "OpenStreetMap / Nominatim + Overpass",
            "source_used":  source_used,
        })
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 400


# ─────────────────────────────────────────────────────────────
# POST /api/parse_png
# Upload d'un PNG de carte → extraction des éléments colorés
# Body : multipart/form-data avec champ 'file' + 'bbox' (JSON)
# ─────────────────────────────────────────────────────────────
@app.route("/api/parse_png", methods=["POST"])
def parse_png():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "Aucun fichier reçu"}), 400

    f    = request.files["file"]
    bbox = json.loads(request.form.get("bbox", "{}"))

    if not bbox:
        return jsonify({"ok": False, "error": "bbox manquante"}), 400

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            f.save(tmp.name)
            result = extract_elements_from_png(tmp.name, bbox)
            os.unlink(tmp.name)

        def pts_to_list(pts):
            return [{"lat": lat, "lon": lon} for lat, lon in pts]

        bins_org  = pts_to_list(result.get("bins_org", []))
        bins_plas = pts_to_list(result.get("bins_plas", []))
        food_poi  = pts_to_list(result.get("food_poi", []))
        parks     = pts_to_list(result.get("parks", []))
        schools   = pts_to_list(result.get("schools", []))

        all_bins = [{"lat": b["lat"], "lon": b["lon"], "type": "organique"} for b in bins_org] + \
                   [{"lat": b["lat"], "lon": b["lon"], "type": "plastique"} for b in bins_plas]

        return jsonify({
            "ok":       True,
            "bins":     all_bins,
            "bins_org": bins_org,
            "food_poi": food_poi,
            "parks":    parks,
            "schools":  schools,
            "counts": {
                "bins_org":  len(bins_org),
                "bins_plas": len(bins_plas),
                "food_poi":  len(food_poi),
                "parks":     len(parks),
                "schools":   len(schools),
            }
        })
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 400


# ─────────────────────────────────────────────────────────────
# POST /api/s2/recommend
# Recommande le nombre et les emplacements des bennes
# Body JSON :
#   { district_name, population, nb_menages, surface_km2 }
# ─────────────────────────────────────────────────────────────
@app.route("/api/s2/recommend", methods=["POST"])
def s2_recommend():
    d = request.get_json(force=True)
    try:
        district  = d.get("district_name", "")
        pop       = int(d.get("population",  10000))
        menages   = int(d.get("nb_menages",  2500))
        surface   = float(d.get("surface_km2", 5.0))
        nb_chiens = float(d.get("nb_chiens",  0.0))

        # Récupérer les données OSM du quartier
        osm = get_district_data(district_name=district)
        bbox       = osm["bbox"]
        food_poi   = osm.get("food_poi", pd.DataFrame())
        schools    = osm.get("schools",  pd.DataFrame())

        from utils.smart_bin_placer import place_bins

        result = place_bins(
            bbox           = bbox,
            population     = pop,
            nb_menages     = menages,
            surface_km2    = surface,
            food_poi_df    = food_poi,
            dog_density_df = None,
            schools_df     = schools,
            nb_chiens      = nb_chiens
        )

        bins_list = df_to_list(result["recommended_bins"])

        return jsonify({
            "ok":              True,
            "district":        district,
            "nb_recommande":   result["bin_count_info"]["nb_recommande"],
            "formule":         result["bin_count_info"]["formule"],
            "contributions":   result["bin_count_info"]["contributions"],
            "nb_restaurants":  result["nb_restaurants"],
            "nb_cafes":        result["nb_cafes"],
            "intersections":   result["intersections_count"],
            "bins":            bins_list,
            "bbox":            bbox,
        })

    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e),
                        "trace": traceback.format_exc()}), 400


# ─────────────────────────────────────────────────────────────
# GET /api/s2/map/<district>
# Retourne une carte Folium HTML interactive avec les bennes
# ─────────────────────────────────────────────────────────────
@app.route("/api/s2/map", methods=["POST"])
def s2_map():
    """
    Génère et retourne une carte HTML Folium interactive.
    Les bennes recommandées y sont affichées aux vrais coins de rue.
    """
    d = request.get_json(force=True)
    try:
        import folium
    except ImportError:
        return jsonify({"ok": False,
                        "error": "folium non installé. Lance : pip install folium"}), 500

    try:
        district  = d.get("district_name", "")
        pop       = int(d.get("population",  10000))
        menages   = int(d.get("nb_menages",  2500))
        surface   = float(d.get("surface_km2", 5.0))
        nb_chiens = float(d.get("nb_chiens",  0.0))
        bins_data = d.get("bins", None)  # bins déjà calculés (optionnel)

        # Si les bins sont passés directement, pas besoin de recalculer
        if bins_data:
            bins_df = pd.DataFrame(bins_data)
            nb_rec  = len(bins_df)
            bbox    = d.get("bbox", {})
            center_lat = (bbox.get("lat_min",0) + bbox.get("lat_max",0)) / 2
            center_lon = (bbox.get("lon_min",0) + bbox.get("lon_max",0)) / 2
        else:
            # Recalculer
            osm = get_district_data(district_name=district)
            bbox       = osm["bbox"]
            food_poi   = osm.get("food_poi", pd.DataFrame())
            schools    = osm.get("schools",  pd.DataFrame())
            center_lat = (bbox["lat_min"] + bbox["lat_max"]) / 2
            center_lon = (bbox["lon_min"] + bbox["lon_max"]) / 2

            from utils.smart_bin_placer import place_bins
            result  = place_bins(bbox, pop, menages, surface,
                                 food_poi, None, schools, nb_chiens)
            bins_df = result["recommended_bins"]
            nb_rec  = result["bin_count_info"]["nb_recommande"]

        # ── Construire la carte Folium ────────────────────────
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles="CartoDB dark_matter"
        )

        # Ajouter chaque benne recommandée
        for i, row in bins_df.iterrows():
            score_pct = int(min(100, row.get("score", 0) / max(
                bins_df["score"].max() if "score" in bins_df else 1, 0.001) * 100))

            popup_html = f"""
            <div style="font-family:monospace;min-width:180px">
                <b style="color:#3fb950">🗑 Benne #{i+1}</b><br>
                <b>Type :</b> Organique<br>
                <b>Score :</b> {score_pct}%<br>
                <b>Source :</b> {row.get('source','OSM')}<br>
                <b>Lat :</b> {row['lat']:.6f}<br>
                <b>Lon :</b> {row['lon']:.6f}
            </div>"""

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=10,
                color="#3fb950",
                fill=True,
                fill_color="#3fb950",
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"Benne #{i+1} — Score {score_pct}%"
            ).add_to(m)

            # Numéro sur la benne
            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:9px;font-weight:bold;color:#fff;'
                         f'background:#3fb950;border-radius:50%;width:16px;height:16px;'
                         f'display:flex;align-items:center;justify-content:center;'
                         f'margin-top:-8px;margin-left:-8px">{i+1}</div>',
                    icon_size=(16, 16)
                )
            ).add_to(m)

        # Titre sur la carte
        title_html = f"""
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
             z-index:1000;background:rgba(13,17,23,.9);color:#e2e8f0;
             padding:8px 18px;border-radius:8px;font-family:monospace;font-size:13px;
             border:1px solid #3fb950">
            🗑 <b>{nb_rec} bennes recommandées</b>
            {f'— {district}' if district else ''}
        </div>"""
        m.get_root().html.add_child(folium.Element(title_html))

        map_html = m._repr_html_()
        return jsonify({"ok": True, "map_html": map_html,
                        "nb_recommande": nb_rec})

    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e),
                        "trace": traceback.format_exc()}), 400


if __name__ == "__main__":
    print("\n🐕 Stray Dogs API — http://localhost:5000")
    print("   Ouvrir test_client.html dans le navigateur\n")
    app.run(debug=True, port=5000)
