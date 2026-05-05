# =============================================================
# pipeline.py
# Pipeline principal : input quartier → 3 solutions
#
# Input  : nom du quartier + (infos complémentaires) + (PNG optionnel)
# Output : S1 nb chiens, S2 bennes optimisées, S3 feeding zones
# =============================================================

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import json

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MDL_DIR, DATA_PRO, DATA_OUT, VIZ_DIR,
    RATIO_BASE, ZONE_COEFF, DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
    GRID_STEP_DEG, N_FEEDING_ZONES, MIN_DIST_FZ_DEG,
    AHP_WEIGHTS, BIN_ORGANIC_RATIO, BIN_CAPACITIES,
    BIN_CAPACITY_PROBA, POINT_NOIR_QUANTILE, RANDOM_SEED
)


# ── Chargement des modèles entraînés ─────────────────────────

def load_models():
    """
    Charge les modèles entraînés depuis models/trained/.
    Retourne None si pas encore entraînés.
    """
    models = {}
    files  = {
        "rf_reg":  "rf_regressor.joblib",
        "gb_reg":  "gb_regressor.joblib",
        "rf_clf":  "rf_classifier.joblib",
        "scaler":  "scaler.joblib",
    }
    for key, fname in files.items():
        path = MDL_DIR / fname
        if path.exists():
            models[key] = joblib.load(path)
        else:
            models[key] = None

    if not all(v is not None for v in models.values()):
        missing = [k for k, v in models.items() if v is None]
        print(f"[PIPELINE] Modèles manquants : {missing}")
        print("[PIPELINE] Lancer : python src/models/train_models.py")
    else:
        print("[PIPELINE] Modèles chargés ✓")

    return models


# ── SOLUTION 1 : Estimation du nombre de chiens ───────────────

def solution1_dog_density(
    population,
    nb_menages,
    surface_km2,
    nb_food_poi,
    nb_bins_org,
    zone_type="residential",
    dist_centre=0.5,
    models=None,
):
    """
    Estime le nombre de chiens et le niveau de risque pour un secteur.

    Args:
        population   : nombre d'habitants
        nb_menages   : nombre de ménages
        surface_km2  : surface du secteur
        nb_food_poi  : nombre de POI alimentaires
        nb_bins_org  : nombre de bennes organiques actuelles
        zone_type    : type de zone (residential, commercial, ...)
        dist_centre  : distance au centre [0, 1]
        models       : dict des modèles (None = estimation manuelle)

    Returns:
        dict {nb_chiens, risk_class, risk_label, confidence, details}
    """
    # Features
    taille_menage = population / max(nb_menages, 1)
    densite_pop   = population / max(surface_km2, 0.01)
    zone_coeff    = ZONE_COEFF.get(zone_type, 1.0)

    features = np.array([[
        population, nb_menages, taille_menage, surface_km2,
        densite_pop, nb_food_poi, nb_bins_org, dist_centre, zone_coeff
    ]])

    # ── Prédiction avec modèles entraînés ─────────────────────
    if models and models.get("rf_reg") and models.get("scaler"):
        scaler      = models["scaler"]
        rf_reg      = models["rf_reg"]
        gb_reg      = models.get("gb_reg")
        rf_clf      = models.get("rf_clf")

        X_scaled = scaler.transform(features)

        # Prédiction régression (ensemble RF + GB)
        pred_rf = float(rf_reg.predict(X_scaled)[0])
        pred_gb = float(gb_reg.predict(X_scaled)[0]) if gb_reg else pred_rf
        nb_chiens = int((pred_rf * 0.6 + pred_gb * 0.4))  # ensemble

        # Prédiction classification
        if rf_clf:
            risk_class  = int(rf_clf.predict(X_scaled)[0])
            risk_proba  = rf_clf.predict_proba(X_scaled)[0]
            confidence  = float(risk_proba.max())
        else:
            risk_class = _rule_based_risk(nb_chiens, population)
            confidence = 0.75

        source = "RF+GB (ensemble)"

    else:
        # ── Fallback : estimation manuelle ─────────────────────
        nb_chiens = int(
            population * RATIO_BASE * zone_coeff +
            nb_food_poi * 2.0 +
            nb_bins_org * 3.0
        )
        risk_class = _rule_based_risk(nb_chiens, population)
        confidence = 0.70
        source     = "estimation manuelle"

    risk_labels = {0: "Faible", 1: "Moyen", 2: "Élevé"}
    risk_colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    return {
        "nb_chiens":   max(0, nb_chiens),
        "risk_class":  risk_class,
        "risk_label":  risk_labels[risk_class],
        "risk_color":  risk_colors[risk_class],
        "confidence":  round(confidence, 3),
        "source":      source,
        "details": {
            "ratio_chien_habitant":   f"1 chien / {int(population/max(nb_chiens,1))} habitants",
            "zone_coeff":             zone_coeff,
            "contribution_food_poi":  nb_food_poi * 2,
            "contribution_bennes":    nb_bins_org * 3,
        }
    }


def _rule_based_risk(nb_chiens, population):
    ratio = nb_chiens / max(population, 1)
    if ratio < 0.05:
        return 0
    elif ratio < 0.10:
        return 1
    else:
        return 2


# ── SOLUTION 2 : Recommandation intelligente des bennes ───────

def solution2_bins_optimization(
    bins_df,
    dog_density_map,
    bbox,
    # Nouveaux paramètres pour la formule de recommandation
    population:     int   = 0,
    nb_menages:     int   = 0,
    surface_km2:    float = 0.0,
    food_poi_df     = None,
    schools_df      = None,
    nb_chiens:      float = 0.0,
):
    """
    Recommande le nombre de bennes organiques et leurs emplacements
    optimaux aux intersections de rues réelles (OSM).

    Nouveautés vs ancienne version :
      - Formule multicritère pour le nombre : pop + ménages + surface
        + restaurants + cafés + bonus densité canine
      - Emplacements aux vrais coins de rue (Overpass API)
      - Score de chaque intersection selon food POI + densité chiens
        - pénalité école
      - Contrainte de distance minimale entre bennes (80m)

    Returns:
        dict {
            bin_count_info   : formule + contributions détaillées,
            recommended_bins : DataFrame lat/lon/score des bennes,
            black_spots      : zones à forte densité canine (héritage S2),
            stats            : statistiques complètes
        }
    """
    from utils.smart_bin_placer import place_bins

    # ── Compatibilité avec l'ancien code (black spots) ────────
    black_spots = pd.DataFrame()
    if bins_df is not None and not bins_df.empty:
        org_bins = bins_df[bins_df["type"].str.lower().str.contains("organ",
                           na=False)].copy()
        if org_bins.empty:
            org_bins = bins_df.copy()
            org_bins["type"] = "organique"

        if dog_density_map is not None and not dog_density_map.empty:
            density_arr = dog_density_map[["lat", "lon", "nb_chiens"]].values

            def nearest_density(lat, lon):
                dists = np.sqrt((density_arr[:, 0] - lat) ** 2 +
                                (density_arr[:, 1] - lon) ** 2)
                return float(density_arr[dists.argmin(), 2])

            org_bins["dog_density"] = org_bins.apply(
                lambda r: nearest_density(r["lat"], r["lon"]), axis=1)
            seuil = org_bins["dog_density"].quantile(POINT_NOIR_QUANTILE)
            org_bins["black_spot"] = org_bins["dog_density"] >= seuil
            black_spots = org_bins[org_bins["black_spot"]]

    # ── Nouveau : placement intelligent ───────────────────────
    placement = place_bins(
        bbox           = bbox,
        population     = population,
        nb_menages     = nb_menages,
        surface_km2    = surface_km2,
        food_poi_df    = food_poi_df,
        dog_density_df = dog_density_map,
        schools_df     = schools_df,
        nb_chiens      = nb_chiens
    )

    count_info     = placement["bin_count_info"]
    recommended    = placement["recommended_bins"]
    nb_recommande  = count_info["nb_recommande"]

    stats = {
        # Nouveau
        "nb_recommande":       nb_recommande,
        "formule":             count_info["formule"],
        "contributions":       count_info["contributions"],
        "intersections_osm":   placement["intersections_count"],
        "nb_restaurants":      placement["nb_restaurants"],
        "nb_cafes":            placement["nb_cafes"],
        # Héritage
        "black_spots":         len(black_spots),
        "bennes_placees":      len(recommended),
    }

    return {
        "bin_count_info":   count_info,
        "recommended_bins": recommended,
        "black_spots":      black_spots,
        "stats":            stats,
    }


# ── SOLUTION 3 : Feeding Zones ────────────────────────────────

def solution3_feeding_zones(
    bbox,
    dog_density_map,
    food_poi_df,
    schools_df,
    parks_df,
    org_bins_df,
    n_zones=N_FEEDING_ZONES,
):
    """
    Identifie les meilleures feeding zones via score AHP.

    Returns:
        DataFrame des N meilleures feeding zones avec scores.
    """
    # Grille
    lat_range = np.arange(bbox["lat_min"], bbox["lat_max"], GRID_STEP_DEG)
    lon_range = np.arange(bbox["lon_min"], bbox["lon_max"], GRID_STEP_DEG)
    grid_pts  = [{"lat": la, "lon": lo}
                 for la in lat_range for lo in lon_range]
    df_grid   = pd.DataFrame(grid_pts)

    if df_grid.empty:
        return pd.DataFrame()

    # Arrays de référence
    def to_arr(df, cols=["lat", "lon"]):
        if df is not None and not df.empty and all(c in df.columns for c in cols):
            return df[cols].values
        return np.empty((0, 2))

    density_arr  = dog_density_map[["lat", "lon", "nb_chiens"]].values \
                   if dog_density_map is not None and not dog_density_map.empty \
                   else None
    max_density  = float(density_arr[:, 2].max()) if density_arr is not None else 1.0

    food_arr    = to_arr(food_poi_df)
    school_arr  = to_arr(schools_df)
    park_arr    = to_arr(parks_df)
    bins_arr    = to_arr(org_bins_df)

    def min_dist_km(lat, lon, arr):
        if arr is None or len(arr) == 0:
            return 99.0
        dists = np.sqrt((arr[:, 0] - lat) ** 2 + (arr[:, 1] - lon) ** 2)
        return float(dists.min()) * 111.0

    scores = []
    for _, cell in df_grid.iterrows():
        la, lo = cell["lat"], cell["lon"]

        # 1. Densité canine (élevée = bien) ──────────────────
        if density_arr is not None:
            d_c = np.sqrt((density_arr[:, 0] - la) ** 2 +
                          (density_arr[:, 1] - lo) ** 2)
            s_dog = float(density_arr[d_c.argmin(), 2]) / max_density
        else:
            s_dog = 0.5

        # 2. Distance écoles (loin = bien) ───────────────────
        d_school = min_dist_km(la, lo, school_arr)
        s_school = min(d_school / 0.5, 1.0)

        # 3. Proximité parc/espace vert (proche = bien) ──────
        d_park   = min_dist_km(la, lo, park_arr)
        s_park   = max(0.0, 1.0 - d_park / 1.0)

        # 4. Distance food POI (loin = bien) ─────────────────
        d_food   = min_dist_km(la, lo, food_arr)
        s_food   = min(d_food / 0.3, 1.0)

        # 5. Distance bennes organiques (loin = bien) ────────
        d_bins   = min_dist_km(la, lo, bins_arr)
        s_bins   = min(d_bins / 0.2, 1.0)

        # 6. Dans la bbox (accès route simulé) ───────────────
        s_road = 1.0 if (bbox["lat_min"] + 0.002 < la < bbox["lat_max"] - 0.002 and
                         bbox["lon_min"] + 0.002 < lo < bbox["lon_max"] - 0.002) else 0.0

        score = (
            AHP_WEIGHTS["dog_density"]   * s_dog   +
            AHP_WEIGHTS["dist_schools"]  * s_school +
            AHP_WEIGHTS["near_green"]    * s_park   +
            AHP_WEIGHTS["dist_food_poi"] * s_food   +
            AHP_WEIGHTS["dist_org_bins"] * s_bins   +
            AHP_WEIGHTS["road_access"]   * s_road
        )
        scores.append(score)

    df_grid["score"] = scores

    # Sélection avec diversité spatiale
    df_sorted = df_grid.nlargest(500, "score")
    selected  = []

    for _, cand in df_sorted.iterrows():
        if len(selected) >= n_zones:
            break
        if not selected:
            selected.append(cand)
            continue
        fz_arr = np.array([[fz["lat"], fz["lon"]] for fz in selected])
        dists  = np.sqrt((fz_arr[:, 0] - cand["lat"]) ** 2 +
                         (fz_arr[:, 1] - cand["lon"]) ** 2)
        if dists.min() >= MIN_DIST_FZ_DEG:
            selected.append(cand)

    if not selected:
        return pd.DataFrame()

    df_fz = pd.DataFrame(selected).reset_index(drop=True)
    df_fz["zone_id"]   = [f"FZ-{i+1:02d}" for i in range(len(df_fz))]
    df_fz["score_pct"] = (df_fz["score"] * 100).round(1)
    df_fz["rayon_m"]   = 400

    return df_fz, df_grid


# ── Pipeline complet ─────────────────────────────────────────

def run_pipeline(
    district_name,
    population,
    nb_menages,
    surface_km2,
    bbox,
    bins_df       = None,
    food_poi_df   = None,
    schools_df    = None,
    parks_df      = None,
    zone_type     = "residential",
    n_zones       = N_FEEDING_ZONES,
    models        = None,
    verbose       = True,
):
    """
    Pipeline complet : 1 appel → 3 solutions.

    Returns:
        dict {s1, s2, s3, meta}
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"PIPELINE — {district_name}")
        print(f"{'='*60}")

    if bins_df is None:
        bins_df = pd.DataFrame()
    if food_poi_df is None:
        food_poi_df = pd.DataFrame()
    if schools_df is None:
        schools_df = pd.DataFrame()
    if parks_df is None:
        parks_df = pd.DataFrame()

    nb_bins_org = len(bins_df[bins_df.get("type", pd.Series()).str.lower().str.contains("organ", na=False)]) \
                  if not bins_df.empty else 0
    nb_food_poi = len(food_poi_df)

    # ── S1 ────────────────────────────────────────────────────
    if verbose:
        print("\n[S1] Estimation densité canine…")
    s1 = solution1_dog_density(
        population=population,
        nb_menages=nb_menages,
        surface_km2=surface_km2,
        nb_food_poi=nb_food_poi,
        nb_bins_org=nb_bins_org,
        zone_type=zone_type,
        models=models,
    )
    if verbose:
        print(f"  → {s1['nb_chiens']} chiens estimés | Risque : {s1['risk_label']} ({s1['confidence']*100:.0f}%)")

    # Carte densité simple (grille uniforme)
    lat_c = (bbox["lat_min"] + bbox["lat_max"]) / 2
    lon_c = (bbox["lon_min"] + bbox["lon_max"]) / 2
    dog_map = pd.DataFrame([{
        "lat": lat_c, "lon": lon_c,
        "nb_chiens": s1["nb_chiens"],
        "district": district_name,
    }])

    # ── S2 ────────────────────────────────────────────────────
    if verbose:
        print("\n[S2] Optimisation des bennes…")
    s2 = solution2_bins_optimization(
        bins_df        = bins_df,
        dog_density_map= dog_map,
        bbox           = bbox,
        population     = int(population),
        nb_menages     = int(nb_menages),
        surface_km2    = float(surface_km2),
        food_poi_df    = food_poi_df,
        schools_df     = schools_df,
        nb_chiens      = float(s1["nb_chiens"]),
    )
    if verbose and s2.get("stats"):
        st = s2["stats"]
        print(f"  → {st.get('nb_recommande', 0)} bennes recommandées | "
              f"{st.get('bennes_placees', 0)} placées | "
              f"{st.get('black_spots', 0)} points noirs")

    # ── S3 ────────────────────────────────────────────────────
    if verbose:
        print("\n[S3] Identification feeding zones…")
    s3_result = solution3_feeding_zones(
        bbox=bbox,
        dog_density_map=dog_map,
        food_poi_df=food_poi_df,
        schools_df=schools_df,
        parks_df=parks_df,
        org_bins_df=bins_df,
        n_zones=n_zones,
    )

    if isinstance(s3_result, tuple):
        df_fz, df_grid = s3_result
    else:
        df_fz, df_grid = s3_result, pd.DataFrame()

    if verbose and not df_fz.empty:
        print(f"  → {len(df_fz)} feeding zones | Meilleur score : {df_fz['score_pct'].max()}%")

    return {
        "s1": s1,
        "s2": s2,
        "s3": {"feeding_zones": df_fz, "grid": df_grid},
        "meta": {
            "district":    district_name,
            "population":  population,
            "nb_menages":  nb_menages,
            "surface_km2": surface_km2,
            "bbox":        bbox,
        }
    }
