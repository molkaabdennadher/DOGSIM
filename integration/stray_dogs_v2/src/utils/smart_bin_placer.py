# ============================================================
#  SMART BIN PLACER — Recommandation réaliste des bennes
#
#  Problèmes corrigés :
#    - Formule : une seule base (population), les autres sont
#      des ajustements multiplicateurs, pas des additions → réaliste
#    - Emplacements : intersections OSM réelles via Overpass,
#      et si échec → fallback autour des food POIs (pas une grille)
# ============================================================

import requests
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


# ─────────────────────────────────────────────────────────────
# Paramètres calibrés (modifiables)
# ─────────────────────────────────────────────────────────────

# Contexte tunisien : 1 benne organique 700L pour 120 habitants
HABITANTS_PAR_BENNE = 150

# Chaque restaurant génère autant de déchets qu'environ 30 ménages
# → +0.25 benne par restaurant
BENNE_PAR_RESTAURANT = 0.25

# Chaque café → +0.12 benne
BENNE_PAR_CAFE = 0.12

# Bonus densité canine : chaque tranche de 10 chiens = +0.5 benne
BENNE_PAR_20_CHIENS = 1.0

# Capacité de chaque benne
CAPACITE_LITRES = 700

# Limites pour un quartier tunisien
MIN_BENNES = 5
MAX_BENNES = 300

# Distance minimale entre deux bennes : 150m
# (80m donnait des clusters visuels — bins à 90m l'un de l'autre passaient quand même)
MIN_DIST_KM = 0.15

# Rayon de couverture d'une benne (zone "déjà servie") : 350m
# Au-delà de cette distance, une benne ne réduit plus le score de ses voisines
COVERAGE_RADIUS_KM = 0.35

# Rayon de recherche autour d'un food POI pour le fallback : 120m
FALLBACK_FOOD_RADIUS_KM = 0.12


# ─────────────────────────────────────────────────────────────
# Utilitaires géographiques
# ─────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance en km entre deux points GPS."""
    R = 6371.0
    dl = radians(lat2 - lat1)
    dg = radians(lon2 - lon1)
    a  = sin(dl/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dg/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _offset_point(lat, lon, dx_m, dy_m):
    """Décaler un point GPS de dx_m mètres (est) et dy_m mètres (nord)."""
    lat2 = lat + (dy_m / 111320)
    lon2 = lon + (dx_m / (111320 * cos(radians(lat))))
    return lat2, lon2


# ─────────────────────────────────────────────────────────────
# 1. Formule réaliste du nombre de bennes
# ─────────────────────────────────────────────────────────────

def recommend_bin_count(population:     int,
                        nb_menages:     int   = 0,
                        surface_km2:    float = 0.0,
                        nb_restaurants: int   = 0,
                        nb_cafes:       int   = 0,
                        nb_chiens:      float = 0.0) -> dict:
    """
    Formule corrigée — une seule base (population), pas de double-comptage.

    N = base_population + ajust_food + ajust_chiens
    """
    base       = population / HABITANTS_PAR_BENNE
    adj_food   = (nb_restaurants * BENNE_PAR_RESTAURANT
                + nb_cafes       * BENNE_PAR_CAFE)
    adj_chiens = (nb_chiens / 20) * BENNE_PAR_20_CHIENS

    total_brut = base + adj_food + adj_chiens
    nb         = int(np.clip(round(total_brut), MIN_BENNES, MAX_BENNES))

    return {
        "nb_recommande":  nb,
        "total_brut":     round(total_brut, 2),
        "contributions": {
            "base_population": round(base, 1),
            "restaurants":     round(nb_restaurants * BENNE_PAR_RESTAURANT, 1),
            "cafes":           round(nb_cafes * BENNE_PAR_CAFE, 1),
            "bonus_chiens":    round(adj_chiens, 1),
        },
        "formule": (
            f"{population} hab ÷ {HABITANTS_PAR_BENNE} = {round(base,1)} "
            f"+ {nb_restaurants} restos×{BENNE_PAR_RESTAURANT} = {round(nb_restaurants*BENNE_PAR_RESTAURANT,1)} "
            f"+ {nb_cafes} cafés×{BENNE_PAR_CAFE} = {round(nb_cafes*BENNE_PAR_CAFE,1)} "
            f"→ total brut {round(total_brut,1)} → {nb} bennes"
        )
    }


# ─────────────────────────────────────────────────────────────
# 2. Intersections réelles via Overpass
# ─────────────────────────────────────────────────────────────

def fetch_street_intersections(bbox: dict, timeout: int = 25) -> pd.DataFrame:
    """
    Récupère les vraies intersections de rues depuis OpenStreetMap.
    Un nœud intersection = nœud partagé par ≥ 2 segments de route.
    """
    s, n = bbox["lat_min"], bbox["lat_max"]
    w, e = bbox["lon_min"], bbox["lon_max"]

    # Requête robuste : routes locales → nœuds partagés par 2+ voies
    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["highway"~"^(primary|secondary|tertiary|residential|living_street|unclassified)$"]
        ({s},{w},{n},{e});
    )->.roads;
    node(w.roads:2-)({s},{w},{n},{e});
    out body;
    """

    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=timeout + 10
        )
        r.raise_for_status()
        elements = r.json().get("elements", [])

        nodes = [
            {"lat": el["lat"], "lon": el["lon"], "source": "osm_intersection"}
            for el in elements if el["type"] == "node"
        ]

        if len(nodes) >= 5:
            df = pd.DataFrame(nodes)
            print(f"[BinPlacer] OSM : {len(df)} intersections trouvées")
            return df
        else:
            print(f"[BinPlacer] OSM : seulement {len(nodes)} nœuds — fallback activé")
            return pd.DataFrame()

    except Exception as e:
        print(f"[BinPlacer] Overpass indisponible ({e}) — fallback activé")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 3. Fallback intelligent — autour des food POIs
#    (PAS une grille — emplacements liés aux sources réelles
#     de déchets organiques)
# ─────────────────────────────────────────────────────────────

def _smart_fallback(bbox: dict, food_poi_df: pd.DataFrame,
                    nb_needed: int) -> pd.DataFrame:
    """
    Si Overpass échoue, place des candidats autour des restaurants/cafés
    (les sources principales de déchets organiques).

    Logique :
      - Pour chaque food POI, générer 2-3 points candidats légèrement décalés
        (simulant des coins de rue proches)
      - Compléter avec quelques points répartis dans la zone si insuffisant
    """
    candidates = []

    # Autour des food POIs
    if food_poi_df is not None and not food_poi_df.empty:
        offsets = [
            (80, 30), (-80, 30), (30, 80), (-30, -80),
            (60, -60), (-60, 60)
        ]
        for _, poi in food_poi_df.iterrows():
            for dx, dy in offsets[:3]:   # 3 candidats par POI
                lat2, lon2 = _offset_point(poi["lat"], poi["lon"], dx, dy)
                # Vérifier que le point est dans la bbox
                if (bbox["lat_min"] < lat2 < bbox["lat_max"] and
                        bbox["lon_min"] < lon2 < bbox["lon_max"]):
                    candidates.append({
                        "lat": lat2, "lon": lon2,
                        "source": "near_food_poi"
                    })

    # Compléter si pas assez de candidats — points le long des axes
    # (pas une grille, mais une dispersion orientée)
    if len(candidates) < nb_needed * 3:
        cx = (bbox["lon_min"] + bbox["lon_max"]) / 2
        cy = (bbox["lat_min"] + bbox["lat_max"]) / 2
        dlon = (bbox["lon_max"] - bbox["lon_min"])
        dlat = (bbox["lat_max"] - bbox["lat_min"])

        angles = np.linspace(0, 2 * np.pi, nb_needed * 4, endpoint=False)
        for angle in angles:
            # Spirale vers l'extérieur depuis le centre
            r_lat = dlat * 0.35 * (0.3 + 0.7 * abs(np.sin(angle * 1.5)))
            r_lon = dlon * 0.35 * (0.3 + 0.7 * abs(np.cos(angle * 1.5)))
            lat2 = cy + r_lat * np.sin(angle)
            lon2 = cx + r_lon * np.cos(angle)
            if (bbox["lat_min"] < lat2 < bbox["lat_max"] and
                    bbox["lon_min"] < lon2 < bbox["lon_max"]):
                candidates.append({
                    "lat": lat2, "lon": lon2,
                    "source": "zone_coverage"
                })

    if not candidates:
        return pd.DataFrame()

    df = pd.DataFrame(candidates).drop_duplicates(subset=["lat", "lon"])
    print(f"[BinPlacer] Fallback : {len(df)} candidats générés")
    return df


# ─────────────────────────────────────────────────────────────
# 4. Scoring des candidats
# ─────────────────────────────────────────────────────────────

def score_candidates(candidates: pd.DataFrame,
                     food_poi_df: pd.DataFrame,
                     dog_density_df: pd.DataFrame,
                     schools_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score initial de chaque candidat — utilisé comme point de départ
    pour la sélection gloutonne dans select_bins().

    Logique corrigée :
      - Score de base 5.0 pour TOUS les candidats → couverture de zone
        assurée même loin des restaurants
      - Bonus food POI (max +2.5 par POI, rayon 300m) → facteur secondaire
      - Bonus densité canine modéré
      - Pénalité école forte (< 50m)

    Le regroupement autour des restaurants était causé par un score de base
    nul + bonus food de 10 : tous les bins allaient vers les POIs.
    Avec base 5.0 et bonus max 2.5, un candidat sans restaurant proche
    conserve un score compétitif, et la distribution dans la zone est naturelle.
    """
    if candidates.empty:
        return candidates

    lats = candidates["lat"].values
    lons = candidates["lon"].values
    # Score de base uniforme → chaque candidat a une valeur intrinsèque
    scores = np.full(len(lats), 5.0)

    # Bonus food POI — facteur secondaire (préférence, pas dominance)
    if food_poi_df is not None and not food_poi_df.empty:
        for _, poi in food_poi_df.iterrows():
            for i in range(len(lats)):
                d = haversine(lats[i], lons[i], poi["lat"], poi["lon"])
                if d < 0.30:
                    # Bonus max +2.5 (vs 10 avant) sur un rayon plus large (300m vs 250m)
                    scores[i] += (1.0 - d / 0.30) * 2.5

    # Bonus densité canine — signal modéré
    if dog_density_df is not None and not dog_density_df.empty:
        for _, z in dog_density_df.iterrows():
            for i in range(len(lats)):
                d = haversine(lats[i], lons[i], float(z["lat"]), float(z["lon"]))
                if d < 0.3:
                    scores[i] += float(z.get("nb_chiens", 1)) * (1.0 - d / 0.3) * 0.15

    # Pénalité école (< 50m) — éliminatoire
    if schools_df is not None and not schools_df.empty:
        for _, sc in schools_df.iterrows():
            for i in range(len(lats)):
                d = haversine(lats[i], lons[i], sc["lat"], sc["lon"])
                if d < 0.05:
                    scores[i] -= 100

    res = candidates.copy()
    res["score"] = scores
    return res.sort_values("score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 5. Sélection finale avec contrainte de distance minimale
# ─────────────────────────────────────────────────────────────

def select_bins(scored: pd.DataFrame, nb: int) -> pd.DataFrame:
    """
    Sélection gloutonne avec pénalité de couverture dynamique.

    Algorithme corrigé :
      Après chaque benne placée, on applique une pénalité aux candidats
      restants dans un rayon de COVERAGE_RADIUS_KM (250m). Cela "repousse"
      les bennes suivantes vers des zones encore non couvertes.

      Ancien problème : simple filtre à 80m → dans un cluster de restaurants,
      les candidats étaient tous à >80m les uns des autres mais tous à <250m
      du même point → résultat : 5 bennes dans le même quartier.

      Avec la pénalité dynamique, chaque benne placée réduit l'attractivité
      de son voisinage, forçant les bennes suivantes à se disperser.
    """
    if scored.empty:
        return pd.DataFrame()

    # Travailler sur un tableau mutable des scores
    remaining = scored.copy().reset_index(drop=True)
    remaining["dyn_score"] = remaining["score"].values.copy()
    selected = []

    while len(selected) < nb and not remaining.empty:
        # Exclure les candidats trop proches des bennes déjà placées
        if selected:
            mask_ok = pd.Series(True, index=remaining.index)
            for s in selected:
                dists = remaining.apply(
                    lambda r: haversine(r["lat"], r["lon"], s["lat"], s["lon"]), axis=1
                )
                mask_ok = mask_ok & (dists >= MIN_DIST_KM)
            remaining = remaining[mask_ok].reset_index(drop=True)
            if remaining.empty:
                break

        # Trier par score dynamique et choisir le meilleur
        remaining = remaining.sort_values("dyn_score", ascending=False).reset_index(drop=True)
        best = remaining.iloc[0]

        selected.append({
            "lat":    round(float(best["lat"]), 6),
            "lon":    round(float(best["lon"]), 6),
            "score":  round(float(best["dyn_score"]), 3),
            "type":   "organique",
            "statut": "recommandé",
            "source": best.get("source", "osm"),
        })

        # Pénalité de couverture : réduire fortement le score des voisins dans COVERAGE_RADIUS_KM
        # Pénalité max = 15.0 → score_base 5.0 - 15.0 = -10 → candidats voisins
        # deviennent non-compétitifs et ne seront choisis qu'en dernier recours,
        # forçant les bennes suivantes à aller dans des zones non couvertes.
        dists_to_best = remaining.apply(
            lambda r: haversine(r["lat"], r["lon"], float(best["lat"]), float(best["lon"])),
            axis=1
        )
        in_coverage = dists_to_best < COVERAGE_RADIUS_KM
        # Pénalité proportionnelle à la proximité (max -15.0 au centre, 0 au bord)
        penalty = (1.0 - dists_to_best[in_coverage] / COVERAGE_RADIUS_KM) * 15.0
        remaining.loc[in_coverage, "dyn_score"] -= penalty

        # Retirer le candidat choisi
        remaining = remaining.iloc[1:].reset_index(drop=True)

    print(f"[BinPlacer] {len(selected)} bennes sélectionnées (demandées : {nb})")
    return pd.DataFrame(selected) if selected else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 6. Fonction principale
# ─────────────────────────────────────────────────────────────

def place_bins(bbox:            dict,
               population:     int,
               nb_menages:     int   = 0,
               surface_km2:    float = 0.0,
               food_poi_df:    pd.DataFrame = None,
               dog_density_df: pd.DataFrame = None,
               schools_df:     pd.DataFrame = None,
               nb_chiens:      float = 0.0) -> dict:
    """
    Pipeline complet : formule → candidats OSM (ou fallback POI) → scoring → sélection.
    """
    # Compter les food POI
    nb_restaurants = nb_cafes = 0
    if food_poi_df is not None and not food_poi_df.empty:
        col = food_poi_df.get("type", food_poi_df.get("category", pd.Series(dtype=str)))
        nb_restaurants = int(col.str.contains("restaurant|fast_food|bakery|snack",
                                               case=False, na=False).sum())
        nb_cafes       = int(col.str.contains("cafe|bar|coffee",
                                               case=False, na=False).sum())

    # 1. Nombre recommandé
    count_info = recommend_bin_count(
        population     = population,
        nb_menages     = nb_menages,
        surface_km2    = surface_km2,
        nb_restaurants = nb_restaurants,
        nb_cafes       = nb_cafes,
        nb_chiens      = nb_chiens
    )
    nb = count_info["nb_recommande"]
    print(f"[BinPlacer] Nombre recommandé : {nb}")
    print(f"[BinPlacer] {count_info['formule']}")

    # 2. Candidats : intersections OSM en premier
    candidates = fetch_street_intersections(bbox)

    # Si OSM a échoué ou retourné trop peu → fallback intelligent
    if candidates.empty or len(candidates) < nb:
        fallback = _smart_fallback(bbox, food_poi_df, nb)
        if candidates.empty:
            candidates = fallback
        else:
            candidates = pd.concat([candidates, fallback], ignore_index=True)

    # 3. Scoring
    scored = score_candidates(candidates, food_poi_df, dog_density_df, schools_df)

    # 4. Sélection finale
    final = select_bins(scored, nb)

    return {
        "bin_count_info":      count_info,
        "recommended_bins":    final,
        "intersections_count": len(candidates),
        "nb_restaurants":      nb_restaurants,
        "nb_cafes":            nb_cafes,
    }
