# =============================================================
# osm_fetcher.py
# Récupération des données OpenStreetMap pour un quartier donné
#
# Fonctions :
#   - get_district_from_name()  : bbox depuis le nom du quartier
#   - get_poi()                 : restaurants, cafés, écoles, parcs
#   - get_road_network()        : réseau routier
#   - get_all_data()            : pipeline complet
# =============================================================

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OSM_FOOD_TAGS, OSM_SCHOOL_TAGS, OSM_PARK_TAGS, DATA_PRO

# ── Dépendances OSM (optionnelles) ───────────────────────────

def _check_osmnx():
    try:
        import osmnx as ox
        return ox
    except ImportError:
        return None

def _check_requests():
    try:
        import requests
        return requests
    except ImportError:
        return None


# ── 1. Récupération de la bbox d'un quartier ──────────────────

def get_district_bbox(district_name, country="Tunisia"):
    """
    Retourne la bounding box d'un quartier via Nominatim (OSM).

    Args:
        district_name : ex "La Marsa", "Sousse Medina", "Carthage"
        country       : pays (default: Tunisia)

    Returns:
        dict {lat_min, lat_max, lon_min, lon_max, center_lat, center_lon}
        ou None si introuvable
    """
    requests = _check_requests()
    if not requests:
        print("[OSM] requests non installé → bbox simulée")
        return None

    query  = f"{district_name}, {country}"
    url    = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": "StrayDogsTunisia/1.0"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()

        if not data:
            print(f"[OSM] Quartier non trouvé : '{query}'")
            return None

        item     = data[0]
        bb       = item.get("boundingbox", [])
        if len(bb) < 4:
            return None

        lat_min, lat_max = float(bb[0]), float(bb[1])
        lon_min, lon_max = float(bb[2]), float(bb[3])

        return {
            "lat_min":    lat_min,
            "lat_max":    lat_max,
            "lon_min":    lon_min,
            "lon_max":    lon_max,
            "center_lat": (lat_min + lat_max) / 2,
            "center_lon": (lon_min + lon_max) / 2,
            "name":       item.get("display_name", district_name),
        }

    except Exception as e:
        print(f"[OSM] Erreur Nominatim : {e}")
        return None


# ── 2. Récupération des POI via Overpass ──────────────────────

def _parse_overpass_elements(elements):
    """Convertit la liste d'éléments Overpass en DataFrame catégorisé."""
    rows = []
    for el in elements:
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if not lat or not lon:
            continue

        tags    = el.get("tags", {})
        amenity = tags.get("amenity", "")
        leisure = tags.get("leisure", "")
        shop    = tags.get("shop", "")
        tag_val = amenity or leisure or shop

        if amenity in OSM_FOOD_TAGS or shop in ["supermarket", "convenience", "butcher", "bakery"]:
            category = "food"
        elif amenity in OSM_SCHOOL_TAGS:
            category = "school"
        elif leisure in OSM_PARK_TAGS:
            category = "park"
        else:
            category = "other"

        rows.append({
            "lat":      lat,
            "lon":      lon,
            "type":     tag_val,
            "nom":      tags.get("name", "Inconnu"),
            "category": category,
            "source":   "osm",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_poi_overpass(bbox, timeout=25):
    """
    Récupère les POI (restaurants, cafés, écoles, parcs, marchés)
    depuis l'API Overpass. Essaie 4 miroirs en POST puis GET.

    Returns:
        DataFrame avec colonnes [lat, lon, type, nom, category]
    """
    requests = _check_requests()
    if not requests:
        return pd.DataFrame()

    bb_str = f"{bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']}"

    food_str   = "|".join(OSM_FOOD_TAGS)
    school_str = "|".join(OSM_SCHOOL_TAGS)
    park_str   = "|".join(OSM_PARK_TAGS)

    query = (
        f"[out:json][timeout:{timeout}];"
        f"("
        f'node["amenity"~"{food_str}"]({bb_str});'
        f'way["amenity"~"{food_str}"]({bb_str});'
        f'node["shop"~"supermarket|convenience|butcher|bakery"]({bb_str});'
        f'node["amenity"~"{school_str}"]({bb_str});'
        f'way["amenity"~"{school_str}"]({bb_str});'
        f'node["leisure"~"{park_str}"]({bb_str});'
        f'way["leisure"~"{park_str}"]({bb_str});'
        f");out center;"
    )

    headers = {"User-Agent": "DOGSIM-TN/1.0 (stray-dog-management; Tunisia)"}

    mirrors = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]

    for mirror in mirrors:
        for method in ("POST", "GET"):
            try:
                print(f"[OSM] {method} → {mirror}")
                if method == "POST":
                    resp = requests.post(
                        mirror, data={"data": query},
                        headers=headers, timeout=timeout + 5
                    )
                else:
                    resp = requests.get(
                        mirror, params={"data": query},
                        headers=headers, timeout=timeout + 5
                    )

                if resp.status_code == 429:
                    print(f"[OSM] {mirror} → 429 rate-limit, miroir suivant")
                    break  # inutile d'essayer GET sur le même miroir

                resp.raise_for_status()
                payload = resp.json()
                elements = payload.get("elements", [])

                if not elements:
                    print(f"[OSM] {mirror} → 0 éléments")
                    break  # même miroir en GET ne donnera pas mieux

                df = _parse_overpass_elements(elements)
                if df.empty:
                    break

                n_f = len(df[df["category"] == "food"])
                n_s = len(df[df["category"] == "school"])
                n_p = len(df[df["category"] == "park"])
                print(f"[OSM] ✓ {len(df)} POI — {n_f} food · {n_s} écoles · {n_p} parcs")
                return df

            except Exception as e:
                print(f"[OSM] Erreur {method} {mirror} : {e}")
                continue  # essaie GET sur le même miroir, puis miroir suivant

    print("[OSM] Tous les miroirs Overpass ont échoué → fallback simulation")
    return pd.DataFrame()


# ── 3. Pipeline complet ───────────────────────────────────────

def get_district_data(district_name=None, bbox=None, country="Tunisia"):
    """
    Pipeline complet : nom du quartier → toutes les données OSM.

    Args:
        district_name : nom du quartier (ex: "La Marsa")
        bbox          : dict bbox si déjà connu (optionnel)
        country       : pays

    Returns:
        dict {
            "bbox"      : bounding box,
            "poi"       : DataFrame POI,
            "food_poi"  : POI alimentaires seulement,
            "schools"   : écoles seulement,
            "parks"     : parcs seulement,
        }
    """
    # ── Récupération de la bbox ─────────────────────────────
    if bbox is None:
        if district_name is None:
            raise ValueError("Fournir district_name ou bbox")
        print(f"[OSM] Recherche : '{district_name}, {country}'")
        bbox = get_district_bbox(district_name, country)

        if bbox is None:
            print(f"[OSM] Quartier introuvable → bbox simulée pour La Marsa")
            bbox = {
                "lat_min": 36.862, "lat_max": 36.930,
                "lon_min": 10.295, "lon_max": 10.350,
                "center_lat": 36.878, "center_lon": 10.322,
                "name": district_name or "Quartier inconnu",
            }

    print(f"[OSM] Zone : {bbox.get('name', 'bbox fournie')}")
    print(f"[OSM] Bbox : lat [{bbox['lat_min']:.3f}, {bbox['lat_max']:.3f}] "
          f"lon [{bbox['lon_min']:.3f}, {bbox['lon_max']:.3f}]")

    # ── Récupération des POI ────────────────────────────────
    df_poi = get_poi_overpass(bbox)

    if df_poi.empty:
        print("[OSM] Aucun POI trouvé → génération simulée")
        df_poi = _simulate_poi(bbox)

    food_poi = df_poi[df_poi["category"] == "food"].reset_index(drop=True)
    schools  = df_poi[df_poi["category"] == "school"].reset_index(drop=True)
    parks    = df_poi[df_poi["category"] == "park"].reset_index(drop=True)

    # ── Sauvegarde ─────────────────────────────────────────
    name_slug = (district_name or "district").replace(" ", "_").lower()
    DATA_PRO.mkdir(parents=True, exist_ok=True)
    df_poi.to_csv(DATA_PRO / f"{name_slug}_poi.csv", index=False)

    return {
        "bbox":     bbox,
        "poi":      df_poi,
        "food_poi": food_poi,
        "schools":  schools,
        "parks":    parks,
    }


# ── 4. Simulation si pas d'accès internet ─────────────────────

def _simulate_poi(bbox, seed=42):
    """Génère des POI simulés dans la bbox si pas d'accès internet."""
    np.random.seed(seed)
    lat_c = (bbox["lat_min"] + bbox["lat_max"]) / 2
    lon_c = (bbox["lon_min"] + bbox["lon_max"]) / 2
    dlat  = (bbox["lat_max"] - bbox["lat_min"]) * 0.4
    dlon  = (bbox["lon_max"] - bbox["lon_min"]) * 0.4

    rows = []
    configs = [
        ("restaurant",   "food",   20),
        ("cafe",         "food",   12),
        ("marketplace",  "food",    3),
        ("school",       "school",  6),
        ("park",         "park",    4),
    ]
    for poi_type, cat, n in configs:
        for i in range(n):
            rows.append({
                "lat":      lat_c + np.random.uniform(-dlat, dlat),
                "lon":      lon_c + np.random.uniform(-dlon, dlon),
                "type":     poi_type,
                "nom":      f"{poi_type.capitalize()} {i+1}",
                "category": cat,
                "source":   "simulé",
            })

    df = pd.DataFrame(rows)
    print(f"[OSM] POI simulés : {len(df)}")
    return df


# ── 5. Extraction depuis PNG (vision) ─────────────────────────

def extract_elements_from_png(png_path, bbox):
    """
    Extrait les éléments placés sur un PNG de carte.
    Utilise la détection de couleurs (marqueurs colorés standards).

    Convention de couleurs :
        Rouge   → bennes organiques
        Bleu    → bennes plastique
        Orange  → restaurants/cafés
        Vert    → parcs/espaces verts
        Jaune   → écoles

    Args:
        png_path : chemin vers le PNG
        bbox     : dict {lat_min, lat_max, lon_min, lon_max}

    Returns:
        dict {bins_org, bins_plas, food_poi, parks, schools}
    """
    try:
        from PIL import Image
        import numpy as np

        img  = Image.open(png_path).convert("RGB")
        arr  = np.array(img)
        h, w = arr.shape[:2]

        def pixel_to_gps(px, py):
            """Convertit coordonnées pixel → GPS."""
            lon = bbox["lon_min"] + (px / w) * (bbox["lon_max"] - bbox["lon_min"])
            lat = bbox["lat_max"] - (py / h) * (bbox["lat_max"] - bbox["lat_min"])
            return lat, lon

        def find_color_pixels(arr, r_range, g_range, b_range, min_area=5):
            """Trouve les pixels d'une couleur et retourne les centroïdes."""
            mask = (
                (arr[:, :, 0] >= r_range[0]) & (arr[:, :, 0] <= r_range[1]) &
                (arr[:, :, 1] >= g_range[0]) & (arr[:, :, 1] <= g_range[1]) &
                (arr[:, :, 2] >= b_range[0]) & (arr[:, :, 2] <= b_range[1])
            )
            ys, xs = np.where(mask)
            if len(xs) < min_area:
                return []

            # Clustering simple pour trouver les centroïdes
            points = np.column_stack([xs, ys])
            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=15, min_samples=3).fit(points)
            centroids = []
            for cl in set(db.labels_) - {-1}:
                mask_cl = db.labels_ == cl
                cx = int(points[mask_cl, 0].mean())
                cy = int(points[mask_cl, 1].mean())
                centroids.append((cx, cy))
            return centroids

        results = {}

        # Bennes organiques (rouge)
        centers = find_color_pixels(arr, (180, 255), (0, 80), (0, 80))
        results["bins_org"] = [pixel_to_gps(x, y) for x, y in centers]

        # Bennes plastique (bleu)
        centers = find_color_pixels(arr, (0, 80), (0, 100), (180, 255))
        results["bins_plas"] = [pixel_to_gps(x, y) for x, y in centers]

        # Restaurants (orange)
        centers = find_color_pixels(arr, (200, 255), (100, 180), (0, 80))
        results["food_poi"] = [pixel_to_gps(x, y) for x, y in centers]

        # Parcs (vert)
        centers = find_color_pixels(arr, (0, 80), (180, 255), (0, 80))
        results["parks"] = [pixel_to_gps(x, y) for x, y in centers]

        # Écoles (jaune)
        centers = find_color_pixels(arr, (200, 255), (200, 255), (0, 80))
        results["schools"] = [pixel_to_gps(x, y) for x, y in centers]

        total = sum(len(v) for v in results.values())
        print(f"[PNG] Éléments extraits depuis {png_path} : {total}")
        for k, v in results.items():
            print(f"  {k:<15} : {len(v)}")

        return results

    except ImportError:
        print("[PNG] PIL non disponible → éléments non extraits depuis PNG")
        return {}
    except Exception as e:
        print(f"[PNG] Erreur : {e}")
        return {}
