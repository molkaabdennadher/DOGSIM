# =============================================================
# config.py — Configuration centrale
# Stray Dogs Tunisia | ESPRIT 3A IA
# =============================================================

import os
from pathlib import Path

# ROOT = dossier stray_dogs_v2/ (parent de src/)
# Path(__file__) = .../stray_dogs_v2/src/config.py toujours
ROOT     = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_SYN = ROOT / "data" / "synthetic"
DATA_PRO = ROOT / "data" / "processed"
DATA_OUT = ROOT / "data" / "outputs"
MDL_DIR  = ROOT / "models" / "trained"
VIZ_DIR  = ROOT / "visualizations"
NB_DIR   = ROOT / "notebooks"

for d in [DATA_RAW, DATA_SYN, DATA_PRO, DATA_OUT, MDL_DIR, VIZ_DIR, NB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ratio national chiens ──────────────────────────────────
CHIENS_TOTAL_TN   = 1_000_000
POPULATION_TN     = 12_000_000
RATIO_BASE        = CHIENS_TOTAL_TN / POPULATION_TN   # 0.0833

# ── Coefficients attractivité canine par type de zone ─────
ZONE_COEFF = {
    "residential":          1.0,
    "commercial":           1.6,
    "retail":               1.5,
    "industrial":           0.7,
    "park":                 0.9,
    "beach":                1.4,
    "market":               1.8,
    "restaurant_zone":      1.7,
    "mixed":                1.3,
    "periphery":            0.8,
    "unknown":              1.0,
}

# ── OSM tags à récupérer ──────────────────────────────────
OSM_POI_TAGS = {
    "amenity": ["restaurant", "cafe", "fast_food", "bar",
                "bakery", "marketplace", "school", "college",
                "university", "hospital", "pharmacy"],
    "shop":    ["supermarket", "convenience", "butcher", "bakery"],
    "leisure": ["park", "garden", "pitch"],
}

OSM_FOOD_TAGS = ["restaurant", "cafe", "fast_food", "bar",
                 "bakery", "marketplace", "supermarket",
                 "convenience", "butcher"]

OSM_SCHOOL_TAGS = ["school", "college", "university", "kindergarten"]
OSM_PARK_TAGS   = ["park", "garden"]

# ── Paramètres modèles ────────────────────────────────────

# Random Forest (S1 — densité canine)
RF_PARAMS = {
    "n_estimators":   300,
    "max_depth":      10,
    "min_samples_split": 2,
    "min_samples_leaf":  1,
    "random_state":   42,
    "n_jobs":         -1,
}

# DBSCAN (S2 — clustering bennes)
DBSCAN_EPS         = 0.4
DBSCAN_MIN_SAMPLES = 3

# Grille pour scores spatiaux
GRID_STEP_DEG = 0.001   # ~100m

# Feeding zones
N_FEEDING_ZONES   = 7
MIN_DIST_FZ_DEG   = 0.008  # ~900m entre zones

# Poids AHP pour feeding zones
AHP_WEIGHTS = {
    "dog_density":      0.30,
    "dist_schools":     0.20,
    "near_green":       0.15,
    "dist_food_poi":    0.15,
    "dist_org_bins":    0.10,
    "road_access":      0.10,
}

# Bennes
BIN_ORGANIC_RATIO    = 0.55   # 55% organiques
BIN_CAPACITIES       = [120, 240]
BIN_CAPACITY_PROBA   = [0.4, 0.6]
POINT_NOIR_QUANTILE  = 0.60   # top 40% densité = point noir

# ── Gouvernorats Tunisie (pour génération synthétique) ────
GOUVERNORATS_TN = {
    "Tunis":         {"pop": 1056247, "lat": 36.819, "lon": 10.166, "type": "urban"},
    "Ariana":        {"pop": 576088,  "lat": 36.865, "lon": 10.194, "type": "urban"},
    "Ben Arous":     {"pop": 705126,  "lat": 36.753, "lon": 10.228, "type": "urban"},
    "Manouba":       {"pop": 384925,  "lat": 36.809, "lon": 9.989,  "type": "periurban"},
    "Nabeul":        {"pop": 787920,  "lat": 36.452, "lon": 10.736, "type": "mixed"},
    "Zaghouan":      {"pop": 180038,  "lat": 36.401, "lon": 10.143, "type": "rural"},
    "Bizerte":       {"pop": 568219,  "lat": 37.274, "lon": 9.873,  "type": "mixed"},
    "Beja":          {"pop": 303276,  "lat": 36.726, "lon": 9.182,  "type": "rural"},
    "Jendouba":      {"pop": 422769,  "lat": 36.502, "lon": 8.778,  "type": "rural"},
    "Kef":           {"pop": 258655,  "lat": 36.174, "lon": 8.709,  "type": "rural"},
    "Siliana":       {"pop": 228714,  "lat": 36.085, "lon": 9.371,  "type": "rural"},
    "Sousse":        {"pop": 674971,  "lat": 35.825, "lon": 10.636, "type": "urban"},
    "Monastir":      {"pop": 548828,  "lat": 35.764, "lon": 10.812, "type": "urban"},
    "Mahdia":        {"pop": 420952,  "lat": 35.505, "lon": 11.062, "type": "mixed"},
    "Sfax":          {"pop": 955421,  "lat": 34.740, "lon": 10.760, "type": "urban"},
    "Kairouan":      {"pop": 570559,  "lat": 35.678, "lon": 10.100, "type": "mixed"},
    "Kasserine":     {"pop": 439243,  "lat": 35.167, "lon": 8.836,  "type": "rural"},
    "Sidi Bouzid":   {"pop": 435044,  "lat": 35.038, "lon": 9.485,  "type": "rural"},
    "Gabes":         {"pop": 374300,  "lat": 33.882, "lon": 10.098, "type": "mixed"},
    "Medenine":      {"pop": 490302,  "lat": 33.355, "lon": 10.502, "type": "mixed"},
    "Tataouine":     {"pop": 163186,  "lat": 32.929, "lon": 10.452, "type": "rural"},
    "Gafsa":         {"pop": 341552,  "lat": 34.425, "lon": 8.779,  "type": "mixed"},
    "Tozeur":        {"pop": 109402,  "lat": 33.919, "lon": 8.130,  "type": "rural"},
    "Kebili":        {"pop": 163761,  "lat": 33.706, "lon": 8.965,  "type": "rural"},
}

RANDOM_SEED = 42
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
