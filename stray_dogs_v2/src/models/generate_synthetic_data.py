# =============================================================
# generate_synthetic_data.py
# Génération du dataset synthétique pour entraîner les modèles
#
# JUSTIFICATION (à présenter en soutenance) :
# -------------------------------------------
# Il n'existe pas de données officielles de comptage de chiens
# errants en Tunisie. Nous adoptons une approche "Synthetic Data
# Generation" documentée dans la littérature (ex: He et al. 2021,
# WHO Dog Population Management Guidelines).
#
# Hypothèses utilisées :
# 1. Ratio national : 1 chien / 12 habitants (OMS, Tunisie 2022)
# 2. La densité canine est MODULÉE par :
#    - La densité de POI alimentaires (restaurants, marchés)
#    - Le type de zone urbaine (commercial > résidentiel > rural)
#    - La taille des ménages (plus de résidents = plus de déchets)
#    - La densité de bennes organiques à proximité
# 3. Du bruit gaussien est ajouté pour simuler la variabilité réelle
# 4. Le dataset couvre les 24 gouvernorats tunisiens (RGPH 2014)
#    avec 50 secteurs synthétiques chacun → 1200 samples
#
# Usage : python generate_synthetic_data.py
# =============================================================

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_SYN, GOUVERNORATS_TN, RATIO_BASE,
    ZONE_COEFF, RANDOM_SEED
)

np.random.seed(RANDOM_SEED)

# ── 1. Paramètres de génération ───────────────────────────────

N_SECTORS_PER_GOV  = 50   # secteurs synthétiques par gouvernorat
NOISE_STD          = 0.15  # bruit gaussien (15% de variation)

URBAN_TYPE_DIST = {
    "urban":     {"residential": 0.3, "commercial": 0.3, "mixed": 0.2, "park": 0.1, "industrial": 0.1},
    "periurban": {"residential": 0.4, "commercial": 0.2, "mixed": 0.2, "park": 0.1, "periphery": 0.1},
    "mixed":     {"residential": 0.4, "commercial": 0.15, "mixed": 0.2, "periphery": 0.15, "park": 0.1},
    "rural":     {"residential": 0.5, "periphery": 0.3, "park": 0.1, "mixed": 0.1},
}

# ── 2. Génération des features ────────────────────────────────

def generate_sector_features(gov_name, gov_info, n_sectors):
    """
    Génère N secteurs synthétiques pour un gouvernorat.
    Chaque secteur a des features réalistes basées sur le type de gouvernorat.
    """
    gov_type   = gov_info["type"]
    gov_pop    = gov_info["pop"]
    gov_lat    = gov_info["lat"]
    gov_lon    = gov_info["lon"]
    zone_dist  = URBAN_TYPE_DIST.get(gov_type, URBAN_TYPE_DIST["mixed"])

    zone_types = list(zone_dist.keys())
    zone_probs = list(zone_dist.values())

    records = []

    for i in range(n_sectors):
        # Tirage du type de zone
        zone_type = np.random.choice(zone_types, p=zone_probs)
        coeff     = ZONE_COEFF.get(zone_type, 1.0)

        # Population du secteur (lognormale autour de pop/n_sectors)
        pop_mean = gov_pop / n_sectors
        pop      = max(500, int(np.random.lognormal(
            mean=np.log(pop_mean), sigma=0.5
        )))

        # Ménages (taille moyenne 3.5 personnes en Tunisie)
        taille_menage = np.random.normal(3.5, 0.5)
        taille_menage = max(1.5, min(7.0, taille_menage))
        nb_menages    = max(50, int(pop / taille_menage))

        # POI alimentaires (corrélés au type de zone)
        poi_base = {
            "commercial": 25, "retail": 20, "market": 30,
            "mixed": 12, "residential": 5, "periphery": 2,
            "park": 2, "industrial": 1, "beach": 15,
        }
        nb_food_poi = max(0, int(np.random.poisson(
            poi_base.get(zone_type, 5)
        )))

        # Bennes organiques dans le secteur
        nb_bennes_org = max(1, int(np.random.poisson(
            pop / 800  # 1 benne / 800 habitants environ
        )))

        # Distance au centre-ville (0 = centre, 1 = lointain)
        dist_centre = np.random.beta(2, 2)

        # Surface en km²
        surface_km2 = max(0.1, np.random.lognormal(mean=0.5, sigma=0.8))

        # Densité de population
        densite_pop = pop / surface_km2

        # Coordonnées GPS (autour du centre du gouvernorat)
        lat = gov_lat + np.random.normal(0, 0.05)
        lon = gov_lon + np.random.normal(0, 0.05)

        # ── Calcul de la cible (nombre de chiens) ─────────────
        # Formule principale
        chiens_base = pop * RATIO_BASE

        # Modulation par zone
        chiens = chiens_base * coeff

        # Modulation par POI alimentaires (chaque restau attire ~2 chiens)
        chiens += nb_food_poi * 2.0

        # Modulation par bennes organiques (chaque benne attire ~3 chiens)
        chiens += nb_bennes_org * 3.0

        # Modulation par densité (zones très denses = plus de chiens)
        if densite_pop > 10000:
            chiens *= 1.2
        elif densite_pop < 500:
            chiens *= 0.7

        # Bruit gaussien pour simuler la variabilité réelle
        noise  = np.random.normal(1.0, NOISE_STD)
        chiens = max(0, int(chiens * noise))

        # Classe de risque (pour la classification)
        if chiens < pop * 0.05:
            risk_class = 0   # faible
        elif chiens < pop * 0.10:
            risk_class = 1   # moyen
        else:
            risk_class = 2   # élevé

        records.append({
            # Identifiants
            "sector_id":       f"{gov_name[:3].upper()}_{i:04d}",
            "gouvernorat":     gov_name,
            "gov_type":        gov_type,
            "zone_type":       zone_type,
            "lat":             round(lat, 5),
            "lon":             round(lon, 5),

            # Features démographiques
            "population":      pop,
            "nb_menages":      nb_menages,
            "taille_menage":   round(taille_menage, 2),
            "surface_km2":     round(surface_km2, 3),
            "densite_pop":     round(densite_pop, 1),

            # Features POI
            "nb_food_poi":     nb_food_poi,
            "nb_bennes_org":   nb_bennes_org,
            "dist_centre":     round(dist_centre, 3),

            # Feature encodée
            "zone_coeff":      coeff,

            # Cibles
            "nb_chiens":       chiens,
            "risk_class":      risk_class,   # 0=faible, 1=moyen, 2=élevé
        })

    return records


# ── 3. Pipeline de génération ─────────────────────────────────

def generate_full_dataset():
    print("=" * 60)
    print("GÉNÉRATION DU DATASET SYNTHÉTIQUE")
    print("Couverture : 24 gouvernorats tunisiens (RGPH 2014)")
    print(f"Secteurs   : {len(GOUVERNORATS_TN)} × {N_SECTORS_PER_GOV} = "
          f"{len(GOUVERNORATS_TN) * N_SECTORS_PER_GOV} samples")
    print("=" * 60)

    all_records = []

    for gov_name, gov_info in GOUVERNORATS_TN.items():
        records = generate_sector_features(gov_name, gov_info, N_SECTORS_PER_GOV)
        all_records.extend(records)
        print(f"  ✓ {gov_name:<20} : {len(records)} secteurs générés "
              f"(pop moy: {int(np.mean([r['population'] for r in records])):,})")

    df = pd.DataFrame(all_records)

    # ── Statistiques ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Dataset total     : {len(df):,} samples")
    print(f"Features          : {len(df.columns)} colonnes")
    print(f"\nDistribution des chiens :")
    print(f"  Min    : {df['nb_chiens'].min():,}")
    print(f"  Max    : {df['nb_chiens'].max():,}")
    print(f"  Médiane: {df['nb_chiens'].median():,.0f}")
    print(f"  Moyenne: {df['nb_chiens'].mean():,.0f}")
    print(f"\nDistribution des classes de risque :")
    for cls, label in [(0, "Faible"), (1, "Moyen"), (2, "Élevé")]:
        n = (df['risk_class'] == cls).sum()
        print(f"  {label:<8} : {n:4d} ({n/len(df)*100:.1f}%)")

    # ── Sauvegarde ────────────────────────────────────────────
    DATA_SYN.mkdir(parents=True, exist_ok=True)
    path = DATA_SYN / "tunisia_sectors_synthetic.csv"
    df.to_csv(path, index=False)
    print(f"\n✅ Dataset sauvegardé : {path}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    df = generate_full_dataset()
    print(f"\nAperçu :")
    print(df[["sector_id", "gouvernorat", "zone_type", "population",
              "nb_food_poi", "nb_chiens", "risk_class"]].head(10).to_string(index=False))
