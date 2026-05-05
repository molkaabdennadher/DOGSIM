# =============================================================
# bin_generator.py
# Génère des bennes simulées pour une bbox donnée
# =============================================================

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BIN_ORGANIC_RATIO, BIN_CAPACITIES, BIN_CAPACITY_PROBA, RANDOM_SEED


def generate_bins_for_bbox(bbox, population, seed=RANDOM_SEED):
    """
    Génère des bennes simulées réparties dans une bbox selon la population.
    Règle : 1 benne organique / 700 habitants, 1 plastique / 1000 habitants.
    """
    np.random.seed(seed)

    nb_org  = max(5, int(population / 700))
    nb_plas = max(3, int(population / 1000))
    total   = nb_org + nb_plas

    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]

    rows = []
    for i in range(nb_org):
        rows.append({
            "id":              f"B_ORG_{i:04d}",
            "lat":             np.random.uniform(lat_min, lat_max),
            "lon":             np.random.uniform(lon_min, lon_max),
            "type":            "organique",
            "capacite_litres": np.random.choice(BIN_CAPACITIES, p=BIN_CAPACITY_PROBA),
            "statut":          "simulé",
        })
    for i in range(nb_plas):
        rows.append({
            "id":              f"B_PLAS_{i:04d}",
            "lat":             np.random.uniform(lat_min, lat_max),
            "lon":             np.random.uniform(lon_min, lon_max),
            "type":            "plastique",
            "capacite_litres": np.random.choice(BIN_CAPACITIES, p=BIN_CAPACITY_PROBA),
            "statut":          "simulé",
        })

    df = pd.DataFrame(rows)
    print(f"[BIN_GEN] {len(df)} bennes générées ({nb_org} organiques, {nb_plas} plastique)")
    return df
