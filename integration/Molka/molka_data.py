# =============================================================
# molka_data.py — Génération du dataset synthétique DOGSIM-TN
# Molka Module | ESPRIT 3A IA
#
# Fonctions exportées :
#   - generate_dataset()      : DataFrame 2015-2023 × 6 quartiers
#   - QUARTIERS_CFG           : config de base par quartier
#   - QUARTIERS               : liste ordonnée des quartiers
#   - facteur_saison(d)       : coefficient saisonnier
#   - label_saison(d)         : étiquette textuelle de saison
#   - taux_rage/parvo/lepto() : taux de maladie
# =============================================================

import numpy as np
import pandas as pd

# ── Configuration des 6 quartiers ────────────────────────────
QUARTIERS_CFG = {
    'Bab Souika': {'base_d': 22, 'base_b': 0.72},
    'La Medina' : {'base_d': 18, 'base_b': 0.65},
    'Le Bardo'  : {'base_d': 15, 'base_b': 0.58},
    'Ariana'    : {'base_d': 12, 'base_b': 0.50},
    'Ben Arous' : {'base_d': 20, 'base_b': 0.68},
    'La Marsa'  : {'base_d':  8, 'base_b': 0.38},
}
QUARTIERS = list(QUARTIERS_CFG.keys())

# ── Calendrier Ramadan (mois de début) ───────────────────────
RAMADAN_MOIS = {
    2015: 7, 2016: 6, 2017: 5, 2018: 5, 2019: 5,
    2020: 4, 2021: 4, 2022: 4, 2023: 3, 2024: 3,
}

# ── Facteur saisonnier (densité canine) ──────────────────────
def facteur_saison(d):
    """Retourne le coefficient multiplicateur saisonnier pour une date d."""
    m = d.month
    y = d.year
    # Ramadan : moindre activité nocturne → densité perçue réduite
    if (y == 2021 and m == 4) or (y == 2022 and m == 4) or (y == 2023 and m == 3):
        return 0.75
    elif m in [6, 7, 8]:   return 1.45   # Été : canicule → errance accrue
    elif m in [12, 1, 2]:  return 1.30   # Hiver : regroupement autour des ordures
    elif m in [3, 4, 5]:   return 1.10   # Printemps : mise bas → abandon
    return 1.00                           # Automne : baseline

def label_saison(d):
    """Retourne l'étiquette de saison pour une date d."""
    m = d.month
    if RAMADAN_MOIS.get(d.year) == m:  return 'Ramadan'
    elif m in [6, 7, 8]:               return 'Été'
    elif m in [12, 1, 2]:              return 'Hiver'
    elif m in [3, 4, 5]:               return 'Printemps'
    return 'Automne'

# ── Taux de maladies ─────────────────────────────────────────
def taux_rage(d, dens):
    """
    Taux rage canine.
    Base 8% : cohérent avec endémicité Tunisie (Institut Pasteur Tunis).
    Pic été ×1.3 : chaleur → déplacements accrus.
    Source saisonnalité : Kalthoum et al. 2021, Vet Med Sci.
    """
    f_ete = 1.3 if d.month in [6, 7, 8] else 1.0
    return min(round(0.08 * f_ete + 0.003 * dens + np.random.uniform(0, 0.02), 4), 1.0)

def taux_parvo(d, dens):
    """Taux parvovirus : pic printemps/hiver (transmission fécale-orale)."""
    base = 0.12
    f = 1.6 if d.month in [3, 4, 5] else (1.3 if d.month in [12, 1, 2] else 1.0)
    return min(round(base * f + 0.002 * dens + np.random.uniform(0, 0.025), 4), 1.0)

def taux_lepto(d, dens):
    """Taux leptospirose : pic automne/hiver (humidité → survie bactérienne)."""
    f = 1.5 if d.month in [10, 11, 12, 1] else 1.0
    return min(round(0.05 * f + 0.001 * dens + np.random.uniform(0, 0.015), 4), 1.0)


# ── Générateur principal ──────────────────────────────────────
def generate_dataset(seed=42):
    """
    Génère le DataFrame synthétique hebdomadaire 2015-2023
    pour les 6 quartiers de Tunis.

    Returns:
        pd.DataFrame avec colonnes :
            date, quartier, saison, densite, densite_chiens,
            taux_rage, taux_parvo, taux_lepto, risque_global, source
    """
    np.random.seed(seed)
    dates = pd.date_range('2015-01-01', '2023-12-31', freq='W')

    records = []
    for q, cfg in QUARTIERS_CFG.items():
        for dt in dates:
            fs   = facteur_saison(dt)
            dens = max(0, round(
                cfg['base_d'] * fs
                + 3 * np.sin(2 * np.pi * dt.dayofyear / 365)
                + np.random.normal(0, 1.5),
                2
            ))
            ra = taux_rage(dt, dens)
            pa = taux_parvo(dt, dens)
            le = taux_lepto(dt, dens)
            records.append({
                'date':           dt,
                'quartier':       q,
                'saison':         label_saison(dt),
                'densite':        dens,
                'densite_chiens': dens,
                'taux_rage':      ra,
                'taux_parvo':     pa,
                'taux_lepto':     le,
                'risque_global':  round((ra + pa + le) / 3, 4),
            })

    df = pd.DataFrame(records)
    df['source'] = 'synthetique'
    return df


# ── Données géographiques ─────────────────────────────────────
GEO = {
    'Bab Souika': {
        'center': (36.8180, 10.1680),
        'bounds': [(36.8100, 10.1580), (36.8260, 10.1780)],
        'color' : '#E24B4A',
        'radius': 0.008,
    },
    'La Medina': {
        'center': (36.7980, 10.1720),
        'bounds': [(36.7880, 10.1580), (36.8080, 10.1860)],
        'color' : '#EF9F27',
        'radius': 0.010,
    },
    'Le Bardo': {
        'center': (36.8090, 10.1430),
        'bounds': [(36.7980, 10.1280), (36.8200, 10.1580)],
        'color' : '#7F77DD',
        'radius': 0.010,
    },
    'Ariana': {
        'center': (36.8620, 10.1920),
        'bounds': [(36.8450, 10.1720), (36.8790, 10.2120)],
        'color' : '#1D9E75',
        'radius': 0.014,
    },
    'Ben Arous': {
        'center': (36.7530, 10.2280),
        'bounds': [(36.7350, 10.2050), (36.7710, 10.2510)],
        'color' : '#D4537E',
        'radius': 0.013,
    },
    'La Marsa': {
        'center': (36.8780, 10.3230),
        'bounds': [(36.8620, 10.3050), (36.8940, 10.3410)],
        'color' : '#378ADD',
        'radius': 0.012,
    },
}


# ── Test rapide ───────────────────────────────────────────────
if __name__ == '__main__':
    df = generate_dataset()
    print(f"Dataset : {len(df)} observations | {df['quartier'].nunique()} quartiers")
    print(f"Période : {df['date'].min().date()} → {df['date'].max().date()}")
    print(df.groupby('quartier')[['densite', 'risque_global']].mean().round(3))
