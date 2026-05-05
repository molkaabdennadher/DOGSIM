# 🐕 Stray Dogs Tunisia — Système IA Complet
**ESPRIT — 3ème année cycle Ingénieur IA | Axe Déchets & Urbanisme**

---

## Principe du système

```
INPUT                              OUTPUT
─────────────────────────────      ─────────────────────────────────────
Nom du quartier                →   S1 : Nombre de chiens estimés
+ Population / Ménages / Surface    + Niveau de risque (Faible/Moyen/Élevé)
+ (PNG de carte optionnel)     →   S2 : Emplacements bennes optimisés
+ (Infos complémentaires)           + Points noirs identifiés
                               →   S3 : Feeding zones recommandées
                                    + Dashboard visuel complet
```

**Fonctionne pour n'importe quel quartier tunisien.**

---

## Structure

```
stray_dogs_v2/
├── notebooks/
│   └── main_notebook.ipynb          ← POINT D'ENTRÉE PRINCIPAL
│
├── src/
│   ├── config.py                    ← Tous les paramètres
│   ├── models/
│   │   ├── generate_synthetic_data.py   ← Génération dataset 1200 samples
│   │   └── train_models.py              ← Entraînement RF + GB + Classifier
│   ├── pipeline/
│   │   └── pipeline.py              ← Cœur du système : input → 3 solutions
│   └── utils/
│       ├── osm_fetcher.py           ← Données OSM + extraction PNG
│       ├── bin_generator.py         ← Générateur de bennes simulées
│       └── visualizer.py           ← Dashboard + carte HTML
│
├── data/
│   ├── raw/                         ← Données brutes (Excel RGPH, PNG)
│   ├── synthetic/                   ← Dataset généré (1200 secteurs)
│   ├── processed/                   ← POI et données intermédiaires
│   └── outputs/                     ← Résultats finaux (CSV, JSON)
│
├── models/trained/                  ← Modèles .joblib entraînés
├── visualizations/                  ← Dashboards PNG + cartes HTML
└── requirements.txt
```

---

## Installation et lancement

```bash
# 1. Créer environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Installer dépendances
pip install -r requirements.txt

# 3. Lancer Jupyter
jupyter notebook notebooks/main_notebook.ipynb
```

---

## Modèles utilisés

| Solution | Modèle | Librairie | Rôle |
|---|---|---|---|
| S1 | **Random Forest Regressor** | scikit-learn | Prédiction nb chiens |
| S1 | **Gradient Boosting Regressor** | scikit-learn | Prédiction nb chiens (ensemble) |
| S1 | **Random Forest Classifier** | scikit-learn | Classification risque 0/1/2 |
| S2 | **DBSCAN** | scikit-learn | Clustering bennes organiques |
| S2 | **P-Median (Facility Location)** | scipy | Optimisation emplacements |
| S3 | **Score multicritère AHP** | numpy | Sélection feeding zones |

---

## Dataset synthétique — justification

**Problème** : Pas de données officielles de comptage de chiens en Tunisie.

**Solution** : Génération synthétique assumée et documentée.

- **1200 secteurs** couvrant les **24 gouvernorats** tunisiens (RGPH 2014)
- **Hypothèses documentées** :
  - Ratio base : 1 chien / 12 habitants (OMS 2022)
  - Modulation par type de zone (commercial × 1.6, balnéaire × 1.4...)
  - Modulation par densité de POI alimentaires (+2 chiens/restaurant)
  - Modulation par nombre de bennes organiques (+3 chiens/benne)
  - Bruit gaussien 15% pour simuler la variabilité réelle
- **Labels** : nb_chiens (régression) + risk_class 0/1/2 (classification)
- **À défendre en soutenance** : approche courante en l'absence de données terrain, utilisée en médecine vétérinaire et en écologie urbaine

---

## Input PNG — convention couleurs

Si tu fournis un PNG de carte avec des marqueurs colorés :

| Couleur | Élément |
|---|---|
| 🔴 Rouge | Bennes organiques |
| 🔵 Bleu | Bennes plastique |
| 🟠 Orange | Restaurants/Cafés |
| 🟢 Vert | Espaces verts/Parcs |
| 🟡 Jaune | Écoles |

---

## Multimodalité

```
TABULAIRE        SPATIAL (OSM)      IMAGE (PNG)       API
──────────       ───────────────    ────────────      ─────────────
RGPH 2014        Réseau routier     Extraction        Google Places
Population       Écoles/Lycées      couleurs          (optionnel)
Ménages          Parcs/Espaces      marqueurs
Surface          Restaurants
                 Bbox quartier
     │                │                 │                 │
     └────────────────┴─────────────────┴─────────────────┘
                              │
                      PIPELINE IA
                 S1 → S2 → S3 → Dashboard
```
