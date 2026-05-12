# DOGSIM — Stray Dog Detection & Management System 🐕

A full-stack AI-powered system for stray dog detection, population estimation, and urban management built with Python, YOLOv8, and machine learning.

## Overview

This project, **DOGSIM**, was developed as part of the coursework for the **Software Engineering & AI program** at **Esprit School of Engineering**. It addresses the urban challenge of stray dog management in Tunisia by combining computer vision, geospatial analysis, predictive machine learning, and LLM-powered advisory tools into an end-to-end pipeline.

---

## Features

- Real-time stray dog **detection and tracking** using **YOLOv8**
- **Population estimation** per neighborhood using Random Forest & Gradient Boosting
- **Time-series forecasting** with SARIMAX for population trends
- **Risk classification** (Low / Medium / High) based on urban density and waste infrastructure
- **Optimal bin placement** via DBSCAN clustering and P-Median facility location
- **Feeding zone recommendations** using multi-criteria AHP scoring
- **Animal aggression detection** using YOLOv8-Pose and ByteTrack
- **Coproscopic analysis** (MobileNet) detecting 6 health conditions
- **Happy Paws** — LLM-powered dog adoption advisor (Ollama llama3 / Groq LLaMA-3.3-70B)
- Interactive **visual dashboards** and HTML maps
- REST **API** built with FastAPI and Flask

---

## ⚡ Lancement rapide

### Prérequis
- Python 3.10+
- [Ollama](https://ollama.com) avec le modèle `llama3` (`ollama pull llama3`)
- Dépendances : `pip install -r requirements.txt`

### Windows — 1 clic
```bat
LANCER_DEMO.bat
```
Lance automatiquement :
- Ollama LLM → `http://localhost:11434`
- DOGSIM-TN Dashboard → `http://localhost:8080`
- Happy Paws → `http://localhost:8000`

### Manuel
```bash
# Terminal 1 — Dashboard principal
python server.py

# Terminal 2 — Happy Paws
cd "recommendation/adoption system/pet-advisor"
.venv/Scripts/uvicorn backend.main:app --port 8000
```

---

## 📊 Performances

| Modèle | Métrique | Résultat |
|--------|----------|----------|
| Random Forest Regressor | R² | 0.93 |
| Gradient Boosting Regressor | R² | 0.94 |
| Gradient Boosting Regressor | MAE | 164 chiens |
| RF Classifier (risque) | Accuracy | 85.8% |
| MobileNet (coproscopie) | Classes | 6 (Normal · Diarrhée · Sang · Vers · …) |

---

## 🛠️ Stack technique

**Machine Learning** · Python · scikit-learn · TensorFlow · statsmodels (SARIMAX)  
**Computer Vision** · YOLOv8n · YOLOv8n-Pose · ByteTrack · OpenCV · Lucas-Kanade  
**Backend** · Flask · FastAPI · Uvicorn · SQLite  
**LLM** · Ollama (llama3 local) · Groq (LLaMA-3.3-70B fallback)  
**Géospatial** · OpenStreetMap Overpass API · DBSCAN · AHP · folium  
**Frontend** · React · HTML/CSS · Jupyter Notebooks  

---

## Directory Structure

```
DOGSIM/
├── stray_dogs_v2/          ← Pipeline ML géospatial (Aziz)
├── cv_module/              ← Module computer vision (Aziz)
│
└── integration/            ← Couche d'intégration complète du système
    ├── server.py                      ← Serveur Flask principal (port 8080)
    ├── LANCER_DEMO.bat                ← Lanceur Windows 1-clic
    ├── DOGSIM-TN Final.html           ← Dashboard principal
    ├── cv_tab.html                    ← Onglet détection CV
    ├── pipeline_tab.html              ← Onglet pipeline ML
    ├── simulation_tab.html            ← Onglet simulation Q-Learning
    ├── timeseries_tab.html            ← Onglet séries temporelles
    ├── yolov8n.pt                     ← Modèle YOLOv8n (détection)
    ├── yolov8n-pose.pt                ← Modèle YOLOv8n-Pose (agression)
    │
    ├── stray_dogs_v2/                 ← Module Aziz (pipeline S1/S2/S3)
    │   ├── api.py                     ← FastAPI endpoint ML
    │   ├── src/
    │   │   ├── pipeline/pipeline.py   ← Pipeline complet S1→S2→S3
    │   │   ├── models/                ← Génération données + entraînement
    │   │   └── utils/                 ← OSM fetcher, visualizer, bin generator
    │   ├── models/trained/            ← rf_regressor, gb_regressor, rf_classifier, scaler (.joblib)
    │   ├── data/
    │   │   ├── raw/                   ← RGPH 2014 (24 gouvernorats)
    │   │   ├── synthetic/             ← Dataset 1200 secteurs générés
    │   │   ├── processed/             ← POI par quartier (CSV)
    │   │   └── outputs/               ← Résultats JSON/CSV (bennes, feeding zones)
    │   └── visualizations/            ← Dashboards PNG + cartes HTML
    │
    ├── cv_module/                     ← Module Aziz (détection infractions CV)
    │   ├── cv_api.py                  ← FastAPI CV endpoint
    │   ├── pipeline.py                ← Pipeline vidéo complet
    │   ├── core/
    │   │   ├── detector.py            ← Détection YOLOv8
    │   │   ├── tracker.py             ← Tracking objets
    │   │   ├── stationarity.py        ← Détection chiens stationnaires
    │   │   ├── eating_detector.py     ← Détection comportement alimentaire
    │   │   ├── backtracker.py         ← Rétro-tracking
    │   │   └── video_buffer.py        ← Buffer vidéo temps réel
    │   ├── utils/infraction_reporter.py ← Génération rapports JSON
    │   └── output/
    │       ├── frames/                ← Captures infractions (approche/dépôt/départ)
    │       └── reports/               ← Rapports JSON par infraction
    │
    ├── Molka/                         ← Module Molka (SARIMAX + Q-Learning)
    │   ├── molka_api.py               ← FastAPI endpoint séries temporelles
    │   ├── molka_sarimax.py           ← Prévision SARIMAX
    │   ├── molka_rl.py                ← Simulation Q-Learning
    │   ├── molka_data.py              ← Préparation données
    │   └── cache/molka_models.joblib  ← Modèles SARIMAX entraînés
    │
    ├── adham/                         ← Module Adham (détection agression)
    │   ├── aggression_api.py          ← FastAPI endpoint agression
    │   ├── aggression_pipeline.py     ← Pipeline YOLOv8-Pose + ByteTrack
    │   └── output/
    │       ├── captures/              ← Frames d'agression détectées (JPG)
    │       └── aggression_events.db   ← Base SQLite des événements
    │
    └── recommendation/                ← Modules recommendation & santé
        ├── adoption system/           ← Module adoption (pet-advisor)
        │   └── pet-advisor/
        │       ├── backend/           ← FastAPI backend adoption
        │       ├── frontend/          ← Interface React
        │       └── data/              ← Dataset adoption
        ├── dr_halim/                  ← Happy Paws — conseiller LLM
        │   ├── backend/
        │   │   ├── main.py            ← FastAPI + Ollama/Groq
        │   │   └── inference.py       ← Inférence race + comportement
        │   ├── frontend/index.html    ← Interface Happy Paws
        │   └── models/
        │       ├── breed_model.h5     ← Modèle classification race
        │       └── skin_model.pth     ← Modèle dermatologie canine
        └── tacheRanim/                ← Module Ranim (coproscopie + chatbot)
            ├── backend/
            │   ├── app.py             ← Flask backend
            │   ├── predictor.py       ← MobileNet coproscopie (6 classes)
            │   └── llm_service.py     ← Chatbot vétérinaire LLM
            ├── frontend/index.html    ← Interface chatbot
            └── model/
                └── model_mobilenet_selles.keras ← Modèle coproscopie
```

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/molkaabdennadher/DOGSIM.git
cd DOGSIM

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r stray_dogs_v2/requirements.txt
```

---

## 📁 Notebooks

| Notebook | Contenu |
|----------|---------|
| `main_notebook.ipynb` | Pipeline complet S1→S2→S3 + dashboard |
| `training_notebook.ipynb` | Entraînement RF · GB · Classifier |
| `feeding_infraction_notebook.ipynb` | Module CV détection infractions |
| `pi-sarimax-ql.ipynb` | Séries temporelles + Q-Learning |
| `animal_aggression_detection.ipynb` | Détection agression |

---

## 👥 Équipe

| Membre | Objectif |
|--------|----------|
| Aziz | Pipeline ML géospatial (S1/S2/S3) + Détection infractions (CV) |
| Molka | Prévision SARIMAX + Simulation Q-Learning |
| Adham | Détection agression animale |
| Ranim | Analyse coproscopique + Chatbot vétérinaire |
| Dr. Halim | Conseiller adoption (Happy Paws) |

---

## Acknowledgments

This project was developed as part of the **Software Engineering & AI curriculum** at **Esprit School of Engineering**, Tunisia. It tackles a real urban challenge using state-of-the-art artificial intelligence and computer vision techniques applied to the Tunisian context.

---

## 📄 Licence

Projet académique — **Esprit School of Engineering** © 2026

---

## Topics

`python` `machine-learning` `deep-learning` `computer-vision` `yolov8` `object-detection` `artificial-intelligence` `fastapi` `geospatial` `data-analysis` `urban-management` `llm` `esprit-school-of-engineering`
