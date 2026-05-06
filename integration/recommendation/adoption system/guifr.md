# 🐾 Happy Paws — Guide de lancement (Python 3.12)

Ce guide couvre tout, de l'installation à l'accès aux trois interfaces : **Maya** (conseiller adoption), **Ali** (gestion refuges), et **Aggression** (détection d'agression).

---

## 0. Prérequis

| Outil | Version | Vérification |
|---|---|---|
| **Python** | 3.12.x (exact) | `py -3.12 --version` |
| **Git** | n'importe | `git --version` |
| **Clé API Groq** | gratuite | [console.groq.com](https://console.groq.com) |
| **Espace disque** | ~3 GB | pour `.venv` + modèles YOLO |

> **Pourquoi Python 3.12 exactement ?**
> Le fichier `.python-version` et `pyproject.toml` imposent `>=3.12,<3.13`.
> `start.py` refusera de démarrer sous toute autre version.

---

## 1. Créer l'environnement virtuel

Ouvre **PowerShell** dans `adoption system\pet-advisor\` et lance le script fourni :

```powershell
cd "adoption system\pet-advisor"
.\setup_python312.ps1
```

Ce script :
- vérifie que Python 3.12 est présent (`py -3.12`)
- crée `.venv` si absent (ou le recrée si la version est mauvaise)
- installe toutes les dépendances depuis `requirements.txt`

**Activation manuelle** (si tu veux travailler dans le terminal) :

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 2. Configurer la clé API Groq

Le fichier `.env` se trouve dans `pet-advisor\.env`. Ouvre-le et remplis :

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Les trois modules (Maya, Ali, Aggression) utilisent tous Groq. Sans cette clé, le serveur démarre mais les agents LLM renvoient une erreur.

**Modèles Groq utilisés :**

| Module | Modèle par défaut | Variable d'env pour changer |
|---|---|---|
| Maya | `meta-llama/llama-4-scout-17b-16e-instruct` | `GROQ_MODEL` |
| Ali | `meta-llama/llama-4-scout-17b-16e-instruct` | `GROQ_MODEL` |
| Aggression (texte) | `llama-3.3-70b-versatile` | `GROQ_TEXT_MODEL` |
| Aggression (vision) | `meta-llama/llama-4-scout-17b-16e-instruct` | `GROQ_VISION_MODEL` |

---

## 3. Artefacts Maya (optionnel — conseiller adoption)

Maya a besoin d'un index d'embeddings généré par le notebook `cosinesimilarity2.ipynb`.
Si tu n'as pas encore ces fichiers, Maya fonctionne en **mode dégradé** (sans recommandations personnalisées).

Pour générer les artefacts :

1. Lance `cosinesimilarity2.ipynb` sur **Kaggle** (GPU T4 gratuit, ~10 min).
2. Télécharge le dossier `artifacts/` généré.
3. Copie son contenu dans `pet-advisor\artifacts\aiAdvisor\`.

Fichiers attendus :

```
pet-advisor\artifacts\aiAdvisor\
├── df_original.parquet
├── pet_emb.npy
├── pet_ids.npy
├── pet_features_shape.txt
├── user_encoder.pt
├── pet_encoder.pt
├── thresholds.json
├── img_emb_raw.npy
├── rich_descriptions.json
└── pet_data_raw.json
```

---

## 4. Lancer le serveur

Depuis `pet-advisor\` avec le `.venv` activé :

```powershell
python start.py
```

Options utiles :

```powershell
python start.py --port 8000         # port par défaut
python start.py --reload            # rechargement auto (développement)
python start.py --port 9000         # autre port
```

Le serveur démarre sur `http://localhost:8000`.

**Logs de démarrage attendus :**

```
INFO  ali.db: Ali DB initialised
INFO  ali.agent: [Ali] Agent ready — model: meta-llama/llama-4-scout-17b-16e-instruct
INFO  aggression.db: Aggression DB initialised
INFO  uvicorn: Application startup complete.
```

> Si tu vois `WARNING Maya embedding artifacts not loaded`, c'est normal si tu n'as pas fait l'étape 3 — le reste fonctionne quand même.

---

## 5. Accéder aux interfaces

| Interface | URL | Description |
|---|---|---|
| **Accueil** | http://localhost:8000 | Page principale |
| **Maya** | http://localhost:8000/maya | Conseiller adoption par IA |
| **Ali** | http://localhost:8000/ali | Gestion des refuges + chatbot |
| **Aggression** | http://localhost:8000/aggression | Détection d'agression vidéo |
| **Health check** | http://localhost:8000/health | `{"status":"ok"}` |

---

## 6. Interface Ali — Premiers pas

Ali fonctionne **sans aucun upload** au démarrage : les 120 refuges seeds sont chargés automatiquement.

**Ce que tu peux faire dès le lancement :**

- Taper « Montre-moi les statistiques » → Ali appelle `get_statistics`
- Taper « Quels refuges sont presque pleins ? » → Ali appelle `get_capacity_risks`
- Taper « Liste les chiens en attente » → Ali appelle `get_waiting_list`

**Pour assigner des chiens :**

1. Clique sur le bouton d'upload dans l'interface Ali.
2. Uploade un fichier `.xlsx` avec les colonnes :
   `PetID, Name, Age, Breed, Gender, Vaccinated, Dewormed, Sterilized, Health, Behaviour, IsPregnant, ExpectedLitterSize, HasSkinDisease`
3. Ali reçoit le résultat et peut l'expliquer en langage naturel.

**Priorités de tri (avant l'affectation) :**

| Condition | Points |
|---|---|
| Blessure grave | +100 |
| Femelle gestante | +85 |
| Maladie de peau | +60 |
| Comportement anormal | +50 |
| Blessure légère | +40 |
| Non vacciné | +15 |
| Non vermifugé | +10 |

---

## 7. Interface Aggression — Utilisation

L'interface Aggression analyse des vidéos de caméras de surveillance.

**Upload direct depuis l'interface :**

1. Va sur http://localhost:8000/aggression
2. Uploade une vidéo (MP4, AVI, MOV, MKV, WebM)
3. Le pipeline YOLO + ByteTrack + LLM analyse la vidéo
4. Le résultat s'affiche : sévérité, type d'agression, actions recommandées

**Depuis le notebook Colab (`animal_aggression_detection_v4.ipynb`) :**

1. Cellule 04 : renseigne ta `GROQ_API_KEY` et le chemin vidéo
2. Cellule 05 : charge les modèles YOLO
3. Cellule 17 : configure `VIDEO_SOURCE` et lance `run_pipeline()`
4. Cellule 23 : appelle `forward_all_incidents()` pour envoyer les incidents détectés au backend

> **Note** : le notebook envoie les incidents à `http://localhost:8000` — le serveur doit être actif quand tu fais `forward_all_incidents()`.

---

## 8. Vérifications rapides après lancement

```powershell
# 1. Health check
curl http://localhost:8000/health
# → {"status":"ok"}

# 2. Statistiques Ali
curl http://localhost:8000/ali/statistics
# → {"total_shelters":120, ...}

# 3. Liste des refuges
curl http://localhost:8000/ali/shelters | python -m json.tool | head -30

# 4. Incidents d'agression (vide au départ)
curl http://localhost:8000/aggression/incidents
```

---

## 9. Structure du projet

```
adoption system/
├── guifr.md                          ← ce guide
├── .python-version                   ← 3.12
├── setup_python312.ps1               ← script setup racine
│
├── Notebooks/
│   ├── NotebookComputerVisionDetectionAgression/
│   │   └── animal_aggression_detection_v4.ipynb  ← notebook Colab
│   ├── NotebookRecommendation/
│   │   └── cosinesimilarity2.ipynb               ← génère artefacts Maya
│   └── NotebookDogFinderTraining/
│       └── dog-finder-training.ipynb
│
└── pet-advisor/
    ├── .env                          ← GROQ_API_KEY ici
    ├── .python-version               ← 3.12
    ├── pyproject.toml
    ├── requirements.txt
    ├── start.py                      ← point d'entrée
    ├── setup_python312.ps1           ← script setup
    ├── cleanup.ps1                   ← nettoyage caches
    ├── yolov8n-pose.pt               ← modèle YOLO (sentinel)
    │
    ├── backend/
    │   ├── main.py                   ← FastAPI, tous les endpoints
    │   ├── maya/                     ← agent conseiller adoption
    │   ├── ali/                      ← agent gestion refuges
    │   └── aggression/               ← agent post-détection LangGraph
    │
    ├── frontend/
    │   ├── index.html
    │   ├── maya.html
    │   ├── ali.html
    │   └── aggression.html
    │
    └── artifacts/
        ├── aiAdvisor/                ← embeddings Maya (à générer)
        ├── aliAdvisor/               ← DB SQLite Ali
        └── agressionDetection/       ← DB SQLite + keyframes + rapports
```

---

## 10. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `Python 3.12 is required` | mauvaise version Python | `py -3.12 -m venv .venv` |
| `GROQ_API_KEY not set` | `.env` manquant ou vide | remplis `GROQ_API_KEY` dans `.env` |
| `Ali agent not initialised` | même cause | idem |
| `WARNING Maya artifacts not loaded` | artefacts absents | lance `cosinesimilarity2.ipynb` (étape 3) |
| `HTTP 503` sur `/aggression/analyze-video` | ultralytics manquant | `pip install ultralytics>=8.2.0` |
| Ali répond vide / erreur JSON | modèle Groq instable | relance ou change `GROQ_MODEL` dans `.env` |
| Page blanche sur `/aggression` | frontend absent | vérifie que `frontend/aggression.html` existe |
| Port 8000 occupé | autre process | `python start.py --port 8001` |
