# Happy Paws — Guide de configuration & Checklist de mise en service

---

## TODO — Actions requises pour que le site soit fonctionnel

Les problèmes ci-dessous sont classés par priorité. Commencer par le **Bloc 1** — il résout la majorité des pannes LLM.

---

### BLOC 1 — API Groq (cause principale de toutes les pannes LLM)

**Symptôme :** Maya répond « Internal error », Ali ne se connecte pas, détection d'agression échoue.

#### TODO 1.1 — Vérifier / régénérer la clé API Groq
1. Aller sur https://console.groq.com → **API Keys**
2. Vérifier que la clé dans `.env` est bien **Active** (pas Revoked)
3. Si expirée → **Create API Key** et remplacer dans `.env` :
   ```
   GROQ_API_KEY=gsk_VOTRE_NOUVELLE_CLE
   ```

#### TODO 1.2 — Vérifier que les noms de modèles sont toujours valides
Les modèles Groq sont mis à jour / renommés régulièrement. Vérifier sur https://console.groq.com/playground → menu **Model** :

| Variable `.env` | Valeur actuelle | Fallback si introuvable |
|---|---|---|
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | `llama-3.3-70b-versatile` |
| `GROQ_TEXT_MODEL` | `llama-3.3-70b-versatile` | `llama3-70b-8192` |
| `GROQ_VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | `llama-3.2-11b-vision-preview` |

Test rapide (PowerShell) pour lister les modèles disponibles :
```powershell
$key = (Get-Content .env | Where-Object { $_ -match "^GROQ_API_KEY=" }) -replace "^GROQ_API_KEY=",""
Invoke-RestMethod -Uri "https://api.groq.com/openai/v1/models" -Headers @{ Authorization = "Bearer $key" } | Select -Expand data | Select id
```

---

### BLOC 2 — Dépendances Python et modèles ML locaux

#### TODO 2.1 — Vérifier l'environnement Python
```powershell
cd "C:\Users\Adham Ferchichi\Documents\recommendation\adoption system\pet-advisor"
.\.venv\Scripts\Activate.ps1
python -c "import fastapi, groq, langgraph, torch, faiss, sentence_transformers, open_clip; print('OK')"
```
Si une erreur : `pip install -r requirements.txt`

#### TODO 2.2 — Pré-télécharger SentenceTransformer (nécessite internet au premier run)
Le modèle `all-MiniLM-L6-v2` est téléchargé depuis HuggingFace au premier démarrage.
Sans connexion internet, le chargement des artéfacts Maya échouera silencieusement.
```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Cached OK')"
```

#### TODO 2.3 — Pré-télécharger CLIP ViT-B/32 (nécessite internet au premier run)
Requis pour la fonctionnalité « chien perdu » de Maya.
```powershell
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); print('CLIP cached OK')"
```

#### TODO 2.4 — (Optionnel) Installer ultralytics pour YOLO complet
```powershell
pip install ultralytics
```
Sans ultralytics, la détection d'agression fonctionne en mode lite (heuristique mouvement). Avec ultralytics, YOLOv8-pose + ByteTrack est activé.

---

### BLOC 3 — Artéfacts ML (cohérence notebook → backend)

**Les artéfacts sont déjà présents** dans `artifacts/aiAdvisor/`. Ce bloc sert à vérifier leur intégrité ou à les régénérer si corrompus.

Vérification rapide :
```powershell
python -c "
import numpy as np
emb = np.load('artifacts/aiAdvisor/pet_emb.npy')
ids = np.load('artifacts/aiAdvisor/pet_ids.npy', allow_pickle=True)
print(f'pet_emb: {emb.shape}  (attendu: (8132, 128))')
print(f'pet_ids: {ids.shape}  (attendu: (8132,))')
import numpy as np; img = np.load('artifacts/aiAdvisor/img_emb_raw.npy')
print(f'img_emb_raw: {img.shape}  (attendu: (8132, 512))')
"
```

**Cohérence notebook `cosinesimilarity2.ipynb` → backend `retrieval.py` :**

Le notebook produit les artéfacts suivants consommés par le backend :

| Fichier | Produit par | Consommé par | Description |
|---|---|---|---|
| `pet_emb.npy` (8132×128) | cosinesimilarity2 section 10 | retrieval.py `_index` | Embeddings fusionnés L2-normalisés |
| `pet_ids.npy` (8132,) | cosinesimilarity2 section 13 | retrieval.py `_pet_ids` | IDs alignés avec pet_emb |
| `user_encoder.pt` | cosinesimilarity2 section 9 | retrieval.py `_user_encoder` | MLP 384-d → 128-d (InfoNCE) |
| `pet_encoder.pt` | cosinesimilarity2 section 9 | (non chargé à l'inférence) | Référence pour réentraînement |
| `df_original.parquet` | cosinesimilarity2 section 2 | retrieval.py `_df` | Métadonnées PetFinder |
| `thresholds.json` | cosinesimilarity2 section 11 | retrieval.py `_thresholds` | Statistiques de distance |
| `pet_features_shape.txt` | cosinesimilarity2 section 8 | (informatif) | Dimension FUSED_DIM |
| `img_emb_raw.npy` (8132×512) | cosinesimilarity2 section 5 | dog_finder.py `_dog_index` | CLIP 512-d bruts |
| `rich_descriptions.json` | cosinesimilarity2 section 12 | retrieval.py `_rich_descriptions` | Descriptions LLM enrichies |
| `pet_data_raw.json` | cosinesimilarity2 section 3 | (debug / référence) | Attributs structurés bruts |
| `pca_text.pkl`, `pca_img.pkl` | cosinesimilarity2 section 6 | NON utilisés à l'inférence | PCA appliquée au training uniquement |
| `chroma_db/` | cosinesimilarity2 section 13 | retrieval.py `_chroma_collection` | Optionnel — FAISS si absent |

**Cohérence notebook `animal_aggression_detection_v3.ipynb` → backend `aggression/` :**

Le notebook est autonome et ne produit pas d'artéfacts persistants pour le backend.
La pipeline complète (EMA + hysteresis + ByteTrack + mosaïque + agent LangGraph) est réimplémentée nativement dans `backend/aggression/video_analyzer.py`.
Les paramètres clés (EMA_ALPHA, HYSTERESIS_HIGH, HYSTERESIS_LOW, MIN_SUSTAINED, etc.) dans le notebook correspondent exactement aux constantes en haut de `video_analyzer.py`.

---

### BLOC 4 — Base de données refuges (Ali)

Vérifier que la base contient des refuges :
```powershell
python -c "
import sqlite3
db = sqlite3.connect('artifacts/aliAdvisor/shelter_assignment.db')
print('Shelters:', db.execute('SELECT COUNT(*) FROM shelters').fetchone()[0])
print('Dogs:    ', db.execute('SELECT COUNT(*) FROM dogs').fetchone()[0])
"
```
Si vide → le serveur recharge automatiquement depuis `data/shelters_seed.json` au démarrage suivant.

---

### BLOC 5 — Détection d'agression — contacts

Éditer `artifacts/agressionDetection/aggression_config.json` pour remplacer les emails de démo par les vrais destinataires :
```json
{
  "ema_alpha": 0.12,
  "hysteresis_high": 0.55,
  "hysteresis_low": 0.28,
  "min_sustained_frames": 10,
  "cooldown_frames": 25,
  "proximity_norm_max": 0.38,
  "track_ttl_frames": 60,
  "mosaic_max_frames": 12,
  "contacts": {
    "hospital_email":      "urgences@hopital-reel.org",
    "hospital_label":      "Service des urgences",
    "animal_rights_email": "alertes@spa-reelle.org",
    "animal_rights_label": "SPA locale"
  }
}
```

---

### BLOC 6 — Démarrage du serveur

```powershell
cd "C:\Users\Adham Ferchichi\Documents\recommendation\adoption system\pet-advisor"
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Logs attendus au démarrage :**
```
[retrieval] Rich descriptions loaded: 8132 pets.
[retrieval] Ready — ChromaDB (8132 pets).   ← ou : Ready — FAISS (8132 pets).
[dog_finder] CLIP ViT-B/32 loaded.
[dog_finder] Zero-shot CLIP index ready: N dogs.
[Ali] Agent ready — model: meta-llama/llama-4-scout-17b-16e-instruct
INFO:     Application startup complete.
```

**Si le démarrage échoue :**
- `WARNING — missing artifacts` → Bloc 3
- `Agent not initialised` → Bloc 1
- `SentenceTransformer download failed` → TODO 2.2 (pas d'internet)
- `open_clip not installed` → TODO 2.3
- Port occupé → `Get-Process -Name python | Stop-Process -Force`

---

## 2. Modèles LLM

| Composant | Modèle par défaut | Variable override |
|---|---|---|
| Maya — adoption advisor | `meta-llama/llama-4-scout-17b-16e-instruct` | `GROQ_MODEL` |
| Ali — shelter coordinator | `meta-llama/llama-4-scout-17b-16e-instruct` | `GROQ_MODEL` |
| Agression — texte | `llama-3.3-70b-versatile` | `GROQ_TEXT_MODEL` |
| Agression — vision | `meta-llama/llama-4-scout-17b-16e-instruct` | `GROQ_VISION_MODEL` |

---

## 3. Référence `.env`

### Obligatoire

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Clé API Groq — obtenir sur https://console.groq.com |

### LLM (optionnel — defaults inclus)

| Variable | Défaut | Fallback |
|---|---|---|
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | `llama-3.3-70b-versatile` |
| `GROQ_TEXT_MODEL` | `llama-3.3-70b-versatile` | `llama3-70b-8192` |
| `GROQ_VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | `llama-3.2-11b-vision-preview` |
| `YOLO_POSE_MODEL` | `yolov8l-pose.pt` | — |

### SMTP — alertes agression (optionnel)

| Variable | Exemple |
|---|---|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | `you@gmail.com` |
| `SMTP_PASS` | mot de passe app Gmail (https://myaccount.google.com/apppasswords) |

---

## 4. Endpoints principaux

| Méthode | URL | Description |
|---|---|---|
| `GET` | `/` | Page d'accueil |
| `GET` | `/maya` | Interface Maya |
| `GET` | `/ali` | Interface Ali |
| `GET` | `/aggression` | Interface détection |
| `GET` | `/health` | `{"status": "ok"}` |
| `POST` | `/auth/login` | Créer session Maya |
| `POST` | `/upload/{session_id}` | Upload image chien perdu |
| `WS` | `/ws/maya/{session_id}` | Chat Maya |
| `WS` | `/ws/ali/{session_id}` | Chat Ali |
| `POST` | `/ali/upload` | Upload Excel → assignation |
| `GET` | `/ali/shelters` | Liste refuges |
| `GET` | `/ali/statistics` | Statistiques |
| `POST` | `/aggression/analyze-video` | Analyser vidéo |
| `GET` | `/aggression/incidents` | Historique incidents |

---

## 5. Dépannage rapide

| Symptôme | Cause | Solution |
|---|---|---|
| Maya : « Internal error. Please check GROQ_API_KEY » | Clé expirée ou modèle introuvable | **Bloc 1** |
| Ali : WS ne se connecte jamais | `GROQ_API_KEY` manquant au démarrage | **Bloc 1** |
| `WARNING — missing artifacts` dans les logs | Artéfacts absents ou corrompus | **Bloc 3** |
| `SentenceTransformer download failed` | Pas d'internet au premier run | **TODO 2.2** |
| CLIP ne se charge pas | `open_clip` non installé ou pas d'internet | **TODO 2.3** |
| Agression HTTP 503 | `ultralytics` non installé | **TODO 2.4** |
| Port 8000 occupé | Ancien process Python actif | `Get-Process -Name python \| Stop-Process -Force` |
| Maya répète les mêmes animaux | ChromaDB corrompu | Supprimer `artifacts/aiAdvisor/chroma_db/`, relancer |
| Emails d'alerte non envoyés | SMTP ou mot de passe app expiré | **Bloc 5** |
| Ali répond mais voit 0 refuges | Base vide | Redémarrer le serveur (seed auto) |
