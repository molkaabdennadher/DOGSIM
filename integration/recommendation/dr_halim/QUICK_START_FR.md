# 🐕 Dr Halim - Configuration Ollama3 llama3

## 📋 Résumé rapide

Vous avez téléchargé **Ollama3**. Voici comment configurer la détection de race de chien avec **llama3**:

---

## ⚡ Démarrage en 5 minutes

### Option 1: Script automatisé (recommandé) ⭐

**Étape 1** - Télécharger et configurer llama3:
```powershell
# En PowerShell Administrateur
cd adoption_system/dr_halim
.\setup_ollama_llama3.ps1
```

Répondez `y` pour démarrer `ollama serve`.

**Étape 2** - Dans un **nouveau terminal PowerShell**, démarrer le backend:
```powershell
cd adoption_system/dr_halim
python backend/main.py
```

**Étape 3** - Tester (dans un **3e terminal**):
```powershell
python test_breed_detection.py
```

---

### Option 2: Commandes manuelles

**Terminal 1** - Démarrer Ollama:
```powershell
ollama pull llama3      # Télécharger (une fois seulement)
ollama serve            # Démarrer le service
```

**Terminal 2** - Démarrer le backend:
```powershell
cd adoption_system/dr_halim
python backend/main.py
```

**Terminal 3** - Tester:
```powershell
python test_breed_detection.py
```

---

### Option 3: One-click launcher

```powershell
cd adoption_system/dr_halim
.\start_dr_halim_with_ollama.ps1
```

Cela ouvre **2 terminaux automatiquement** (Ollama + Backend).

---

## 🔧 Configuration

### Fichiers créés:

| Fichier | Objectif |
|---------|----------|
| `OLLAMA_SETUP.md` | Documentation complète d'Ollama |
| `setup_ollama_llama3.ps1` | Installation et téléchargement de llama3 |
| `start_dr_halim_with_ollama.ps1` | Launcher "one-click" (Ollama + Backend) |
| `test_breed_detection.py` | Script de test de l'API |

### Code modifié automatiquement:

- ✅ `backend/main.py` — Ollama intégré à l'endpoint `/predict-breed`
- ✅ `OLLAMA_URL = "http://127.0.0.1:11434/api/generate"` — Déjà configuré
- ✅ Modèle utilisé: `llama3` — Prêt à l'emploi

---

## 📍 URLs de l'API

Une fois lancé:

```
🔷 Ollama API:        http://127.0.0.1:11434
🟦 Dr Halim Backend:  http://localhost:8000

Endpoints:
  POST /predict-breed    → Détection de race (avec recommandation Ollama)
  POST /predict-skin     → Analyse de peau
  POST /predict-behavior → Analyse de comportement
  GET  /                → Health check
```

---

## 🧪 Test simple

### Avec curl:
```bash
curl -X POST "http://localhost:8000/predict-breed" \
  -F "file=@dog_image.jpg"
```

### Avec Python:
```python
import requests

with open("dog_image.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/predict-breed",
        files={"file": f}
    )
    print(resp.json())
```

---

## 🐢 Si c'est lent (sur CPU)

C'est **normal** ! Ollama sur CPU prend **30-120 secondes** par réponse.

Pour accélérer:
1. **GPU NVIDIA** → Installez CUDA + `ollama serve --gpu` (5-30x plus rapide)
2. **Modèle plus petit** → Remplacez `llama3` par `mistral` ou `neural-chat`
3. **Réponses courtes** → Modifiez le `num_predict` dans `main.py`

---

## ⚠️ Troubleshooting

### ❌ "Connection refused" / "Cannot connect to Ollama"
```powershell
# Vérifier qu'Ollama est en cours d'exécution
Get-Process ollama

# Vérifier le port 11434
netstat -ano | findstr "11434"

# Redémarrer Ollama
ollama serve
```

### ❌ "llama3 not found"
```powershell
ollama pull llama3
ollama list  # Vérifier que llama3 apparaît
```

### ❌ Backend Python ne démarre pas
```powershell
# Vérifier les dépendances
cd adoption_system/dr_halim
pip install -r requirements.txt

# Relancer
python backend/main.py
```

### ❌ "ModuleNotFoundError: No module named 'fastapi'"
```powershell
pip install fastapi uvicorn
```

---

## 📊 Flux de détection de race

```
Image du chien
    ↓
[TensorFlow EfficientNet] ← Détection de race rapide (< 1s)
    ↓
Race: "Labrador Retriever" (confiance: 92%)
    ↓
[Ollama3 llama3] ← Génération de recommandation (30-120s)
    ↓
Recommandation: "Les Labradors sont excellents pour..."
    ↓
Réponse JSON à l'utilisateur
```

---

## 📝 Fichiers modifiés

### `backend/main.py` (snippet)

```python
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Dans /predict-breed:
try:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": "llama3", "prompt": prompt, "stream": False},
        timeout=120,
    )
    recommendation = resp.json().get("response", fallback_message)
except Exception:
    recommendation = "Ollama not available - install and start Ollama"
```

---

## ✅ Checklist avant d'utiliser

- [ ] Ollama installé (`ollama --version` fonctionne)
- [ ] llama3 téléchargé (`ollama list` affiche llama3)
- [ ] Ollama en cours d'exécution (`ollama serve` actif dans un terminal)
- [ ] Backend Dr Halim lancé (`python backend/main.py` sans erreurs)
- [ ] API accessible (`http://localhost:8000/` retourne `{"status": "ok"}`)
- [ ] Premier test réussi (`python test_breed_detection.py` avec succès)

---

## 🎯 Prochaines étapes

1. **Démarrer les services** → Suivez "Démarrage en 5 minutes"
2. **Tester l'API** → `python test_breed_detection.py`
3. **Intégrer au frontend** → Utilisez les endpoints `/predict-*`
4. **Optimiser les performances** → Voir "Si c'est lent" section

---

## 📞 Support

Si vous rencontrez des problèmes:

1. Vérifiez `OLLAMA_SETUP.md` pour la documentation détaillée
2. Consultez la section "Troubleshooting" ci-dessus
3. Vérifiez que les ports `8000` (backend) et `11434` (Ollama) sont libres
4. Redémarrez les services en cas de doute

---

**Bon courage! 🚀**
