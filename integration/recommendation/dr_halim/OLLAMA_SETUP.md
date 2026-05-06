# Configuration Ollama3 pour Dr Halim

## 1️⃣ Installation et démarrage d'Ollama

### Windows
1. **Télécharger** : https://ollama.ai
2. **Installer** et redémarrer le PC
3. **Vérifier l'installation** :
   ```powershell
   ollama --version
   ```

### Démarrer le service Ollama
```powershell
# Option 1: Commande simple (en PowerShell Administrateur)
ollama serve

# Option 2: En tant que service Windows (automatique au démarrage)
# L'installeur configure Ollama comme service
```

## 2️⃣ Télécharger le modèle llama3

Une fois Ollama est en cours d'exécution:
```powershell
ollama pull llama3
```

**Temps estimé**: 5-15 minutes selon votre connexion internet
**Taille du modèle**: ~4.7 GB

### Vérifier les modèles installés:
```powershell
ollama list
```

Vous devriez voir:
```
NAME               ID              SIZE      MODIFIED
llama3:latest      365c0bd3c000    4.7GB     2 days ago
```

## 3️⃣ Vérifier la connexion

Ollama s'exécute par défaut sur `http://127.0.0.1:11434`

### Test API simple:
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" | ConvertTo-Json
```

Ou en Python:
```python
import requests
resp = requests.get("http://127.0.0.1:11434/api/tags")
print(resp.json())
```

## 4️⃣ Configuration pour Dr Halim

### Vérifier `main.py`:
- ✅ `OLLAMA_URL = "http://127.0.0.1:11434/api/generate"` — déjà configuré
- ✅ Le modèle utilisé est `"llama3"` — déjà correct

### Démarrer le backend Dr Halim:
```powershell
cd dr_halim
python backend/main.py
```

## 5️⃣ Tester l'intégration

### Curl:
```bash
curl -X POST "http://localhost:8000/predict-breed" \
  -F "file=@path/to/dog_image.jpg"
```

### Python:
```python
import requests

with open("dog_image.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/predict-breed",
        files={"file": f}
    )
print(resp.json())
```

## ⚠️ Troubleshooting

### Ollama ne démarre pas
```powershell
# Vérifier le statut du service
Get-Service | Where-Object {$_.Name -like "*ollama*"}

# Redémarrer le service
Restart-Service -Name "OllamaService" -Force
```

### "Connection refused" ou erreur HTTP
1. Vérifier que Ollama est en cours d'exécution:
   ```powershell
   Get-Process | grep ollama
   ```
2. Vérifier le port:
   ```powershell
   netstat -ano | findstr "11434"
   ```

### La génération de texte est lente
- C'est normal pour Ollama sur CPU
- Temps estimé: 30-120 secondes par réponse
- Utilisez le paramètre `timeout=120` dans `requests.post()` (déjà configuré)

## 📝 Paramètres d'optimisation (optionnel)

Modifier `main.py` pour ajuster la réponse d'Ollama:

```python
resp = requests.post(
    OLLAMA_URL,
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 300,      # Limiter la longueur de réponse
            "temperature": 0.7,      # Contrôler la créativité (0-1)
        }
    },
    timeout=120,
)
```

## ✅ Checklist finale

- [ ] Ollama3 installé
- [ ] `ollama pull llama3` exécuté avec succès
- [ ] `ollama serve` en cours d'exécution
- [ ] `http://127.0.0.1:11434/api/tags` répond avec llama3
- [ ] Backend Dr Halim démarre sans erreurs
- [ ] Test `/predict-breed` retourne recommandation Ollama

---

**Notes**: 
- Ollama peut être arrêté/redémarré sans redémarrer le backend Dr Halim (l'API gère les erreurs gracieusement)
- Pour une meilleure performance en production, considérez un GPU (NVIDIA CUDA)
