"""
test_breed_detection.py
Teste l'endpoint /predict-breed avec Ollama3
"""

import sys
import json
import time
from pathlib import Path

import requests
from PIL import Image

# ======================================================================
#  CONFIGURATION
# ======================================================================
BACKEND_URL = "http://localhost:8000"
BREED_ENDPOINT = f"{BACKEND_URL}/predict-breed"
OLLAMA_API = "http://127.0.0.1:11434/api/tags"

# ======================================================================
#  HELPERS
# ======================================================================
def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def check_ollama():
    """Vérifier que Ollama est en cours d'exécution."""
    try:
        resp = requests.get(OLLAMA_API, timeout=5)
        data = resp.json()
        models = [m.get("name") for m in data.get("models", [])]
        
        if "llama3:latest" in models or any("llama3" in m for m in models):
            print("✅ Ollama est actif avec llama3")
            return True
        else:
            print(f"❌ llama3 non trouvé. Modèles disponibles: {models}")
            return False
    except Exception as e:
        print(f"❌ Ollama ne répond pas: {e}")
        return False

def check_backend():
    """Vérifier que le backend Dr Halim est actif."""
    try:
        resp = requests.get(f"{BACKEND_URL}/", timeout=5)
        if resp.status_code == 200:
            print(f"✅ Backend Dr Halim actif: {resp.json()}")
            return True
    except Exception as e:
        print(f"❌ Backend ne répond pas: {e}")
        return False

def test_breed_detection(image_path):
    """Test la détection de race avec une image."""
    if not Path(image_path).exists():
        print(f"❌ Image non trouvée: {image_path}")
        return None
    
    # Vérifier que c'est une image valide
    try:
        img = Image.open(image_path)
        print(f"✅ Image chargée: {img.size} {img.mode}")
    except Exception as e:
        print(f"❌ Image invalide: {e}")
        return None
    
    # Envoyer à l'API
    print(f"\n📤 Envoi de l'image à {BREED_ENDPOINT}...")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            start = time.time()
            resp = requests.post(BREED_ENDPOINT, files=files, timeout=300)
            elapsed = time.time() - start
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"\n✅ Réponse reçue en {elapsed:.1f} secondes:")
            print(f"\n  Détection de race:")
            print(f"    - Race: {result['breed']}")
            print(f"    - Confiance: {result['confidence']:.2%}")
            
            if "recommendation" in result:
                print(f"\n  Recommandation Ollama:")
                rec = result['recommendation']
                if len(rec) > 300:
                    print(f"    {rec[:300]}...")
                else:
                    print(f"    {rec}")
            
            return result
        else:
            print(f"❌ Erreur API: {resp.status_code}")
            print(f"   Réponse: {resp.text}")
            return None
    except requests.Timeout:
        print("❌ Timeout - la requête a pris trop longtemps")
        print("   (Ollama peut être lent sur CPU, essayez avec timeout_sec=300)")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

# ======================================================================
#  MAIN
# ======================================================================
def main():
    print_header("TEST - Dr Halim Breed Detection avec Ollama3")
    
    # Étape 1: Vérifier Ollama
    print("1️⃣  Vérification d'Ollama...")
    if not check_ollama():
        print("\n⚠️  Ollama n'est pas prêt.")
        print("   Exécutez en PowerShell: ollama serve")
        return 1
    
    # Étape 2: Vérifier backend
    print("\n2️⃣  Vérification du backend Dr Halim...")
    if not check_backend():
        print("\n⚠️  Backend Dr Halim n'est pas prêt.")
        print("   Exécutez: cd dr_halim && python backend/main.py")
        return 1
    
    # Étape 3: Chercher une image de test
    print("\n3️⃣  Recherche d'image de test...")
    test_images = list(Path(".").glob("**/*.jpg")) + \
                  list(Path(".").glob("**/*.png"))
    
    if not test_images:
        print("❌ Aucune image .jpg ou .png trouvée dans le répertoire courant")
        print("\n   Téléchargez une image de chien et réessayez, ou utilisez:")
        print("   python test_breed_detection.py <chemin_image>")
        return 1
    
    image_path = test_images[0]
    print(f"📸 Image de test: {image_path}")
    
    # Étape 4: Test
    print("\n4️⃣  Test de détection de race...")
    result = test_breed_detection(str(image_path))
    
    if result:
        print_header("✅ TEST RÉUSSI")
        print("Le système est prêt pour la détection de race avec Ollama3")
        return 0
    else:
        print_header("❌ TEST ÉCHOUÉ")
        return 1

if __name__ == "__main__":
    image_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if image_arg:
        # Mode: test avec une image spécifique
        print_header("Test avec image fournie")
        result = test_breed_detection(image_arg)
        sys.exit(0 if result else 1)
    else:
        sys.exit(main())
