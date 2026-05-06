#!/usr/bin/env python3
"""
Configuration check script for Happy Paws Pet Advisor.
Verifies Python version, dependencies, artifacts, and API keys.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load .env explicitly
env_file = ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    print(f"Warning: .env not found at {env_file}")

def check_python():
    """Verify Python 3.12."""
    v = sys.version_info
    status = "✓" if v[:2] == (3, 12) else "✗"
    print(f"{status} Python: {v.major}.{v.minor}.{v.micro}")
    return v[:2] == (3, 12)

def check_dependencies():
    """Verify core packages are installed."""
    packages = [
        'fastapi', 'uvicorn', 'openai', 'groq', 
        'langgraph', 'langchain_core',
        'torch', 'sentence_transformers', 'open_clip',
        'faiss', 'numpy', 'pandas', 'pydantic'
    ]
    all_ok = True
    print("\nDépendances Python:")
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — MISSING, run: pip install -r requirements.txt")
            all_ok = False
    return all_ok

def check_artifacts():
    """Verify ML artifacts exist and have correct shapes."""
    print("\nArtéfacts ML (aiAdvisor):")
    artifacts_dir = ROOT / "artifacts" / "aiAdvisor"
    
    checks = [
        ("pet_emb.npy", (8132, 128)),
        ("pet_ids.npy", (8132,)),
        ("img_emb_raw.npy", (8132, 512)),
        ("user_encoder.pt", None),
        ("pet_encoder.pt", None),
        ("df_original.parquet", None),
        ("thresholds.json", None),
        ("pet_features_shape.txt", None),
    ]
    
    all_ok = True
    for filename, expected_shape in checks:
        path = artifacts_dir / filename
        if not path.exists():
            print(f"  ✗ {filename} — MISSING")
            all_ok = False
            continue
        
        try:
            if filename.endswith('.npy'):
                import numpy as np
                arr = np.load(path, allow_pickle=True)
                if expected_shape and arr.shape != expected_shape:
                    print(f"  ✗ {filename} — shape {arr.shape}, expected {expected_shape}")
                    all_ok = False
                else:
                    print(f"  ✓ {filename} — shape {arr.shape}")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {filename} — {size_mb:.1f} MB")
        except Exception as e:
            print(f"  ✗ {filename} — ERROR: {e}")
            all_ok = False
    
    return all_ok

def check_ml_models():
    """Pre-download SentenceTransformer and CLIP."""
    print("\nModèles ML (à télécharger une fois):")
    all_ok = True
    
    # SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"  ✓ SentenceTransformer ('all-MiniLM-L6-v2')")
    except Exception as e:
        print(f"  ✗ SentenceTransformer — ERROR: {e}")
        all_ok = False
    
    # CLIP
    try:
        import open_clip
        model, preprocess, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        print(f"  ✓ CLIP (ViT-B-32)")
    except Exception as e:
        print(f"  ✗ CLIP — ERROR: {e}")
        all_ok = False
    
    return all_ok

def check_groq_api():
    """Verify Groq API key and model availability."""
    print("\nGroq API Configuration:")
    
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("  ✗ GROQ_API_KEY — NOT SET in .env")
        return False
    
    if api_key.startswith("gsk_"):
        print(f"  ✓ GROQ_API_KEY — présente (gsk_...)")
    else:
        print(f"  ✗ GROQ_API_KEY — format invalide (doit commencer par 'gsk_')")
        return False
    
    # Try to list models
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        models = client.models.list()
        available_models = [m.id for m in models.data]
        
        print(f"\n  Modèles disponibles ({len(available_models)}):")
        for model_id in sorted(available_models):
            print(f"    - {model_id}")
        
        # Check configured models
        print("\n  Modèles configurés:")
        configured = {
            'GROQ_MODEL': os.environ.get('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct'),
            'GROQ_TEXT_MODEL': os.environ.get('GROQ_TEXT_MODEL', 'llama-3.3-70b-versatile'),
            'GROQ_VISION_MODEL': os.environ.get('GROQ_VISION_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct'),
        }
        
        all_configured_ok = True
        for var, model_id in configured.items():
            if model_id in available_models:
                print(f"    ✓ {var} = {model_id}")
            else:
                print(f"    ✗ {var} = {model_id} — NOT AVAILABLE")
                all_configured_ok = False
        
        return all_configured_ok
        
    except Exception as e:
        print(f"  ✗ ERROR connecting to Groq API: {e}")
        print(f"    Vérifier la clé API sur https://console.groq.com/keys")
        return False

def check_env_file():
    """Verify .env file exists."""
    env_path = ROOT / ".env"
    if env_path.exists():
        print("✓ Fichier .env présent")
        return True
    else:
        print("✗ Fichier .env MANQUANT")
        return False

def main():
    print("=" * 70)
    print("VÉRIFICATION DE CONFIGURATION — Happy Paws Pet Advisor")
    print("=" * 70)
    
    checks = {
        "Fichier .env": check_env_file,
        "Python 3.12": check_python,
        "Dépendances Python": check_dependencies,
        "Artifacts ML": check_artifacts,
        "Modèles ML": check_ml_models,
        "Groq API": check_groq_api,
    }
    
    results = {}
    for name, check_fn in checks.items():
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n✗ {name} — ERREUR: {e}")
            results[name] = False
        print()
    
    print("=" * 70)
    print("RÉSUMÉ:")
    print("=" * 70)
    
    all_ok = True
    for name, result in results.items():
        status = "✓ OK" if result else "✗ PROBLÈME"
        print(f"  {status:8} — {name}")
        all_ok = all_ok and result
    
    print("=" * 70)
    if all_ok:
        print("✓ CONFIGURATION COMPLÈTE — l'application peut démarrer!")
        print("\nPour lancer l'application:")
        print("  python start.py              # port 8000, sans auto-reload")
        print("  python start.py --reload     # développement avec auto-reload")
        print("  python start.py --port 9000  # port personnalisé")
        return 0
    else:
        print("✗ DES PROBLÈMES DÉTECTÉS — voir ci-dessus")
        return 1

if __name__ == "__main__":
    sys.exit(main())
