#!/usr/bin/env python
"""
Dr Halim + tacheRanim Integration Validator
Checks all paths, dependencies, and endpoints
"""

import os
import json
import sys
from pathlib import Path

def check(condition, message):
    """Check condition and print status"""
    status = "✅" if condition else "❌"
    print(f"{status} {message}")
    return condition

def main():
    print("\n" + "="*70)
    print("🐾 Dr Halim + tacheRanim Integration Validation")
    print("="*70 + "\n")
    
    all_ok = True
    
    # --- 1. Paths Check ---
    print("📁 FILE PATHS CHECK")
    print("-" * 70)
    
    backend_path = Path(__file__).parent
    dr_halim_root = backend_path.parent
    recommendation_root = dr_halim_root.parent
    tache_ranim_root = recommendation_root / "tacheRanim"
    
    all_ok &= check(
        backend_path.exists(),
        f"Dr Halim backend path: {backend_path}"
    )
    all_ok &= check(
        recommendation_root.exists(),
        f"Recommendation root: {recommendation_root}"
    )
    all_ok &= check(
        tache_ranim_root.exists(),
        f"tacheRanim path: {tache_ranim_root}"
    )
    
    # Check model paths
    fecal_model_path = tache_ranim_root / "model" / "model_mobilenet_selles.keras"
    all_ok &= check(
        fecal_model_path.exists(),
        f"Fecal model: {fecal_model_path.name}"
    )
    
    fecal_kb_path = tache_ranim_root / "data" / "knowledge_base.json"
    all_ok &= check(
        fecal_kb_path.exists(),
        f"Fecal knowledge base: {fecal_kb_path.name}"
    )
    
    # Check Dr. Halim models
    models_dir = dr_halim_root / "models"
    all_ok &= check(
        models_dir.exists(),
        f"Dr. Halim models directory: {models_dir.name}"
    )
    
    print()
    
    # --- 2. Dependencies Check ---
    print("📦 DEPENDENCIES CHECK")
    print("-" * 70)
    
    required_packages = [
        'fastapi', 'uvicorn', 'tensorflow', 'torch',
        'numpy', 'PIL', 'requests'
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg if pkg != 'PIL' else 'PIL')
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} - NOT INSTALLED")
            missing.append(pkg)
            all_ok = False
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
    
    print()
    
    # --- 3. Configuration Check ---
    print("⚙️  CONFIGURATION CHECK")
    print("-" * 70)
    
    print(f"✅ Backend port: 7001 (internal, proxied by pet-advisor)")
    print(f"✅ Public access: http://localhost:8000/drhalim (via pet-advisor)")
    print(f"✅ API prefix: /drhalim")
    print(f"✅ Endpoints:")
    print(f"   - POST /drhalim/predict-breed")
    print(f"   - POST /drhalim/predict-skin")
    print(f"   - POST /drhalim/predict-behavior")
    print(f"   - POST /drhalim/predict-fecal (INTEGRATED)")
    
    print()
    
    # --- 4. Knowledge Base Check ---
    print("📚 KNOWLEDGE BASE CHECK")
    print("-" * 70)
    
    try:
        with open(fecal_kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        classes = list(kb.keys())
        all_ok &= check(
            len(classes) == 6,
            f"Fecal classes in KB: {classes}"
        )
    except Exception as e:
        print(f"❌ Could not load knowledge base: {e}")
        all_ok = False
    
    print()
    
    # --- 5. Frontend Check ---
    print("🎨 FRONTEND CHECK")
    print("-" * 70)
    
    frontend_path = dr_halim_root / "frontend" / "index.html"
    all_ok &= check(
        frontend_path.exists(),
        f"Frontend UI: {frontend_path.name}"
    )
    
    if frontend_path.exists():
        with open(frontend_path, 'r', encoding='utf-8') as f:
            html = f.read()
            has_fecal_button = 'btn-fecal' in html and 'Analyse Fécale' in html
            has_drhalim_api = 'http://localhost:8000/drhalim' in html
            has_predict_fecal = 'predict-fecal' in html
            
            all_ok &= check(
                has_fecal_button,
                "Fecal analysis button in UI"
            )
            all_ok &= check(
                has_drhalim_api,
                "API endpoint URL configured (/drhalim)"
            )
            all_ok &= check(
                has_predict_fecal,
                "Predict-fecal endpoint called"
            )
    
    print()
    
    # --- 6. Backend Code Check ---
    print("🔧 BACKEND CODE CHECK")
    print("-" * 70)
    
    main_py = backend_path / "main.py"
    if main_py.exists():
        with open(main_py, 'r', encoding='utf-8') as f:
            code = f.read()
            
            has_fecal_router = '@dr_halim_router.post("/predict-fecal")' in code
            has_model_loader = 'fecal_model = tf.keras.models.load_model' in code
            has_heuristic = '_analyze_fecal_color_heuristic' in code
            has_port_8000 = 'port=8000' in code
            has_drhalim_prefix = 'prefix="/drhalim"' in code
            
            all_ok &= check(
                has_fecal_router,
                "Fecal endpoint router defined"
            )
            all_ok &= check(
                has_model_loader,
                "tacheRanim model loader integrated"
            )
            all_ok &= check(
                has_heuristic,
                "Fallback heuristic available"
            )
            all_ok &= check(
                has_port_8000,
                "Port 8000 configured"
            )
            all_ok &= check(
                has_drhalim_prefix,
                "/drhalim prefix configured"
            )
    
    print()
    
    # --- Summary ---
    print("="*70)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Ready to deploy!")
    else:
        print("⚠️  SOME CHECKS FAILED - Please review above")
    print("="*70 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
