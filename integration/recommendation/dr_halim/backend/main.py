"""
Dr Halim - Unified AI Vet Backend
Endpoints:
  POST /predict-breed    -> breed detection + Ollama recommendation
  POST /predict-skin     -> skin/coat analysis
  POST /predict-behavior -> behaviour analysis (rage risk)
  POST /predict-fecal    -> fecal analysis (tacheRanim MobileNet + GROQ)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import asyncio
import colorsys
import io
import json
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from PIL import Image

# --- TensorFlow -----------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    decode_predictions as _eff_decode,
    preprocess_input as _eff_preprocess,
)

# --- PyTorch (skin model) -------------------------------------------------
import torch
import torchvision.transforms as transforms
from torchvision import models as torch_models

# --- Behaviour inference --------------------------------------------------
from inference import (
    BEHAV_ANORMAL_IDX,
    extract_audio,
    extract_frames,
    has_voice_activity,
    late_fusion_decision,
    predict_behavior,
    predict_voice,
)

# --- GROQ (fecal LLM) -----------------------------------------------------
try:
    from groq import Groq as _Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# ======================================================================
#  PATHS
# ======================================================================
_HERE            = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR       = os.path.normpath(os.path.join(_HERE, "..", "models"))
_TACHE_RANIM_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "tacheRanim"))
OLLAMA_URL       = "http://127.0.0.1:11434/api/generate"   # kept for reference, not used
GROQ_API_KEY_BREED = "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"
GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_BREED   = "llama-3.3-70b-versatile"

# ======================================================================
#  FASTAPI APP
# ======================================================================
app = FastAPI(title="Dr Halim - AI Vet API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router with NO prefix — the proxy already strips /drhalim from the path
router = APIRouter()

# ======================================================================
#  LOAD MODELS AT STARTUP
# ======================================================================

# --- Fecal model (tacheRanim MobileNet) ----------------------------------
print("Loading fecal matter model ...")
fecal_model  = None
_FECAL_KB    = {}
_FECAL_CLASSES = ["Blood", "Diarrhoea", "LackOfWater", "Normal", "SoftPoop", "Worms"]

_fecal_model_path = os.path.join(_TACHE_RANIM_ROOT, "model", "model_mobilenet_selles.keras")
_fecal_kb_path    = os.path.join(_TACHE_RANIM_ROOT, "data", "knowledge_base.json")

try:
    if os.path.exists(_fecal_model_path):
        fecal_model = tf.keras.models.load_model(_fecal_model_path)
        with open(_fecal_kb_path, "r", encoding="utf-8") as _f:
            _FECAL_KB = json.load(_f)
        print("Fecal model ready.")
    else:
        print(f"WARNING: Fecal model not found at {_fecal_model_path}")
except Exception as _e:
    print(f"WARNING: Could not load fecal model: {_e}")

# --- Breed model (EfficientNetB0 pre-trained ImageNet) -------------------
print("Loading breed model ...")
breed_model = EfficientNetB0(weights="imagenet", include_top=True)

with open(os.path.join(MODELS_DIR, "class_names.json"), "r") as _f:
    _class_names = json.load(_f)

_synset_to_breed: dict[str, str] = {
    cn.split("-")[0]: cn.split("-", 1)[1].replace("_", " ").title()
    for cn in _class_names if "-" in cn
}
print("Breed model ready.")

# --- Skin model (PyTorch EfficientNet-B0) --------------------------------
print("Loading skin model ...")
DEVICE = torch.device("cpu")
skin_model = torch_models.efficientnet_b0(weights=None)
skin_model.classifier[1] = torch.nn.Linear(skin_model.classifier[1].in_features, 2)
skin_model.load_state_dict(
    torch.load(
        os.path.join(MODELS_DIR, "skin_model.pth"),
        map_location=DEVICE,
        weights_only=True,
    )
)
skin_model.to(DEVICE).eval()

with open(os.path.join(MODELS_DIR, "skin_class_names.json"), "r") as _f:
    skin_classes = json.load(_f)

skin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
print("Skin model ready.")
print("All Dr Halim models loaded.\n")

# ======================================================================
#  HELPERS
# ======================================================================

def _preprocess_breed(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.resize((224, 224)), dtype=np.float32)
    return _eff_preprocess(np.expand_dims(arr, axis=0))


def _groq_fecal_recommendation(top_class: str, kb_info: dict) -> str | None:
    """Call GROQ for a French vet recommendation on the fecal result."""
    if not _GROQ_AVAILABLE:
        return None
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        # Try loading from tacheRanim .env
        env_path = os.path.join(_TACHE_RANIM_ROOT, ".env")
        if os.path.exists(env_path):
            for line in open(env_path).read().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        return None
    try:
        client = _Groq(api_key=api_key)
        contexte = (
            f"Classe détectée: {top_class}\n"
            f"Urgence: {kb_info.get('urgence', 'INCONNUE')}\n"
            f"Maladies probables: {', '.join(kb_info.get('maladies_probables', []))}\n"
            f"Traitements: {', '.join(kb_info.get('traitement', []))}"
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Tu es un assistant vétérinaire expert. Réponds en français de façon claire et empathique."},
                {"role": "user", "content": f"Analyse ces résultats et donne une recommandation en 3-4 phrases:\n{contexte}"},
            ],
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as _e:
        print(f"GROQ fecal error: {_e}")
        return None


def _breed_recommendation(breed_name: str) -> str:
    """Conseil de race : Ollama en priorité (validé prof), Groq en fallback."""
    import requests as _req

    prompt_fr = (
        f"Tu es Dr Halim, vétérinaire IA bienveillant.\n"
        f"Race détectée : {breed_name}.\n"
        f"Rédige un conseil d'adoption en français, chaleureux et concis (5-6 phrases).\n"
        f"Couvre : 1) Présentation  2) Tempérament  3) Enfants ?  "
        f"4) Appartement ?  5) Exercice  6) Alimentation & entretien.\n"
        f"Commence directement, sans salutation."
    )

    # ── 1. Ollama (priorité — validé) ──────────────────────────
    for _model in ("llama3:latest", "llama3"):
        try:
            r = _req.post(
                OLLAMA_URL,
                json={"model": _model, "prompt": prompt_fr, "stream": False},
                timeout=90,
            )
            if r.status_code == 200:
                result = r.json().get("response", "").strip()
                if result:
                    print(f"[breed] Ollama OK (model={_model})")
                    return result
            else:
                print(f"[breed] Ollama {_model} → HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e_ollama:
            print(f"[breed] Ollama {_model} erreur : {e_ollama}")
            break
    print("[breed] Ollama indisponible — bascule sur Groq")

    # ── 2. Groq (fallback si Ollama absent/éteint) ──────────────
    try:
        r = _req.post(
            GROQ_API_URL,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY_BREED}",
            },
            json={
                "model": GROQ_MODEL_BREED,
                "messages": [
                    {"role": "system", "content": "Tu es un vétérinaire expert. Réponds en français, de façon claire et empathique."},
                    {"role": "user",   "content": prompt_fr},
                ],
                "max_tokens": 500,
                "temperature": 0.7,
            },
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()["choices"][0]["message"]["content"].strip()
        print("[breed] Groq fallback OK")
        return result
    except Exception as e_groq:
        print(f"[breed] Groq erreur : {e_groq}")
        return (
            f"Conseil indisponible pour {breed_name}. "
            f"Démarrez Ollama (`ollama serve`) ou vérifiez la connexion internet."
        )


def _color_profile(pil_img: Image.Image) -> dict:
    w, h = pil_img.size
    mx, my = int(w * 0.2), int(h * 0.2)
    crop = pil_img.crop((mx, my, w - mx, h - my))
    arr  = np.array(crop.resize((64, 64)), dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    h_v, s_v, v_v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
    return {
        "r": round(float(r), 3), "g": round(float(g), 3), "b": round(float(b), 3),
        "hue_deg": round(h_v * 360, 1),
        "saturation": round(s_v, 3),
        "brightness": round(v_v, 3),
    }

# ======================================================================
#  ENDPOINTS
# ======================================================================

@app.get("/")
def health():
    return {"status": "ok", "message": "Dr Halim API is running"}


# --- 1. Breed detection ------------------------------------------------
@router.post("/predict-breed")
async def predict_breed(file: UploadFile = File(...)):
    contents = await file.read()
    img      = Image.open(io.BytesIO(contents)).convert("RGB")
    arr      = _preprocess_breed(img)
    preds    = breed_model.predict(arr, verbose=0)
    decoded  = _eff_decode(preds, top=10)[0]

    breed_name, confidence = None, 0.0
    for synset_id, imagenet_name, prob in decoded:
        if synset_id in _synset_to_breed:
            breed_name = _synset_to_breed[synset_id]
            confidence = float(prob)
            break
    if breed_name is None:
        _, imagenet_name, prob = decoded[0]
        breed_name = imagenet_name.split(",")[0].replace("_", " ").title()
        confidence = float(prob)

    recommendation = _breed_recommendation(breed_name)

    return {"breed": breed_name, "confidence": confidence, "recommendation": recommendation}


# --- 2. Skin analysis --------------------------------------------------
@router.post("/predict-skin")
async def predict_skin(file: UploadFile = File(...)):
    contents = await file.read()
    img      = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor   = skin_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(skin_model(tensor), dim=1)
        idx   = int(torch.argmax(probs).item())
        conf  = float(probs[0][idx])
    return {"skin": skin_classes[idx], "confidence": conf}


# --- 3. Behaviour analysis ---------------------------------------------
_VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
_IMAGE_EXT = (".jpg", ".jpeg", ".png")
_ALL_EXT   = _VIDEO_EXT + _IMAGE_EXT


@router.post("/predict-behavior")
async def predict_behavior_route(file: UploadFile = File(...)):
    fname = (file.filename or "").lower()
    if not any(fname.endswith(e) for e in _ALL_EXT):
        raise HTTPException(400, f"Unsupported format. Accepted: {', '.join(_ALL_EXT)}")

    is_image = any(fname.endswith(e) for e in _IMAGE_EXT)

    with tempfile.TemporaryDirectory() as tmpdir:
        ext        = os.path.splitext(file.filename or "file.mp4")[1]
        input_path = os.path.join(tmpdir, f"input{ext}")
        audio_path = os.path.join(tmpdir, "audio.wav")

        content = await file.read()
        with open(input_path, "wb") as fh:
            fh.write(content)

        if is_image:
            frames, voice_present, voix_result = [Image.open(input_path).convert("RGB")], False, None
            input_type   = "image"
            sound_status = "n/a"
        else:
            frames = extract_frames(input_path, max_frames=16)
            if not frames:
                raise HTTPException(422, "Could not extract frames. Video may be corrupted.")
            audio_ok, sound_status = extract_audio(input_path, audio_path)
            if audio_ok:
                voice_active  = has_voice_activity(audio_path)
                voice_present = voice_active
                sound_status  = "sound_detected" if voice_active else "silent_audio"
            else:
                voice_present = False
            voix_result = predict_voice(audio_path) if voice_present else None
            input_type  = "video"

        behav_result = predict_behavior(frames, use_yolo=not is_image, anormal_idx=BEHAV_ANORMAL_IDX)
        decision     = late_fusion_decision(behav_result, voix_result, voice_present)

        return JSONResponse(content={
            "decision":        str(decision["decision"]),
            "mode":            str(decision["mode"]),
            "score":           float(decision["score"]),
            "confidence":      float(decision["confidence"]),
            "score_behav":     float(decision["score_behav"]),
            "score_voix":      float(decision["score_voix"]) if decision["score_voix"] is not None else None,
            "top_behav":       str(decision["top_behav"]),
            "top_voix":        str(decision["top_voix"]) if decision["top_voix"] is not None else None,
            "threshold":       float(decision["threshold"]),
            "reason":          str(decision["reason"]),
            "contrib_behav":   float(decision["contrib_behav"]),
            "contrib_voix":    float(decision["contrib_voix"]) if decision["contrib_voix"] is not None else None,
            "voice_present":   bool(voice_present),
            "sound_status":    str(sound_status),
            "input_type":      str(input_type),
            "frames_analyzed": int(len(frames)),
            "probs_behav":     behav_result["probs_fusion"].tolist(),
            "probs_voix":      voix_result["probs_fusion"].tolist() if voix_result is not None else None,
        })


# --- 4. Fecal analysis (tacheRanim) ------------------------------------
_FECAL_STATUS_MAP = {
    "Normal":      "NORMAL",
    "SoftPoop":    "INCERTAIN",
    "LackOfWater": "INCERTAIN",
    "Blood":       "ANORMAL",
    "Diarrhoea":   "ANORMAL",
    "Worms":       "ANORMAL",
}
_URGENCY_COLORS = {"HAUTE": "danger", "MOYENNE": "warning", "FAIBLE": "accent", "AUCUNE": "primary"}


@router.post("/predict-fecal")
async def predict_fecal(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Photo uniquement (JPG ou PNG).")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(422, "Impossible de lire l'image fournie.")

    # ── ML prediction (tacheRanim MobileNet) ──
    if fecal_model is not None:
        arr         = np.expand_dims(np.array(img.resize((224, 224))) / 255.0, axis=0)
        confidences = fecal_model.predict(arr, verbose=0)[0]
        top_idx     = int(np.argmax(confidences))
        top_class   = _FECAL_CLASSES[top_idx]
        top_conf    = float(confidences[top_idx])

        detected = [_FECAL_CLASSES[i] for i, c in enumerate(confidences) if c > 0.35] or [top_class]
        kb_info  = _FECAL_KB.get(top_class, {})
        status   = _FECAL_STATUS_MAP.get(top_class, "INCERTAIN")

        conditions = [_FECAL_KB.get(cls, {}).get("nom", cls) for cls in detected]
        traitement = kb_info.get("traitement", [])

        # GROQ recommendation (blocking call in thread to avoid freezing event loop)
        recommandation_llm = await asyncio.to_thread(_groq_fecal_recommendation, top_class, kb_info)

        advice = recommandation_llm or "\n".join(traitement)

        return JSONResponse(content={
            "status":            status,
            "conditions":        conditions,
            "advice":            advice,
            "confidence":        top_conf,
            "model_used":        "tacheRanim",
            "detected_class":    top_class,
            "all_detections":    detected,
            "urgence":           kb_info.get("urgence", "AUCUNE"),
            "maladies_probables": kb_info.get("maladies_probables", []),
            "traitement":        traitement,
            "recommandation_llm": recommandation_llm,
            "color_profile":     _color_profile(img),
        })

    # ── Fallback: heuristic colour analysis ──
    cp   = _color_profile(img)
    h_deg, s_val, v_val = cp["hue_deg"], cp["saturation"], cp["brightness"]

    if v_val < 0.18:
        status, conditions, advice, conf = "ANORMAL", ["Selles noires (melena suspectée)"], "Consultez un vétérinaire en urgence.", 0.83
    elif s_val < 0.15 and v_val > 0.72:
        status, conditions, advice, conf = "ANORMAL", ["Selles pâles / blanches"], "Possible obstruction biliaire. Consultation recommandée.", 0.78
    elif (h_deg < 18 or h_deg > 340) and s_val > 0.35:
        status, conditions, advice, conf = "ANORMAL", ["Sang rouge vif (hématochézie)"], "Urgence vétérinaire immédiate.", 0.88
    elif 42 < h_deg < 85 and s_val > 0.28:
        status, conditions, advice, conf = "ANORMAL", ["Selles jaunâtres / verdâtres"], "Transit accéléré ou excès de bile. Surveillance 24h.", 0.72
    elif 15 < h_deg < 45 and v_val > 0.15:
        status, conditions, advice, conf = "NORMAL", ["Couleur brun normale"], "Selles normales. Continuez une alimentation équilibrée.", 0.80
    else:
        status, conditions, advice, conf = "INCERTAIN", ["Couleur atypique"], "Évaluation vétérinaire directe conseillée.", 0.55

    return JSONResponse(content={
        "status": status, "conditions": conditions, "advice": advice,
        "confidence": conf, "model_used": "heuristic",
        "detected_class": None, "all_detections": [],
        "urgence": "HAUTE" if status == "ANORMAL" else "AUCUNE",
        "maladies_probables": [], "traitement": [advice],
        "recommandation_llm": None,
        "color_profile": cp,
    })


# ======================================================================
#  REGISTER ROUTER
# ======================================================================
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7001, reload=False)
