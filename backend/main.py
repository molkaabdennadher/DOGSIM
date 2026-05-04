import json
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# TensorFlow (race)
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

# PyTorch (skin)
import torch
import torchvision.transforms as transforms
from torchvision import models

app = FastAPI()

# Autoriser le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔵 LOAD MODEL RACE
# =========================
breed_model = tf.keras.models.load_model("../models/breed_model.h5")

with open("../models/class_names.json", "r") as f:
    class_names = json.load(f)

# =========================
# 🔴 LOAD MODEL SKIN
# =========================
device = torch.device("cpu")

skin_model = models.efficientnet_b0(pretrained=False)
skin_model.classifier[1] = torch.nn.Linear(skin_model.classifier[1].in_features, 2)
skin_model.load_state_dict(torch.load("../models/skin_model.pth", map_location=device))
skin_model.to(device)
skin_model.eval()
with open("../models/skin_class_names.json", "r") as f:
    skin_classes = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# 🚀 API
# =========================
@app.post("/predict-breed")
async def predict_breed(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    img_resized = img.resize((224, 224))
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    breed_pred = breed_model.predict(img_array)
    breed_index = np.argmax(breed_pred)

    breed_name = class_names[breed_index]
    breed_name = breed_name.split("-")[1].replace("_", " ") if "-" in breed_name else breed_name

    breed_conf = float(np.max(breed_pred))

    # ✅ RECOMMANDATION ICI (OBJECTIF 1)
    prompt = f"""
Tu es un assistant intelligent pour faciliter l’adoption des chiens.

La race prédite est : {breed_name}.

Donne une recommandation claire et utile en français pour une personne qui veut adopter ce chien.

Structure obligatoire :
1. Présentation courte de la race
2. Tempérament
3. Adapté ou non à une famille avec enfants
4. Adapté ou non à la vie en appartement
5. Besoin d’activité physique
6. Conseils d’alimentation
7. Conseils de soins
8. Pour quel type de propriétaire cette race convient le mieux

Réponse courte, claire, naturelle, sans inventer de détails trop techniques.
"""

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        recommendation = response.json().get("response", "Aucune recommandation")

    except Exception as e:
        print("ERREUR OLLAMA :", e)
        recommendation = "LLM non disponible"

    return {
        "breed": breed_name,
        "confidence": breed_conf,
        "recommendation": recommendation
    }

@app.post("/predict-skin")
async def predict_skin(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    img_torch = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        skin_pred = skin_model(img_torch)
        probs = torch.softmax(skin_pred, dim=1)
        skin_index = torch.argmax(probs).item()
        skin_name = skin_classes[skin_index]
        skin_conf = float(probs[0][skin_index])

   


    return {
        "skin": skin_name,
        "confidence": skin_conf,
    }