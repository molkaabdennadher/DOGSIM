from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
import io, json, os

load_dotenv()

# Chemin du répertoire du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

app = FastAPI()
model = tf.keras.models.load_model(os.path.join(PROJECT_ROOT, 'model', 'model_mobilenet_selles.keras'))
client_groq = None

def get_groq_client():
    global client_groq
    if client_groq is None:
        client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return client_groq

with open(os.path.join(PROJECT_ROOT, 'data', 'knowledge_base.json'), encoding='utf-8') as f:
    knowledge_base = json.load(f)

CLASSES = ['Blood', 'Diarrhoea', 'LackOfWater', 'Normal', 'SoftPoop', 'Worms']
IMG_SIZE = (224, 224)

def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    confiances = model.predict(arr)[0]
    classes_detectees = [CLASSES[i] for i, c in enumerate(confiances) if c > 0.5]
    if not classes_detectees:
        classes_detectees = [CLASSES[np.argmax(confiances)]]
    return classes_detectees, confiances

def generer_recommandation(contexte):
    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Tu es un assistant vétérinaire expert. Réponds en français."},
            {"role": "user", "content": f"Analyse ces résultats et donne une recommandation claire:\n{contexte}"}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content
    return response.choices[0].message.content

@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(PROJECT_ROOT, 'frontend', 'index.html'), encoding="utf-8").read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    classes_detectees, confiances = predict_image(contents)

    cls_principale = classes_detectees[0]
    info = knowledge_base[cls_principale]

    contexte = f"Classe détectée: {cls_principale}\nUrgence: {info['urgence']}\n"
    contexte += f"Maladies: {', '.join(info['maladies_probables'])}\n"
    contexte += f"Traitements: {', '.join(info['traitement'])}"

    recommandation = generer_recommandation(contexte)

    return {
        "classes_detectees": classes_detectees,
        "confiances": {c: round(float(f), 3) for c, f in zip(CLASSES, confiances)},
        "urgence": info["urgence"],
        "traitement": info["traitement"],
        "recommandation_llm": recommandation
    }