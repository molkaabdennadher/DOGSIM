# Dr Halim — AI Veterinarian Chatbot

A unified chatbot that combines three AI-powered dog analysis tools into a single interface.

---

## Features

| Analysis | Input | Description |
|---|---|---|
| 🐕 Detect Breed | Photo | Identifies the dog breed + adoption recommendation (via Ollama) |
| 🔬 Skin Check | Photo | Detects skin or coat problems |
| 🎬 Behaviour Analysis | Photo or Video | Detects abnormal behaviour (rage risk) using visual + audio fusion |

---

## Project Structure

```
dr_halim/
├── backend/
│   ├── main.py            ← Unified FastAPI server (all 3 endpoints)
│   ├── inference.py       ← Behaviour analysis module (from zaineb)
│   └── requirements.txt
├── frontend/
│   └── index.html         ← Dr Halim chatbot UI (open in browser)
└── README.md
```

The models are loaded from their **original locations** in the Downloads folder:
- **Breed & Skin models** → `Downloads/mayseneZaineb/maysese/models/`
- **Behaviour & Voice models** → `Downloads/mayseneZaineb/zaineb/models/`

---

## Setup & Launch

### 1. Install dependencies

```bash
cd dr_halim/backend
pip install -r requirements.txt
```

### 2. Start the backend

```bash
cd dr_halim/backend
python main.py
```

The API will be available at `http://localhost:8000`.

### 3. Open the chatbot

Open `dr_halim/frontend/index.html` directly in your browser.

---

## Optional — Breed Recommendations (Ollama)

The breed detection endpoint generates adoption advice using a local LLM.
To enable it:

1. Download and install [Ollama](https://ollama.com)
2. Pull the llama3 model: `ollama pull llama3`
3. Make sure Ollama is running before starting the backend

Without Ollama the breed detection still works — it just skips the recommendation text.

---

## API Endpoints

| Method | Endpoint | Input | Description |
|---|---|---|---|
| GET | `/` | — | Health check |
| POST | `/predict-breed` | image file | Breed detection + recommendation |
| POST | `/predict-skin` | image file | Skin/coat analysis |
| POST | `/predict-behavior` | image or video file | Behaviour analysis |

---

## Model Paths

If you move the model folders, update the path constants at the top of:
- `backend/main.py` → `MAYSESE_MODELS`
- `backend/inference.py` → `MODELS_DIR`
