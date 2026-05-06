# 🏗️ ARCHITECTURE - Happy Paws Unified Platform

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     🌐 USER BROWSER                                 │
│                http://localhost:8000                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│           🖥️ PET-ADVISOR (Main FastAPI Server - Port 8000)         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Frontend Routes:                                                   │
│  ├─ GET  /              → Landing page (index.html)               │
│  ├─ GET  /maya          → Maya chat UI                            │
│  ├─ GET  /ali           → Ali shelter coordinator UI              │
│  ├─ GET  /aggression    → Aggression detection UI                 │
│  └─ GET  /dr-halim      → Dr Halim veterinarian UI ⭐            │
│                                                                      │
│  Backend API Routes:                                                │
│  ├─ /ws/*               → WebSocket connections                   │
│  ├─ /auth/*             → Authentication                          │
│  ├─ /maya/*             → Maya endpoints                          │
│  ├─ /ali/*              → Ali endpoints                           │
│  ├─ /aggression/*       → Aggression analysis                     │
│  └─ /drhalim/*   ┐      → Dr Halim PROXY ⭐                       │
│     (proxy)      │                                                  │
│                  └────────────────────┐                             │
│                                        │                             │
│                                        ▼                             │
│     ┌──────────────────────────────────────────────────┐             │
│     │  Subprocess: Dr Halim (Port 7001 - Internal)    │             │
│     ├──────────────────────────────────────────────────┤             │
│     │                                                  │             │
│     │  Models:                                        │             │
│     │  • Breed detection (EfficientNetB0)            │             │
│     │  • Skin analysis (PyTorch)                     │             │
│     │  • Behaviour analysis (custom CNN + audio)     │             │
│     │  • Fecal analysis (tacheRanim ML) ⭐           │             │
│     │                                                  │             │
│     │  POST /predict-breed                           │             │
│     │  POST /predict-skin                            │             │
│     │  POST /predict-behavior                        │             │
│     │  POST /predict-fecal    ← Integrated! 🎉       │             │
│     │                                                  │             │
│     └──────────────────────────────────────────────────┘             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Port Mapping

| Service | Port | Accessibility | Purpose |
|---------|------|----------------|---------|
| **pet-advisor** | 8000 | Public (`0.0.0.0`) | Main entry point |
| **Dr Halim** | 7001 | Local (`127.0.0.1`) | Internal subprocess |
| **pet-advisor proxy** | 8000 | Public | Routes `/drhalim/*` → `localhost:7001/*` |

## Request Flow

### Example: Fecal Analysis Upload

```
User Browser (localhost:8000/dr-halim)
           │
           ├─ Loads UI: http://localhost:8000/dr-halim
           │            ↓ (serves index.html via pet-advisor)
           │
           ├─ Uploads image
           │            ↓
           └─→ POST http://localhost:8000/drhalim/predict-fecal
               │
               │ (FastAPI proxy route in pet-advisor)
               ├─ Takes request
               ├─ Forwards to Dr Halim subprocess
               └─→ http://localhost:7001/predict-fecal
                   │
                   └─→ Dr Halim backend processes
                       • Loads tacheRanim ML model
                       • Classifies stool (Blood/Diarrhoea/Normal/etc)
                       • Returns result with advice
                   │
                   ├─ Returns response to pet-advisor proxy
                   │
               └─→ pet-advisor returns to browser
                   │
               JSON: { status: "NORMAL", conditions: [...], advice: "...", ... }
```

## Integration Points

### 1. **tacheRanim Integration**
- ✅ ML model: `../../tacheRanim/model/model_mobilenet_selles.keras`
- ✅ KB: `../../tacheRanim/data/knowledge_base.json`
- ✅ 6-class classification (Blood, Diarrhoea, LackOfWater, Normal, SoftPoop, Worms)
- ✅ Location: Integrated into Dr Halim's `/predict-fecal` endpoint

### 2. **Dr Halim Integration**
- ✅ Location: Separate service launched as subprocess by pet-advisor
- ✅ Port: 7001 (internal)
- ✅ Public path: `/drhalim/*` (proxied by pet-advisor)
- ✅ Frontend: Served at `/dr-halim` by pet-advisor

### 3. **Frontend Access**
- ✅ URL: `http://localhost:8000/dr-halim`
- ✅ API calls: `POST http://localhost:8000/drhalim/predict-fecal`
- ✅ Backend: Automatically routed to `http://localhost:7001/predict-fecal`

## Startup Sequence

```
1. User runs: uvicorn backend.main:app --port 8000
   └─ Starts pet-advisor on port 8000

2. pet-advisor startup event:
   ├─ Load artifacts (Maya, dog finder, aggression DB)
   └─ Launch Dr Halim subprocess
      └─ python ../../dr_halim/backend/main.py
         └─ Starts Dr Halim on port 7001
            ├─ Load breed model
            ├─ Load skin model
            ├─ Load behavior models
            ├─ Load fecal model (tacheRanim)
            ├─ Load fecal KB
            └─ Ready to accept requests

3. User accesses: http://localhost:8000/dr-halim
   ├─ pet-advisor serves index.html
   └─ Frontend loads and ready

4. User uploads image and clicks "Analyse Fécale"
   ├─ Frontend sends: POST /drhalim/predict-fecal
   ├─ pet-advisor proxy receives request
   ├─ Forwards to: http://localhost:7001/predict-fecal
   ├─ Dr Halim processes
   └─ Result returned to browser
```

## Key Configuration Files

### pet-advisor/backend/main.py (Lines ~160-195)
```python
_DRHALIM_ROOT    = Path(...) / "dr_halim"
_DRHALIM_PYTHON  = _DRHALIM_ROOT / "venv" / "Scripts" / "python.exe"
_DRHALIM_MAIN    = _DRHALIM_ROOT / "backend" / "main.py"
_DRHALIM_URL     = "http://localhost:7001"  # ← Where Dr Halim listens

@app.api_route("/drhalim/{path:path}", methods=["GET", "POST", ...])
async def drhalim_proxy(request, path):
    # Proxies all /drhalim/* requests to http://localhost:7001/*

@app.on_event("startup")
async def startup():
    # Launches Dr Halim as subprocess
    subprocess.Popen([python, main.py], cwd=...)
```

### dr_halim/backend/main.py (Bottom)
```python
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7001, reload=False)
    #         Listens only on localhost ↑ ↑ Internal port
```

### dr_halim/frontend/index.html (Line ~464)
```javascript
const API = 'http://localhost:8000/drhalim';
//      User's browser calls pet-advisor proxy
```

## Why This Architecture?

| Aspect | Benefit |
|--------|---------|
| **Unified Port** | Single entry point for users (8000) |
| **Subprocess** | Dr Halim isolated, easy to restart |
| **Proxy Pattern** | Transparent routing, no frontend changes needed |
| **Graceful Degradation** | If Dr Halim crashes, error page is returned |
| **Modular** | Can run Dr Halim standalone or via pet-advisor |
| **Scalable** | Can add more subprocesses (Dr Zaineb, etc.) |

## Testing the Integration

### 1. Check Service Status
```bash
# Check if pet-advisor is running
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# Check if Dr Halim is available
curl http://localhost:8000/drhalim/health
# Expected: {"status": "ok", "message": "Dr Halim API is running"}
```

### 2. Test Fecal Analysis
```bash
curl -X POST http://localhost:8000/drhalim/predict-fecal \
  -F "file=@test_image.jpg"
# Expected: JSON with status, conditions, advice, confidence, etc.
```

### 3. Test via Browser
1. Open `http://localhost:8000/dr-halim`
2. Upload an image
3. Click "Analyse Fécale"
4. Check browser DevTools → Network tab for request to `/drhalim/predict-fecal`

## Troubleshooting

### "Cannot reach Dr Halim" (503 error)
- Check: Is subprocess running? Look for process ID in logs
- Fix: Wait a few seconds for models to load
- Restart: Kill both processes and start pet-advisor again

### "Module not found" in Dr Halim
- Check: Is `../../tacheRanim` path correct?
- Fix: Run from `adoption system/` folder

### Port 8000 already in use
- Check: `netstat -ano | findstr :8000` (Windows)
- Kill: `taskkill /PID <pid> /F`
- Change: Edit port in `main.py`

### Dr Halim models not loading
- Check: Logs should show "Fecal model ready" or warning
- Fix: If warning, heuristic fallback will work
- Verify: `../../tacheRanim/model/model_mobilenet_selles.keras` exists

---

**Last Updated:** 2026-05-04  
**Architecture Status:** ✅ Unified and Integrated  
**Integration:** tacheRanim → Dr Halim → pet-advisor
