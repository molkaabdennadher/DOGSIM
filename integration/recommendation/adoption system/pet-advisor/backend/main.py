"""
main.py — Unified FastAPI entry point for Happy Paws Pet Advisor.

Modules
-------
Maya   — AI pet adoption advisor (chat + lost-dog search)
Ali    — Shelter assignment coordinator (spreadsheet upload + chat)
Aggression — Dog aggression detection (video analysis + incidents)

Start with:
    uvicorn backend.main:app --reload --port 8000

Endpoints
---------
GET   /                                     → landing page (index.html)
GET   /maya                                 → Maya chat UI
GET   /ali                                  → Ali shelter UI
GET   /aggression                           → Aggression detection UI

# Maya
POST  /auth/login                           → create session
POST  /upload/{session_id}                  → upload dog image
WS    /ws/maya/{session_id}                 → Maya real-time chat
GET   /pets/{pet_id}                        → full pet details
GET   /feedback/{user_email}                → user preference profile

# Ali
POST  /ali/upload                           → upload Excel, assign dogs
WS    /ws/ali/{session_id}                  → Ali real-time chat
GET   /ali/shelters                         → list shelters
POST  /ali/shelters                         → add shelter
GET   /ali/shelters/{id}                    → shelter details
PATCH /ali/shelters/{id}                    → update shelter field
GET   /ali/dogs                             → list dogs
POST  /ali/dogs/{pet_id}/release            → release dog
GET   /ali/statistics                       → aggregate stats
GET   /ali/waiting-list                     → waiting-list dogs

# Aggression
POST  /aggression/analyze-video             → upload video → detect
POST  /aggression/incident                  → submit pre-computed incident
GET   /aggression/incidents                 → list incidents
GET   /aggression/incidents/{id}            → single incident
GET   /aggression/incidents/{id}/keyframes  → frame mosaic JPEG

GET   /health                               → liveness check
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from types import GeneratorType
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import subprocess
import httpx

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# ── Maya imports ───────────────────────────────────────────────────────────────
from backend.maya.agent import PetAdvisorAgent
from backend.maya import feedback as maya_feedback
from backend.maya.schemas import AggressionIncidentRequest, LoginRequest, LoginResponse, UploadResponse
from backend.maya.session import create_session, get_session, save_session
from backend.maya.tools.dog_finder import load_dog_artifacts
from backend.maya.tools.retrieval import load_artifacts, get_pet_details

# ── Ali imports ────────────────────────────────────────────────────────────────
import backend.ali.db as ali_db
from backend.ali.agent import init_agent as ali_init_agent, chat as ali_chat
from backend.ali.schemas import AddShelterRequest, UpdateShelterRequest
from backend.ali.tools.excel_parser import parse_file as parse_spreadsheet, SUPPORTED_EXTENSIONS
from backend.ali.optimizer import run_assignment

# ── Aggression imports (optional — module lives in DOGSIM-TN) ─────────────────
try:
    from backend import aggression
except ImportError:
    aggression = None  # aggression intégré dans DOGSIM-TN, pas nécessaire ici

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")
log = logging.getLogger("happy_paws")

ROOT_DIR       = Path(__file__).parent.parent
FRONTEND_DIR   = ROOT_DIR / "frontend"
DR_HALIM_HTML  = Path(__file__).parent.parent.parent.parent / "dr_halim" / "frontend" / "index.html"
UPLOADS_DIR    = ROOT_DIR / "uploads"
ARTIFACTS_BASE = ROOT_DIR / "artifacts"
KEYFRAMES_DIR  = ARTIFACTS_BASE / "agressionDetection" / "keyframes"
ALI_UPLOAD_DIR = ARTIFACTS_BASE / "aliAdvisor" / "uploads"

for d in (UPLOADS_DIR, KEYFRAMES_DIR, ALI_UPLOAD_DIR):
    d.mkdir(exist_ok=True, parents=True)


# ── Artifact migration (flat → subdirs, one-time idempotent) ──────────────────

def _migrate_artifacts() -> None:
    """One-time idempotent migration: move flat artifacts into subdirectories."""
    ai_dir  = ARTIFACTS_BASE / "aiAdvisor"
    agg_dir = ARTIFACTS_BASE / "agressionDetection"
    ali_dir = ARTIFACTS_BASE / "aliAdvisor"
    for d in (ai_dir, agg_dir, ali_dir):
        d.mkdir(exist_ok=True, parents=True)

    def _mv(dest: Path, names: list[str]) -> None:
        for f in names:
            src, dst = ARTIFACTS_BASE / f, dest / f
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
                log.info("Migrated %s -> artifacts/%s/", f, dest.name)

    # aiAdvisor: embeddings, encoders, metadata, raw CLIP images, descriptions
    _mv(ai_dir, [
        "df_original.parquet", "pet_emb.npy", "user_encoder.pt", "pet_encoder.pt",
        "pet_features_shape.txt", "pet_ids.npy", "thresholds.json",
        "feedback.db", "sessions.db",
        "img_emb_raw.npy", "rich_descriptions.json", "pet_data_raw.json",
        "pca_text.pkl", "pca_img.pkl",
    ])
    _mv(agg_dir, ["aggression_config.json", "aggression_incidents.db"])
    for name in ("aggression_outbox", "aggression_reports"):
        src, dst = ARTIFACTS_BASE / name, agg_dir / name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Happy Paws", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ── Dr Halim subprocess + proxy ───────────────────────────────────────────────
_DRHALIM_ROOT    = Path(__file__).parent.parent.parent.parent / "dr_halim"
_DRHALIM_PYTHON  = _DRHALIM_ROOT / "venv" / "Scripts" / "python.exe"
_DRHALIM_MAIN    = _DRHALIM_ROOT / "backend" / "main.py"
_DRHALIM_URL     = "http://localhost:7001"
_drhalim_proc: subprocess.Popen | None = None


@app.api_route("/drhalim/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def drhalim_proxy(request: Request, path: str):
    body    = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.request(
                method=request.method,
                url=f"{_DRHALIM_URL}/{path}",
                headers=headers,
                content=body,
            )
        return Response(content=resp.content, status_code=resp.status_code,
                        media_type=resp.headers.get("content-type"))
    except httpx.ConnectError:
        return JSONResponse(
            {"detail": "Dr Halim is still loading models, please retry in a moment."},
            status_code=503,
        )


@app.on_event("startup")
async def startup():
    global _drhalim_proc
    _migrate_artifacts()

    if _DRHALIM_PYTHON.exists() and _DRHALIM_MAIN.exists():
        try:
            # Kill any stale process on port 7001 before starting a fresh one.
            # This ensures the updated inference.py is always loaded.
            import copy, signal as _signal
            try:
                import psutil
                for conn in psutil.net_connections(kind="tcp"):
                    if conn.laddr.port == 7001 and conn.pid:
                        try:
                            psutil.Process(conn.pid).terminate()
                            log.info("Killed stale process on port 7001 (pid=%s)", conn.pid)
                        except Exception:
                            pass
            except ImportError:
                # psutil not available — fall back to OS-level kill on Windows
                try:
                    subprocess.run(
                        ["powershell", "-Command",
                         "Get-NetTCPConnection -LocalPort 7001 -ErrorAction SilentlyContinue "
                         "| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"],
                        capture_output=True, timeout=5,
                    )
                    log.info("Cleared port 7001 via PowerShell")
                except Exception:
                    pass

            import time as _time
            _time.sleep(1)   # brief pause so the OS releases the port

            child_env = copy.deepcopy(os.environ)
            _drhalim_proc = subprocess.Popen(
                [str(_DRHALIM_PYTHON), str(_DRHALIM_MAIN)],
                cwd=str(_DRHALIM_ROOT / "backend"),
                env=child_env,
            )
            log.info("Dr Halim subprocess started (pid=%s)", _drhalim_proc.pid)
        except Exception as e:
            log.warning("Could not start Dr Halim subprocess: %s", e)
    else:
        log.warning("Dr Halim venv not found at %s", _DRHALIM_PYTHON)

    # Maya artifacts — optional (only needed for adoption advisor)
    try:
        load_artifacts()
    except Exception as e:
        log.warning("Maya embedding artifacts not loaded (Maya will run in fallback mode): %s", e)
    try:
        load_dog_artifacts()
    except Exception as e:
        log.warning("Dog artifacts not loaded: %s", e)

    # Always-required databases
    try:
        aggression.init_db()
    except Exception as e:
        log.error("Aggression DB init failed: %s", e)
    try:
        ali_db.init_db()
    except Exception as e:
        log.error("Ali DB init failed: %s", e)

    _seed_shelters_if_empty()

    # Ali agent — requires GROQ_API_KEY
    try:
        ali_init_agent()
    except Exception as e:
        log.warning("Ali agent not initialised (check GROQ_API_KEY in .env): %s", e)


@app.on_event("shutdown")
async def shutdown():
    global _drhalim_proc
    if _drhalim_proc and _drhalim_proc.poll() is None:
        _drhalim_proc.terminate()
        log.info("Dr Halim subprocess terminated")


def _seed_shelters_if_empty():
    shelters = ali_db.get_all_shelters()
    seed = ROOT_DIR / "data" / "shelters_seed.json"
    if not shelters and seed.exists():
        seed_data = json.loads(seed.read_text())
        for s in seed_data:
            ali_db.add_shelter(s)
        shelters = ali_db.get_all_shelters()
        log.info("Seeded %d shelters", len(seed_data))

    if len(shelters) < 120:
        added = _seed_generated_shelters(existing=shelters, target=120)
        if added:
            log.info("Seeded %d generated shelters to restore demo capacity", added)


def _seed_generated_shelters(existing: list[dict], target: int = 120) -> int:
    existing_ids = {str(s.get("id")) for s in existing}
    existing_names = {str(s.get("name")) for s in existing}
    cities = ["Tunis", "Ariana", "Ben Arous", "Manouba", "Sfax", "Sousse", "Monastir"]
    expertise = ["beginner", "intermediate", "expert"]
    shelter_types = ["public", "private", "NGO"]
    added = 0

    for i in range(1, target + 1):
        if len(existing) + added >= target:
            break
        shelter_id = f"R{i:03d}"
        name = f"Refuge_{i}"
        if shelter_id in existing_ids or name in existing_names:
            continue
        city = cities[(i * 3) % len(cities)]
        level = expertise[i % len(expertise)]
        capacity = 18 + ((i * 7) % 55)
        ali_db.add_shelter({
            "id": shelter_id,
            "name": name,
            "location": city,
            "capacity": capacity,
            "monthly_budget": 3500 + ((i * 430) % 9000),
            "expertise_level": level,
            "staff_count": 3 + (i % 10),
            "shelter_type": shelter_types[i % len(shelter_types)],
            "has_vet": level == "expert" or i % 4 == 0,
        })
        added += 1

    return added


maya_agent = PetAdvisorAgent()


def _consume_agent_generator(result):
    """Return a generator's final value, keeping compatibility with older agents."""
    last_value = None
    while True:
        try:
            last_value = next(result)
        except StopIteration as exc:
            return exc.value if exc.value is not None else last_value


# ── Static pages ───────────────────────────────────────────────────────────────

def _serve(name: str):
    p = FRONTEND_DIR / name
    return FileResponse(str(p)) if p.exists() else JSONResponse({"page": name})


@app.get("/")
async def landing():       return _serve("index.html")

@app.get("/maya")
async def maya_page():     return _serve("maya.html")

@app.get("/ali")
async def ali_page():      return _serve("ali.html")

@app.get("/aggression")
async def aggression_page(): return _serve("aggression.html")

@app.get("/dr-halim")
async def dr_halim_page():
    if DR_HALIM_HTML.exists():
        return FileResponse(str(DR_HALIM_HTML))
    return JSONResponse({"error": "Dr. Halim interface not found"}, status_code=404)

@app.get("/health")
async def health():        return {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# MAYA ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    if not req.email.strip():
        raise HTTPException(400, "Email is required.")
    session = create_session(user_name=req.name, user_email=req.email.strip().lower())
    return LoginResponse(session_id=session.session_id,
                         message=f"Welcome, {req.name}! Your session is ready.")


@app.post("/upload/{session_id}", response_model=UploadResponse)
async def upload_image(session_id: str, file: UploadFile = File(...)):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/jpg"}:
        raise HTTPException(400, "Only JPEG/PNG/WEBP images are accepted")
    image_id = f"{uuid.uuid4()}.jpg"
    (UPLOADS_DIR / image_id).write_bytes(await file.read())
    return UploadResponse(image_id=image_id)


@app.get("/pets/{pet_id}")
async def get_pet(pet_id: str):
    details = get_pet_details(pet_id)
    if not details or details.get("error"):
        raise HTTPException(404, "Pet not found")
    return details


@app.get("/feedback/{user_email}")
async def get_feedback(user_email: str):
    profile = maya_feedback.build_profile(user_email.strip().lower())
    return {
        "user_email":           profile.user_email,
        "total_rejections":     profile.total_rejections,
        "excluded_pet_ids":     list(profile.excluded_pet_ids),
        "avoided_breeds":       list(profile.avoided_breeds),
        "avoided_sizes":        list(profile.avoided_sizes),
        "avoided_animal_types": list(profile.avoided_animal_types),
        "rejection_reasons":    profile.rejection_reasons,
        "summary_for_llm":      maya_feedback.summarize_for_llm(profile),
    }


@app.websocket("/ws/maya/{session_id}")
async def maya_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = get_session(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found."})
        await websocket.close()
        return

    if not session.messages:
        await websocket.send_json({
            "type": "message", "role": "assistant",
            "content": (
                f"Hi {session.user_name}! 🐾 I'm Maya, your pet adoption advisor. "
                "I'm here to help you find your perfect companion. "
                "What can I help you with today?"
            ),
        })

    try:
        while True:
            data     = await websocket.receive_json()
            message  = data.get("message", "").strip()
            image_id = data.get("image_id")
            if not message:
                continue
            await websocket.send_json({"type": "typing", "active": True})
            try:
                result = await asyncio.to_thread(maya_agent.run, session, message, image_id)
                if isinstance(result, GeneratorType):
                    result = await asyncio.to_thread(_consume_agent_generator, result)
                save_session(session)
                await websocket.send_json({"type": "typing", "active": False})
                if result.text:
                    await websocket.send_json({"type": "message", "role": "assistant", "content": result.text})
                if result.pets:
                    await websocket.send_json({"type": "pets", "data": result.pets[:3]})
                if result.found_dog:
                    await websocket.send_json({"type": "found_dog", "data": result.found_dog})
            except Exception as exc:
                log.exception("Maya agent error for session %s", session_id)
                save_session(session)
                await websocket.send_json({"type": "typing", "active": False})
                await websocket.send_json({"type": "error", "message": "Internal error. Please check GROQ_API_KEY."})
    except WebSocketDisconnect:
        save_session(session)


# ══════════════════════════════════════════════════════════════════════════════
# ALI ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ali/upload")
async def ali_upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".xlsx":
        raise HTTPException(
            400,
            "Only .xlsx files are accepted. Please export your spreadsheet as "
            "Excel (.xlsx) before uploading.",
        )
    tmp = ALI_UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    try:
        with tmp.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        dogs, warnings = parse_spreadsheet(tmp)
        if not dogs:
            return JSONResponse({"status": "no_dogs", "warnings": warnings})
        from groq import Groq
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
        result = run_assignment(dogs, groq_client)
        return JSONResponse({
            "status": "ok",
            "total": result.total,
            "assigned_count": result.assigned_count,
            "waiting_count": result.waiting_count,
            "assigned": result.assigned,
            "waiting": result.waiting,
            "warnings": warnings,
        })
    finally:
        if tmp.exists():
            tmp.unlink()


@app.websocket("/ws/ali/{session_id}")
async def ali_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        # ── Greeting: query live DB state on connect ───────────────────────
        greeting = (
            "Hello! Briefly introduce yourself in 1–2 sentences, then call "
            "get_statistics and get_capacity_risks to report the current state "
            "of the system. If the DB is empty (no shelters or dogs), say so "
            "and explain that the user can upload an Excel file to get started. "
            "Keep the whole reply under 120 words."
        )
        async for chunk in ali_chat(greeting, session_id):
            await websocket.send_text(chunk)

        # ── Normal message loop ────────────────────────────────────────────
        while True:
            data    = await websocket.receive_text()
            message = json.loads(data).get("message", "")
            if not message.strip():
                continue
            async for chunk in ali_chat(message, session_id):
                await websocket.send_text(chunk)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "content": str(exc)}))


@app.get("/ali/shelters")
def ali_list_shelters():
    return ali_db.get_all_shelters()


@app.get("/ali/shelters/{shelter_id}")
def ali_get_shelter(shelter_id: str):
    s = ali_db.get_shelter(shelter_id)
    if not s:
        raise HTTPException(404, f"Shelter '{shelter_id}' not found.")
    return s


@app.post("/ali/shelters")
def ali_add_shelter(req: AddShelterRequest):
    data = req.model_dump()
    if data.get("id") is None:
        data.pop("id", None)
    shelter_id = ali_db.add_shelter(data)
    return {"status": "created", "shelter_id": shelter_id}


@app.patch("/ali/shelters/{shelter_id}")
def ali_update_shelter(shelter_id: str, req: UpdateShelterRequest):
    if not ali_db.get_shelter(shelter_id):
        raise HTTPException(404, f"Shelter '{shelter_id}' not found.")
    ali_db.update_shelter_field(shelter_id, req.field, req.value)
    return {"status": "updated", "shelter_id": shelter_id, "field": req.field}


@app.get("/ali/dogs")
def ali_list_dogs(status: Optional[str] = Query(None), shelter_id: Optional[str] = Query(None)):
    if shelter_id:
        return ali_db.get_dogs_in_shelter(shelter_id)
    kwargs = {}
    if status:
        kwargs["status"] = status
    return ali_db.get_dogs_by_criteria(**kwargs)


@app.post("/ali/dogs/{pet_id}/release")
def ali_release_dog(pet_id: str):
    dog = ali_db.get_dog(pet_id)
    if not dog:
        raise HTTPException(404, f"Dog '{pet_id}' not found.")
    if dog["status"] != "assigned":
        raise HTTPException(400, f"Dog '{pet_id}' is not currently assigned.")
    result = ali_db.release_dog(pet_id)
    if not result:
        raise HTTPException(500, "Release failed.")
    return {"status": "released", "pet_id": pet_id, "previous_shelter": result["freed_shelter_id"]}


@app.get("/ali/statistics")
def ali_statistics():
    stats = ali_db.get_statistics()   # flat dict from db
    return {
        "total_shelters":  stats.get("total_shelters",  0),
        "total_capacity":  stats.get("total_capacity",  0),
        "total_free_spots": stats.get("total_free_spots", 0),
        "total_budget":    stats.get("total_budget",    0),
        "total_cost":      stats.get("total_cost",      0),
        "total_dogs":      stats.get("total_dogs",      0),
        "assigned_dogs":   stats.get("assigned",        0),
        "waiting_dogs":    stats.get("waiting",         0),
        "released_dogs":   stats.get("released",        0),
        "serious_injury":  stats.get("serious_injury",  0),
        "pregnant":        stats.get("pregnant",        0),
        "skin_disease":    stats.get("skin_disease",    0),
        "abnormal":        stats.get("abnormal",        0),
    }


@app.get("/ali/waiting-list")
def ali_waiting_list():
    return ali_db.get_waiting_list()


# ══════════════════════════════════════════════════════════════════════════════
# AGGRESSION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def _save_keyframe_mosaic(incident_id: str, key_frames_b64: str) -> str:
    if not key_frames_b64:
        return ""
    try:
        raw  = base64.b64decode(key_frames_b64)
        path = KEYFRAMES_DIR / f"{incident_id}.jpg"
        path.write_bytes(raw)
        return str(path)
    except Exception as exc:
        log.warning("Could not save keyframe mosaic for %s: %s", incident_id, exc)
        return ""


@app.post("/aggression/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...), location: str = Form("")):
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}:
        raise HTTPException(400, "Accepted formats: MP4, AVI, MOV, MKV, WebM, MPEG")
    video_path = UPLOADS_DIR / f"{uuid.uuid4()}{ext}"
    video_path.write_bytes(await file.read())
    try:
        from backend.aggression.video_analyzer import analyze_video as _analyze
        incident_dict = await asyncio.to_thread(_analyze, str(video_path), location.strip())
        key_frames_b64 = incident_dict.pop("key_frames_b64", "") or ""
        incident_id    = incident_dict.get("incident_id", "")
        kf_path        = _save_keyframe_mosaic(incident_id, key_frames_b64)
        # Always run the LangGraph agent so report + email are generated for
        # any detected activity. Only return "no_incident" for truly empty videos.
        if incident_dict.get("ema_score", 0) < 0.05 and not incident_dict.get("evidence_list"):
            return {
                "no_incident": True,
                "message": "No activity detected in this video.",
                "ema_score": incident_dict["ema_score"],
                "severity": "none",
                "evidence_list": [],
            }
        final_state = await asyncio.to_thread(aggression.process_incident, incident_dict, key_frames_b64) if aggression else {}
        if kf_path:
            final_state["keyframes_path"] = kf_path
        sev = final_state.get("severity", "minor")
        ema = incident_dict.get("ema_score", 0)
        return {
            "incident_id":       incident_id,
            "severity":          sev,
            "severity_rationale":final_state.get("severity_rationale"),
            "incident_type":     incident_dict.get("incident_type"),
            "aggression_type":   incident_dict.get("incident_type"),  # alias for frontend
            "ema_score":         ema,
            "sustained_frames":  incident_dict.get("sustained_frames"),
            "evidence_list":     incident_dict.get("evidence_list", []),
            # LLM-panel fields expected by aggression.html
            "is_aggression":     ema >= 0.28 or sev in ("serious", "critical"),
            "llm_confidence":    min(ema * 1.1, 0.99),
            "key_observations":  incident_dict.get("evidence_list", []),
            "recommended_action":final_state.get("severity_rationale", ""),
            "report_markdown":   final_state.get("report_markdown", ""),
            "contact_decision":  final_state.get("contact_decision", []),
            "notifications_sent":final_state.get("notifications_sent", []),
            "db_event_id":       final_state.get("db_event_id"),
            "actions_taken":     final_state.get("actions_taken", []),
            "has_keyframes":     bool(kf_path),
        }
    except ImportError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        log.exception("Video analysis failed")
        raise HTTPException(500, f"Analysis error: {exc}")
    finally:
        video_path.unlink(missing_ok=True)


@app.post("/aggression/incident")
async def post_incident(req: AggressionIncidentRequest):
    if aggression is None:
        raise HTTPException(503, "Aggression module not available — integrated in DOGSIM-TN")
    payload   = req.model_dump()
    image_b64 = payload.pop("image_jpeg_b64", "")
    final     = await asyncio.to_thread(aggression.process_incident, payload, image_b64)
    incident_id = payload.get("incident_id", "")
    kf_path     = _save_keyframe_mosaic(incident_id, image_b64)
    if kf_path:
        final["keyframes_path"] = kf_path
    return {
        "incident_id":   incident_id,
        "severity":      final.get("severity"),
        "rationale":     final.get("severity_rationale"),
        "contact_decision": final.get("contact_decision", []),
        "notifications_sent": final.get("notifications_sent", []),
        "report_markdown":  final.get("report_markdown", ""),
        "db_event_id":      final.get("db_event_id"),
        "actions_taken":    final.get("actions_taken", []),
        "has_keyframes":    bool(kf_path),
    }


@app.get("/aggression/incidents")
async def list_incidents(limit: int = Query(50, le=200)):
    if aggression is None:
        raise HTTPException(503, "Aggression module not available — integrated in DOGSIM-TN")
    return aggression.list_incidents(limit=limit)


@app.get("/aggression/incidents/{incident_id}")
async def get_incident(incident_id: str):
    if aggression is None:
        raise HTTPException(503, "Aggression module not available — integrated in DOGSIM-TN")
    inc = aggression.get_incident(incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    return inc


@app.get("/aggression/incidents/{incident_id}/keyframes")
async def get_keyframes(incident_id: str):
    if aggression is None:
        raise HTTPException(503, "Aggression module not available — integrated in DOGSIM-TN")
    inc = aggression.get_incident(incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    stored = inc.get("keyframes_path", "")
    if not stored:
        raise HTTPException(404, "No keyframe stored for this incident")
    kf_path = Path(stored)
    if not kf_path.exists():
        raise HTTPException(404, "Keyframe file not found on disk")
    return FileResponse(str(kf_path), media_type="image/jpeg")
