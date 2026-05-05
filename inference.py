"""
inference.py - Dog behaviour analysis for Dr Halim.
Loads behaviour + voice models from dr_halim/models/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
import numpy as np
import librosa
import cv2
from PIL import Image
from ultralytics import YOLO

# ======================================================================
#  DEVICE
# ======================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
#  CLASSES
# ======================================================================
BEHAV_CLASSES = sorted([
    "tail_wagging", "playing", "sitting", "standing", "eating", "lying",
    "restlessness", "paralysis", "incoordination", "digging", "barking",
    "hyper_salivation", "bone_in_throat", "dropped_jaw",
    "sudden_aggression", "seizure",
])
BEHAV_ANORMAL = [
    "restlessness", "paralysis", "incoordination", "hyper_salivation",
    "bone_in_throat", "dropped_jaw", "sudden_aggression", "seizure",
]
BEHAV_ANORMAL_IDX = [BEHAV_CLASSES.index(c) for c in BEHAV_ANORMAL]

VOIX_CLASSES  = ["bark", "distress", "growl", "grunt", "whine"]
VOIX_ANORMAL  = ["growl", "distress"]
VOIX_ANORMAL_IDX = [VOIX_CLASSES.index(c) for c in VOIX_ANORMAL]

# ======================================================================
#  LATE FUSION PARAMETERS  (identical to training notebook)
# ======================================================================
W_BEHAV         = 0.6
W_VOIX          = 0.4
THRESHOLD_MULTI = 0.50
THRESHOLD_MONO  = 0.55
SAFETY_RULE     = 0.70

# ======================================================================
#  MODEL PATHS  -  all inside dr_halim/models/
# ======================================================================
_HERE      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(_HERE, "..", "models"))

BEHAV_EFF_PATH  = os.path.join(MODELS_DIR, "comportement", "efficientnet_best.pth")
BEHAV_RES_PATH  = os.path.join(MODELS_DIR, "comportement", "resnet50_best.pth")
BEHAV_YOLO_PATH = os.path.join(MODELS_DIR, "comportement", "yolov8_best.pt")
VOIX_EFF_PATH   = os.path.join(MODELS_DIR, "voix", "efficientnet_phase2_best.pth")
VOIX_RES_PATH   = os.path.join(MODELS_DIR, "voix", "resnet50_phase2_best.pth")

# ======================================================================
#  MODEL LOADERS  (weights=None replaces deprecated pretrained=False)
# ======================================================================

def _load_state(path: str) -> dict:
    """Load state dict - handles both weights_only=True and legacy format."""
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except Exception:
        return torch.load(path, map_location=DEVICE, weights_only=False)


def load_behav_efficientnet():
    model = tv_models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, 16),
    )
    model.load_state_dict(_load_state(BEHAV_EFF_PATH))
    return model.to(DEVICE).eval()


def load_behav_resnet50():
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 16),
    )
    model.load_state_dict(_load_state(BEHAV_RES_PATH))
    return model.to(DEVICE).eval()


def load_voix_efficientnet():
    model = tv_models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, 5),
    )
    model.load_state_dict(_load_state(VOIX_EFF_PATH))
    return model.to(DEVICE).eval()


def load_voix_resnet50():
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 5),
    )
    model.load_state_dict(_load_state(VOIX_RES_PATH))
    return model.to(DEVICE).eval()


# ======================================================================
#  LOAD ALL MODELS AT IMPORT TIME
# ======================================================================
print(f"Loading behaviour/voice models on {DEVICE} ...")
behav_eff  = load_behav_efficientnet()
behav_res  = load_behav_resnet50()
behav_yolo = YOLO(BEHAV_YOLO_PATH)
voix_eff   = load_voix_efficientnet()
voix_res   = load_voix_resnet50()
print("Behaviour and voice models ready.")

# ======================================================================
#  IMAGE TRANSFORM
# ======================================================================
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ======================================================================
#  FRAME EXTRACTION
# ======================================================================

def extract_frames(video_path: str, max_frames: int = 16) -> list:
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step   = max(1, total // max_frames)
    frames = []
    idx    = 0
    while cap.isOpened() and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += step
    cap.release()
    return frames

# ======================================================================
#  AUDIO EXTRACTION
# ======================================================================

def _find_ffmpeg() -> str | None:
    """Locate the FFmpeg executable, even when launched as a subprocess of another
    application (e.g. pet-advisor) that may not have the user's full shell PATH.

    Search order:
      1. shutil.which  — respects the current process PATH
      2. Registry      — reads HKLM/HKCU Software\FFmpeg (rarely set, but possible)
      3. WinGet packages dir  — the known install location for Gyan.FFmpeg via winget
      4. Common static paths  — chocolatey, scoop, manual installs, Program Files
      5. Every directory in PATH  — explicit fallback if which() misses something
    """
    import subprocess
    import shutil
    import glob

    def _probe(exe: str) -> bool:
        try:
            r = subprocess.run([exe, "-version"], capture_output=True, timeout=5)
            return r.returncode == 0
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return False

    # ── 1. shutil.which (fastest, respects current PATH) ─────────────────────
    found = shutil.which("ffmpeg")
    if found and _probe(found):
        return found

    candidates: list[str] = []

    # ── 2. WinGet packages directory (Gyan.FFmpeg installed via winget) ───────
    winget_base = os.path.expanduser(
        r"~\AppData\Local\Microsoft\WinGet\Packages"
    )
    if os.path.isdir(winget_base):
        # glob all Gyan.FFmpeg* subdirs and pick the first ffmpeg.EXE/ffmpeg.exe
        for exe in glob.glob(
            os.path.join(winget_base, "Gyan.FFmpeg*", "**", "ffmpeg.EXE"),
            recursive=True,
        ):
            candidates.append(exe)
        for exe in glob.glob(
            os.path.join(winget_base, "Gyan.FFmpeg*", "**", "ffmpeg.exe"),
            recursive=True,
        ):
            candidates.append(exe)

    # ── 3. WinGet links / scoop shims ────────────────────────────────────────
    candidates += [
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"),
        os.path.expanduser(r"~\scoop\shims\ffmpeg.exe"),
        os.path.expanduser(r"~\scoop\apps\ffmpeg\current\bin\ffmpeg.exe"),
    ]

    # ── 4. Common static install locations ───────────────────────────────────
    candidates += [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
        r"C:\Users\Public\ffmpeg\bin\ffmpeg.exe",
    ]

    # ── 5. Explicit PATH scan (fallback if shutil.which missed it) ────────────
    for d in os.environ.get("PATH", "").split(os.pathsep):
        for name in ("ffmpeg.exe", "ffmpeg"):
            candidates.append(os.path.join(d, name))

    for c in candidates:
        if c and os.path.isfile(c) and _probe(c):
            return c

    return None


_FFMPEG_PATH: str | None = _find_ffmpeg()
if _FFMPEG_PATH:
    print(f"[audio] FFmpeg found: {_FFMPEG_PATH}")
else:
    print("[audio] WARNING: FFmpeg not found — audio analysis disabled.")
    print("[audio]   Install via winget: winget install Gyan.FFmpeg")


def extract_audio(video_path: str, output_path: str) -> tuple[bool, str]:
    """Extract audio from a video file to a WAV file.

    Returns:
        (success: bool, status: str)
        status values: "ok" | "ffmpeg_missing" | "no_audio_stream" |
                       "extraction_failed" | "empty_wav"
    """
    import subprocess
    ffmpeg = _FFMPEG_PATH
    if not ffmpeg:
        print("[audio] FFmpeg not found — cannot extract audio.")
        return False, "ffmpeg_missing"

    # Probe whether the video has an audio stream
    probe_cmd = [ffmpeg, "-i", video_path]
    try:
        probe = subprocess.run(probe_cmd, capture_output=True, timeout=15)
    except Exception as e:
        print(f"[audio] FFmpeg probe error: {e}")
        return False, "extraction_failed"

    probe_out = probe.stderr.decode("utf-8", errors="replace")
    if "Audio:" not in probe_out:
        print(f"[audio] No audio stream found in {os.path.basename(video_path)}")
        return False, "no_audio_stream"

    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
        output_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode != 0:
            err_msg = r.stderr.decode("utf-8", errors="replace")[-300:]
            print(f"[audio] FFmpeg extraction failed (exit {r.returncode}): {err_msg}")
            return False, "extraction_failed"
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
            print("[audio] WAV file empty or missing after extraction.")
            return False, "empty_wav"
        print(f"[audio] Audio extracted OK ({os.path.getsize(output_path)} bytes)")
        return True, "ok"
    except Exception as e:
        print(f"[audio] FFmpeg error: {e}")
        return False, "extraction_failed"


def has_voice_activity(audio_path: str) -> bool:
    """Return True if the WAV file is valid and long enough to analyse.

    No amplitude check — the voice model handles quiet/compressed audio.
    We only skip if the file is missing, too small, or unreadable.
    """
    try:
        file_size = os.path.getsize(audio_path)
        if file_size < 500:
            print(f"[audio] WAV file too small ({file_size} bytes) — skipping.")
            return False
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        print(f"[audio] WAV OK — size={file_size}B  duration={duration:.2f}s")
        return duration >= 0.1   # at least 100 ms → run the voice model
    except Exception as e:
        print(f"[audio] has_voice_activity error: {e}")
        return False


def wav_to_melspec(audio_path: str):
    y, sr  = librosa.load(audio_path, sr=22050, duration=3.0)
    target = sr * 3
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                             hop_length=512, n_fft=2048)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    norm   = ((mel_db - mel_db.min()) /
              (mel_db.max() - mel_db.min()) * 255).astype(np.uint8)
    img    = Image.fromarray(norm).convert("RGB").resize((224, 224))
    tensor = IMG_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    return tensor, img

# ======================================================================
#  BEHAVIOUR INFERENCE
# ======================================================================

def predict_behavior(frames: list, use_yolo: bool = True,
                      anormal_idx: list = None) -> dict:
    if anormal_idx is None:
        anormal_idx = BEHAV_ANORMAL_IDX

    all_eff, all_res, all_yolo = [], [], []
    best_frame, best_score = None, -1

    for frame in frames:
        tensor = IMG_TRANSFORM(frame).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p_eff = torch.softmax(behav_eff(tensor), dim=1).cpu().numpy()[0]
            p_res = torch.softmax(behav_res(tensor), dim=1).cpu().numpy()[0]

        p_yolo = np.zeros(len(BEHAV_CLASSES))
        if use_yolo:
            tmp_dir  = os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp"))
            tmp_path = os.path.join(tmp_dir, f"yolo_{os.getpid()}.jpg")
            frame.save(tmp_path)
            try:
                r_yolo     = behav_yolo(tmp_path, verbose=False)
                yolo_probs = r_yolo[0].probs.data.cpu().numpy()
                yolo_names = r_yolo[0].names
                for yi, yn in yolo_names.items():
                    if yn in BEHAV_CLASSES:
                        p_yolo[BEHAV_CLASSES.index(yn)] = yolo_probs[yi]
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        all_eff.append(p_eff)
        all_res.append(p_res)
        all_yolo.append(p_yolo)

        score = ((0.4*p_eff + 0.3*p_res + 0.3*p_yolo) if use_yolo
                 else (0.571*p_eff + 0.429*p_res))[anormal_idx].sum()
        if score > best_score:
            best_score, best_frame = score, frame

    mean_eff  = np.mean(all_eff,  axis=0)
    mean_res  = np.mean(all_res,  axis=0)
    mean_yolo = np.mean(all_yolo, axis=0)
    fusion    = (0.4*mean_eff + 0.3*mean_res + 0.3*mean_yolo if use_yolo
                 else 0.571*mean_eff + 0.429*mean_res)

    return {
        "probs_fusion"  : fusion,
        "score_anormal" : float(fusion[anormal_idx].sum()),
        "top_class"     : BEHAV_CLASSES[int(fusion.argmax())],
        "best_frame"    : best_frame,
        "probs_eff"     : mean_eff,
    }

# ======================================================================
#  VOICE INFERENCE
# ======================================================================

def predict_voice(audio_path: str) -> dict:
    tensor, mel_img = wav_to_melspec(audio_path)
    with torch.no_grad():
        p_eff = torch.softmax(voix_eff(tensor), dim=1).cpu().numpy()[0]
        p_res = torch.softmax(voix_res(tensor), dim=1).cpu().numpy()[0]
    fusion = 0.5*p_eff + 0.5*p_res
    return {
        "probs_fusion"  : fusion,
        "score_anormal" : float(fusion[VOIX_ANORMAL_IDX].sum()),
        "top_class"     : VOIX_CLASSES[int(fusion.argmax())],
        "mel_image"     : mel_img,
        "probs_eff"     : p_eff,
    }

# ======================================================================
#  LATE FUSION DECISION
# ======================================================================

def late_fusion_decision(behav_result: dict,
                          voix_result: dict = None,
                          voice_present: bool = True) -> dict:
    s_b       = behav_result["score_anormal"]
    top_behav = behav_result["top_class"]

    if voice_present and voix_result is not None:
        s_v           = voix_result["score_anormal"]
        top_voix      = voix_result["top_class"]
        score         = W_BEHAV * s_b + W_VOIX * s_v
        threshold     = THRESHOLD_MULTI
        mode          = "multimodal"
        contrib_behav = round(W_BEHAV * s_b / score * 100, 1) if score > 0 else 60.0
        contrib_voix  = round(W_VOIX  * s_v / score * 100, 1) if score > 0 else 40.0
    else:
        s_v = top_voix = None
        score         = s_b
        threshold     = THRESHOLD_MONO
        mode          = "behaviour_only"
        contrib_behav = 100.0
        contrib_voix  = 0.0

    if s_b >= SAFETY_RULE:
        decision = "ANORMAL"
        reason   = f"Safety rule: '{top_behav}' is highly abnormal ({s_b:.0%})"
    else:
        decision = "ANORMAL" if score >= threshold else "NORMAL"
        reason   = (f"Fusion score {score:.0%} "
                    f"{'ge' if decision == 'ANORMAL' else 'lt'} "
                    f"threshold {threshold:.0%}")

    return {
        "decision"     : decision,
        "mode"         : mode,
        "score"        : round(score, 3),
        "confidence"   : round(score if decision == "ANORMAL" else 1 - score, 3),
        "score_behav"  : round(s_b, 3),
        "score_voix"   : round(s_v, 3) if s_v is not None else None,
        "top_behav"    : top_behav,
        "top_voix"     : top_voix,
        "threshold"    : threshold,
        "reason"       : reason,
        "contrib_behav": contrib_behav,
        "contrib_voix" : contrib_voix,
    }
