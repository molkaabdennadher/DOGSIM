"""
Lost dog recognition tool — CLIP ViT-B/32 zero-shot.

Strategy
--------
Encode the user's uploaded photo with CLIP ViT-B/32 and cosine-search
against the raw 512-d CLIP embeddings of every shelter dog stored in
``artifacts/aiAdvisor/img_emb_raw.npy``.

CLIP zero-shot delivers ~75–85% accuracy for clear photos with no training
required and no extra model file to maintain.

Loading
-------
1. CLIP ViT-B/32 backbone is loaded at startup.
2. ``img_emb_raw.npy``     — raw 512-d CLIP embeddings per pet.
3. ``pet_ids.npy``         — pet ID array aligned with img_emb_raw.
4. ``df_original.parquet`` — metadata for building the match response.
Dog-only filtering is applied via ``Type == 1`` from df_original.
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image

ARTIFACTS = Path(__file__).parent.parent.parent.parent / "artifacts" / "aiAdvisor"
UPLOADS_DIR = Path(__file__).parent.parent.parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

CLIP_DIM = 512  # CLIP ViT-B/32 output dimension

# Confidence tiers for the user-facing match label.
THRESHOLD_HIGH   = 0.82   # very likely the same dog
THRESHOLD_MEDIUM = 0.70   # possible match — worth investigating
THRESHOLD_MIN    = 0.55   # below this → assume "not in shelter"

# ── Module-level singletons ────────────────────────────────────────────────────
_clip_model   = None
_clip_preproc = None
_dog_index: faiss.Index | None = None
_dog_ids:   list[str]          = []
_dog_df:    pd.DataFrame | None = None


# ── Loading ────────────────────────────────────────────────────────────────────

def load_dog_artifacts() -> None:
    """Load CLIP and the dog FAISS index. Called once at server startup."""
    global _clip_model, _clip_preproc, _dog_index, _dog_df

    # CLIP backbone
    try:
        import open_clip
        _clip_model, _, _clip_preproc = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_model.eval()
        print("[dog_finder] CLIP ViT-B/32 loaded.")
    except ImportError:
        print("[dog_finder] WARNING — open_clip not installed. Run: pip install open-clip-torch")
        return

    # Metadata (shared with retrieval.py)
    df_path = ARTIFACTS / "df_original.parquet"
    if not df_path.exists():
        print("[dog_finder] WARNING — df_original.parquet missing. Re-run cosinesimilarity2.ipynb.")
        return
    _dog_df = pd.read_parquet(df_path)

    # Raw 512-d CLIP embeddings (all pets) + aligned pet IDs
    raw_path = ARTIFACTS / "img_emb_raw.npy"
    ids_path = ARTIFACTS / "pet_ids.npy"
    if not raw_path.exists() or not ids_path.exists():
        print("[dog_finder] WARNING — img_emb_raw.npy or pet_ids.npy missing in aiAdvisor/.")
        return

    all_emb = np.load(raw_path).astype("float32")
    all_ids = np.load(ids_path, allow_pickle=True)

    # Filter dogs only (Type == 1)
    dog_mask = _dog_df["Type"].values == 1
    dog_emb  = all_emb[dog_mask]
    _dog_ids[:] = [str(pid) for pid in all_ids[dog_mask]]

    _build_index(dog_emb)
    print(f"[dog_finder] Zero-shot CLIP index ready: {_dog_index.ntotal:,} dogs.")


def _build_index(dog_emb: np.ndarray) -> None:
    global _dog_index
    faiss.normalize_L2(dog_emb)
    _dog_index = faiss.IndexFlatIP(CLIP_DIM)
    _dog_index.add(dog_emb)


# ── Query ──────────────────────────────────────────────────────────────────────

def find_lost_dog(image_id: str) -> dict | None:
    """Compare the uploaded image against all shelter dogs.

    Returns the best match dict if confidence >= THRESHOLD_MIN, else None.
    Returns an error dict if the finder is not ready.
    """
    if _clip_model is None or _dog_index is None:
        return {"error": "Dog finder not ready. Check that img_emb_raw.npy is in artifacts/aiAdvisor/."}

    image_path = UPLOADS_DIR / image_id
    if not image_path.exists():
        return {"error": f"Uploaded image '{image_id}' not found."}

    query_vec = _encode_image(image_path)
    faiss.normalize_L2(query_vec)

    D, I = _dog_index.search(query_vec, 5)
    best_score = float(D[0][0])
    best_idx   = int(I[0][0])

    if best_score < THRESHOLD_MIN:
        return None

    confidence = (
        "high"   if best_score >= THRESHOLD_HIGH   else
        "medium" if best_score >= THRESHOLD_MEDIUM else
        "low"
    )

    pet_id = _dog_ids[best_idx]
    rows   = _dog_df[_dog_df["PetID"] == pet_id]
    if rows.empty:
        return None
    row = rows.iloc[0]

    return {
        "pet_id":      pet_id,
        "name":        str(row.get("Name", "Unknown")),
        "breed":       str(row.get("Breed1", "Unknown")),
        "age_label":   _age_label(int(row.get("Age", 0))),
        "gender":      {1: "Male", 2: "Female"}.get(row.get("Gender"), "Unknown"),
        "description": str(row.get("Description", ""))[:300],
        "similarity":  round(best_score, 3),
        "confidence":  confidence,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _encode_image(path: Path) -> np.ndarray:
    """Encode a single image to a 512-d CLIP embedding (shape: 1 × 512)."""
    image = _clip_preproc(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = _clip_model.encode_image(image).cpu().numpy().astype("float32")
    return emb  # shape (1, 512)


def _age_label(months: int) -> str:
    if months < 12:
        return f"{months} months"
    years = months // 12
    return f"{years} year{'s' if years > 1 else ''}"
