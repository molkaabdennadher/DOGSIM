"""
Pet retrieval tool — ChromaDB first, FAISS fallback.

Artifacts exported by cosinesimilarity2.ipynb (Gemini descriptions, dynamic PCA)
must be placed in artifacts/aiAdvisor/.

Search pipeline
---------------
1. Encode the natural-language query with MiniLM (text_model).
2. Project it through user_encoder into the shared 128-d space.
3. If ChromaDB available: collection.query() with metadata + feedback filtering.
   Else: FAISS top-(k * 10) candidates → hard filters → feedback re-rank.
4. Subtract per-pet penalty derived from the caller's UserPreferenceProfile.
5. Return top-k.

Why ChromaDB first?
-------------------
- Metadata built-in (no separate _df lookup for core fields)
- Cosine-aware index (no manual normalize_L2 needed)
- Persistent local collection, scales better than FAISS for large datasets
- Falls back to FAISS if chroma_db/ does not exist (backward compatibility)

Embedding space (v3 — Gemini descriptions, adaptive PCA)
---------------------------------------------------------
- TEXT:   MiniLM on Gemini-generated rich descriptions (384-d)
          → PCA(n_components=0.90 variance, dynamic dim)
- IMAGE:  CLIP ViT-B/32 (512-d)
          → PCA(n_components=0.90 variance, dynamic dim)
- Fusion: hstack([text_pca, img_pca]) = FUSED_DIM (dynamic)
- PetEncoder:  FUSED_DIM → 128-d (L2-normalized, InfoNCE)
- UserEncoder: 384-d (MiniLM) → 128-d (L2-normalized)

Note: PCA is applied at notebook training time only. At inference, the
user encoder maps raw MiniLM (384-d) directly to 128-d — no PCA needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from backend.maya.schemas import UserPreferenceProfile

ARTIFACTS = Path(__file__).parent.parent.parent.parent / "artifacts" / "aiAdvisor"

EMBED_DIM = 128   # Shared latent space dimension
TEXT_DIM  = 384   # MiniLM output dimension

# PetFinder column → integer code mappings (must match cosinesimilarity2.ipynb)
TYPE_MAP   = {"dog": 1, "cat": 2, "rabbit": 3, "bird": 4, "other": 5}
SIZE_MAP   = {"small": [1], "medium": [2], "large": [3, 4], "any": [1, 2, 3, 4]}
GENDER_MAP = {"male": 1, "female": 2}

# Soft re-rank: each matching avoid signal subtracts this from cosine score.
# 0.05 is gentle enough to keep borderline pets in range while reliably pushing
# multi-signal matches below unpenalised alternatives.
FEEDBACK_PENALTY = 0.05


class Encoder(nn.Module):
    """Two-layer MLP projector — must match the architecture used in cosinesimilarity2.ipynb."""

    def __init__(self, in_dim: int, out_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ── Module-level singletons (loaded once at startup via load_artifacts()) ──────

_index:             faiss.Index | None         = None
_chroma_client:     object | None              = None   # chromadb.PersistentClient
_chroma_collection: object | None              = None   # chromadb Collection
_using_chroma:      bool                       = False
_df:                pd.DataFrame | None        = None
_pet_ids:           np.ndarray | None          = None
_user_encoder:      Encoder | None             = None
_text_model:        SentenceTransformer | None = None
_thresholds:        dict | None                = None
_rich_descriptions: dict | None               = None   # pet_id → Gemini description


# ── Startup ───────────────────────────────────────────────────────────────────

def load_artifacts() -> None:
    """Load all ML artifacts into memory. Called once at server startup.

    Tries ChromaDB first, falls back to FAISS. Both paths load the same
    user encoder and sentence-transformer so query-time code is identical.
    """
    global _index, _chroma_client, _chroma_collection, _using_chroma, \
    _df, _pet_ids, _user_encoder, _text_model, _thresholds, _rich_descriptions

    required = ["pet_emb.npy", "user_encoder.pt", "df_original.parquet",
                "pet_ids.npy", "thresholds.json"]
    missing  = [f for f in required if not (ARTIFACTS / f).exists()]
    if missing:
        print(f"[retrieval] WARNING — missing artifacts: {missing}")
        print("[retrieval] Run cosinesimilarity2.ipynb on Kaggle and export artifacts first.")
        return

    # ── Rich descriptions (Gemini, optional — falls back to raw Description col) ─
    rich_path = ARTIFACTS / "rich_descriptions.json"
    if rich_path.exists():
        _rich_descriptions = json.loads(rich_path.read_text())
        print(f"[retrieval] Rich descriptions loaded: {len(_rich_descriptions):,} pets.")
    else:
        _rich_descriptions = {}
        print("[retrieval] rich_descriptions.json not found — using raw Description column.")

    # ── Try ChromaDB (preferred) ───────────────────────────────────────────────
    chroma_dir = ARTIFACTS / "chroma_db"
    if chroma_dir.exists():
        try:
            import chromadb
            _chroma_client     = chromadb.PersistentClient(path=str(chroma_dir))
            _chroma_collection = _chroma_client.get_collection("pets")
            _using_chroma      = True
            print(f"[retrieval] ChromaDB loaded: {_chroma_collection.count():,} pets")
        except Exception as exc:
            print(f"[retrieval] WARNING - ChromaDB failed: {exc}  -> falling back to FAISS.")
            _using_chroma = False
    else:
        _using_chroma = False

    # ── Common artifacts (both backends need these) ────────────────────────────
    pet_emb     = np.load(ARTIFACTS / "pet_emb.npy").astype("float32")
    _pet_ids    = np.load(ARTIFACTS / "pet_ids.npy", allow_pickle=True)
    _df         = pd.read_parquet(ARTIFACTS / "df_original.parquet")
    _thresholds = json.loads((ARTIFACTS / "thresholds.json").read_text())

    # ── FAISS index (always built — used as fallback or when ChromaDB absent) ──
    faiss.normalize_L2(pet_emb)
    _index = faiss.IndexFlatIP(EMBED_DIM)
    _index.add(pet_emb)

    # ── User-side encoder (maps 384-d MiniLM → 128-d shared space) ────────────
    _user_encoder = Encoder(TEXT_DIM, EMBED_DIM)
    # weights_only=False: required for torch >= 2.4 when the checkpoint was
    # saved with torch.save(state_dict) using an older version.
    _user_encoder.load_state_dict(
        torch.load(ARTIFACTS / "user_encoder.pt", map_location="cpu", weights_only=False)
    )
    _user_encoder.eval()

    # ── Sentence transformer (same model as used in the notebook) ──────────────
    # Try local cache first (no network); fall back to download on first run.
    try:
        _text_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    except Exception:
        try:
            _text_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            print(f"[retrieval] WARNING — SentenceTransformer download failed: {exc}")
            print("[retrieval] Ensure internet access on first run, or pre-download the model.")
            return

    if _using_chroma:
        print(f"[retrieval] Ready — ChromaDB ({_chroma_collection.count():,} pets).")
    else:
        print(f"[retrieval] Ready — FAISS ({_index.ntotal:,} pets).")


def _is_ready() -> bool:
    """Return True if the retrieval backend has been loaded successfully."""
    if _using_chroma:
        return _chroma_collection is not None and _df is not None
    return _index is not None and _df is not None


# ── Internal: ChromaDB search ─────────────────────────────────────────────────

def _search_chroma(
    user_vec:     np.ndarray,
    excluded_ids: set[str],
    preferences:  dict,
    profile:      UserPreferenceProfile | None,
    k:            int = 5,
) -> list[dict]:
    """Search ChromaDB with hard-filter + feedback re-rank.

    Over-fetches by 10× so filtering still leaves enough candidates.
    Converts ChromaDB cosine distance → similarity (1 − distance).
    """
    results = _chroma_collection.query(
        query_embeddings=user_vec.tolist(),
        n_results=k * 10,
        include=["metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    animal_type = preferences.get("animal_type", "any")
    max_fee     = preferences.get("max_fee", 0)
    max_age     = preferences.get("max_age_months", 0)
    size_pref   = preferences.get("size", "any")

    candidates: list[tuple[float, dict]] = []

    for pet_id, distance, meta in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        # ChromaDB cosine distance → cosine similarity
        similarity = 1.0 - float(distance)

        # ── Hard exclusions ────────────────────────────────────────────────────
        if pet_id in excluded_ids:
            continue
        if animal_type != "any" and TYPE_MAP.get(animal_type) != meta.get("animal_type"):
            continue
        if max_fee > 0 and meta.get("fee", 0) > max_fee:
            continue
        if max_age > 0 and meta.get("age_months", 0) > max_age:
            continue
        if size_pref != "any":
            if meta.get("size") not in SIZE_MAP.get(size_pref, [1, 2, 3, 4]):
                continue

        # ── Soft feedback penalty ──────────────────────────────────────────────
        # Look up the full row for breed/size/type penalty calculation
        mask = _df["PetID"] == pet_id
        row  = _df[mask].iloc[0] if mask.any() else None
        penalty  = _feedback_penalty(row, profile) if row is not None else 0.0
        adjusted = similarity - penalty

        candidates.append((adjusted, {
            "pet_id":           pet_id,
            "name":             meta.get("name", "Unknown"),
            "animal_type":      "Dog" if meta.get("animal_type") == 1 else "Cat",
            "breed":            meta.get("breed", ""),
            "size":             _size_label(meta.get("size")),
            "age_months":       meta.get("age_months", 0),
            "age_label":        _age_label(meta.get("age_months", 0)),
            "gender":           {1: "Male", 2: "Female", 3: "Mixed"}.get(meta.get("gender"), "Unknown"),
            "fee":              float(meta.get("fee", 0)),
            "description":      str(meta.get("description", ""))[:300],
            "similarity_score": round(similarity, 3),
        }))

    # Re-sort by adjusted score and return top-k
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in candidates[:k]]


# ── Public API ────────────────────────────────────────────────────────────────

def search_pets(
    preferences:  dict,
    excluded_ids: list[str],
    profile:      UserPreferenceProfile | None = None,
    k:            int = 5,
) -> list[dict]:
    """Return up to k pets matching the preference dict.

    Routes to ChromaDB (preferred) or FAISS (fallback). In both cases:
      - hard filters are applied first (animal_type, max_fee, max_age, size)
      - soft feedback penalties are subtracted from cosine scores
      - results are sorted by adjusted score

    Args:
        preferences: dict with keys:
            query_description (str): natural-language preference string built by Maya
            animal_type (str):       "dog" | "cat" | "any"
            max_fee (int):           max adoption fee in local currency (0 = no limit)
            max_age_months (int):    max age in months (0 = no limit)
            size (str):              "small" | "medium" | "large" | "any"
        excluded_ids: pet IDs rejected this session (cleared when session ends)
        profile:      cross-session feedback profile from feedback.build_profile()
        k:            number of pets to return
    """
    if not _is_ready():
        return []

    # Encode the natural-language query the LLM built for us
    query_text = preferences.get("query_description", "a friendly pet")
    query_emb  = _text_model.encode([query_text])[0]
    query_t    = torch.from_numpy(query_emb).float().unsqueeze(0)

    with torch.no_grad():
        user_vec = _user_encoder(query_t).numpy().astype("float32")
    faiss.normalize_L2(user_vec)

    # Union of this-session rejections + persistent cross-session avoids
    skip_ids = set(excluded_ids)
    if profile:
        skip_ids.update(profile.excluded_pet_ids)

    # ── ChromaDB path ──────────────────────────────────────────────────────────
    if _using_chroma:
        return _search_chroma(user_vec, skip_ids, preferences, profile, k)

    # ── FAISS path ─────────────────────────────────────────────────────────────
    D, I = _index.search(user_vec, k * 10)   # over-fetch for filtering room

    animal_type = preferences.get("animal_type", "any")
    max_fee     = preferences.get("max_fee", 0)
    max_age     = preferences.get("max_age_months", 0)
    size_pref   = preferences.get("size", "any")

    candidates: list[tuple[float, dict]] = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(_df):
            continue

        row    = _df.iloc[idx]
        pet_id = str(_pet_ids[idx])

        if pet_id in skip_ids:
            continue
        if animal_type != "any" and TYPE_MAP.get(animal_type) != row.get("Type"):
            continue
        if max_fee > 0 and row.get("Fee", 0) > max_fee:
            continue
        if max_age > 0 and row.get("Age", 0) > max_age:
            continue
        if size_pref != "any":
            if row.get("MaturitySize") not in SIZE_MAP.get(size_pref, [1, 2, 3, 4]):
                continue

        adjusted = float(score) - _feedback_penalty(row, profile)
        candidates.append((adjusted, _row_to_dict(row, pet_id, float(score))))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in candidates[:k]]


def get_pet_details(pet_id: str) -> dict:
    """Return full metadata for a single pet by PetID. Used by GET /pets/{pet_id}."""
    if not _is_ready():
        return {"error": "Artifacts not loaded"}

    mask = _df["PetID"] == pet_id
    if not mask.any():
        return {"error": f"Pet {pet_id} not found"}

    row = _df[mask].iloc[0]
    return _row_to_dict(row, pet_id, similarity_score=1.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _feedback_penalty(row, profile: UserPreferenceProfile | None) -> float:
    """Compute soft score penalty based on the user's systematic avoid signals.

    Each avoid signal (breed, size, animal type) that matches this pet adds
    FEEDBACK_PENALTY to the total. Pets matching multiple signals are pushed
    down proportionally while remaining in the result set.
    """
    if profile is None or profile.total_rejections == 0:
        return 0.0

    penalty = 0.0
    breed = str(row.get("Breed1", "")).strip().lower()
    size  = _size_label(row.get("MaturitySize"))
    atype = "dog" if row.get("Type") == 1 else "cat"

    if breed and breed in profile.avoided_breeds:
        penalty += FEEDBACK_PENALTY
    if size and size in profile.avoided_sizes:
        penalty += FEEDBACK_PENALTY
    if atype in profile.avoided_animal_types:
        penalty += FEEDBACK_PENALTY
    return penalty


def _size_label(maturity_size) -> str:
    """Convert PetFinder MaturitySize code (1–4) to a string label."""
    return {1: "small", 2: "medium", 3: "large", 4: "large"}.get(maturity_size, "")


def _age_label(months: int) -> str:
    """Convert age in months to a human-readable label."""
    if months < 3:
        return "Newborn"
    if months < 12:
        return f"{months} months"
    years = months // 12
    return f"{years} year{'s' if years > 1 else ''}"


def _row_to_dict(row, pet_id: str, similarity_score: float) -> dict:
    """Serialise a DataFrame row to the standard pet result dict.

    Uses the Gemini-generated rich description when available (from
    rich_descriptions.json), falling back to the raw PetFinder description.
    Used by both the FAISS and ChromaDB code paths.
    """
    type_label   = "Dog" if row.get("Type") == 1 else "Cat"
    gender_label = {1: "Male", 2: "Female", 3: "Mixed"}.get(row.get("Gender"), "Unknown")
    age_months   = int(row.get("Age", 0))

    # Prefer Gemini description; fall back to raw column
    description = (
        _rich_descriptions.get(str(pet_id))
        or str(row.get("Description", ""))
    )[:300]

    return {
        "pet_id":           pet_id,
        "name":             str(row.get("Name", "Unknown")) or "Unknown",
        "animal_type":      type_label,
        "breed":            str(row.get("Breed1", "")),
        "size":             _size_label(row.get("MaturitySize")),
        "age_months":       age_months,
        "age_label":        _age_label(age_months),
        "gender":           gender_label,
        "fee":              float(row.get("Fee", 0)),
        "description":      description,
        "similarity_score": round(similarity_score, 3),
    }
