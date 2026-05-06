# Pet Advisor — Setup & Production Guide

End-to-end procedure to bring the system from a fresh clone to a running advisor that recommends adoptable pets and recognises lost dogs from a photo.

The project has three moving parts:

1. **`cosinesimilarity2.ipynb`** — builds the multimodal pet retrieval index (FAISS) and calibrates similarity thresholds.
2. **`dog_finder_training.ipynb`** — trains a Siamese network that re-identifies a specific dog from a photo.
3. **`pet-advisor/`** — the live application: a FastAPI backend running a LangGraph agent (Maya) that calls those two indexes through tools, plus a single-page frontend.

The notebooks run on **Kaggle** (the PetFinder dataset is mounted there), they produce an `artifacts/` folder, and the backend consumes those artifacts locally. The repo will not work end-to-end until you have run *both* notebooks and copied their outputs into `pet-advisor/artifacts/`.

---

## 0. Prerequisites

| Tool                      | Version                | Notes                                                          |
|---------------------------|------------------------|----------------------------------------------------------------|
| Python                    | 3.10 or 3.11           | 3.12 works but PyTorch wheels lag a few weeks behind.          |
| pip                       | latest                 |                                                                |
| Git                       | any                    |                                                                |
| Kaggle account            | free                   | Required to access the PetFinder dataset and to run the notebooks with a free GPU. |
| Ollama                    | 0.3+                   | Local LLM server. Pull `llama3.2` once installed.              |
| ~6 GB free disk           | for artifacts + models | Approximate sizes: pet_emb=8 MB, df=10 MB, img_emb_raw=120 MB, dog_emb_reid=8 MB, llama3.2=2 GB. |

> **Why Ollama and not the Anthropic / OpenAI API?** You explicitly chose to keep the LLM local for this project. The whole agent layer is OpenAI-compatible, so the only thing to change to switch providers is the `OLLAMA` constant and the API key in `backend/agent.py`.

---

## 1. Clone, install, and create the local Python environment

```bash
git clone <your-repo-url>
cd "adoption system/pet-advisor"

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Pinning rationale (see `requirements.txt`):

- `langgraph >= 0.2.50` — the agent orchestration. Justification in §6 below.
- `openai >= 1.54.0` — wire-format compatibility with Ollama's chat-completion endpoint, including tool calls.
- `faiss-cpu` — same index implementation used in the notebooks; switching to `faiss-gpu` is safe but unnecessary for ~15 K pets.
- `open-clip-torch` — the local equivalent of OpenAI's CLIP repo (which is unmaintained and pip-unfriendly).

**Verify** the install:

```bash
python -c "import langgraph, openai, faiss, torch, sentence_transformers, open_clip; print('OK')"
```

---

## 2. Install Ollama and pull `llama3.2`

The agent talks to Ollama through its OpenAI-compatible HTTP endpoint at `http://localhost:11434/v1`.

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Windows: download the installer from https://ollama.com
```

**Verify**:

```bash
ollama run llama3.2 "Say hello in three words."
```

If you see a coherent answer, you are good. The pet-advisor will not start without this — it spawns a real client connection on the first chat turn.

> **Tool-calling caveat.** `llama3.2` (3B) supports OpenAI-style tool calls but occasionally emits malformed JSON arguments. The agent guards against this (`json.JSONDecodeError → empty args`), but if you see degraded behaviour, switch to `llama3.1:8b-instruct` (`ollama pull llama3.1:8b-instruct`) and update `MODEL` in `backend/agent.py` — bigger model, same protocol.

---

## 3. Get the PetFinder dataset onto Kaggle

The notebooks expect the data at:

```
/kaggle/input/competitions/petfinder-adoption-prediction/train/train.csv
/kaggle/input/competitions/petfinder-adoption-prediction/train_images/{PetID}-{n}.jpg
```

1. Go to <https://www.kaggle.com/competitions/petfinder-adoption-prediction>.
2. Accept the rules, then either fork an existing notebook or upload `cosinesimilarity2.ipynb` and `dog_finder_training.ipynb` to a new notebook.
3. In the notebook sidebar, **Add Data → Competitions → "PetFinder.my Adoption Prediction"**. Make sure the input path matches the constants `DATA_DIR` and `IMAGE_DIR` at the top of each notebook — adjust them if Kaggle mounts the dataset under a slightly different path.

Running locally is possible but discouraged: the dataset is ~3 GB of images and CLIP encoding without a GPU takes hours.

---

## 4. Run `cosinesimilarity2.ipynb` (the retrieval index)

On Kaggle, switch the accelerator to **GPU T4 x2** (free tier), then **Run All**. The notebook should finish in ~10 minutes.

What it produces, in order:

| Section              | What happens                                                       | Output                |
|----------------------|--------------------------------------------------------------------|-----------------------|
| 3. Tabular           | OHE + IQR + StandardScaler                                         | `tab` array           |
| 4. Text (MiniLM)     | 384-d sentence embeddings → PCA → 64-d                             | `text_emb`            |
| 5. Image (CLIP)      | 512-d CLIP embeddings (kept raw) + PCA → 64-d                      | `img_emb_raw`, `img_emb_pca` |
| 6. Fusion            | concat + per-block re-norm                                         | `pet_features`        |
| 7. Encoders          | MLP → 128-d shared space                                           |                       |
| 9. InfoNCE training  | 3 epochs × 200 steps                                               | trained encoders      |
| 10. FAISS index      | L2-normalise + IndexFlatIP                                         | `index`, `pet_emb`    |
| 11. Threshold calib. | percentile-based, no hardcoded values                              | `THRESHOLDS`          |
| 13. Export           | writes `/kaggle/working/artifacts/`                                | (see below)           |
| 14. Verification     | confirms each artifact exists and is non-empty                     | sanity printout       |

**Verifications you must run before downloading:**

- The training loss decreases each epoch (should land roughly between 0.4 and 1.5 by epoch 3).
- The threshold-distribution histogram has visible spread between the green (P25) and red (P75) lines.
- Section 12 (`Sanity-check retrieval`) returns plausibly different pets for the three test prompts.
- Section 14 ends with `All artifacts OK.`

When all four pass, click **Output → Download** in Kaggle to grab the `artifacts/` folder.

> **Common failure**: `img_emb_raw.npy` is empty. The previous version of the notebook had a bug where the raw CLIP embeddings were overwritten by their PCA-reduced version. The current notebook keeps `img_emb_raw` and `img_emb_pca` as separate variables — if you see this in your run, you are looking at a stale notebook. Re-pull the latest version.

Copy the downloaded folder into the repo:

```bash
mkdir -p pet-advisor/artifacts
unzip ~/Downloads/artifacts.zip -d pet-advisor/artifacts
```

---

## 5. Run `dog_finder_training.ipynb` (the Siamese re-ID network)

Same drill: open in Kaggle, GPU T4 x2, **Run All**. ~25 minutes for 10 epochs on the default 8 K triplets.

What is different from a stock Siamese training, and why it matters here:

- **Multi-image positives.** PetFinder gives up to ~10 photos per dog. The triplet sampler picks two distinct photos of the same dog as the (anchor, positive) — that is the raw signal we are after.
- **Augmentations on the anchor and positive only** (horizontal flip, mild colour jitter). Strong enough to teach invariance, conservative enough to preserve breed-distinguishing features. The negative is left untouched so it stays a "real" point in the visual distribution.
- **Semi-hard negative mining.** A naive triplet loss with random negatives plateaus quickly because most triplets are trivial. Inside each batch we replace the random negative with the hardest still-trainable one — the FaceNet trick. Expect Rank-1 to jump 5–15 absolute points compared to the random-negative baseline.
- **Clean train / val split.** PetIDs are partitioned 85 / 15 with a fixed seed. A val pet is never seen during training, so Rank-1 measures actual generalisation.
- **Multi-view gallery aggregation at inference.** When evaluating (and at export time) we average the embeddings of all photos of a dog except the one used as the query. This is exactly what production does — it ought to match how the metric is computed.

**Acceptance checks before downloading:**

- Loss curve monotonically (or near-monotonically) decreasing.
- Rank-1 ≥ 0.55 on val. Rank-5 ≥ 0.78. mAP ≥ 0.62. (Lower numbers usually mean a too-small `n_triplets` or a broken split — re-check.)
- Section 10 ends with `All dog-finder artifacts OK.`

Outputs to copy into `pet-advisor/artifacts/`:

- `dog_encoder.pt`           — projector weights, the backend auto-detects this file.
- `dog_emb_reid.npy`         — multi-view 128-d embeddings of every dog. Preferred path in `dog_finder.py`.
- `dog_ids_reid.npy`         — PetID list aligned with the above.

> The backend's loading order in `dog_finder.py` is documented in its docstring. Short version: if `dog_emb_reid.npy` exists, it is used directly; otherwise the backend re-projects `img_emb_raw.npy` through `dog_encoder.pt`; otherwise CLIP zero-shot.

---

## 6. The agent layer — what was changed and why

Three product asks shaped the rewrite, all already implemented in `pet-advisor/backend/`:

### Tâche 1 — Fully agentic chatbot

`backend/agent.py` now contains *no* hardcoded "after N exchanges, search" or "after N rejections, ask why" rules. Those have been deleted from the system prompt and replaced with **decision principles** ("decide each step whether another short question would dramatically narrow the space, or whether you have enough signal to suggest something genuinely helpful"). The LLM controls the loop through its tool-calling.

Why this matters: the previous prompt over-prescribed the path and the bot felt scripted. With the rules removed and the principles surfaced, the same model picks better moments to ask vs. recommend, and the behaviour gracefully scales when you swap in a stronger LLM.

### Tâche 2 — Persistent rejection feedback

`backend/feedback.py` adds a separate sqlite store (`artifacts/feedback.db`) keyed by **user email**. Every time the agent calls `handle_rejection`, the rejection is persisted with structured attributes (breed, size, animal_type, age_months) on top of the human-readable reason.

This signal feeds back in two places:

1. **At search time** — `retrieval.search_pets` accepts a `UserPreferenceProfile` and applies a soft penalty (`FEEDBACK_PENALTY = 0.05`) for each "systematic avoid" that matches a candidate. A pet with a perfect cosine score that hits one matching avoid stays competitive; one that hits multiple avoids falls out of top-k. The threshold for "systematic" is currently 2 occurrences (`feedback.SYSTEMATIC_THRESHOLD`) — tune it as you collect data.
2. **In the system prompt** — `feedback.summarize_for_llm` produces a one-paragraph natural-language memory ("User has consistently rejected: large dogs; reasons: too much shedding") that is appended to the system prompt. The LLM sees the user's history *across sessions* and adapts its phrasing.

You can introspect what the system has learned about a user with `GET /feedback/{user_email}`.

### Tâche 3 — LangGraph

The agent loop is now a typed `StateGraph` with two nodes (`agent`, `tools`) and a conditional edge that lets the LLM decide whether to keep calling tools or to emit a final reply.

**Why LangGraph and not the alternatives**, given the answers you gave during planning:

- **LangGraph** — explicit state, durable checkpointing if/when you want it, native tool routing, very thin wrapper around the OpenAI-compatible call. Ideal for "one agent, many tools, multi-turn, persistent feedback memory". Picked.
- **LlamaIndex Workflows** — event-bus model, lighter, but less mature for the conditional-router patterns we use here.
- **Anthropic Agent SDK / Claude Agent SDK** — excellent ergonomics but assumes the Anthropic API. We are staying local.
- **AutoGen** — multi-agent by default. Overkill for one advisor and adds conversational latency that hurts the chat UX.

If/when you want true durable state across backend restarts (so a session survives a redeploy), wrap `self.graph = self._build_graph()` in `agent.py` with `SqliteSaver` from `langgraph.checkpoint.sqlite`.

---

## 7. Run the application

From the repository root:

```bash
cd "adoption system/pet-advisor"
source .venv/bin/activate
ollama serve &                         # if not already running
uvicorn backend.main:app --reload --port 8000
```

Open <http://localhost:8000>. You should see the chat UI.

**End-to-end smoke test:**

1. Sign in with any name + a real-looking email. The email is the key for cross-session memory, so use the same one for repeat tests.
2. Type `I want a small dog for an apartment, I work long hours`. The bot should ask one focused follow-up before calling `search_pets`.
3. Reject the first suggestion with `Too big`. Watch the backend logs: you should see a `handle_rejection` tool call, and the next search result should not include that pet.
4. Disconnect, reconnect with the same email, start a new chat. Open `http://localhost:8000/feedback/<your-email>` in another tab — you should see the rejected pet, breed, and the LLM-readable summary.

For the lost dog flow:

1. Click the upload icon, drop a clear photo of a dog (ideally one you also see in the shelter list).
2. Type `I lost my dog`. The bot should immediately call `find_lost_dog` and either return a match with a confidence label or honestly say "no dog matching this photo was found".

---

## 8. Verification checklist (per layer)

| Layer       | Check                                                                                  | How                                                       |
|-------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------|
| Notebooks   | All cells in §14 (cosine) and §10 (dog finder) print `All artifacts OK.`               | Re-run the verification cell.                             |
| Artifacts   | `pet-advisor/artifacts/` contains the 8 cosine files + the 3 dog-finder files.         | `ls -lh pet-advisor/artifacts`                             |
| Backend boot| `[retrieval] Loaded N pets. Ready.` and `[dog_finder] Dog index ready: N dogs.`         | `uvicorn` startup logs.                                   |
| Agent       | The agent never asks two questions in one message.                                     | Manual chat.                                              |
| Feedback    | Same email, two sessions: the second session no longer suggests the rejected pet.      | Manual chat.                                              |
| Persistence | `GET /feedback/<email>` returns the running list of rejections.                         | `curl http://localhost:8000/feedback/your@email.com`      |
| Lost dog    | Uploading the *same* image as a known shelter dog returns `confidence ≥ medium`.       | Manual upload.                                            |

---

## 9. Open assumptions / things to revisit

These are the points where I had to choose without explicit confirmation. Each is noted at the call site too.

- **`FEEDBACK_PENALTY = 0.05`** in `retrieval.py` is a heuristic. Once you have ~50 rejections in production, plot the cosine-score distribution before/after the penalty and tune.
- **`SYSTEMATIC_THRESHOLD = 2`** in `feedback.py`: a breed must be rejected twice to be considered an "avoid" signal. Lower it to 1 if you want the system to react faster but expect more false-positive avoids.
- **`MAX_ITER = 6`** in `agent.py` caps the agent's tool-calling depth per turn. Llama3.2 occasionally loops; if you switch to a stronger model you can lower this.
- The frontend keeps `/auth/login` payload format unchanged, but the email is now **mandatory**. Older clients that posted `email: ""` will get a 400 — wire a friendly error message in the UI if you support those.
- The `dog_finder.py` siamese projector architecture is hardcoded to match the notebook (256 hidden, 128 out). If you tweak the notebook architecture, mirror the change in `_load_siamese`.

---

## 10. Troubleshooting quick reference

| Symptom                                                              | Likely cause / fix                                                            |
|----------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `[retrieval] WARNING — missing artifacts: [...]`                     | Re-export from `cosinesimilarity2.ipynb`, copy into `pet-advisor/artifacts/`. |
| `[dog_finder] WARNING — neither re-ID nor raw embeddings found.`     | Same — also check that `df_original.parquet` is present.                       |
| Agent answers in plain text but never calls tools                    | Ollama tool-calling glitch. Restart `ollama serve`, or move to `llama3.1:8b`. |
| WebSocket closes immediately after login                             | Backend not running, or the artifacts failed to load — check `uvicorn` logs. |
| "Connection refused" from agent → Ollama                              | `ollama serve` is not running, or firewall blocks `localhost:11434`.          |
| Lost-dog matches always low confidence                               | Re-train with more `n_triplets` or check that the multi-view export ran.       |
| Same pet keeps being suggested despite repeated rejections           | Confirm the email passed to `/auth/login` is consistent across sessions.       |
