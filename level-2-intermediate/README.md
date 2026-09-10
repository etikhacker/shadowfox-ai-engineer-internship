# Grounded — Document Q&A (ShadowFox AI Engineer Internship, Intermediate Level)

A document-based question-answering assistant: upload PDF/TXT/MD files,
ask questions, and get answers grounded in retrieved chunks — not the
model's general knowledge.

## Architecture

```
frontend/index.html          FastAPI backend (backend/main.py)
──────────────────────       ────────────────────────────────────
1. Upload file        ───►   extract text → chunk → embed (Gemini) → FAISS
2. Ask question        ───►  embed question → FAISS search (top-k) →
                              grounded prompt → Gemini generateContent
                       ◄───   answer + retrieved source chunks
```

Backend: **FastAPI** for the API, **FAISS** (`IndexFlatIP` wrapped in
`IndexIDMap` for deletion support) for vector search, **Gemini
`text-embedding-004`** for embeddings, and **Gemini `gemini-2.0-flash`**
for the grounded answer. Frontend: a single static HTML/JS page — no
build step.

## Why a backend this time

The beginner-level app called Gemini directly from the browser, which
means the API key lives in the page. That's fine for a single prompt-in,
text-out feature, but this level needs to hold document state (chunks +
vectors) across requests and run a proper retrieval step before
generation — that's server-side responsibility. The `GEMINI_API_KEY`
now lives only in `backend/.env` and is never sent to the browser.

## Pipeline in detail

### 1. Text extraction
`.pdf` files are parsed page-by-page with `pypdf`; `.txt`/`.md` are read
directly. If extraction yields no text (e.g. a scanned, image-only PDF),
the upload is rejected with a clear error instead of silently indexing
nothing.

### 2. Chunking strategy
Sentences are grouped greedily into ~**900-character** chunks with a
**150-character** overlap (`chunk_text` in `main.py`):

- 900 characters (~150–200 words) is small enough to keep each chunk
  topically focused — so a similarity search doesn't retrieve a chunk
  where the relevant sentence is buried among unrelated ones — while
  staying large enough to preserve surrounding context for the model to
  reason with.
- Splitting on sentence boundaries (not a hard character cut) avoids
  slicing a sentence in half.
- A ~15–17% overlap means a fact stated right at a chunk boundary is
  still fully readable from at least one of the two neighboring chunks,
  instead of being split across both.

### 3. Embeddings + indexing
Each chunk is embedded with Gemini's `text-embedding-004`
(`batchEmbedContents`, one request per document for efficiency),
L2-normalized, and added to a FAISS `IndexFlatIP` — with normalized
vectors, inner product search is equivalent to cosine similarity.
Vectors are ID-mapped so a document can be removed later without
rebuilding the whole index.

### 4. Retrieval
A question is embedded the same way, then FAISS returns the **top 4**
closest chunks (`TOP_K` in `main.py`) by cosine similarity, optionally
restricted to specific `document_ids` if the caller wants to scope the
search.

### 5. Grounded generation
The retrieved chunks are numbered (`[Source 1] filename, chunk 2 …`) and
passed to Gemini inside a fixed system instruction that:
- restricts the model to answering **only** from the given excerpts,
- requires it to say plainly when the documents don't cover the
  question, instead of falling back on general knowledge,
- asks it to reference sources inline (`[Source 1]`) where relevant.

This is what keeps the answer "grounded" — the model never sees the raw
question without the retrieved context attached, and it's explicitly
told not to answer from anything else.

### 6. Reducing hallucination risk
Three things work together here:
- **Retrieval-first**: the model only ever receives the top-matching
  chunks, not the whole document, so there's less room for it to drift.
- **Explicit refusal instruction**: if similarity search returns nothing
  relevant, the backend short-circuits with "the documents don't seem to
  contain anything relevant" and skips generation entirely.
- **Low temperature** (`0.2`) for the generation call, since this is a
  factual-retrieval task, not creative writing.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents/upload` | Upload a file (multipart), returns `document_id` + chunk count |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/documents/{id}` | Remove a document and its vectors |
| `POST` | `/reset` | Clear all documents and the index |
| `POST` | `/ask` | `{ "question": "...", "document_ids": [optional] }` → answer + sources |
| `GET` | `/health` | Basic liveness/status check |

All error paths return a JSON `{"detail": "..."}` with an appropriate
HTTP status (`400` for bad input, `404` for unknown document, `502` for
a failed upstream Gemini call) — the frontend surfaces `detail` directly
instead of a generic failure message.

## Running it

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # then paste your Gemini API key into .env
uvicorn main:app --reload   # runs on http://localhost:8000
```

Then open `frontend/index.html` in a browser (the backend URL field
defaults to `http://localhost:8000`).

## Validation & error handling covered

- Empty file, unsupported file type, or unreadable/scanned PDF → `400`
  with a specific message.
- Empty question, or asking with zero documents uploaded → `400`.
- Unknown `document_ids` passed to `/ask` → `400`.
- Missing server-side API key → `500` with a clear message instead of a
  confusing downstream failure.
- Failed Gemini calls (embedding or generation) → `502` with the
  upstream error message truncated and surfaced to the frontend.
- Frontend never uses `alert()` for these cases (upload failures are the
  one exception, kept as a lightweight `alert` since they interrupt an
  in-progress action) and disables the Ask button with a "Thinking…"
  state while a request is in flight.

## Week 7 — LLM judgement endpoint

This project now includes a narrow, non-chat endpoint: `POST /triage`. It classifies one support message into a closed set of categories and routes it to a suggested team.

### Job card

- Input: `{ "text": "string, 1-2000 characters" }`
- Categories: `billing`, `bug`, `feature`, `other`
- Urgency: `low`, `normal`, `high`
- Suggested team: `support`, `engineering`, `product`, `other`
- Confidence: numeric value between `0.0` and `1.0`
- The service returns `other` with low confidence when the message is ambiguous.

### Output contract

```json
{
  "category": "billing",
  "urgency": "normal",
  "suggested_team": "support",
  "confidence": 0.98,
  "reason": "The message describes a duplicate charge.",
  "meta": {
    "provider": "openrouter",
    "repaired": false
  }
}
```

### Reliability controls

The model output is parsed and validated with Pydantic before it is returned. Invalid JSON receives one bounded repair attempt. Provider calls use an explicit 8-second timeout and at most two retries, and retries are limited to timeouts, HTTP 429, and HTTP 5xx responses. Every request appends a JSONL cost record with the provider, model, repair flag, and estimated cost. Set `LLM_ENABLED=0` to activate the kill switch. Set `LLM_STUB_MODE=1` to run locally without spending provider quota.

The provider lane is OpenRouter. Configure `OPENROUTER_API_KEY` in a local `.env` file, keep `.env` untracked, and use made-up test messages because free endpoints may use prompts for training.

### Run locally

```bash
cd level-2-intermediate/backend
python3 -m pip install -r requirements.txt
cp .env.example .env
# Set OPENROUTER_API_KEY and set LLM_STUB_MODE=0 for a real provider call.
uvicorn main:app --reload
```

To run the local no-cost stub:

```bash
curl -X POST http://localhost:8000/triage \
  -H 'Content-Type: application/json' \
  -d '{"text":"I was charged twice for my subscription."}'
```

To run the real OpenRouter path, set `LLM_STUB_MODE=0` and use the same curl request. Never commit the API key.

### Tests and evaluation

Run the eight required test cases with:

```bash
python3 -m unittest discover -s tests -v
```

Run the deterministic eight-case local evaluation with:

```bash
python3 run_eval.py
```

The current stub evaluation score is **8/8 (1.0)**. This score checks the fixed examples without spending provider quota; a real-provider evaluation should be run separately with synthetic messages after the provider key is configured.
