"""
Grounded — Document Q&A backend (ShadowFox AI Engineer Internship, Intermediate Level)

Pipeline: upload -> extract text -> chunk -> embed (Gemini) -> store in a FAISS
index -> on a question, embed it, retrieve the closest chunks, and generate an
answer that is grounded strictly in those chunks.
"""

import io
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = "models/gemini-embedding-001"
GEN_MODEL = "models/gemini-3.6-flash"
EMBED_DIM = 768

CHUNK_SIZE = 900       # characters per chunk — see README for the reasoning
CHUNK_OVERLAP = 150    # characters carried into the next chunk
TOP_K = 4              # chunks retrieved per question

app = FastAPI(title="Grounded — Document Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev-only; restrict this in a real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- schemas --

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    document_ids: Optional[List[str]] = None  # None = search across all documents
    lang: str = "en"  # "en" or "az" — controls the generated answer's language


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


# ------------------------------------------------------------ vector store --

class VectorStore:
    """Thin wrapper around a FAISS flat index with removable, ID-mapped vectors."""

    def __init__(self, dim: int):
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.meta: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> List[int]:
        ids = np.arange(self._next_id, self._next_id + len(metadatas)).astype("int64")
        self.index.add_with_ids(vectors, ids)
        for i, m in zip(ids, metadatas):
            self.meta[int(i)] = m
        self._next_id += len(metadatas)
        return ids.tolist()

    def remove_document(self, document_id: str) -> None:
        ids_to_remove = [i for i, m in self.meta.items() if m["document_id"] == document_id]
        if ids_to_remove:
            self.index.remove_ids(np.array(ids_to_remove, dtype="int64"))
            for i in ids_to_remove:
                del self.meta[i]

    def search(self, vector: np.ndarray, k: int, allowed_doc_ids: Optional[List[str]] = None):
        if self.index.ntotal == 0:
            return []
        # over-fetch when filtering to a subset of documents, then trim to k
        fetch_k = min(self.index.ntotal, k * 5) if allowed_doc_ids else min(self.index.ntotal, k)
        scores, ids = self.index.search(vector.reshape(1, -1), fetch_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            m = self.meta.get(int(idx))
            if m is None:
                continue
            if allowed_doc_ids and m["document_id"] not in allowed_doc_ids:
                continue
            results.append((m, float(score)))
            if len(results) >= k:
                break
        return results


store = VectorStore(EMBED_DIM)
documents: Dict[str, Dict[str, Any]] = {}   # document_id -> {filename, chunks}


# ---------------------------------------------------- extraction & chunking --

def extract_text(filename: str, raw: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return raw.decode("utf-8", errors="ignore")
    raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are supported.")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Greedily group sentences into ~chunk_size character blocks, carrying the
    trailing ~overlap characters of each chunk into the next one so a fact
    split across a chunk boundary is still retrievable from either side."""
    sentences = split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            tail: List[str] = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s)
            current, current_len = tail, tail_len
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------- Gemini I/O --

def require_api_key() -> None:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")


def embed_texts(texts: List[str]) -> np.ndarray:
    require_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/{EMBED_MODEL}:batchEmbedContents?key={GEMINI_API_KEY}"
    body = {
        "requests": [
            {
                "model": EMBED_MODEL,
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": EMBED_DIM,
            }
            for t in texts
        ]
    }
    resp = requests.post(url, json=body, timeout=60)
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {resp.text[:300]}")
    data = resp.json()
    vectors = np.array([e["values"] for e in data["embeddings"]], dtype="float32")
    faiss.normalize_L2(vectors)  # so inner product == cosine similarity
    return vectors


def generate_answer(question: str, context_blocks: List[str], lang: str = "en") -> str:
    require_api_key()
    context = "\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"
    language_line = (
        "CRITICAL: Write your entire answer in Azerbaijani (Azərbaycan dilində), "
        "even though the source excerpts below are in English or another language. "
        "Translate the relevant facts into Azerbaijani yourself — never answer in the "
        "source excerpts' language. Keep [Source N] markers as-is."
        if lang == "az"
        else "Write your entire answer in English, regardless of the source excerpts' language."
    )
    system_instruction = (
        "You are a document question-answering assistant. Answer strictly using the "
        "provided source excerpts below. If the excerpts do not contain the answer, say "
        "clearly that the uploaded documents don't cover it — never fall back on outside "
        "knowledge to fill the gap. Reference sources inline like [Source 1] where relevant. "
        "Keep the answer concise. " + language_line
    )
    user_prompt = f"Question: {question}\n\nSource excerpts:\n{context}\n\n{language_line}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{GEN_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700},
    }
    resp = requests.post(url, json=body, timeout=60)
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"Generation request failed: {resp.text[:300]}")
    data = resp.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


# -------------------------------------------------------------------- routes --

@app.get("/health")
def health():
    return {"status": "ok", "documents": len(documents), "chunks": store.index.ntotal}


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    return [
        DocumentInfo(document_id=doc_id, filename=info["filename"], chunks=info["chunks"])
        for doc_id, info in documents.items()
    ]


@app.post("/documents/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="The uploaded file has no filename.")

    text = extract_text(file.filename, raw)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No extractable text found — this may be a scanned/image-only PDF.",
        )

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not split this document into chunks.")

    vectors = embed_texts(chunks)
    document_id = str(uuid.uuid4())[:8]
    metadatas = [
        {"document_id": document_id, "filename": file.filename, "chunk_index": i, "text": c}
        for i, c in enumerate(chunks)
    ]
    store.add(vectors, metadatas)
    documents[document_id] = {"filename": file.filename, "chunks": len(chunks)}

    return UploadResponse(document_id=document_id, filename=file.filename, chunks_created=len(chunks))


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    store.remove_document(document_id)
    del documents[document_id]
    return {"status": "deleted", "document_id": document_id}


@app.post("/reset")
def reset():
    global store, documents
    store = VectorStore(EMBED_DIM)
    documents = {}
    return {"status": "reset"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not documents:
        raise HTTPException(status_code=400, detail="Upload at least one document before asking questions.")
    if payload.document_ids:
        unknown = [d for d in payload.document_ids if d not in documents]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown document_ids: {unknown}")

    q_vector = embed_texts([question])[0]
    hits = store.search(q_vector, TOP_K, allowed_doc_ids=payload.document_ids)

    if not hits:
        no_hits_msg = (
            "Yüklənmiş sənədlərdə bu sualla bağlı heç nə tapılmadı."
            if payload.lang == "az"
            else "The uploaded documents don't seem to contain anything relevant to this question."
        )
        return AskResponse(answer=no_hits_msg, sources=[])

    context_blocks = [
        f"[Source {i + 1} — {m['filename']}, chunk {m['chunk_index']}]\n{m['text']}"
        for i, (m, _) in enumerate(hits)
    ]
    answer = generate_answer(question, context_blocks, lang=payload.lang)

    sources = [
        SourceChunk(
            document_id=m["document_id"],
            filename=m["filename"],
            chunk_index=m["chunk_index"],
            text=m["text"],
            score=score,
        )
        for m, score in hits
    ]
    return AskResponse(answer=answer, sources=sources)
