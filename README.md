# UserGuide-RAG-with-LanceDB

Retrieval-Augmented Generation over `docs/User_Guide.pdf` using:
- Docling for PDF parsing (only Docling is used for document processing)
- SentenceTransformers `google/embeddinggemma-300m` for embeddings
- LanceDB for vector storage and hybrid search (vector + BM25)
- FlagEmbedding reranker `BAAI/bge-reranker-v2-m3`
- OpenRouter `qwen/qwen3-30b-a3b-instruct-2507` for final answer generation

## Quickstart (Windows)

1) Create and activate a virtual environment
```bat
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies
```bat
pip install -r requirements.txt
```

3) Configure environment
- Copy the template and fill in your keys
```bat
copy env.example .env
```
- Open `.env` and set:
  - `OPENROUTER_API_KEY=...` (required)
  - `HF_TOKEN=...` (required for gated models on Hugging Face)
  - Optional: tune paths and parameters (see Configuration)

4) Ingest the manual into LanceDB
```bat
python scripts/ingest_user_guide.py
```

5) Run the API (serves Web UI)
```bat
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

6) Open Web UI
```
http://localhost:8000/
```
The chat UI supports:
- Markdown rendering with code highlighting
- Clickable citations opening the PDF page
- Russian UI labels and editing of your messages
- Top‑K slider with persistent value (localStorage)

7) Ask a question (REST example)
```bat
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Where do I manage Favorites on the Home Page?\",\"top_k\":100}"
```

## Project layout
```
app/            FastAPI app (serves Web UI at / and static at /assets, /docs)
  └─ static/    Chat UI (index.html, app.js, styles.css)
rag/            Core RAG pipeline modules
scripts/        CLI helpers (ingestion script)
docs/           Input PDF(s) (served at /docs for citations)
```

## Configuration (.env)
Required:
- `OPENROUTER_API_KEY` — OpenRouter API key
- `HF_TOKEN` — Hugging Face token (for model downloads)

Optional (defaults shown):
- `LANCEDB_PATH=./lancedb_data`
- `LANCEDB_TABLE=user_guide`
- `EMBEDDING_MODEL_ID=google/embeddinggemma-300m`
- `RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3`
- `LLM_MODEL_ID=qwen/qwen3-30b-a3b-instruct-2507`
- `OPENROUTER_ENDPOINT=https://openrouter.ai/api/v1/chat/completions`
- `DEVICE=cuda` (auto-falls back to `cpu` if CUDA not available)
- `BATCH_SIZE_EMBED=64`
- `BATCH_SIZE_RERANK=16`
- `CHUNK_SIZE_TOKENS=1000`
- `CHUNK_OVERLAP_TOKENS=200`

Use `env.example` as a reference.

## How it works
- Parsing: PDF is parsed exclusively with Docling. If Python API is unavailable, a Docling CLI fallback exports Markdown.
- Normalization: text is lowercased and normalized to NFKC.
- Chunking: token-based windows of `CHUNK_SIZE_TOKENS` with `CHUNK_OVERLAP_TOKENS` overlap.
- Storage: chunks are embedded (L2-normalized) and stored in LanceDB with metadata.
- Retrieval: hybrid search = 0.8 (vector cosine) + 0.2 (BM25) with top_k candidates.
- Reranking: BGE reranker v2-m3 selects top 15.
- Context assembly: concatenates with headers `S#{serial} — {filename}, p.{page}:` under ~3500 token budget.
- Generation: OpenRouter `qwen/qwen3-30b-a3b-instruct-2507` answers using only provided CONTEXT.

If the answer is not found in the context, the system responds with:
```
I don't have enough information in the manual to answer.
```

## API
- Endpoint: `POST /ask`
- Request JSON:
```json
{
  "query": "string",
  "top_k": 100
}
```
- Response JSON:
```json
{
  "answer": "string",
  "citations": [
    {
      "serial": 1,
      "filename": "User_Guide.pdf",
      "page": 10,
      "chunk_id": "...",
      "start": 0,
      "end": 1234
    }
  ],
  "latency_ms": 1234
}
```

## Troubleshooting
- 500 Internal Server Error on `/ask`:
  - Ensure you ran ingestion: `python scripts/ingest_user_guide.py`
  - Verify `.env` keys are set and valid.
- Model downloads fail:
  - Make sure `HF_TOKEN` has access to required repos.
- CPU fallback:
  - If CUDA is not available, the pipeline automatically uses CPU.
- Docling import issues:
  - CLI fallback is used automatically. Ensure `docling` is installed and available on PATH.

## Notes
- Input PDF path is `docs/User_Guide.pdf` by default (adjust script or path if you change it).
- LanceDB data persists under `lancedb_data/` (ignored by Git).
- `.gitignore` includes typical Python and project-specific entries.

## Frontend (React + TypeScript)
- Source in `frontend/` (Vite + React 18 + TS)
- Dev: `cd frontend && npm install && npm run dev` → open `http://localhost:5173/`
- Build for FastAPI: `cd frontend && npm install && npm run build` then open `http://localhost:8000/`
- Build output goes to `app/static/` and PDF is served under `/docs`.
