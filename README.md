# UserGuide-RAG-with-LanceDB

Retrieval-Augmented Generation over `docs/User_Guide.pdf` using:
- Docling for PDF parsing (only Docling is used for document processing)
- SentenceTransformers `google/embeddinggemma-300m` for embeddings
- LanceDB for vector storage and hybrid search (vector + BM25 via bm25s)
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
  - Optional: tune paths and parameters (see Configuration). Key options:
    - Ingestion flags: `PREFER_DOCLING_API`, `ENABLE_CLI_FALLBACK`, `ENABLE_OCR`, `PRESERVE_CASE`, `STRIP_HEADERS_FOOTERS`, `MERGE_HYPHENATION`, `TABLE_MODE`, `MAX_PARALLEL_PAGES`, `ALLOWED_EXTS`
    - Rerank meta weights: `RERANK_BONUS_HEADING`, `RERANK_BONUS_TABLE`, `RERANK_BONUS_CODE`, `RERANK_BONUS_LIST`, `RERANK_BONUS_MATH`, `RERANK_SECTION_DEPTH_PENALTY`, `RERANK_MAX_META_BONUS`

4) Ingest the manual into LanceDB
```bat
python scripts/ingest_user_guide.py
```

4.1) Ingest any file or directory (multi‑format)
```bat
python scripts/ingest_path.py <path-to-file-or-dir>
```

5) Run the API (serves Web UI)
```bat
uvicorn app.main:app --port 8000
```

6) Open Web UI
```
http://localhost:8000/
```
The chat UI supports:
- Markdown rendering with code highlighting
- Clickable citations opening the PDF page
- Russian UI labels and editing of your messages
- Modes: internal (manual only) and web search only (checkbox in settings)
- Top‑K slider with persistent value (applies to internal mode only)

7) Ask a question (REST examples)
```bat
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Where do I manage Favorites on the Home Page?\",\"top_k\":5,\"web_search\":false}"
```

```bat
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Новости по стандарту API 6A 2024\",\"web_search\":true}"
```

## Project layout
```
app/            FastAPI app (serves Web UI at / and static at /assets, /docs)
  └─ static/    Chat UI (index.html, app.js, styles.css)
rag/            Core RAG pipeline modules
scripts/        CLI helpers (ingestion scripts)
docs/           Input PDF(s) (served at /docs for citations)
```

## Configuration (.env)
Required:
- `OPENROUTER_API_KEY` — OpenRouter API key
- `HF_TOKEN` — Hugging Face token (for model downloads)

Optional (defaults shown):
- Core paths and models
  - `LANCEDB_PATH=./lancedb_data`
  - `LANCEDB_TABLE=user_guide`
  - `EMBEDDING_MODEL_ID=google/embeddinggemma-300m`
  - `RERANKER_MODEL_ID=BAAI/bge-reranker-v2-m3`
  - `LLM_MODEL_ID=qwen/qwen3-30b-a3b-instruct-2507`
  - `OPENROUTER_ENDPOINT=https://openrouter.ai/api/v1/chat/completions`
  - `DEVICE=cuda` (auto-falls back to `cpu` if CUDA not available)
  - `BATCH_SIZE_EMBED=64`
  - `BATCH_SIZE_RERANK=16`
- Ingestion
  - `CHUNK_SIZE_TOKENS=1000`
  - `CHUNK_OVERLAP_TOKENS=200`
  - `PREFER_DOCLING_API=true`
  - `ENABLE_CLI_FALLBACK=true`
  - `ENABLE_OCR=true` (requires Tesseract installed; otherwise skip)
  - `PRESERVE_CASE=true`
  - `STRIP_HEADERS_FOOTERS=true`
  - `MERGE_HYPHENATION=true`
  - `TABLE_MODE=md` (md|html|csv)
  - `MAX_PARALLEL_PAGES=4`
  - `ALLOWED_EXTS=.pdf,.docx,.pptx,.xlsx,.html,.htm,.txt`
- Rerank meta weights
  - `RERANK_BONUS_HEADING=0.10`
  - `RERANK_BONUS_TABLE=0.06`
  - `RERANK_BONUS_CODE=0.06`
  - `RERANK_BONUS_LIST=0.02`
  - `RERANK_BONUS_MATH=0.05`
  - `RERANK_SECTION_DEPTH_PENALTY=0.015`
  - `RERANK_MAX_META_BONUS=0.18`

Use `env.example` as a reference.

## How it works
- Parsing: documents are parsed with Docling API; CLI is used as fallback; optional OCR for scans.
- Normalization: Unicode NFKC, с гибкими опциями (case, переносы, хедеры/футеры).
- Chunking: семантические блоки (заголовки/абзацы/списки/таблицы/код/формулы) + токенный бюджет/перекрытие.
- Storage: эмбеддинги и метаданные (section_path, element_type, lang, content_hash) сохраняются в LanceDB.
- Retrieval: гибрид (вектор cosine + BM25 via bm25s); итоговое объединение и нормализация скорингов.
- Reranking: BGE v2‑m3 с мета‑бонусами по типу элемента и штрафом за глубину секции.
- Context assembly: заголовки вида `S#{serial} — {filename}, стр. {page}, {section_path} [element_type] (lang):` с токенным бюджетом ~3500.
- Generation: OpenRouter `qwen/qwen3-30b-a3b-instruct-2507` отвечает строго по CONTEXT.

If the answer is not found in the context (internal mode), the system responds with:
```
Ответ не найден в документах.
```

### Web mode
When web mode is selected, only internet search is used. If no results are found, the answer is:

```
Ответ не найден в интернете.
```

Configuration (env):
- `WEB_SEARCH_MAX_RESULTS=5`
- `WEB_FETCH_TOP_N=3`
- `WEB_CONTEXT_MAX_TOKENS=3200`

## API
- Endpoint: `POST /ask`
- Request JSON:
```json
{
  "query": "string",
  "session_id": "chat-session-uuid",
  "top_k": 100,
  "web_search": false
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
