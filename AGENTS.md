# Repository Guidelines

## Project Structure & Module Organization
- `app/` - FastAPI app (`main.py`) exposing `POST /ask` and `/healthz`.
- `rag/` - Core pipeline: `pipeline.py`, `embedding.py`, `reranker.py`, `store.py`, `doc_ingest.py`, `config.py`, `utils.py`.
- `scripts/` - CLIs (e.g., `ingest_user_guide.py`).
- `docs/` - Input PDFs (default: `docs/User_Guide.pdf`).
- `lancedb_data/` - Local LanceDB storage (generated, ignored by Git).
- Tests (when added): `tests/` with `test_*.py` files.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv` then `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux).
- Install deps: `pip install -r requirements.txt`.
- Configure env: `copy env.example .env` (Windows) or `cp env.example .env`; set `OPENROUTER_API_KEY`, `HF_TOKEN`.
- Ingest PDF: `python scripts/ingest_user_guide.py`.
- Run API (dev): `uvicorn app.main:app --reload --port 8000`.
- Test (if present): `pytest -q` (target >=80% coverage where practical).

## Coding Style & Naming Conventions
- Python 3.10+; follow PEP 8 with 4-space indentation and type hints.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Keep functions focused; avoid side effects in `rag/` (except controlled I/O in `store.py`).
- Prefer explicit errors over silent failures; handle external services defensively.
- Imports grouped: stdlib, third-party, local (`rag.*`).

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/`, name `test_*.py`.
- Mock external calls (HTTP to OpenRouter, Docling, LanceDB) and GPU/cuda checks.
- Provide minimal fixtures for sample chunks; avoid committing large PDFs.
- Run: `pytest -q`; add `-k` to filter and `-x` to stop early.

## Commit & Pull Request Guidelines
- Commits (recommended): Conventional Commits (e.g., `feat: add hybrid rerank`, `fix: handle docling CLI fallback`).
- PRs should include: clear description, linked issues, how to test (commands/endpoints), and notes on performance or model changes.
- Verify before merging: ingestion works, API runs, tests pass, docs updated, no secrets in Git.

## Security & Configuration Tips
- Do not commit `.env` or `lancedb_data/`. Required vars: `OPENROUTER_API_KEY`, `HF_TOKEN`.
- Tune models and paths via `rag/config.py` or environment; CPU fallback occurs if CUDA is unavailable.

## Architecture Overview (Quick)
Ingest (Docling) -> Embed (SentenceTransformers) -> Store/Retrieve (LanceDB + BM25) -> Rerank (BGE) -> Assemble context -> Generate (OpenRouter).

