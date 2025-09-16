from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
from pydantic import BaseModel

from rag.pipeline import RAGPipeline
from rag.config import settings

app = FastAPI(title="UserGuide-RAG-with-LanceDB")

# CORS for local dev (Vite) and configurable origins
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()


class AskRequest(BaseModel):
	query: str
	top_k: Optional[int] = None
	web_search: Optional[bool] = False


class AskResponse(BaseModel):
	answer: str
	citations: list
	latency_ms: int


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
	# Docs-first: try internal docs. Only if insufficient and web_search is enabled -> web.
	# Clamp top_k to at least 1, fallback to settings.default_top_k
	tk = int(req.top_k) if (req.top_k and req.top_k >= 1) else int(getattr(settings, "default_top_k", 200))
	res = pipeline.answer_internal(req.query, top_k=tk)
	try:
		ans_text = (res.get("answer") or "").strip()
		insufficient = pipeline.is_insufficient_answer(ans_text)
		if insufficient and req.web_search:
			res = pipeline.answer_web(req.query)
	except Exception:
		pass
	# Developer telemetry (not returned to client)
	try:
		telemetry = res.pop("_telemetry", None)
		if telemetry:
			print(f"[telemetry] {telemetry}")
	except Exception:
		pass
	return AskResponse(**res)


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
	return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=int(os.getenv("PORT", "8000")))

# Serve static chat UI
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")
    # Serve docs (PDF) for clickable citations
    DOCS_DIR = Path(__file__).parent.parent / "docs"
    if DOCS_DIR.exists():
        app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")


@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse(
        """
        <!doctype html>
        <html lang='en'>
          <head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
          <title>UserGuide RAG</title></head>
          <body style="font-family: ui-sans-serif, system-ui; padding: 24px;">
            <h2>Frontend not built</h2>
            <p>Run:</p>
            <pre>cd frontend && npm install && npm run build</pre>
            <p>Then refresh this page.</p>
          </body>
        </html>
        """,
        status_code=200,
    )
