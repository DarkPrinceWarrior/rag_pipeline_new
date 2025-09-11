from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

from rag.pipeline import RAGPipeline

app = FastAPI(title="UserGuide-RAG-with-LanceDB")

pipeline = RAGPipeline()


class AskRequest(BaseModel):
	query: str
	top_k: Optional[int] = 100


class AskResponse(BaseModel):
	answer: str
	citations: list
	latency_ms: int


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
	res = pipeline.answer(req.query, top_k=req.top_k or 100)
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
    return FileResponse(str(index_file))
