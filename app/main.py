from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
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
	uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
