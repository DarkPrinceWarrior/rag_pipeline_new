from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import httpx

from .config import settings
from .doc_ingest import ingest_pdf_to_chunks
from .embedding import EmbeddingModel
from .reranker import Reranker
from .store import LanceDBStore, LanceRecord
from .utils import get_tokenizer, sliding_token_windows


@dataclass
class Citation:
	serial: int
	filename: str
	page: int
	chunk_id: str
	start: int
	end: int


class RAGPipeline:
	def __init__(self):
		self.store = LanceDBStore()
		self.embedder = EmbeddingModel()
		self.reranker = Reranker()

	def ingest_pdf(self, pdf_path: str) -> int:
		chunks = ingest_pdf_to_chunks(pdf_path)
		vectors = self.embedder.embed([c["text"] for c in chunks])
		records = []
		for i, c in enumerate(chunks):
			records.append(
				LanceRecord(
					id=c["chunk_id"],
					text=c["text"],
					vector=vectors[i],
					filename=c["filename"],
					page=int(c["page"]),
					chunk_id=c["chunk_id"],
					source_path=c["source_path"],
				)
			)
		self.store.upsert(records)
		return len(records)

	def _assemble_context(
		self,
		query: str,
		candidates: List[Tuple[Dict, float]],
		max_context_tokens: int = 3500,
	) -> Tuple[str, List[Citation]]:
		# Prepare reranker inputs
		passages_with_meta: List[Tuple[str, dict]] = []
		for hit, _ in candidates:
			meta = {
				"filename": hit["filename"],
				"page": int(hit["page"]),
				"chunk_id": hit["chunk_id"],
			}
			passages_with_meta.append((hit["text"], meta))
		reranked = self.reranker.rerank(query, passages_with_meta, top_n=15)
		# Assemble with headers and token budget
		header_template = "S#{serial} — {filename}, p.{page}: "
		enc = get_tokenizer()
		budget = max_context_tokens
		context_parts: List[str] = []
		citations: List[Citation] = []
		serial = 0
		for text, meta, score in reranked:
			serial += 1
			header = header_template.format(serial=serial, filename=meta["filename"], page=meta["page"])
			segment = header + text
			seg_tokens = len(enc.encode(segment))
			if seg_tokens > budget:
				# trim the text to fit
				max_text_tokens = max(0, budget - len(enc.encode(header)))
				if max_text_tokens <= 0:
					break
				trimmed_windows = sliding_token_windows(text, max_text_tokens, 0, enc=enc)
				if trimmed_windows:
					segment = header + trimmed_windows[0][2]
					seg_tokens = len(enc.encode(segment))
				else:
					break
			if seg_tokens <= budget:
				start_offset = 0
				end_offset = len(text)
				citations.append(
					Citation(
						serial=serial,
						filename=meta["filename"],
						page=meta["page"],
						chunk_id=meta["chunk_id"],
						start=start_offset,
						end=end_offset,
					)
				)
				context_parts.append(segment)
				budget -= seg_tokens
			else:
				break
		context = "\n\n".join(context_parts)
		return context, citations

	def answer(self, query: str, top_k: int = 100) -> Dict:
		start = time.time()
		qv = self.embedder.embed_query(query)
		candidates = self.store.hybrid_search(query, qv, top_k=top_k, weights=(0.8, 0.2))
		context, citations = self._assemble_context(query, candidates, max_context_tokens=3500)
		answer = self._generate_with_openrouter(query, context)
		latency_ms = int((time.time() - start) * 1000)
		return {
			"answer": answer,
			"citations": [c.__dict__ for c in citations],
			"latency_ms": latency_ms,
		}

	def _generate_with_openrouter(self, user_query: str, context: str) -> str:
		prompt = (
			"You are a precise Retrieval-Augmented Generation assistant.\n"
			"Answer strictly using the provided CONTEXT from User Guide.pdf.\n"
			"If the answer is not in the context, say: \"I don't have enough information in the manual to answer.\"\n"
			"Always:\n- show a concise answer first,\n- then bullet key steps,\n- then cite pages as (S# — filename, p.X).\n"
			"Avoid hallucinations; do not invent page numbers.\n\n"
			f"QUESTION:\n{user_query}\n\nCONTEXT:\n{context}"
		)
		headers = {
			"Authorization": f"Bearer {settings.openrouter_api_key}",
			"Content-Type": "application/json",
			"X-Title": "UserGuide-RAG-with-LanceDB",
		}
		body = {
			"model": settings.llm_model_id,
			"temperature": 0.2,
			"top_p": 0.95,
			"max_tokens": 1024,
			"messages": [
				{"role": "system", "content": "You are a helpful, precise assistant."},
				{"role": "user", "content": prompt},
			],
		}
		with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
			resp = client.post(settings.__dict__.get("openrouter_endpoint", "https://openrouter.ai/api/v1/chat/completions"), headers=headers, content=json.dumps(body))
			resp.raise_for_status()
			data = resp.json()
			return data["choices"][0]["message"]["content"].strip()
