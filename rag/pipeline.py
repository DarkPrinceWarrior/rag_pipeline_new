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
from .web_search import build_web_context


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
	) -> Tuple[str, List[Citation], float, int]:
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
		best_score = float(reranked[0][2]) if reranked else 0.0
		# Assemble with headers and token budget
		header_template = "S#{serial} — {filename}, стр. {page}: "
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
		enc2 = get_tokenizer()
		context_tokens = len(enc2.encode(context)) if context else 0
		return context, citations, best_score, context_tokens

	def answer(self, query: str, top_k: int = 100) -> Dict:
		start = time.time()
		qv = self.embedder.embed_query(query)
		candidates = self.store.hybrid_search(query, qv, top_k=top_k, weights=(0.8, 0.2))
		context, citations, best_score, context_tokens = self._assemble_context(query, candidates, max_context_tokens=3500)
		answer = None
		# Decide if we should fallback to web search
		use_web = False
		if settings.enable_web_fallback:
			insufficient_manual = (not candidates) or (best_score < settings.manual_min_rerank_score) or (context_tokens < settings.manual_min_context_tokens)
			use_web = bool(insufficient_manual)
		if use_web:
			web_ctx, web_results = build_web_context(
				query,
				max_results=settings.web_search_max_results,
				fetch_top_n=settings.web_fetch_top_n,
				max_tokens=settings.web_context_max_tokens,
			)
			if web_ctx:
				generated = self._generate_with_openrouter_web(query, web_ctx)
				notice = "В мануале нет данных — ищем в интернете.\n\n"
				sources_lines = []
				for r in web_results[: settings.web_fetch_top_n]:
					title = r.title or r.url
					sources_lines.append(f"- [{title}]({r.url})")
				sources_md = ("\n\nИсточники:\n" + "\n".join(sources_lines)) if sources_lines else ""
				answer = notice + generated + sources_md
				citations = []
			else:
				answer = self._generate_with_openrouter(query, context)
		else:
			answer = self._generate_with_openrouter(query, context)
		latency_ms = int((time.time() - start) * 1000)
		return {
			"answer": answer,
			"citations": [c.__dict__ for c in citations],
			"latency_ms": latency_ms,
		}

	def _generate_with_openrouter(self, user_query: str, context: str) -> str:
		prompt = (
			"Ты — точный ассистент Retrieval‑Augmented Generation.\n"
			"Отвечай строго, используя только раздел CONTEXT из User_Guide.pdf.\n"
			"Если ответа нет в контексте, скажи: \"У меня недостаточно информации в руководстве, чтобы ответить.\"\n"
			"Всегда сначала дай краткий ответ, затем перечисли ключевые шаги пунктами.\n"
			"Не указывай страницы, номера S# или ссылки на локальные документы.\n"
			"Формулы оформляй в LaTeX: inline $...$, block $$...$$. Не используй кастомные обёртки.\n"
			"Отвечай только на русском языке.\n\n"
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

	def _generate_with_openrouter_web(self, user_query: str, web_context: str) -> str:
		prompt = (
			"Ты — точный ассистент с веб‑дополнением.\n"
			"PDF‑мануал не содержит нужного ответа.\n"
			"Используй ТОЛЬКО раздел WEB CONTEXT ниже для ответа.\n"
			"Всегда:\n- начни с краткого ответа,\n- затем перечисли ключевые шаги/факты пунктами,\n- избегай спекуляций; если что-то неясно — так и скажи.\n"
			"Не выдумывай источники.\n"
			"Формулы оформляй в LaTeX: inline $...$, block $$...$$. Не используй кастомные обёртки.\n"
			"Отвечай только на русском языке.\n\n"
			f"QUESTION:\n{user_query}\n\nWEB CONTEXT:\n{web_context}"
		)
		headers = {
			"Authorization": f"Bearer {settings.openrouter_api_key}",
			"Content-Type": "application/json",
			"X-Title": "UserGuide-RAG-with-LanceDB-WebFallback",
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
