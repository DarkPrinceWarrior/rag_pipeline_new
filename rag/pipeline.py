from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import httpx

from .config import settings
from .doc_ingest import ingest_pdf_to_chunks
from .memory import MemoryManager
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
		self.memory = MemoryManager()

	def ingest_pdf(self, pdf_path: str) -> int:
		chunks = ingest_pdf_to_chunks(pdf_path)
		vectors = self.embedder.embed([c["text"] for c in chunks])
		records = []
		# Пропуск неизменённых документов по хэшу содержимого (если таблица поддерживает поле)
		content_hash = None
		if chunks:
			content_hash = chunks[0].get("content_hash")
			filename = chunks[0].get("filename")
			if content_hash and filename and hasattr(self.store, "has_file_with_hash"):
				try:
					if self.store.has_file_with_hash(str(filename), str(content_hash)):
						return 0
				except Exception:
					pass
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
					section_path=c.get("section_path"),
					element_type=c.get("element_type"),
					content_hash=c.get("content_hash"),
					ocr_used=c.get("ocr_used"),
					doc_id=c.get("doc_id"),
					lang=c.get("lang"),
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
				"section_path": hit.get("section_path"),
				"element_type": hit.get("element_type"),
				"lang": hit.get("lang"),
			}
			passages_with_meta.append((hit["text"], meta))
		reranked = self.reranker.rerank(query, passages_with_meta, top_n=15)
		best_score = float(reranked[0][2]) if reranked else 0.0
		# Assemble with headers and token budget
		header_template = "S#{serial} — {filename}, стр. {page}{section}{etype}{lang}: "
		enc = get_tokenizer()
		budget = max_context_tokens
		context_parts: List[str] = []
		citations: List[Citation] = []
		serial = 0
		for text, meta, score in reranked:
			serial += 1
			section_suffix = ""
			if "section_path" in meta and meta.get("section_path"):
				section_suffix = f", {meta['section_path']}"
			etype_suffix = ""
			if meta.get("element_type"):
				etype_suffix = f" [{meta['element_type']}]"
			lang_suffix = ""
			if meta.get("lang") and meta.get("lang") != "other":
				lang_suffix = f" ({meta['lang']})"
			header = header_template.format(serial=serial, filename=meta["filename"], page=meta["page"], section=section_suffix, etype=etype_suffix, lang=lang_suffix)
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
		user_ref = user_id or settings.memory_default_user_id
		memories = self.memory.fetch(query, user_ref)
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
				generated = self._generate_with_openrouter_web(query, web_ctx, memories=memories)
				answer = generated
				citations = []
			else:
				answer = self._generate_with_openrouter(query, context, memories=memories)
		else:
			answer = self._generate_with_openrouter(query, context, memories=memories)
		latency_ms = int((time.time() - start) * 1000)
		return {
			"answer": answer,
			"citations": [c.__dict__ for c in citations],
			"latency_ms": latency_ms,
		}

	def is_insufficient_answer(self, answer: str) -> bool:
		"""Heuristic detection that the LLM declared lack of info in the manual.

		Returns True if the answer indicates the manual lacks sufficient info.
		"""
		try:
			text = (answer or "").strip().lower()
			if not text:
				return True
			# Phrase emitted per system prompt when docs lack info
			if "недостаточно информации в руководстве" in text:
				return True
			# Built-in internal fallback text when we had zero context
			if "ответ не найден в документах" in text:
				return True
			return False
		except Exception:
			return False

	def answer_internal(self, query: str, top_k: int = 5, user_id: str | None = None) -> Dict:
		start = time.time()
		user_ref = user_id or settings.memory_default_user_id
		memories = self.memory.fetch(query, user_ref)
		qv = self.embedder.embed_query(query)
		candidates = self.store.hybrid_search(query, qv, top_k=top_k, weights=(0.8, 0.2))
		context, citations, best_score, context_tokens = self._assemble_context(query, candidates, max_context_tokens=3500)
		if (not candidates) or (not context) or context_tokens <= 0:
			latency_ms = int((time.time() - start) * 1000)
			telemetry = {
				"selected_mode": "internal",
				"query": query,
				"top_k": top_k,
				"num_candidates": len(candidates),
				"best_score": float(best_score) if isinstance(best_score, (int, float)) else 0.0,
				"context_tokens": int(context_tokens),
				"used_memories": len(memories),
			}
			fallback_answer = "??? ?? ?????? ? ???????."
			self.memory.store_exchange(
				query,
				fallback_answer,
				user_ref,
				metadata={"mode": "internal", "reason": "insufficient_context"},
			)
			return {
				"answer": fallback_answer,
				"citations": [],
				"latency_ms": latency_ms,
				"_telemetry": telemetry,
			}
		answer = self._generate_with_openrouter(query, context, memories=memories)
		latency_ms = int((time.time() - start) * 1000)
		telemetry = {
			"selected_mode": "internal",
			"query": query,
			"top_k": top_k,
			"num_candidates": len(candidates),
			"best_score": float(best_score) if isinstance(best_score, (int, float)) else 0.0,
			"context_tokens": int(context_tokens),
			"num_citations": len(citations),
			"used_memories": len(memories),
		}
		self.memory.store_exchange(query, answer, user_ref, metadata={"mode": "internal"})
		return {
			"answer": answer,
			"citations": [c.__dict__ for c in citations],
			"latency_ms": latency_ms,
			"_telemetry": telemetry,
		}

	def answer_web(self, query: str, user_id: str | None = None) -> Dict:
		start = time.time()
		user_ref = user_id or settings.memory_default_user_id
		memories = self.memory.fetch(query, user_ref)
		web_ctx, web_results = build_web_context(
			query,
			max_results=settings.web_search_max_results,
			fetch_top_n=settings.web_fetch_top_n,
			max_tokens=settings.web_context_max_tokens,
		)
		if not web_ctx:
			latency_ms = int((time.time() - start) * 1000)
			telemetry = {
				"selected_mode": "web",
				"query": query,
				"num_web_results": len(web_results),
				"used_memories": len(memories),
			}
			fallback_answer = "??? ?? ?????? ? ?????."
			self.memory.store_exchange(
				query,
				fallback_answer,
				user_ref,
				metadata={"mode": "web", "reason": "no_web_results"},
			)
			return {
				"answer": fallback_answer,
				"citations": [],
				"latency_ms": latency_ms,
				"_telemetry": telemetry,
			}
		generated = self._generate_with_openrouter_web(query, web_ctx, memories=memories)
		answer = generated
		latency_ms = int((time.time() - start) * 1000)
		telemetry = {
			"selected_mode": "web",
			"query": query,
			"num_web_results": len(web_results),
			"web_result_urls": [r.url for r in web_results],
			"used_memories": len(memories),
		}
		self.memory.store_exchange(query, answer, user_ref, metadata={"mode": "web"})
		return {
			"answer": answer,
			"citations": [],
			"latency_ms": latency_ms,
			"_telemetry": telemetry,
		}

	def _generate_with_openrouter(self, user_query: str, context: str, memories: List[str] | None = None) -> str:
		memory_section = ""
		if memories:
			formatted = MemoryManager.to_prompt_section(memories)
			if formatted:
				memory_section = f"  <memories>\n{formatted}\n  </memories>\n"
		prompt = (
			"<prompt>\n"
			"  <persona>\n"
			"    Ты — ведущий инженер-аналитик в нефтегазовой отрасли.\n"
			"  </persona>\n"
			"  <task>\n"
			"    Ты — точный ассистент Retrieval‑Augmented Generation. Используй ТОЛЬКО раздел <context>.\n"
			"    Если ответа нет в контексте, явно скажи: \"У меня недостаточно информации в руководстве, чтобы ответить.\"\n"
			"  </task>\n"
			"  <constraints>\n"
			"    - Сначала краткий ответ, затем ключевые шаги пунктами.\n"
			"    - Не указывай страницы, номера S# и ссылки на локальные документы.\n"
			"    - Формулы: LaTeX (inline $...$, block $$...$$), без кастомных обёрток.\n"
			"    - Ответ только на русском языке.\n"
			"  </constraints>\n"
			f"{memory_section}"
			f"  <question>{user_query}</question>\n"
			f"  <context>{context}</context>\n"
			"</prompt>"
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

	def _generate_with_openrouter_web(self, user_query: str, web_context: str, memories: List[str] | None = None) -> str:
		memory_section = ""
		if memories:
			formatted = MemoryManager.to_prompt_section(memories)
			if formatted:
				memory_section = f"  <memories>\n{formatted}\n  </memories>\n"
		prompt = (
			"<prompt>\n"
			"  <persona>\n"
			"    Ты — ведущий инженер-аналитик в нефтегазовой отрасли.\n"
			"  </persona>\n"
			"  <task>\n"
			"    PDF‑мануал не содержит ответа. Используй ТОЛЬКО раздел <web_context>.\n"
			"  </task>\n"
			"  <constraints>\n"
			"    - Начни с краткого ответа, затем ключевые шаги/факты пунктами.\n"
			"    - Не выдумывай источники.\n"
			"    - Формулы: LaTeX (inline $...$, block $$...$$), без кастомных обёрток.\n"
			"    - Ответ только на русском языке.\n"
			"  </constraints>\n"
			f"{memory_section}"
			f"  <question>{user_query}</question>\n"
			f"  <web_context>{web_context}</web_context>\n"
			"</prompt>"
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
