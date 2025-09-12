from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import lancedb

# Используем быструю реализацию BM25S при наличии
try:
	from bm25s import BM25 as BM25S  # type: ignore
	_HAS_BM25S = True
except ImportError:  # pragma: no cover - опциональная зависимость
	BM25S = None  # type: ignore
	_HAS_BM25S = False

from .config import settings


@dataclass
class LanceRecord:
	id: str
	text: str
	vector: List[float]
	filename: str
	page: int
	chunk_id: str
	source_path: str


class LanceDBStore:
	def __init__(self, db_path: str | None = None, table_name: str | None = None):
		self.db_path = db_path or settings.lancedb_path
		self.table_name = table_name or settings.lancedb_table
		Path(self.db_path).mkdir(parents=True, exist_ok=True)
		self.db = lancedb.connect(self.db_path)
		self.table = self.db.open_table(self.table_name) if self.table_name in self.db.table_names() else None

	def _ensure_table(self, sample_rows: List[dict]):
		if self.table is None:
			self.table = self.db.create_table(self.table_name, data=sample_rows, mode="overwrite")
			try:
				self.table.create_index(num_partitions=1, vector_column="vector", metric="cosine")
			except Exception:
				pass

	def upsert(self, records: List[LanceRecord]) -> None:
		if not records:
			return
		payload = [
			{
				"id": r.id,
				"text": r.text,
				"vector": r.vector,
				"filename": r.filename,
				"page": int(r.page),
				"chunk_id": r.chunk_id,
				"source_path": r.source_path,
			}
			for r in records
		]
		if self.table is None:
			self._ensure_table(sample_rows=payload[:10])
		# Purge existing rows for the same filenames to avoid stale duplicates
		try:
			fns = sorted({p.get("filename") for p in payload if p.get("filename")})
			for fn in fns:
				self.table.delete(f"filename == '{fn}'")
		except Exception:
			pass
		ids = [p["id"] for p in payload]
		if len(ids) == 1:
			self.table.delete(f"id == '{ids[0]}'")
		elif len(ids) > 1:
			self.table.delete(f"id in {tuple(ids)}")
		self.table.add(payload)

	# Предкомпилированный регэксп для токенизации (unicode-слова)
	_TOKENIZER_RE = re.compile(r"\w+", re.UNICODE)

	def _tokenize(self, text: str) -> List[str]:
		"""Токенизация текста: unicode-слова, нижний регистр.

		Без внешних зависимостей и скачиваний; подходит для русского/английского.
		"""
		return [m.group(0) for m in self._TOKENIZER_RE.finditer(text.lower())]

	def _bm25_scores(self, query: str, candidates: List[Dict]) -> Dict[str, float]:
		"""Вычислить BM25-оценки только для переданных кандидатов.

		Исключает чтение всей таблицы и ускоряет работу при больших базах.
		Возвращает словарь id -> нормализованный скор [0..1].
		"""
		if not _HAS_BM25S or not candidates:
			return {}
		ids: List[str] = []
		texts: List[str] = []
		for h in candidates:
			hid = h.get("id")
			txt = h.get("text")
			if hid is None or txt is None:
				continue
			ids.append(str(hid))
			texts.append(str(txt))
		if not ids or not texts:
			return {}
		tokenized_corpus = [self._tokenize(t) for t in texts]
		if not tokenized_corpus:
			return {}
		q_tokens = self._tokenize(query)
		if not q_tokens:
			return {i: 0.0 for i in ids}
		# Единообразная инициализация BM25S: BM25S() + index(corpus)
		try:
			bm25 = BM25S()  # type: ignore
			bm25.index(tokenized_corpus)  # type: ignore
		except Exception:
			return {i: 0.0 for i in ids}
		# Получаем баллы
		scores_list: List[float]
		if hasattr(bm25, "get_scores"):
			try:
				scores_list = list(bm25.get_scores(q_tokens=q_tokens))  # type: ignore
			except TypeError:
				# Некоторые версии принимают позиционный параметр
				scores_list = list(bm25.get_scores(q_tokens))  # type: ignore
		elif hasattr(bm25, "retrieve"):
			try:
				res = bm25.retrieve(q_tokens, k=len(ids))  # type: ignore
				# Ожидаемые варианты: (indices, scores) или список пар
				indices: List[int] = []
				vals: List[float] = []
				if isinstance(res, tuple) and len(res) >= 2:
					indices, vals = list(res[0]), list(res[1])
				elif isinstance(res, list) and res and isinstance(res[0], (list, tuple)):
					indices = [int(x[0]) for x in res]
					vals = [float(x[1]) for x in res]
				else:
					indices, vals = [], []
				scores_list = [0.0] * len(ids)
				for idx, val in zip(indices, vals):
					if 0 <= idx < len(scores_list):
						scores_list[idx] = float(val)
			except Exception:
				scores_list = [0.0] * len(ids)
		elif hasattr(bm25, "query"):
			try:
				res = bm25.query(q_tokens, k=len(ids))  # type: ignore
				indices: List[int] = []
				vals: List[float] = []
				if isinstance(res, tuple) and len(res) >= 2:
					indices, vals = list(res[0]), list(res[1])
				elif isinstance(res, dict) and "scores" in res and "indices" in res:
					indices = list(res["indices"])  # type: ignore
					vals = list(res["scores"])  # type: ignore
				scores_list = [0.0] * len(ids)
				for idx, val in zip(indices, vals):
					if 0 <= idx < len(scores_list):
						scores_list[idx] = float(val)
			except Exception:
				scores_list = [0.0] * len(ids)
		else:
			scores_list = [0.0] * len(ids)
		# Мин-макс нормализация
		max_s = float(np.max(scores_list)) if scores_list else 1.0
		if max_s == 0:
			return {ids[i]: 0.0 for i in range(len(ids))}
		return {ids[i]: float(scores_list[i] / max_s) for i in range(len(ids))}

	def vector_search(self, query_vector: List[float], top_k: int) -> List[Dict]:
		if self.table is None:
			return []
		try:
			res = (
				self.table.search(query_vector)
				.select(["id", "text", "vector", "filename", "page", "chunk_id", "source_path"])
				.metric("cosine")
				.limit(top_k)
				.to_list()
			)
			return res or []
		except Exception:
			return []

	@staticmethod
	def _cosine_sim(query: np.ndarray, vec: np.ndarray) -> float:
		if query.shape[0] != vec.shape[0]:
			min_dim = min(query.shape[0], vec.shape[0])
			query = query[:min_dim]
			vec = vec[:min_dim]
		return float(np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec) + 1e-12))

	def hybrid_search(self, query: str, query_vector: List[float], top_k: int, weights: Tuple[float, float] = (0.8, 0.2)) -> List[Tuple[Dict, float]]:
		vector_hits = self.vector_search(query_vector, top_k=top_k)
		if not vector_hits:
			return []
		bm25_scores = self._bm25_scores(query, candidates=vector_hits)
		q = np.array(query_vector, dtype=np.float32)
		vec_scores: Dict[str, float] = {}
		for h in vector_hits:
			vec = h.get("vector")
			if vec is None:
				continue
			v = np.array(vec, dtype=np.float32)
			vec_scores[h["id"]] = self._cosine_sim(q, v)
		if vec_scores:
			min_vs = min(vec_scores.values())
			max_vs = max(vec_scores.values())
			range_vs = (max_vs - min_vs) or 1.0
			vec_scores = {k: (v - min_vs) / range_vs for k, v in vec_scores.items()}
		alpha, beta = weights
		combined: List[Tuple[Dict, float]] = []
		for h in vector_hits:
			vid = h.get("id")
			if vid is None:
				continue
			vs = vec_scores.get(vid, 0.0)
			bs = bm25_scores.get(vid, 0.0)
			s = alpha * vs + beta * bs
			combined.append((h, s))
		combined.sort(key=lambda x: x[1], reverse=True)
		return combined[:top_k]
