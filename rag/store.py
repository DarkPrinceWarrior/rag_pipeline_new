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
	section_path: str | None = None
	element_type: str | None = None
	content_hash: str | None = None
	ocr_used: bool | None = None
	doc_id: str | None = None
	lang: str | None = None


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
		payload_full = [
			{
				"id": r.id,
				"text": r.text,
				"vector": r.vector,
				"filename": r.filename,
				"page": int(r.page),
				"chunk_id": r.chunk_id,
				"source_path": r.source_path,
				"section_path": r.section_path,
				"element_type": r.element_type,
				"content_hash": r.content_hash,
				"ocr_used": r.ocr_used,
				"doc_id": r.doc_id,
				"lang": r.lang,
			}
			for r in records
		]
		# Минимальный набор полей для обратной совместимости схемы
		payload_min = [
			{
				"id": p["id"],
				"text": p["text"],
				"vector": p["vector"],
				"filename": p["filename"],
				"page": p["page"],
				"chunk_id": p["chunk_id"],
				"source_path": p["source_path"],
			}
			for p in payload_full
		]
		if self.table is None:
			self._ensure_table(sample_rows=payload_full[:10])
		# Purge existing rows for the same filenames to avoid stale duplicates
		try:
			fns = sorted({p.get("filename") for p in payload_full if p.get("filename")})
			for fn in fns:
				self.table.delete(f"filename == '{fn}'")
		except Exception:
			pass
		ids = [p["id"] for p in payload_full]
		if len(ids) == 1:
			self.table.delete(f"id == '{ids[0]}'")
		elif len(ids) > 1:
			self.table.delete(f"id in {tuple(ids)}")
		# Пытаемся добавить с расширенными полями; при несовместимости схемы — минимальный набор
		try:
			self.table.add(payload_full)
		except Exception:
			self.table.add(payload_min)

	def has_file_with_hash(self, filename: str, content_hash: str) -> bool:
		"""Проверить, есть ли в таблице документ с таким именем и хэшем содержимого.

		Использует to_pandas() только для узкого поднабора колонок; при ошибках возвращает False.
		"""
		if self.table is None:
			return False
		try:
			df = self.table.to_pandas()
			if df is None or df.empty:
				return False
			cols = set(df.columns)
			if not {"filename", "content_hash"}.issubset(cols):
				return False
			sub = df[["filename", "content_hash"]].dropna()
			mask = (sub["filename"].astype(str) == str(filename)) & (sub["content_hash"].astype(str) == str(content_hash))
			return bool(mask.any())
		except Exception:
			return False

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
		Возвращает словарь id -> сырая оценка BM25 (без нормализации).
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
		# Возвращаем сырые значения BM25; нормализация выполняется на этапе гибридного склеивания
		return {ids[i]: float(scores_list[i]) for i in range(len(ids))}

	def vector_search(self, query_vector: List[float], top_k: int) -> List[Dict]:
		if self.table is None:
			return []
		try:
			# Пытаемся выбрать расширенные метаданные, при несовместимости схемы — базовый набор
			try:
				res = (
					self.table.search(query_vector)
					.select(["id", "text", "vector", "filename", "page", "chunk_id", "source_path", "section_path", "element_type", "content_hash", "ocr_used"])
					.metric("cosine")
					.limit(top_k)
					.to_list()
				)
			except Exception:
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
		"""Гибридный поиск: используем готовые score из LanceDB и BM25.

		- Для векторной части используем поле score из LanceDB (.metric("cosine")),
		  где меньшие значения лучше (distance). Конвертируем в 
		  «больше — лучше» через отрицание и нормализуем робастно.
		- Для BM25 используем сырые баллы и нормализуем робастно.
		- Смешивание линейное по весам (alpha для векторов, beta для BM25).
		"""
		vector_hits = self.vector_search(query_vector, top_k=top_k)
		if not vector_hits:
			return []
		bm25_scores_raw = self._bm25_scores(query, candidates=vector_hits)

		# Собираем сырые векторные баллы: предпочитаем score из LanceDB; фолбэк — ручной косинус
		ids: List[str] = []
		raw_vec: List[float] = []
		q = np.array(query_vector, dtype=np.float32)
		for h in vector_hits:
			vid = h.get("id")
			if vid is None:
				continue
			ids.append(str(vid))
			if "score" in h and h["score"] is not None:
				# В LanceDB для cosine метрики score — это distance (меньше — лучше)
				try:
					raw_val = -float(h["score"])  # инвертируем знак, чтобы больше было лучше
				except Exception:
					raw_val = 0.0
				raw_vec.append(raw_val)
			else:
				# Фолбэк на случай старого драйвера без score
				vec = h.get("vector")
				if vec is None:
					raw_vec.append(0.0)
				else:
					v = np.array(vec, dtype=np.float32)
					raw_vec.append(self._cosine_sim(q, v))

		# Строим массив сырых BM25 по тем же id
		raw_bm25: List[float] = [float(bm25_scores_raw.get(i, 0.0)) for i in ids]

		def _robust_scale(values: List[float]) -> List[float]:
			"""Робастная нормализация: (x - median) / (1.4826 * MAD).

			При вырожденности MAD — фолбэк на z-score; при нуле и там, и там — возвращаем нули.
			"""
			if not values:
				return []
			arr = np.array(values, dtype=np.float64)
			med = float(np.median(arr))
			mad = float(np.median(np.abs(arr - med)))
			den = 1.4826 * mad
			if den > 0:
				return [float((x - med) / den) for x in arr]
			# Фолбэк: z-score
			mu = float(np.mean(arr))
			sigma = float(np.std(arr))
			if sigma > 0:
				return [float((x - mu) / sigma) for x in arr]
			return [0.0 for _ in arr]

		vec_scaled = _robust_scale(raw_vec)
		bm25_scaled = _robust_scale(raw_bm25)
		vec_scores = {ids[i]: vec_scaled[i] if i < len(vec_scaled) else 0.0 for i in range(len(ids))}
		bm25_scores = {ids[i]: bm25_scaled[i] if i < len(bm25_scaled) else 0.0 for i in range(len(ids))}

		alpha, beta = weights
		combined: List[Tuple[Dict, float]] = []
		for h in vector_hits:
			vid = h.get("id")
			if vid is None:
				continue
			vid_str = str(vid)
			vs = float(vec_scores.get(vid_str, 0.0))
			bs = float(bm25_scores.get(vid_str, 0.0))
			s = alpha * vs + beta * bs
			combined.append((h, s))
		combined.sort(key=lambda x: x[1], reverse=True)
		return combined[:top_k]
