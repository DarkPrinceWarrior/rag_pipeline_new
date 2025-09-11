from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import lancedb
from rank_bm25 import BM25Okapi

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
		ids = [p["id"] for p in payload]
		if len(ids) == 1:
			self.table.delete(f"id == '{ids[0]}'")
		elif len(ids) > 1:
			self.table.delete(f"id in {tuple(ids)}")
		self.table.add(payload)

	def _all_texts_and_ids(self) -> Tuple[List[str], List[str]]:
		if self.table is None:
			return [], []
		df = self.table.to_pandas()
		if df is None or df.empty:
			return [], []
		cols = list(df.columns)
		id_col = "id" if "id" in cols else None
		text_col = "text" if "text" in cols else None
		if id_col is None or text_col is None:
			return [], []
		df = df[[id_col, text_col]].dropna()
		return df[id_col].astype(str).tolist(), df[text_col].astype(str).tolist()

	def _bm25_scores(self, query: str, candidate_ids: List[str] | None = None) -> Dict[str, float]:
		texts, ids = self._all_texts_and_ids()
		if not texts or not ids:
			return {}
		if candidate_ids is not None:
			cand = set(candidate_ids)
			masked = [(t, i) for t, i in zip(texts, ids) if i in cand]
			if not masked:
				return {}
			texts, ids = [m[0] for m in masked], [m[1] for m in masked]
		tokenized_corpus = [t.lower().split() for t in texts]
		if not tokenized_corpus:
			return {}
		bm25 = BM25Okapi(tokenized_corpus)
		scores = bm25.get_scores(query.lower().split())
		max_s = float(np.max(scores)) if len(scores) else 1.0
		if max_s == 0:
			return {ids[i]: 0.0 for i in range(len(ids))}
		return {ids[i]: float(scores[i] / max_s) for i in range(len(ids))}

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
		candidate_ids = [h.get("id") for h in vector_hits if h.get("id") is not None]
		bm25_scores = self._bm25_scores(query, candidate_ids=None)
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
