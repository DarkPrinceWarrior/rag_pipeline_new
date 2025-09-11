from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import torch
from FlagEmbedding import FlagReranker

from .config import settings


class Reranker:
	def __init__(self, model_id: str | None = None, device: str | None = None, use_fp16: bool = True, score_norm: str = "sigmoid"):
		self.model_id = model_id or settings.reranker_model_id
		preferred_device = device or settings.device
		if preferred_device == "cuda" and not torch.cuda.is_available():
			preferred_device = "cpu"
		self.device = preferred_device
		self.use_fp16 = use_fp16 and (self.device == "cuda")
		self.score_norm = score_norm
		self.model = FlagReranker(self.model_id, use_fp16=self.use_fp16, device=self.device)

	def _normalize(self, score: float) -> float:
		if self.score_norm == "sigmoid":
			return 1.0 / (1.0 + math.exp(-score))
		return score

	def score_pairs(self, query: str, passages: Iterable[str], batch_size: int | None = None, max_length: int | None = 1024) -> List[float]:
		pairs = [[query, p] for p in passages]
		scores = self.model.compute_score(pairs, batch_size=(batch_size or settings.batch_size_rerank), max_length=max_length)
		return [self._normalize(s) for s in scores]

	def rerank(self, query: str, passages_with_meta: List[Tuple[str, dict]], top_n: int) -> List[Tuple[str, dict, float]]:
		texts = [t for t, _ in passages_with_meta]
		scores = self.score_pairs(query, texts)
		items = [(texts[i], passages_with_meta[i][1], scores[i]) for i in range(len(texts))]
		items.sort(key=lambda x: x[2], reverse=True)
		return items[:top_n]
