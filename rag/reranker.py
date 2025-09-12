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
		items: List[Tuple[str, dict, float]] = []
		for i in range(len(texts)):
			base = scores[i]
			meta = passages_with_meta[i][1] or {}
			etype = (meta.get("element_type") or "paragraph").lower()
			section_path = meta.get("section_path") or ""
			# Бонусы/штрафы по типу элемента
			bonus = 0.0
			if "heading" in etype:
				bonus += settings.rerank_bonus_heading
			if "table" in etype:
				bonus += settings.rerank_bonus_table
			if "code" in etype:
				bonus += settings.rerank_bonus_code
			if "list" in etype:
				bonus += settings.rerank_bonus_list
			if "math" in etype:
				bonus += settings.rerank_bonus_math
			if "paragraph" in etype:
				bonus += settings.rerank_bonus_paragraph
			# Наказание за глубину секции (чем глубже, тем немного ниже)
			depth = 0
			if section_path:
				depth = max(0, len([p for p in section_path.split(" > ") if p.strip()]) - 1)
				bonus -= min(settings.rerank_section_depth_penalty * depth, settings.rerank_max_meta_bonus)
			# Ограничием общий бонус/штраф
			if bonus > settings.rerank_max_meta_bonus:
				bonus = settings.rerank_max_meta_bonus
			if bonus < -settings.rerank_max_meta_bonus:
				bonus = -settings.rerank_max_meta_bonus
			items.append((texts[i], meta, float(base + bonus)))
		items.sort(key=lambda x: x[2], reverse=True)
		return items[:top_n]
