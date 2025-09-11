from __future__ import annotations

import os
from typing import Iterable, List

import torch
from sentence_transformers import SentenceTransformer

from .config import settings


class EmbeddingModel:
	def __init__(self, model_id: str | None = None, device: str | None = None, normalize: bool = True):
		self.model_id = model_id or settings.embedding_model_id
		preferred_device = device or settings.device
		if preferred_device == "cuda" and not torch.cuda.is_available():
			preferred_device = "cpu"
		self.device = preferred_device
		self.normalize = normalize
		if settings.hf_token:
			os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", settings.hf_token)
		self.model = SentenceTransformer(self.model_id, device=self.device)
		self.model.max_seq_length = 512

	def embed(self, texts: Iterable[str], batch_size: int | None = None) -> List[List[float]]:
		batch_size = batch_size or settings.batch_size_embed
		emb = self.model.encode(
			list(texts),
			batch_size=batch_size,
			normalize_embeddings=self.normalize,
			convert_to_numpy=True,
			show_progress_bar=False,
		)
		return emb.tolist()

	def embed_query(self, text: str) -> List[float]:
		return self.embed([text])[0]

	@property
	def dim(self) -> int:
		return int(self.model.get_sentence_embedding_dimension())
