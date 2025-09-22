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
		# sentence-transformers >=2.7 поддерживает prompt= для instruct моделей;
		# если параметр недоступен, префиксуем тексты вручную.
		items = list(texts)
		kwargs = dict(
			batch_size=batch_size,
			normalize_embeddings=self.normalize,
			convert_to_numpy=True,
			show_progress_bar=False,
		)
		try:
			# Попытка использовать нативный prompt=
			emb = self.model.encode(items, prompt=None, **kwargs)
		except TypeError:
			# Фолбэк без prompt=
			emb = self.model.encode(items, **kwargs)
		return emb.tolist()

	def embed_query(self, text: str) -> List[float]:
		prefix = settings.embed_query_prompt or ""
		q = f"{prefix}{text}" if prefix else text
		# Если модель поддерживает prompt=, используем его вместо префикса
		try:
			emb = self.model.encode([text], prompt=prefix or None, batch_size=1, normalize_embeddings=self.normalize, convert_to_numpy=True, show_progress_bar=False)
		except TypeError:
			emb = self.model.encode([q], batch_size=1, normalize_embeddings=self.normalize, convert_to_numpy=True, show_progress_bar=False)
		return emb[0].tolist()

	@property
	def dim(self) -> int:
		return int(self.model.get_sentence_embedding_dimension())
