from __future__ import annotations

from typing import Iterable, List, Tuple

import tiktoken


def get_tokenizer(model: str = "gpt-4o-mini"):
	# tiktoken needs a model name to infer encoding; using a common one for consistency
	try:
		return tiktoken.encoding_for_model(model)
	except Exception:
		return tiktoken.get_encoding("cl100k_base")


def tokenize_text(text: str, enc=None) -> List[int]:
	if enc is None:
		enc = get_tokenizer()
	return enc.encode(text)


def detokenize(tokens: Iterable[int], enc=None) -> str:
	if enc is None:
		enc = get_tokenizer()
	return enc.decode(list(tokens))


def sliding_token_windows(
	text: str,
	max_tokens: int,
	overlap_tokens: int,
	enc=None,
) -> List[Tuple[int, int, str]]:
	"""Return list of (start_token, end_token_exclusive, text_segment)."""
	enc = enc or get_tokenizer()
	tokens = enc.encode(text)
	segments: List[Tuple[int, int, str]] = []
	if max_tokens <= 0:
		return [(0, len(tokens), text)]
	start = 0
	while start < len(tokens):
		end = min(start + max_tokens, len(tokens))
		segment_text = enc.decode(tokens[start:end])
		segments.append((start, end, segment_text))
		if end == len(tokens):
			break
		start = end - overlap_tokens
		if start < 0:
			start = 0
	return segments
