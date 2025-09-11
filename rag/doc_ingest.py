from __future__ import annotations

import os
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List

from .config import settings
from .utils import sliding_token_windows


def _normalize_text(text: str) -> str:
	# Normalize unicode and lowercase
	return unicodedata.normalize("NFKC", text).lower()


def _read_pdf_with_docling_api(pdf_path: str | Path):
	try:
		from docling.document_converter import DocumentConverter  # type: ignore
	except Exception as e:
		raise ImportError(f"Docling API unavailable: {e}")
	converter = DocumentConverter()
	result = converter.convert(str(pdf_path))
	return result


def _read_pdf_with_docling_cli(pdf_path: str | Path) -> str:
	# Fallback: use the docling CLI to get markdown
	with tempfile.TemporaryDirectory() as tmpdir:
		out_md = Path(tmpdir) / "out.md"
		cmd = [
			"docling",
			str(pdf_path),
			"--output",
			str(out_md),
			"--format",
			"markdown",
		]
		proc = subprocess.run(cmd, capture_output=True, text=True)
		if proc.returncode != 0:
			raise RuntimeError(f"Docling CLI failed: {proc.stderr}")
		return Path(out_md).read_text(encoding="utf-8", errors="ignore")


def _docling_pages_to_markdown_from_api(result) -> List[str]:
	# Try multiple known export methods defensively
	doc = getattr(result, "document", None) or getattr(result, "doc", None)
	if doc is None:
		raise RuntimeError("Docling result has no document")
	# Try export methods
	for method_name in ("export_markdown", "export_to_markdown", "export_markdown_pages"):
		if hasattr(doc, method_name):
			exported = getattr(doc, method_name)()
			if isinstance(exported, str):
				return [exported]
			try:
				return list(exported)
			except Exception:
				pass
	# Fallback to plain text
	for method_name in ("export_text", "export_to_text"):
		if hasattr(doc, method_name):
			text = getattr(doc, method_name)()
			return [text]
	raise RuntimeError("Docling export functions not found")


def ingest_pdf_to_chunks(
	pdf_path: str | Path,
	chunk_tokens: int | None = None,
	overlap_tokens: int | None = None,
) -> List[Dict]:
	"""Return list of chunk dicts with fields: id, text, page, filename, source_path."""
	pdf_path = Path(pdf_path)
	if not pdf_path.exists():
		raise FileNotFoundError(pdf_path)

	pages_md: List[str]
	try:
		result = _read_pdf_with_docling_api(pdf_path)
		pages_md = _docling_pages_to_markdown_from_api(result)
	except Exception:
		# CLI fallback -> single markdown string
		md = _read_pdf_with_docling_cli(pdf_path)
		pages_md = [md]

	chunk_tokens = chunk_tokens or settings.chunk_size_tokens
	overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

	filename = pdf_path.name
	source_path = str(pdf_path)

	records: List[Dict] = []
	serial = 0
	for page_index, page_md in enumerate(pages_md, start=1):
		page_md = _normalize_text(page_md)
		windows = sliding_token_windows(
			page_md, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens
		)
		for _, _, segment in windows:
			serial += 1
			records.append(
				{
					"chunk_id": f"{filename}::p{page_index}::c{serial}",
					"text": segment,
					"page": page_index,
					"filename": filename,
					"source_path": source_path,
				}
			)
	return records