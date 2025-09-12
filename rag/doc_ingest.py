from __future__ import annotations

import os
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List

from .config import settings
from .utils import sliding_token_windows
import re


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


def _split_markdown_into_pages(md: str) -> List[str]:
    """Best-effort split of a whole-document Markdown into pages.

    Heuristics handle common separators emitted by converters:
    - Form feed characters (\f)
    - Lines starting with 'Page N' or '# Page N'
    - Horizontal rule style delimiters around 'Page N'
    If no markers are found, returns a single-page list.
    """
    if not md:
        return [""]

    # 1) Split on form-feed if present
    if "\f" in md:
        parts = [p.strip() for p in md.split("\f")]
        return [p for p in parts if p]

    # 2) Normalize line endings and scan for page markers
    lines = md.splitlines()
    page_starts = []
    page_header_patterns = [
        re.compile(r"^\s*#{0,3}\s*page\s+(\d+)\s*$", re.IGNORECASE),
        re.compile(r"^\s*page\s+(\d+)\s*$", re.IGNORECASE),
        re.compile(r"^-{3,}\s*page\s+(\d+)\s*-{3,}\s*$", re.IGNORECASE),
    ]
    for i, line in enumerate(lines):
        for pat in page_header_patterns:
            m = pat.match(line)
            if m:
                page_starts.append(i)
                break

    # Build pages based on detected starts
    if page_starts:
        page_starts = sorted(set(page_starts))
        page_starts.append(len(lines))
        pages = []
        for a, b in zip(page_starts[:-1], page_starts[1:]):
            seg = "\n".join(lines[a:b]).strip()
            if seg:
                pages.append(seg)
        if pages:
            return pages

    # 3) Fallback to a single page
    return [md]


def _docling_pages_to_markdown_from_api(result) -> List[str]:
    """Export markdown per page when possible, else fallback to single blob.

    We prefer page-wise export to preserve correct page numbers for citations.
    """
    # Locate underlying document object
    doc = getattr(result, "document", None) or getattr(result, "doc", None)
    if doc is None:
        raise RuntimeError("Docling result has no document")

    def _to_list(obj):
        if obj is None:
            return None
        if isinstance(obj, (list, tuple)):
            return [str(x) for x in obj]
        try:
            lst = list(obj)
            return [str(x) for x in lst]
        except Exception:
            return None

    # 1) Try page-wise markdown first (common in Docling versions)
    for method_name in ("export_markdown_pages", "export_to_markdown_pages", "export_pages_markdown"):
        if hasattr(doc, method_name):
            exported = getattr(doc, method_name)()
            pages = _to_list(exported)
            if pages and len(pages) > 0:
                return pages

    # 2) Fallback to whole-document markdown
    for method_name in ("export_markdown", "export_to_markdown"):
        if hasattr(doc, method_name):
            exported = getattr(doc, method_name)()
            if isinstance(exported, str) and exported.strip():
                # Try to split into pages if possible
                pages = _split_markdown_into_pages(exported)
                return pages if pages else [exported]
            pages = _to_list(exported)
            if pages and len(pages) > 0:
                return pages

    # 3) Try iterating explicit page objects if available
    pages_attr = getattr(doc, "pages", None)
    if pages_attr is not None:
        try:
            pages_list = list(pages_attr)
        except Exception:
            pages_list = None
        if pages_list:
            out: List[str] = []
            for p in pages_list:
                for m in ("export_markdown", "export_to_markdown", "export_text", "export_to_text"):
                    if hasattr(p, m):
                        try:
                            txt = getattr(p, m)()
                            if txt and isinstance(txt, str):
                                out.append(txt)
                                break
                        except Exception:
                            pass
            if out:
                return out

    # 4) Last resort: plain text
    for method_name in ("export_text", "export_to_text"):
        if hasattr(doc, method_name):
            text = getattr(doc, method_name)()
            try:
                s = str(text)
            except Exception:
                s = ""
            pages = _split_markdown_into_pages(s)
            return pages if pages else [s]
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
		# CLI fallback -> try to split into pages heuristically
		md = _read_pdf_with_docling_cli(pdf_path)
		pages_md = _split_markdown_into_pages(md)

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
