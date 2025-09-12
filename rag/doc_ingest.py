from __future__ import annotations

import os
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List
import hashlib

from .config import settings
from .utils import sliding_token_windows, get_tokenizer
import re


def _normalize_text(text: str) -> str:
	"""Нормализация текста для LLM без изменения регистра.

	Выполняет Unicode NFKC, опционально правит переносы с дефисом и пробелы/пустые строки.
	"""
	s = unicodedata.normalize("NFKC", text)
	# Опционально приводим к нижнему регистру
	if not settings.preserve_case:
		s = s.lower()
	# Склеивание переносов с дефисом: опционально
	if settings.merge_hyphenation:
		s = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", s, flags=re.UNICODE)
	# Унификация переводов строк
	s = s.replace("\r\n", "\n").replace("\r", "\n")
	# Заменяем табы на пробелы
	s = s.replace("\t", " ")
	# Схлопываем множественные пробелы внутри строки
	s = re.sub(r"[ ]{2,}", " ", s)
	# Схлопываем более двух пустых строк в одну
	s = re.sub(r"\n{3,}", "\n\n", s)
	return s


def _classify_block(lines: List[str]) -> str:
	"""Определить тип элемента для сегмента Markdown.

	Возможные значения: heading, list, table, code, math, paragraph.
	"""
	if not lines:
		return "paragraph"
	first = lines[0].strip()
	if first.startswith("$$") and lines[-1].strip().endswith("$$"):
		return "math"
	if first.startswith("```") and lines[-1].strip().endswith("```"):
		return "code"
	if re.match(r"^\s*#{1,6}\s+", first):
		return "heading"
	if all(l.strip().startswith("|") or re.search(r"\|", l) for l in lines):
		return "table"
	if all(re.match(r"^\s*([\-*+]|\d+\.)\s+", l) for l in lines if l.strip()):
		return "list"
	return "paragraph"


def _split_markdown_into_blocks(md: str) -> List[Dict]:
	"""Разбить Markdown-страницу на семантические блоки.

	Сохраняет кодовые блоки/формулы единым фрагментом, группирует списки и таблицы.
	"""
	lines = md.splitlines()
	blocks: List[Dict] = []
	buf: List[str] = []
	inside_code = False
	inside_math = False
	for line in lines + [""]:
		strip = line.strip()
		if strip.startswith("$$"):
			inside_math = not inside_math
			buf.append(line)
			if not inside_math:
				blocks.append({"type": "math", "text": "\n".join(buf).strip()})
				buf = []
			continue
		if strip.startswith("```"):
			inside_code = not inside_code
			buf.append(line)
			if not inside_code:
				blocks.append({"type": "code", "text": "\n".join(buf).strip()})
				buf = []
			continue
		if inside_code or inside_math:
			buf.append(line)
			continue
		# Пустая строка — граница абзаца
		if strip == "":
			if buf:
				btype = _classify_block(buf)
				blocks.append({"type": btype, "text": "\n".join(buf).strip()})
				buf = []
			continue
		buf.append(line)
	if buf:
		btype = _classify_block(buf)
		blocks.append({"type": btype, "text": "\n".join(buf).strip()})
	return [b for b in blocks if b.get("text")]


def _strip_headers_footers(pages_md: List[str]) -> List[str]:
	"""Удалить повторяющиеся хедеры/футеры, встречающиеся на большинстве страниц.

	Эвристика: строки длиной 3..120 символов, совпадающие на >=70% страниц
	и находящиеся в первой/последней строке, удаляются.
	"""
	if not settings.strip_headers_footers or not pages_md:
		return pages_md
	from collections import Counter
	lines_per_page = [p.splitlines() for p in pages_md]
	if not lines_per_page:
		return pages_md
	header_counter: Counter[str] = Counter()
	footer_counter: Counter[str] = Counter()
	for lines in lines_per_page:
		if not lines:
			continue
		first = lines[0].strip()
		last = lines[-1].strip()
		if 3 <= len(first) <= 120:
			header_counter[first] += 1
		if 3 <= len(last) <= 120:
			footer_counter[last] += 1
	threshold = max(1, int(0.7 * len(lines_per_page)))
	common_headers = {s for s, c in header_counter.items() if c >= threshold}
	common_footers = {s for s, c in footer_counter.items() if c >= threshold}
	cleaned: List[str] = []
	for lines in lines_per_page:
		out = list(lines)
		if out and out[0].strip() in common_headers:
			out = out[1:]
		if out and out[-1].strip() in common_footers:
			out = out[:-1]
		cleaned.append("\n".join(out))
	return cleaned


def _convert_table_text(md_table: str, mode: str) -> str:
	"""Конвертировать markdown-таблицу в формат md|html|csv.

	Если структура не распознана, возвращает исходный md_table.
	"""
	mode = (mode or "md").lower()
	if mode == "md":
		return md_table
	lines = [l.strip() for l in md_table.splitlines() if l.strip()]
	# Убираем разделитель заголовков типа | --- | --- |
	lines = [l for l in lines if not re.match(r"^\|?\s*:?-{2,}.*", l)]
	rows: List[List[str]] = []
	for l in lines:
		if "|" in l:
			cells = [c.strip() for c in l.strip("|").split("|")]
			rows.append(cells)
		else:
			return md_table
	if mode == "csv":
		csv_lines = []
		for r in rows:
			csv_cells = []
			for c in r:
				cc = c.replace('"', '""')
				if "," in cc or '"' in cc or "\n" in cc:
					cc = f'"{cc}"'
				csv_cells.append(cc)
			csv_lines.append(",".join(csv_cells))
		return "\n".join(csv_lines)
	# html
	parts = ["<table>"]
	for r in rows:
		cells_html = []
		for c in r:
			c2 = re.sub(r">", "&gt;", re.sub(r"<", "&lt;", c))
			cells_html.append(f"<td>{c2}</td>")
		parts.append("<tr>" + "".join(cells_html) + "</tr>")
	parts.append("</table>")
	return "".join(parts)


def _detect_lang(text: str) -> str:
	"""Простая детекция языка: ru/en/other по соотношению кириллицы/латиницы."""
	if not text:
		return "other"
	cyr = len(re.findall(r"[А-Яа-яЁё]", text))
	lat = len(re.findall(r"[A-Za-z]", text))
	if cyr > lat:
		return "ru"
	if lat > cyr:
		return "en"
	return "other"


def _build_section_paths(blocks: List[Dict]) -> List[str]:
	"""Построить путь заголовков (H1>H2>H3) для каждого блока."""
	path: List[str] = []
	paths: List[str] = []
	for b in blocks:
		text = b.get("text", "")
		m = re.match(r"^\s*(#{1,6})\s+(.*)$", text)
		if m:
			level = len(m.group(1))
			title = m.group(2).strip()
			while len(path) >= level:
				path.pop() if path else None
			path.append(title)
			paths.append(" > ".join(path))
			continue
		paths.append(" > ".join(path))
	return paths


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


def _simple_ocr_from_pdf(pdf_path: str | Path) -> str:
	"""Простой OCR-фолбэк: извлечь текст из изображений страниц PDF.

	Использует pdf2image и pytesseract при наличии; при ошибке возвращает пустую строку.
	"""
	try:
		from pdf2image import convert_from_path  # type: ignore
		import pytesseract  # type: ignore
	except Exception:
		return ""
	texts: List[str] = []
	try:
		images = convert_from_path(str(pdf_path))
		for img in images:
			try:
				txt = pytesseract.image_to_string(img)
				if txt and isinstance(txt, str):
					texts.append(txt)
			except Exception:
				continue
		return "\n\n".join(texts)
	except Exception:
		return ""


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


def ingest_document_to_chunks(
	pdf_path: str | Path,
	chunk_tokens: int | None = None,
	overlap_tokens: int | None = None,
) -> List[Dict]:
	"""Вернуть список чанков с полями: chunk_id, text, page, filename, source_path, section_path, element_type."""
	pdf_path = Path(pdf_path)
	if not pdf_path.exists():
		raise FileNotFoundError(pdf_path)

	pages_md: List[str]
	ocr_used = False
	try:
		if settings.prefer_docling_api:
			result = _read_pdf_with_docling_api(pdf_path)
			pages_md = _docling_pages_to_markdown_from_api(result)
			pages_md = _strip_headers_footers(pages_md)
		else:
			raise RuntimeError("API disabled by config")
	except Exception:
		md = ""
		if settings.enable_cli_fallback:
			try:
				md = _read_pdf_with_docling_cli(pdf_path)
			except Exception:
				md = ""
		if not md or not md.strip():
			if settings.enable_ocr:
				ocr_blob = _simple_ocr_from_pdf(pdf_path)
				if ocr_blob and ocr_blob.strip():
					pages_md = _split_markdown_into_pages(ocr_blob)
					pages_md = _strip_headers_footers(pages_md)
					ocr_used = True
				else:
					pages_md = [""]
			else:
				pages_md = [""]
		else:
			pages_md = _split_markdown_into_pages(md)
			pages_md = _strip_headers_footers(pages_md)

	chunk_tokens = chunk_tokens or settings.chunk_size_tokens
	overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

	filename = pdf_path.name
	source_path = str(pdf_path)
	content_hash = hashlib.sha256("\n".join(pages_md).encode("utf-8", errors="ignore")).hexdigest()

	def _process_one_page(page_index: int, page_md: str) -> List[Dict]:
		page_md = _normalize_text(page_md)
		blocks = _split_markdown_into_blocks(page_md)
		section_paths = _build_section_paths(blocks)
		enc_local = get_tokenizer()
		doc_id = filename
		acc_text: List[str] = []
		acc_types: List[str] = []
		acc_path = ""
		acc_tokens = 0
		c_serial = 0
		page_records: List[Dict] = []
		for i, b in enumerate(blocks):
			b_text = b["text"]
			b_type = b.get("type", "paragraph")
			# Конвертация таблиц при необходимости
			if b_type == "table":
				b_text = _convert_table_text(b_text, settings.table_mode)
			b_path = section_paths[i]
			b_tokens = len(enc_local.encode(b_text))
			if b_tokens > chunk_tokens:
				for _, _, seg in sliding_token_windows(b_text, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens):
					c_serial += 1
					page_records.append(
						{
							"chunk_id": f"{filename}::p{page_index}::c{c_serial}",
							"text": seg,
							"page": page_index,
							"filename": filename,
							"source_path": source_path,
							"section_path": b_path,
							"element_type": b_type,
							"doc_id": doc_id,
							"lang": _detect_lang(seg),
							"ocr_used": ocr_used,
							"content_hash": content_hash,
						}
					)
				acc_text, acc_types, acc_path, acc_tokens = [], [], "", 0
				continue
			if acc_tokens + b_tokens <= chunk_tokens:
				acc_text.append(b_text)
				acc_types.append(b_type)
				acc_path = b_path or acc_path
				acc_tokens += b_tokens
			else:
				if acc_text:
					c_serial += 1
					page_records.append(
						{
							"chunk_id": f"{filename}::p{page_index}::c{c_serial}",
							"text": "\n\n".join(acc_text).strip(),
							"page": page_index,
							"filename": filename,
							"source_path": source_path,
							"section_path": acc_path,
							"element_type": ",".join(sorted(set(acc_types))) or "paragraph",
							"doc_id": doc_id,
							"lang": _detect_lang("\n\n".join(acc_text).strip()),
							"ocr_used": ocr_used,
							"content_hash": content_hash,
						}
					)
				acc_text, acc_types, acc_path, acc_tokens = [b_text], [b_type], b_path, b_tokens
		if acc_text:
			c_serial += 1
			page_records.append(
				{
					"chunk_id": f"{filename}::p{page_index}::c{c_serial}",
					"text": "\n\n".join(acc_text).strip(),
					"page": page_index,
					"filename": filename,
					"source_path": source_path,
					"section_path": acc_path,
					"element_type": ",".join(sorted(set(acc_types))) or "paragraph",
					"doc_id": doc_id,
					"lang": _detect_lang("\n\n".join(acc_text).strip()),
					"ocr_used": ocr_used,
					"content_hash": content_hash,
				}
			)
		return page_records

	max_workers = max(1, settings.max_parallel_pages)
	if max_workers == 1:
		all_records: List[Dict] = []
		for page_index, page_md in enumerate(pages_md, start=1):
			all_records.extend(_process_one_page(page_index, page_md))
		return all_records
	else:
		results_by_page: Dict[int, List[Dict]] = {}
		with ThreadPoolExecutor(max_workers=max_workers) as ex:
			futures = {ex.submit(_process_one_page, i, md): i for i, md in enumerate(pages_md, start=1)}
			for fut in futures:
				page_i = futures[fut]
				try:
					results_by_page[page_i] = fut.result()
				except Exception:
					results_by_page[page_i] = []
		all_records: List[Dict] = []
		for page_i in sorted(results_by_page.keys()):
			all_records.extend(results_by_page[page_i])
		return all_records


def ingest_pdf_to_chunks(
	pdf_path: str | Path,
	chunk_tokens: int | None = None,
	overlap_tokens: int | None = None,
) -> List[Dict]:
	"""Совместимость: делегирует в ingest_document_to_chunks."""
	return ingest_document_to_chunks(pdf_path, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
