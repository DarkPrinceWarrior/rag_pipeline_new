from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import httpx
from tavily import TavilyClient
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from .config import settings


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    content: Optional[str] = None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_url(url: str) -> str:
    """Упрощённая нормализация URL: нижний регистр схемы/хоста и удаление трекинг‑параметров."""
    try:
        s = urlsplit(url)
        query_pairs = [
            (k, v)
            for (k, v) in parse_qsl(s.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in {"gclid", "fbclid", "mc_cid", "mc_eid"}
        ]
        new_query = urlencode(query_pairs, doseq=True)
        return urlunsplit((s.scheme.lower(), s.netloc.lower(), s.path, new_query, ""))
    except Exception:
        return url


_tavily_client: Optional[TavilyClient] = None


def _get_tavily_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    return _tavily_client


def web_search(query: str, max_results: int = 5, timeout: float = 10.0) -> List[WebResult]:
    # Подготовка токенов запроса для фильтрации нерелевантных результатов
    filter_enabled = settings.web_search_filter_enabled
    normalized_query = re.sub(r"[^\w\sА-Яа-яЁё-]", " ", query).lower()
    raw_tokens = [t for t in re.split(r"\s+", normalized_query) if t]
    stopwords = {w.strip() for w in settings.web_search_stopwords_ru.split(",") if w.strip()}
    tokens = [t for t in raw_tokens if t not in stopwords and len(t) >= settings.web_search_min_token_len]

    results: List[WebResult] = []
    try:
        client = _get_tavily_client()
        # Префиксируем запрос профессиональной ролью из настроек
        prefixed_query = f"{settings.web_search_query_prefix} {query}".strip()
        # Расширенные параметры поиска
        resp = client.search(
            query=prefixed_query,
            search_depth=settings.web_search_depth,
            max_results=max_results,
            include_raw_content=settings.web_search_include_raw_content,
            chunks_per_source=settings.web_search_chunks_per_source,
        )
        items = resp.get("results", []) if isinstance(resp, dict) else []
        seen: set[str] = set()
        for r in items:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            if not url:
                continue
            norm_url = _normalize_url(url)
            if norm_url in seen:
                continue
            seen.add(norm_url)
            # Tavily returns page content snippet in `content`
            snippet = (r.get("content") or r.get("snippet") or "").strip()
            if filter_enabled and tokens:
                hay = f"{title} {snippet} {norm_url}".lower()
                num_hits = sum(1 for t in tokens if t in hay)
                if num_hits < settings.web_search_required_token_matches:
                    continue
            results.append(WebResult(title=_clean_text(title), url=norm_url, snippet=_clean_text(snippet)))
    except Exception:
        return results
    return results


def fetch_page_text(url: str, timeout: float = 10.0, max_chars: int = 4000) -> Optional[str]:
    headers = {
        "User-Agent": settings.web_user_agent,
    }
    try:
        t = settings.web_http_timeout_seconds if timeout is None else timeout
        m = settings.web_fetch_max_chars if max_chars is None else max_chars
        with httpx.Client(timeout=httpx.Timeout(t), verify=settings.web_http_verify_tls) as client:
            resp = client.get(url, headers=headers, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
    except Exception:
        return None

    # Very lightweight HTML-to-text using regex fallbacks to avoid heavy deps
    try:
        # Remove scripts/styles
        html = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", " ", html, flags=re.IGNORECASE)
        # Strip tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = _clean_text(text)
        if len(text) > m:
            text = text[:m]
        return text
    except Exception:
        return None


def build_web_context(query: str, max_results: int = 5, fetch_top_n: int = 3, max_tokens: int = 3200) -> tuple[str, List[WebResult]]:
    results = web_search(query, max_results=max_results)
    if not results:
        return "", []

    # Fetch a few pages for richer context
    for i, r in enumerate(results[: max(0, fetch_top_n) ]):
        content = fetch_page_text(r.url)
        if content:
            results[i].content = content

    # Assemble context with headers, keeping a token-like budget by length heuristic
    parts: List[str] = []
    budget = max_tokens * 3  # rough heuristic: char budget ~ 3x tokens
    serial = 0
    for r in results:
        serial += 1
        header = f"W#{serial} — {r.title or r.url} ({r.url})\n"
        body = r.content or r.snippet
        if not body:
            continue
        segment = header + body
        if len(segment) > budget:
            segment = segment[:budget]
        parts.append(segment)
        budget -= len(segment)
        if budget <= 0:
            break

    context = "\n\n".join(parts)
    return context, results

