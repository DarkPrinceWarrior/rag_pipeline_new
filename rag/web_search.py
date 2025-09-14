from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import httpx
from duckduckgo_search import DDGS
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


def web_search(query: str, max_results: int = 5, timeout: float = 10.0) -> List[WebResult]:
    region = settings.ddg_region
    safesearch = settings.ddg_safesearch
    timelimit = settings.ddg_timelimit or None
    backend = settings.ddg_backend
    timeout = settings.ddg_timeout_seconds if timeout is None else timeout
    proxy = settings.ddg_proxy or None
    # Подготовка токенов запроса для фильтрации нерелевантных словарных результатов
    filter_enabled = settings.web_search_filter_enabled
    normalized_query = re.sub(r"[^\w\sА-Яа-яЁё-]", " ", query).lower()
    raw_tokens = [t for t in re.split(r"\s+", normalized_query) if t]
    stopwords = {w.strip() for w in settings.web_search_stopwords_ru.split(",") if w.strip()}
    tokens = [t for t in raw_tokens if t not in stopwords and len(t) >= settings.web_search_min_token_len]

    def run_search(backend_name: str) -> List[WebResult]:
        results: List[WebResult] = []
        try:
            ctx = None
            if proxy:
                try:
                    ctx = DDGS(timeout=timeout, proxies=proxy)  # type: ignore[arg-type]
                except TypeError:
                    try:
                        ctx = DDGS(timeout=timeout, proxy=proxy)  # older versions
                    except TypeError:
                        ctx = DDGS(timeout=timeout)
            else:
                ctx = DDGS(timeout=timeout)

            with ctx as ddgs:  # type: ignore[var-annotated]
                seen: set[str] = set()
                try:
                    iter_results = ddgs.text(
                        query,
                        region=region,
                        safesearch=safesearch,
                        timelimit=timelimit,
                        backend=backend_name,
                        max_results=max_results,
                    )  # type: ignore
                except TypeError:
                    # Обратная совместимость для старых версий без параметра backend
                    iter_results = ddgs.text(
                        query,
                        region=region,
                        safesearch=safesearch,
                        timelimit=timelimit,
                        max_results=max_results,
                    )  # type: ignore

                for r in iter_results:
                    title = r.get("title") or ""
                    url = r.get("href") or r.get("url") or ""
                    if not url:
                        continue
                    norm_url = _normalize_url(url)
                    if norm_url in seen:
                        continue
                    seen.add(norm_url)
                    snippet = r.get("body") or r.get("snippet") or ""
                    # Фильтрация по наличию терминов из запроса в заголовке/сниппете/URL
                    if filter_enabled and tokens:
                        hay = f"{title} {snippet} {norm_url}".lower()
                        num_hits = sum(1 for t in tokens if t in hay)
                        if num_hits < settings.web_search_required_token_matches:
                            continue
                    results.append(
                        WebResult(title=_clean_text(title), url=norm_url, snippet=_clean_text(snippet))
                    )
        except Exception:
            return results
        return results

    out = run_search(backend)
    if not out and backend != settings.ddg_alt_backend:
        out = run_search(settings.ddg_alt_backend)
    return out


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

