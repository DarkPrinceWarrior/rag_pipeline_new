from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import httpx
from duckduckgo_search import DDGS


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    content: Optional[str] = None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def web_search(query: str, max_results: int = 5, timeout: float = 10.0) -> List[WebResult]:
    results: List[WebResult] = []
    try:
        with DDGS(timeout=timeout) as ddgs:
            for r in ddgs.text(query, max_results=max_results, safesearch="moderate"):  # type: ignore
                title = r.get("title") or ""
                url = r.get("href") or r.get("url") or ""
                snippet = r.get("body") or r.get("snippet") or ""
                if not url:
                    continue
                results.append(WebResult(title=_clean_text(title), url=url, snippet=_clean_text(snippet)))
    except Exception:
        return results
    return results


def fetch_page_text(url: str, timeout: float = 10.0, max_chars: int = 4000) -> Optional[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
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
        if len(text) > max_chars:
            text = text[:max_chars]
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

