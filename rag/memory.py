from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from rag.config import settings

try:
    from mem0 import Memory as Mem0Memory
except ImportError:  # pragma: no cover - mem0 is optional at runtime
    Mem0Memory = None


logger = logging.getLogger(__name__)


class MemoryManager:
    """Wrapper around Mem0 that enforces OpenRouter usage and safe fallbacks."""

    def __init__(self) -> None:
        self._settings = settings
        self.enabled = bool(self._settings.memory_enabled and Mem0Memory is not None)
        self._memory: Any = None
        if not self.enabled:
            if Mem0Memory is None:
                logger.warning("mem0ai package is not available; memory is disabled.")
            else:
                logger.info("Memory subsystem disabled via configuration.")
            return

        self._prepare_environment()
        try:
            self._memory = Mem0Memory.from_config(self._build_config())
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Failed to initialize Mem0 memory store", exc_info=exc)
            self.enabled = False
            self._memory = None

    def _prepare_environment(self) -> None:
        """Ensure Mem0 only talks to OpenRouter."""
        openrouter_key = self._settings.openrouter_api_key
        os.environ.setdefault("OPENROUTER_API_KEY", openrouter_key)
        os.environ.setdefault("OPENAI_API_KEY", openrouter_key)

        base_url = self._settings.openrouter_endpoint
        suffix = "/chat/completions"
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
        base_url = base_url.rstrip("/")

        os.environ.setdefault("OPENROUTER_API_BASE", base_url)
        os.environ.setdefault("OPENAI_BASE_URL", base_url)

    def _build_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self._settings.memory_model_id,
                    "temperature": self._settings.memory_temperature,
                    "max_tokens": self._settings.memory_max_tokens,
                    "openrouter_base_url": os.environ.get("OPENROUTER_API_BASE"),
                },
            }
        }
        if self._settings.memory_site_url and self._settings.memory_app_name:
            cfg["llm"]["config"].update(
                {
                    "site_url": self._settings.memory_site_url,
                    "app_name": self._settings.memory_app_name,
                }
            )
        return cfg

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self._memory is not None)

    def fetch(self, query: str, user_id: str) -> List[str]:
        if not self.is_ready:
            return []
        try:
            result = self._memory.search(query=query, user_id=user_id, limit=self._settings.memory_search_limit)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Mem0 search failed", exc_info=exc)
            return []
        memories: List[str] = []
        for item in result.get("results", []):
            mem = item.get("memory")
            if mem:
                memories.append(str(mem).strip())
        return memories

    def store_messages(self, messages: List[Dict[str, str]], user_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.is_ready:
            return
        payload_meta: Dict[str, Any] = {}
        payload_meta.update(metadata or {})
        if self._settings.memory_category and "category" not in payload_meta:
            payload_meta["category"] = self._settings.memory_category
        try:
            self._memory.add(messages, user_id=user_id, metadata=payload_meta)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Mem0 add failed", exc_info=exc)

    def store_exchange(self, user_message: str, assistant_message: str, user_id: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        self.store_messages(messages, user_id=user_id, metadata=metadata)

    @staticmethod
    def to_prompt_section(memories: List[str]) -> str:
        if not memories:
            return ""
        lines = [f"- {mem}" for mem in memories if mem]
        return "\n".join(lines)
