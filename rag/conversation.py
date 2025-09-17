from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lancedb
import httpx

from .config import settings


@dataclass
class ConversationMessage:
    """Сообщение диалога, хранящееся в LanceDB."""

    id: str
    session_id: str
    user_id: str
    role: str  # user | assistant | system
    content: str
    created_at: float


class ConversationStore:
    """Хранилище кратковременной истории диалога на LanceDB.

    - Персистентно и без костылей: та же БД, что и документы.
    - Два стола: сообщения (`conversations`) и саммари (`conversation_summaries`).
    """

    def __init__(self) -> None:
        self.db = lancedb.connect(settings.lancedb_path)
        self.messages_table_name = getattr(settings, "conversation_table", "conversations")
        self.summary_table_name = getattr(settings, "conversation_summary_table", "conversation_summaries")
        self._messages = self._open_or_none(self.messages_table_name)
        self._summaries = self._open_or_none(self.summary_table_name)

    def _open_or_none(self, table_name: str):
        try:
            if table_name in self.db.table_names():
                return self.db.open_table(table_name)
            return None
        except Exception:
            return None

    def _ensure_messages(self) -> None:
        if self._messages is None:
            sample = [
                {
                    "id": str(uuid.uuid4()),
                    "session_id": "s",
                    "user_id": "u",
                    "role": "user",
                    "content": "",
                    "created_at": float(time.time()),
                }
            ]
            self._messages = self.db.create_table(self.messages_table_name, data=sample, mode="overwrite")

    def _ensure_summaries(self) -> None:
        if self._summaries is None:
            sample = [
                {
                    "session_id": "s",
                    "summary": "",
                    "updated_at": float(time.time()),
                }
            ]
            self._summaries = self.db.create_table(self.summary_table_name, data=sample, mode="overwrite")

    def append(self, session_id: str, user_id: str, role: str, content: str) -> None:
        if not settings.conversation_enabled:
            return
        self._ensure_messages()
        try:
            payload = {
                "id": str(uuid.uuid4()),
                "session_id": str(session_id),
                "user_id": str(user_id),
                "role": str(role),
                "content": str(content),
                "created_at": float(time.time()),
            }
            self._messages.add([payload])
        except Exception:
            pass

    def fetch_recent(self, session_id: str, limit: int) -> List[ConversationMessage]:
        if not settings.conversation_enabled or self._messages is None:
            return []
        rows: List[Dict] = []
        try:
            # Предпочитаем табличное API без векторного поиска
            df = self._messages.to_pandas()
            if df is not None and not df.empty:
                sub = df[df["session_id"].astype(str) == str(session_id)]
                if not sub.empty:
                    sub = sub.sort_values(by=["created_at"], ascending=True)
                    rows = sub.tail(limit).to_dict(orient="records")  # type: ignore
        except Exception:
            try:
                # Fallback: полное чтение с последующей фильтрацией (при малых объёмах ок)
                all_rows = self._messages.to_list()  # type: ignore
                rows = [r for r in all_rows if str(r.get("session_id")) == str(session_id)]
                rows.sort(key=lambda r: float(r.get("created_at", 0.0)))
                rows = rows[-limit:]
            except Exception:
                rows = []
        if not rows:
            return []
        rows.sort(key=lambda r: float(r.get("created_at", 0.0)))
        messages: List[ConversationMessage] = []
        for r in rows:
            try:
                messages.append(
                    ConversationMessage(
                        id=str(r.get("id")),
                        session_id=str(r.get("session_id")),
                        user_id=str(r.get("user_id")),
                        role=str(r.get("role")),
                        content=str(r.get("content")),
                        created_at=float(r.get("created_at")),
                    )
                )
            except Exception:
                continue
        return messages[-limit:]

    def get_summary(self, session_id: str) -> Optional[str]:
        if not settings.conversation_enabled or self._summaries is None:
            return None
        try:
            df = self._summaries.to_pandas()
            if df is not None and not df.empty:
                sub = df[df["session_id"].astype(str) == str(session_id)]
                if not sub.empty:
                    sub = sub.sort_values(by=["updated_at"], ascending=False)
                    row = sub.iloc[0].to_dict()  # type: ignore
                    return str(row.get("summary") or "").strip() or None
        except Exception:
            try:
                rows = self._summaries.to_list()  # type: ignore
                rows = [r for r in rows if str(r.get("session_id")) == str(session_id)]
                rows.sort(key=lambda r: float(r.get("updated_at", 0.0)), reverse=True)
                if rows:
                    return str(rows[0].get("summary") or "").strip() or None
            except Exception:
                return None
        return None

    def upsert_summary(self, session_id: str, summary: str) -> None:
        if not settings.conversation_enabled:
            return
        self._ensure_summaries()
        try:
            self._summaries.delete(f"session_id == '{session_id}'")
        except Exception:
            pass
        try:
            self._summaries.add([
                {
                    "session_id": str(session_id),
                    "summary": str(summary),
                    "updated_at": float(time.time()),
                }
            ])
        except Exception:
            pass

    @staticmethod
    def to_prompt_section(messages: List[ConversationMessage]) -> str:
        """Сериализация сообщений в компактный блок промпта."""
        if not messages:
            return ""
        lines: List[str] = []
        for m in messages:
            role = m.role.strip().lower()
            prefix = "U" if role == "user" else ("A" if role == "assistant" else "S")
            # Минимизируем размер промпта, сохраняя семантику
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)


class ConversationSummarizer:
    """Обновляет краткое саммари диалога через OpenRouter.

    Используется редко (например, каждые N обменов), чтобы не тратить токены на каждую итерацию.
    """

    @staticmethod
    def should_update(turn_index: int) -> bool:
        n = max(1, int(getattr(settings, "conversation_summary_every_n_turns", 4)))
        return (turn_index % n) == 0

    @staticmethod
    def build_prompt(previous_summary: Optional[str], recent_history: str) -> str:
        summary_block = f"  <previous_summary>\n{previous_summary}\n  </previous_summary>\n" if previous_summary else ""
        return (
            "<prompt>\n"
            "  <task>Сформируй краткое саммари диалога для последующего использования в качестве контекста беседы.</task>\n"
            "  <constraints>\n"
            "    - Пиши по-русски.\n"
            "    - Сожми информацию, сохрани факты, имена, решения и открытые вопросы.\n"
            "    - Формат: 3–7 пунктов, без воды.\n"
            "  </constraints>\n"
            f"{summary_block}"
            f"  <recent_history>\n{recent_history}\n  </recent_history>\n"
            "</prompt>"
        )

    @staticmethod
    def summarize(previous_summary: Optional[str], recent_messages: List[ConversationMessage]) -> Optional[str]:
        if not recent_messages:
            return previous_summary
        recent_history = ConversationStore.to_prompt_section(recent_messages)
        prompt = ConversationSummarizer.build_prompt(previous_summary, recent_history)
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": "Conversation-Summarizer",
        }
        body = {
            "model": getattr(settings, "conversation_summary_model_id", settings.llm_model_id),
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": int(getattr(settings, "conversation_summary_max_tokens", 256)),
            "messages": [
                {"role": "system", "content": "You are a precise note-taking assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            with httpx.Client(timeout=httpx.Timeout(40.0)) as client:
                resp = client.post(settings.openrouter_endpoint, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                return str(data["choices"][0]["message"]["content"]).strip()
        except Exception:
            return previous_summary


