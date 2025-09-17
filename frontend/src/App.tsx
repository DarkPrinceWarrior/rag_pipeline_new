import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import type { AskRequest, AskResponse, Citation } from './types'

function useAutosize(ref: React.RefObject<HTMLTextAreaElement>, value: string) {
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = el.scrollHeight + 'px'
  }, [ref, value])
}

const CHAT_STORAGE_KEY = 'rag_chats_v1'
const CURRENT_CHAT_KEY = 'rag_current_chat_id'
const DEFAULT_CHAT_TITLE = 'Новый чат'
const MAX_TITLE_LENGTH = 60

type ChatMessage = { role: 'user' | 'assistant' | 'system'; content: string; citations?: Citation[] }
type ChatSession = { id: string; title: string; messages: ChatMessage[]; createdAt: number; updatedAt: number }

function transformMath(content: string): string {
  let out = content.replace(/:\s*\[(\s*[\s\S]*?\s*)\]/g, (_m, g1) => `$$\n${g1}\n$$`)
  out = out.replace(/:\s*\((\s*[\s\S]*?\s*)\)/g, (_m, g1) => `$${g1}$`)
  return out
}

function createId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `chat-${Date.now().toString(36)}-${Math.random().toString(16).slice(2)}`
}

function deriveTitle(text: string): string {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (!normalized) return DEFAULT_CHAT_TITLE
  if (normalized.length <= MAX_TITLE_LENGTH) return normalized
  return `${normalized.slice(0, MAX_TITLE_LENGTH - 1).trimEnd()}…`
}

function createEmptyChat(): ChatSession {
  const now = Date.now()
  return { id: createId(), title: DEFAULT_CHAT_TITLE, messages: [], createdAt: now, updatedAt: now }
}

function normalizeMessage(value: unknown): ChatMessage | null {
  if (!value || typeof value !== 'object') return null
  const maybe = value as Partial<ChatMessage>
  if (maybe.role !== 'user' && maybe.role !== 'assistant' && maybe.role !== 'system') return null
  if (typeof maybe.content !== 'string') return null
  const msg: ChatMessage = { role: maybe.role, content: maybe.content }
  if (maybe.role === 'assistant' && Array.isArray(maybe.citations)) {
    msg.citations = maybe.citations.filter((c): c is Citation => !!c && typeof (c as Citation).chunk_id === 'string')
  }
  return msg
}

function normalizeChat(value: unknown): ChatSession | null {
  if (!value || typeof value !== 'object') return null
  const maybe = value as Partial<ChatSession> & { messages?: unknown[] }
  const id = typeof maybe.id === 'string' && maybe.id ? maybe.id : createId()
  const rawMessages = Array.isArray(maybe.messages) ? maybe.messages : []
  const messages = rawMessages.map(normalizeMessage).filter((v): v is ChatMessage => v != null)
  const createdAt = typeof maybe.createdAt === 'number' ? maybe.createdAt : Date.now()
  const updatedAt = typeof maybe.updatedAt === 'number' ? maybe.updatedAt : createdAt
  const titleSource = typeof maybe.title === 'string' ? maybe.title.trim() : ''
  const title = titleSource || (messages[0] ? deriveTitle(messages[0].content) : DEFAULT_CHAT_TITLE)
  return { id, title, messages, createdAt, updatedAt }
}

function loadInitialData(): { chats: ChatSession[]; currentId: string } {
  if (typeof window !== 'undefined') {
    try {
      const raw = window.localStorage.getItem(CHAT_STORAGE_KEY)
      if (raw) {
        const parsed = JSON.parse(raw)
        const chats = Array.isArray(parsed) ? parsed.map(normalizeChat).filter((c): c is ChatSession => c != null) : []
        if (chats.length > 0) {
          const savedId = window.localStorage.getItem(CURRENT_CHAT_KEY)
          const currentId = savedId && chats.some((c) => c.id === savedId) ? savedId : chats[0].id
          return { chats, currentId }
        }
      }
    } catch (err) {
      console.warn('Failed to load stored chats', err)
    }
  }
  const fallback = createEmptyChat()
  return { chats: [fallback], currentId: fallback.id }
}

function useApiBase() {
  return ''
}

export default function App() {
  const initialDataRef = useRef(loadInitialData())
  const [chats, setChats] = useState<ChatSession[]>(initialDataRef.current.chats)
  const [currentChatId, setCurrentChatId] = useState(initialDataRef.current.currentId)
  const apiBase = useApiBase()
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(() => {
    const saved = typeof window !== 'undefined' ? window.localStorage.getItem('top_k') : null
    return saved ? parseInt(saved, 10) : 200
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editingDraft, setEditingDraft] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const editRef = useRef<HTMLTextAreaElement>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const gearRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const bubbleRefs = useRef<Record<number, HTMLDivElement | null>>({})
  const [userMinWidths, setUserMinWidths] = useState<Record<number, number>>({})
  const [webSearch, setWebSearch] = useState(() => {
    const saved = typeof window !== 'undefined' ? window.localStorage.getItem('web_search') : null
    return saved ? saved === 'true' : false
  })

  const currentChat = chats.find((chat) => chat.id === currentChatId)
  const messages = currentChat?.messages ?? []

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  useAutosize(inputRef, query)
  useAutosize(editRef, editingDraft)

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('top_k', String(topK))
  }, [topK])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('web_search', String(webSearch))
  }, [webSearch])

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!settingsOpen) return
      const target = e.target as Node
      if (panelRef.current && !panelRef.current.contains(target) && gearRef.current && !gearRef.current.contains(target)) {
        setSettingsOpen(false)
      }
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [settingsOpen])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chats))
  }, [chats])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (currentChatId) {
      window.localStorage.setItem(CURRENT_CHAT_KEY, currentChatId)
    }
  }, [currentChatId])

  useEffect(() => {
    if (!currentChat && chats.length > 0) {
      setCurrentChatId(chats[0].id)
    }
  }, [currentChat, chats])

  useEffect(() => {
    bubbleRefs.current = {}
    setUserMinWidths({})
    setEditingIndex(null)
    setEditingDraft('')
    setSettingsOpen(false)
    setError(null)
    setQuery('')
  }, [currentChatId])

  function onSelectChat(id: string) {
    if (id === currentChatId) return
    setCurrentChatId(id)
  }

  function onCreateChat() {
    const next = createEmptyChat()
    setChats((prev) => [...prev, next])
    setCurrentChatId(next.id)
  }

  function onEditMessage(index: number) {
    if (!messages[index] || messages[index].role !== 'user') return
    const el = bubbleRefs.current[index]
    if (el && el.offsetWidth) {
      setUserMinWidths((prev) => ({ ...prev, [index]: el.offsetWidth }))
    }
    setEditingIndex(index)
    setEditingDraft(messages[index].content)
  }

  function onCancelEdit() {
    setEditingIndex(null)
    setEditingDraft('')
  }

  async function onSaveEdit(index: number) {
    const q = editingDraft.trim()
    if (!q || loading || !currentChat) return
    const activeChatId = currentChat.id
    setLoading(true)
    setError(null)
    setChats((prev) =>
      prev.map((chat) => {
        if (chat.id !== activeChatId) return chat
        const nextMessages = [...chat.messages]
        if (!nextMessages[index] || nextMessages[index].role !== 'user') return chat
        nextMessages[index] = { role: 'user', content: q }
        const trimmed = nextMessages.slice(0, index + 1)
        const now = Date.now()
        const title = index === 0 ? deriveTitle(q) : chat.title
        return { ...chat, messages: trimmed, title, updatedAt: now }
      })
    )
    try {
      const req: AskRequest = { query: q, top_k: topK, web_search: webSearch }
      const resp = await fetch(`${apiBase}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      })
      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(`HTTP ${resp.status}: ${text}`)
      }
      const json = (await resp.json()) as AskResponse
      setChats((prev) =>
        prev.map((chat) => {
          if (chat.id !== activeChatId) return chat
          const now = Date.now()
          return {
            ...chat,
            messages: [...chat.messages, { role: 'assistant', content: json.answer, citations: json.citations }],
            updatedAt: now
          }
        })
      )
    } catch (err: any) {
      setError(err?.message ?? 'Ошибка запроса')
    } finally {
      setLoading(false)
      setEditingIndex(null)
      setEditingDraft('')
    }
  }

  async function onAsk(e?: React.FormEvent) {
    e?.preventDefault()
    const q = query.trim()
    if (!q || loading || !currentChat) return
    const activeChatId = currentChat.id
    setLoading(true)
    setError(null)
    setChats((prev) =>
      prev.map((chat) => {
        if (chat.id !== activeChatId) return chat
        const now = Date.now()
        const nextMessages = [...chat.messages, { role: 'user', content: q }]
        const title = chat.messages.length === 0 ? deriveTitle(q) : chat.title
        return { ...chat, messages: nextMessages, title, updatedAt: now }
      })
    )
    setQuery('')
    try {
      const req: AskRequest = { query: q, top_k: topK, web_search: webSearch }
      const resp = await fetch(`${apiBase}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      })
      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(`HTTP ${resp.status}: ${text}`)
      }
      const json = (await resp.json()) as AskResponse
      setChats((prev) =>
        prev.map((chat) => {
          if (chat.id !== activeChatId) return chat
          const now = Date.now()
          return {
            ...chat,
            messages: [...chat.messages, { role: 'assistant', content: json.answer, citations: json.citations }],
            updatedAt: now
          }
        })
      )
    } catch (err: any) {
      setError(err?.message ?? 'Ошибка запроса')
    } finally {
      setLoading(false)
    }
  }

  function copyText(text: string) {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      navigator.clipboard.writeText(text).catch(() => {})
    }
  }

  const isEmpty = messages.length === 0 && !loading && !error

  const composerForm = (variant: 'hero' | 'sticky') => (
    <form className={`composer-row${variant === 'hero' ? ' large' : ''}`} onSubmit={onAsk}>
      <textarea
        ref={inputRef}
        placeholder="Сформулируйте запрос..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            onAsk()
          }
        }}
      />
      <div className="actions">
        <button
          ref={gearRef}
          type="button"
          aria-label="Параметры"
          className="icon-button"
          onClick={() => setSettingsOpen((v) => !v)}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c0 .66.26 1.3.73 1.77.47.47 1.11.73 1.77.73H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
        {settingsOpen && (
          <div ref={panelRef} className="settings-panel" role="dialog" aria-label="Параметры запроса">
            <div className="settings-row">
              <div className="settings-label">Веб-поиск</div>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={webSearch}
                  onChange={(e) => setWebSearch(e.target.checked)}
                />
                <span className="slider-switch"></span>
              </label>
            </div>
            <div className="settings-row">
              <div className="settings-label">top_k</div>
              <div className="settings-value">{topK}</div>
            </div>
            <input
              aria-label="top_k"
              className="slider"
              type="range"
              min={1}
              max={200}
              step={1}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value, 10))}
            />
          </div>
        )}
      </div>
      <button disabled={loading || !query.trim()}>{loading ? 'Думаю…' : 'Спросить'}</button>
    </form>
  )

  return (
    <div className="page">
      <div className="chat-tabs" role="navigation" aria-label="История чатов">
        <div className="chat-tabs-list">
          {chats.map((chat) => (
            <button
              key={chat.id}
              type="button"
              className={`chat-tab${chat.id === currentChatId ? ' active' : ''}`}
              onClick={() => onSelectChat(chat.id)}
              title={chat.title}
            >
              {chat.title}
            </button>
          ))}
        </div>
        <button type="button" className="chat-tab new" onClick={onCreateChat}>
          + Новый чат
        </button>
      </div>

      <div className="container">
        <div className="app-title" role="heading" aria-level={1}>Agent RAG</div>
        <main className="chat" aria-live="polite">
          {isEmpty ? (
            <div className="hero">
              {composerForm('hero')}
            </div>
          ) : (
            <>
              {messages.map((m, i) => (
                <div className={`msg-item ${m.role}`} key={i}>
                  <div className={`message ${m.role}`}>
                    <div
                      className={`bubble ${m.role === 'assistant' ? 'prose' : ''} ${m.role === 'user' ? 'uncopyable' : ''}`}
                      ref={m.role === 'user' ? (el) => { bubbleRefs.current[i] = el } : undefined}
                      onCopy={m.role === 'user' ? (ev) => ev.preventDefault() : undefined}
                      onCut={m.role === 'user' ? (ev) => ev.preventDefault() : undefined}
                      onContextMenu={m.role === 'user' ? (ev) => ev.preventDefault() : undefined}
                      style={m.role === 'user' && editingIndex === i ? { minWidth: userMinWidths[i] } : undefined}
                    >
                      {m.role === 'user' && editingIndex === i ? (
                        <div className="inline-editor">
                          <textarea
                            ref={editRef}
                            value={editingDraft}
                            onChange={(e) => setEditingDraft(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault()
                                onSaveEdit(i)
                              }
                            }}
                            placeholder="Уточните запрос..."
                          />
                          <div className="editor-actions">
                            <button type="button" disabled={loading || !editingDraft.trim()} onClick={() => onSaveEdit(i)}>
                              Обновить
                            </button>
                            <button type="button" className="ghost-btn" onClick={onCancelEdit}>
                              Отмена
                            </button>
                          </div>
                        </div>
                      ) : m.role === 'assistant' ? (
                        <>
                          <ReactMarkdown
                            className="prose"
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                          >
                            {transformMath(m.content)}
                          </ReactMarkdown>
                        </>
                      ) : (
                        m.content
                      )}
                      {m.role === 'user' && editingIndex !== i && (
                        <button type="button" className="edit-pill" onClick={() => onEditMessage(i)}>
                          Редактировать
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="msg-toolbar">
                    {m.role !== 'user' && (
                      <button
                        type="button"
                        className="msg-toolbar-btn"
                        onClick={() => copyText(m.content)}
                        aria-label="Копировать"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <span>Скопировать</span>
                      </button>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="message system">
                  <div className="bubble thinking">Думаю</div>
                </div>
              )}
              {error && (
                <div className="message system">
                  <div className="bubble error">Ошибка: {error}</div>
                </div>
              )}
            </>
          )}
        </main>

        {!isEmpty && (
          <div className="composer">
            {composerForm('sticky')}
          </div>
        )}
      </div>
    </div>
  )
}
