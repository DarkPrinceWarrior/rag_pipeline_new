import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import type { AskRequest, AskResponse, Citation } from './types'

function useApiBase2() {
  return ''
}

function CitationList2({ citations }: { citations: Citation[] }) {
  if (!citations?.length) return null
  const bySerial = [...citations].sort((a, b) => a.serial - b.serial)
  return (
    <div className="citations">
      <div className="muted small">Источники</div>
      <ul>
        {bySerial.map((c) => (
          <li key={c.chunk_id}>
            <a href={`/docs/${encodeURIComponent(c.filename)}#page=${c.page}`} target="_blank" rel="noreferrer">
              S{c.serial} — {c.filename}, стр. {c.page}
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}

type ChatMessage2 = { role: 'user' | 'assistant' | 'system'; content: string; citations?: Citation[] }

function transformMath(content: string): string {
  // Convert custom :[ ... ] to $$...$$ and :( ... ) to $...$
  // Block-style first (greedy across newlines within brackets)
  let out = content.replace(/:\s*\[(\s*[\s\S]*?\s*)\]/g, (_m, g1) => `$$\n${g1}\n$$`)
  // Inline-style
  out = out.replace(/:\s*\((\s*[\s\S]*?\s*)\)/g, (_m, g1) => `$${g1}$`)
  return out
}

function App2() {
  const apiBase = useApiBase2()
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(() => {
    const saved = localStorage.getItem('top_k')
    return saved ? parseInt(saved) : 100
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage2[]>([])
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editingDraft, setEditingDraft] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const gearRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const bubbleRefs = useRef<Record<number, HTMLDivElement | null>>({})
  const [userMinHeights, setUserMinHeights] = useState<Record<number, number>>({})

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    localStorage.setItem('top_k', String(topK))
  }, [topK])

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!settingsOpen) return
      const t = e.target as Node
      if (panelRef.current && !panelRef.current.contains(t) && gearRef.current && !gearRef.current.contains(t)) {
        setSettingsOpen(false)
      }
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [settingsOpen])

  function copyText(text: string) {
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).catch(() => {})
    }
  }

  function onEditMessage(index: number) {
    const m = messages[index]
    if (!m || m.role !== 'user') return
    const el = bubbleRefs.current[index]
    if (el && el.offsetHeight) {
      setUserMinHeights((prev) => ({ ...prev, [index]: el.offsetHeight }))
    }
    setEditingIndex(index)
    setEditingDraft(m.content)
  }

  function onCancelEdit() {
    setEditingIndex(null)
    setEditingDraft('')
  }

  async function onSaveEdit(index: number) {
    const q = editingDraft.trim()
    if (!q || loading) return
    setLoading(true)
    setError(null)
    setMessages((prev) => {
      const next = [...prev]
      next[index] = { role: 'user', content: q }
      return next.slice(0, index + 1)
    })
    try {
      const req: AskRequest = { query: q, top_k: topK }
      const resp = await fetch(`${apiBase}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      })
      if (!resp.ok) {
        const t = await resp.text()
        throw new Error(`HTTP ${resp.status}: ${t}`)
      }
      const json = (await resp.json()) as AskResponse
      setMessages((prev) => [...prev, { role: 'assistant', content: json.answer, citations: json.citations }])
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
    if (!q || loading) return
    setLoading(true)
    setError(null)
    setMessages((prev) => [...prev, { role: 'user', content: q }])
    setQuery('')
    try {
      const req: AskRequest = { query: q, top_k: topK }
      const resp = await fetch(`${apiBase}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      })
      if (!resp.ok) {
        const t = await resp.text()
        throw new Error(`HTTP ${resp.status}: ${t}`)
      }
      const json = (await resp.json()) as AskResponse
      setMessages((prev) => [...prev, { role: 'assistant', content: json.answer, citations: json.citations }])
    } catch (err: any) {
      setError(err?.message ?? 'Ошибка запроса')
    } finally {
      setLoading(false)
    }
  }

  const isEmpty = messages.length === 0 && !loading && !error

  return (
    <div className="container">
      <main className="chat" aria-live="polite">
        <div className="chat-header">
          <div className="title">Agent RAG</div>
          <div className="actions">
            <button
              ref={gearRef}
              type="button"
              aria-label="Настройки"
              className="icon-button"
              onClick={() => setSettingsOpen((v) => !v)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c0 .66.26 1.3.73 1.77.47.47 1.11.73 1.77.73H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
              </svg>
            </button>
            {settingsOpen && (
              <div ref={panelRef} className="settings-panel" role="dialog" aria-label="Настройки чата">
                <div className="settings-row">
                  <div className="settings-label">top_k</div>
                  <div className="settings-value">{topK}</div>
                </div>
                <input
                  aria-label="top_k"
                  className="slider"
                  type="range"
                  min={10}
                  max={200}
                  step={10}
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                />
              </div>
            )}
          </div>
        </div>

        {isEmpty ? (
          <div className="hero">
            <form className="composer-row large" onSubmit={onAsk}>
              <textarea
                ref={inputRef}
                placeholder="Спросите что-нибудь о руководстве..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    onAsk();
                  }
                }}
              />
              <button disabled={loading || !query.trim()}>{loading ? 'Думаю...' : 'Отправить'}</button>
            </form>
          </div>
        ) : (
          <>
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.role}`}>
                <div
                  className={`bubble ${m.role === 'user' ? 'uncopyable' : ''}`}
                  ref={m.role === 'user' ? (el) => { bubbleRefs.current[i] = el } : undefined}
                  onCopy={m.role === 'user' ? (e) => e.preventDefault() : undefined}
                  onCut={m.role === 'user' ? (e) => e.preventDefault() : undefined}
                  onContextMenu={m.role === 'user' ? (e) => e.preventDefault() : undefined}
                  style={m.role === 'user' && editingIndex === i ? { minHeight: userMinHeights[i] } : undefined}
                >
                  {m.role === 'user' && editingIndex === i ? (
                    <div className="editor">
                      <textarea
                        className="editor-text"
                        value={editingDraft}
                        onChange={(e) => setEditingDraft(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSaveEdit(i) } }}
                        placeholder="Отредактируйте сообщение..."
                      />
                      <div className="editor-actions">
                        <button type="button" disabled={loading || !editingDraft.trim()} onClick={() => onSaveEdit(i)}>Сохранить</button>
                        <button type="button" className="ghost-btn" onClick={onCancelEdit}>Отмена</button>
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
                      <CitationList citations={m.citations || []} />
                    </>
                  ) : (
                    m.content
                  )}
                </div>
                <div className="msg-toolbar">
                  {m.role !== 'user' && (
                    <button
                      type="button"
                      className="msg-toolbar-btn"
                      onClick={() => copyText(m.content)}
                      aria-label="Copy"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                      </svg>
                      <span>Копировать</span>
                    </button>
                  )}
                  {m.role === 'user' && editingIndex !== i && (
                    <button
                      type="button"
                      className="msg-toolbar-btn edit"
                      onClick={() => onEditMessage(i)}
                      aria-label="Edit"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 20h9"></path>
                        <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4Z"></path>
                      </svg>
                      <span>Редактировать</span>
                    </button>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="message system">
                <div className="bubble thinking">Думаю…</div>
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

      {(!isEmpty) && (
        <div className="composer">
          <form className="composer-row" onSubmit={onAsk}>
            <textarea
              ref={inputRef}
              placeholder="Спросите что-нибудь о руководстве..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  onAsk();
                }
              }}
            />
            <div className="actions">
              <button
                ref={gearRef}
                type="button"
                aria-label="Настройки"
                className="icon-button"
                onClick={() => setSettingsOpen((v) => !v)}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/>
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c0 .66.26 1.3.73 1.77.47.47 1.11.73 1.77.73H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                </svg>
              </button>
              {settingsOpen && (
                <div ref={panelRef} className="settings-panel" role="dialog" aria-label="Настройки чата">
                  <div className="settings-row">
                    <div className="settings-label">top_k</div>
                    <div className="settings-value">{topK}</div>
                  </div>
                  <input
                    aria-label="top_k"
                    className="slider"
                    type="range"
                    min={10}
                    max={200}
                    step={10}
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                  />
                </div>
              )}
            </div>
            <button disabled={loading || !query.trim()}>{loading ? 'Думаю...' : 'Отправить'}</button>
          </form>
        </div>
      )}
    </div>
  )
}

function useApiBase() {
  // When served from FastAPI, use relative. During dev, Vite proxy handles it.
  return ''
}

function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations?.length) return null
  const bySerial = [...citations].sort((a, b) => a.serial - b.serial)
  return (
    <div className="citations">
      <div className="muted small">Источники</div>
      <ul>
        {bySerial.map((c) => (
          <li key={c.chunk_id}>
            <a href={`/docs/${encodeURIComponent(c.filename)}#page=${c.page}`} target="_blank" rel="noreferrer">
              S{c.serial} — {c.filename}, стр. {c.page}
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}

type ChatMessage = { role: 'user' | 'assistant' | 'system'; content: string; citations?: Citation[] }

export default function App() {
  const apiBase = useApiBase()
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(() => {
    const saved = localStorage.getItem('top_k')
    return saved ? parseInt(saved) : 100
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editingDraft, setEditingDraft] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const gearRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const bubbleRefs = useRef<Record<number, HTMLDivElement | null>>({})
  const [userMinWidths, setUserMinWidths] = useState<Record<number, number>>({})

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    localStorage.setItem('top_k', String(topK))
  }, [topK])

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!settingsOpen) return
      const t = e.target as Node
      if (panelRef.current && !panelRef.current.contains(t) && gearRef.current && !gearRef.current.contains(t)) {
        setSettingsOpen(false)
      }
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [settingsOpen])

  function onEditMessage(index: number) {
    const m = messages[index]
    if (!m || m.role !== 'user') return
    const el = bubbleRefs.current[index]
    if (el && el.offsetWidth) {
      setUserMinWidths((prev) => ({ ...prev, [index]: el.offsetWidth }))
    }
    setEditingIndex(index)
    setEditingDraft(m.content)
  }

  function onCancelEdit() {
    setEditingIndex(null)
    setEditingDraft('')
  }

  async function onSaveEdit(index: number) {
    const q = editingDraft.trim()
    if (!q || loading) return
    setLoading(true)
    setError(null)
    setMessages((prev) => {
      const next = [...prev]
      next[index] = { role: 'user', content: q }
      return next.slice(0, index + 1)
    })
    try {
      const req: AskRequest = { query: q, top_k: topK }
      const resp = await fetch(`${apiBase}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      })
      if (!resp.ok) {
        const t = await resp.text()
        throw new Error(`HTTP ${resp.status}: ${t}`)
      }
      const json = (await resp.json()) as AskResponse
      setMessages((prev) => [...prev, { role: 'assistant', content: json.answer, citations: json.citations }])
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
    if (!q || loading) return
    setLoading(true)
    setError(null)
    setMessages((prev) => [...prev, { role: 'user', content: q }])
    setQuery('')
    try {
      const req: AskRequest = { query: q, top_k: topK }
      const resp = await fetch(`${apiBase}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      })
      if (!resp.ok) {
        const t = await resp.text()
        throw new Error(`HTTP ${resp.status}: ${t}`)
      }
      const json = (await resp.json()) as AskResponse
      setMessages((prev) => [...prev, { role: 'assistant', content: json.answer, citations: json.citations }])
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

  const isEmpty = messages.length === 0 && !error

  return (
    <div className="container">
      <div className="app-title" role="heading" aria-level={1}>Agent RAG</div>
      <main className="chat" aria-live="polite">
        {isEmpty && (
          <div className="hero">
            <form className="composer-row large" onSubmit={onAsk}>
              <textarea
                ref={inputRef}
                placeholder="Спросите что-нибудь о руководстве..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    onAsk();
                  }
                }}
              />
              <div className="actions">
                <button
                  ref={gearRef}
                  type="button"
                  aria-label="Настройки"
                  className="icon-button"
                  onClick={() => setSettingsOpen((v) => !v)}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c0 .66.26 1.3.73 1.77.47.47 1.11.73 1.77.73H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                  </svg>
                </button>
                {settingsOpen && (
                  <div ref={panelRef} className="settings-panel" role="dialog" aria-label="Настройки чата">
                    <div className="settings-row">
                      <div className="settings-label">top_k</div>
                      <div className="settings-value">{topK}</div>
                    </div>
                    <input
                      aria-label="top_k"
                      className="slider"
                      type="range"
                      min={10}
                      max={200}
                      step={10}
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value))}
                    />
                  </div>
                )}
              </div>
              <button disabled={loading || !query.trim()}>{loading ? 'Думаю...' : 'Отправить'}</button>
            </form>
          </div>
        )}

        {!isEmpty && (
          <>
            {messages.map((m, i) => (
              <div className={`msg-item ${m.role}`} key={i}>
                <div className={`message ${m.role}`}>
                  <div
                    className={`bubble ${m.role === 'assistant' ? 'prose' : ''}`}
                    ref={m.role === 'user' ? (el) => { bubbleRefs.current[i] = el } : undefined}
                    style={m.role === 'user' && editingIndex === i ? { minWidth: userMinWidths[i] } : undefined}
                  >
                    {m.role === 'user' && editingIndex === i ? (
                      <div className="inline-editor">
                        <textarea
                          value={editingDraft}
                          onChange={(e) => setEditingDraft(e.target.value)}
                          placeholder="Измените сообщение..."
                        />
                        <div className="editor-actions">
                          <button type="button" disabled={loading || !editingDraft.trim()} onClick={() => onSaveEdit(i)}>Сохранить</button>
                          <button type="button" className="ghost-btn" onClick={onCancelEdit}>Отмена</button>
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
                        <CitationList2 citations={m.citations || []} />
                      </>
                    ) : (
                      m.content
                    )}
                    {m.role === 'user' && editingIndex !== i && (
                      <button type="button" className="edit-pill" onClick={() => onEditMessage(i)}>Редактировать</button>
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
                      <span>Копировать</span>
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

      {(!isEmpty) && (
        <div className="composer">
          <form className="composer-row" onSubmit={onAsk}>
            <textarea
              ref={inputRef}
              placeholder="Спросите что-нибудь о руководстве..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  onAsk();
                }
              }}
            />
            <div className="actions">
              <button
                ref={gearRef}
                type="button"
                aria-label="Настройки"
                className="icon-button"
                onClick={() => setSettingsOpen((v) => !v)}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/>
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9c0 .66.26 1.3.73 1.77.47.47 1.11.73 1.77.73H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                </svg>
              </button>
              {settingsOpen && (
                <div ref={panelRef} className="settings-panel" role="dialog" aria-label="Настройки чата">
                  <div className="settings-row">
                    <div className="settings-label">top_k</div>
                    <div className="settings-value">{topK}</div>
                  </div>
                  <input
                    aria-label="top_k"
                    className="slider"
                    type="range"
                    min={10}
                    max={200}
                    step={10}
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                  />
                </div>
              )}
            </div>
            <button disabled={loading || !query.trim()}>{loading ? 'Думаю...' : 'Отправить'}</button>
          </form>
        </div>
      )}
    </div>
  )
}
