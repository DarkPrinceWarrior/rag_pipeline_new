(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const messagesEl = $('#messages');
  const inputEl = $('#input');
  const formEl = $('#composer');
  const sendBtn = $('#send');
  const topkEl = $('#topk');
  const topkValueEl = $('#topk-value');
  const clearBtn = $('#clear-btn');

  function scrollToBottom() {
    try {
      if (messagesEl && typeof messagesEl.scrollTo === 'function') {
        messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
      } else if (messagesEl) {
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    } catch (_) {
      if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  const state = { msgs: [] };

  const template = $('#msg-template');
  function renderMarkdownToHtml(markdownText) {
    const md = (window.marked && window.marked.parse) ? window.marked.parse(markdownText) : markdownText;
    const clean = (window.DOMPurify && window.DOMPurify.sanitize) ? window.DOMPurify.sanitize(md, { USE_PROFILES: { html: true } }) : md;
    return clean;
  }

  function highlightCodeBlocks(root) {
    try {
      if (!window.hljs) return;
      $$("pre code", root).forEach((el) => {
        window.hljs.highlightElement(el);
      });
    } catch (_) {}
  }

  function createMsg(role, content, meta = '', sources = []) {
    const node = template.content.firstElementChild.cloneNode(true);
    node.classList.add(role);
    const contentEl = $('.content', node);
    if (role === 'assistant') {
      contentEl.innerHTML = renderMarkdownToHtml(content);
      highlightCodeBlocks(contentEl);
    } else {
      contentEl.textContent = content;
    }
    $('.meta', node).textContent = meta;
    const sourcesEl = $('.sources', node);
    sources.forEach((s) => {
      const a = document.createElement('a');
      a.className = 'source-chip';
      a.href = `/docs/${encodeURIComponent(s.filename)}#page=${encodeURIComponent(s.page)}`;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.title = `${s.filename}, страница ${s.page}`;
      a.textContent = `S#${s.serial} ${s.filename}, стр.${s.page}`;
      sourcesEl.appendChild(a);
    });
    const copyBtn = $('.copy', node);
    if (copyBtn) {
      copyBtn.classList.add('action');
      copyBtn.textContent = 'Копировать';
      copyBtn.title = 'Копировать';
      copyBtn.addEventListener('click', () => {
        const el = $('.content', node);
        let text = '';
        if (el) {
          text = (el.innerText || el.textContent || '').trim();
        }
        navigator.clipboard.writeText(text).catch(() => {});
      });
    }
    return node;
  }

  function typingMsg() {
    const node = template.content.firstElementChild.cloneNode(true);
    node.classList.add('assistant');
    const c = $('.content', node);
    c.innerHTML = '<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span> Думаю...</span>';
    const metaRow = $('.meta-row', node);
    if (metaRow) metaRow.style.display = 'none';
    return node;
  }

  function push(role, payload) {
    const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const node = createMsg(role, payload.content, ts, payload.sources || []);
    messagesEl.appendChild(node);
    const msg = { role, content: payload.content, node, edited: false };
    state.msgs.push(msg);
    const index = state.msgs.length - 1;
    attachActions(index);
    scrollToBottom();
  }

  function attachActions(index) {
    const msg = state.msgs[index];
    if (!msg) return;
    const node = msg.node;
    if (msg.role === 'user') {
      const actions = $('.actions', node);
      const editBtn = document.createElement('button');
      editBtn.className = 'edit action ghost';
      editBtn.textContent = 'Редактировать';
      editBtn.title = 'Редактировать сообщение';
      actions.appendChild(editBtn);
      editBtn.addEventListener('click', () => beginEdit(index));
      const contentEl = $('.content', node);
      if (contentEl) contentEl.addEventListener('dblclick', () => beginEdit(index));
    }
  }

  function setMetaEdited(node) {
    const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const meta = $('.meta', node);
    if (meta) meta.textContent = `${ts} — изменено`;
  }

  function beginEdit(index) {
    const msg = state.msgs[index];
    if (!msg || msg.role !== 'user') return;
    const node = msg.node;
    if (node.classList.contains('editing')) return;
    node.classList.add('editing');

    const bubble = $('.bubble', node);
    const current = msg.content;
    const editWrap = document.createElement('div');
    editWrap.className = 'edit-area';
    const textarea = document.createElement('textarea');
    textarea.className = 'edit-box';
    textarea.value = current;
    editWrap.appendChild(textarea);

    const actions = $('.actions', node);
    const copyBtn = $('.copy', node);
    const editBtn = $('.edit', node);
    if (copyBtn) copyBtn.style.display = 'none';
    if (editBtn) editBtn.style.display = 'none';

    const saveBtn = document.createElement('button');
    saveBtn.className = 'btn-save action';
    saveBtn.textContent = 'Сохранить';
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn-cancel ghost action';
    cancelBtn.textContent = 'Отмена';
    const editActions = document.createElement('div');
    editActions.className = 'edit-actions';
    editActions.appendChild(saveBtn);
    editActions.appendChild(cancelBtn);
    editWrap.appendChild(editActions);
    bubble.appendChild(editWrap);

    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.preventDefault(); cancelBtn.click(); }
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveBtn.click(); }
    });

    cancelBtn.addEventListener('click', () => {
      bubble.removeChild(editWrap);
      if (actions) {
        actions.removeChild(saveBtn);
        actions.removeChild(cancelBtn);
      }
      if (copyBtn) copyBtn.style.display = '';
      if (editBtn) editBtn.style.display = '';
      node.classList.remove('editing');
    });

    saveBtn.addEventListener('click', async () => {
      const newText = textarea.value.trim();
      if (!newText) return;
      if (newText === msg.content) { cancelBtn.click(); return; }
      msg.content = newText;
      const contentEl = $('.content', node);
      if (contentEl) contentEl.textContent = newText;
      msg.edited = true;
      setMetaEdited(node);

      // Remove messages after the edited user message
      for (let i = state.msgs.length - 1; i > index; i--) {
        state.msgs[i].node.remove();
        state.msgs.pop();
      }

      inputEl.disabled = true; sendBtn.disabled = true;
      const typing = typingMsg();
      messagesEl.appendChild(typing);
      scrollToBottom();
      try {
        const min = parseInt(topkEl.min || '1', 10) || 1;
        const max = parseInt(topkEl.max || '100', 10) || 100;
        const fallback = parseInt(topkEl.value || '20', 10) || 20;
        const topK = Math.min(max, Math.max(min, parseInt(topkEl.value, 10) || fallback));
        const res = await fetch('/ask', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: newText, top_k: topK })
        });
        if (!res.ok) throw new Error('Request failed');
        const data = await res.json();
        typing.remove();
        push('assistant', { content: data.answer, sources: data.citations || [] });
      } catch (err) {
        typing.remove();
        push('assistant', { content: 'Ошибка запроса. Проверьте сервер и .env.' });
      } finally {
        inputEl.disabled = false; sendBtn.disabled = false;
      }

      bubble.removeChild(editWrap);
      if (actions) {
        actions.removeChild(saveBtn);
        actions.removeChild(cancelBtn);
      }
      if (copyBtn) copyBtn.style.display = '';
      if (editBtn) editBtn.style.display = '';
      node.classList.remove('editing');
    });

    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  }

  function resetChat() {
    state.msgs = [];
    messagesEl.innerHTML = '';
    inputEl.value = '';
    inputEl.focus();
  }
  if (clearBtn) clearBtn.addEventListener('click', resetChat);

  function autosize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(160, inputEl.scrollHeight) + 'px';
  }
  inputEl.addEventListener('input', autosize);
  autosize();

  // Инициализация и сохранение Top-K (слайдер)
  (function initTopK() {
    try {
      const saved = window.localStorage.getItem('topk');
      const min = parseInt(topkEl.min || '1', 10) || 1;
      const max = parseInt(topkEl.max || '100', 10) || 100;
      const fallback = parseInt(topkEl.value || '20', 10) || 20;
      let value = fallback;
      if (saved != null) {
        const n = parseInt(saved, 10);
        if (!Number.isNaN(n)) value = Math.min(max, Math.max(min, n));
      }
      topkEl.value = String(value);
      if (topkValueEl) topkValueEl.textContent = String(value);
      topkEl.addEventListener('input', () => {
        const v = parseInt(topkEl.value, 10) || fallback;
        if (topkValueEl) topkValueEl.textContent = String(v);
        try { window.localStorage.setItem('topk', String(v)); } catch (_) {}
      });
    } catch (_) {}
  })();

  formEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;

    const min = parseInt(topkEl.min || '1', 10) || 1;
    const max = parseInt(topkEl.max || '100', 10) || 100;
    const fallback = parseInt(topkEl.value || '20', 10) || 20;
    const topK = Math.min(max, Math.max(min, parseInt(topkEl.value, 10) || fallback));
    push('user', { content: text });

    inputEl.value = '';
    autosize();
    inputEl.disabled = true;
    sendBtn.disabled = true;

    const typing = typingMsg();
    messagesEl.appendChild(typing);
    scrollToBottom();

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, top_k: topK })
      });
      if (!res.ok) throw new Error('Request failed');
      const data = await res.json();

      typing.remove();
      push('assistant', { content: data.answer, sources: data.citations || [] });
    } catch (err) {
      typing.remove();
      push('assistant', { content: 'Ошибка запроса. Проверьте сервер и .env.' });
    } finally {
      inputEl.disabled = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  });

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      formEl.requestSubmit();
    }
  });
})();

