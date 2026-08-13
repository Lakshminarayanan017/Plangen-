/* Plangen UI — wired to the PlanGen API on the same origin.
 *
 * Flow: create a session, feed the brief to step 1, and keep answering the
 * parser's questions until it reports `success`. Then run the pipeline and poll
 * until the sheets come back. This is the same flow as before; only the DOM
 * this file drives has changed (the chat-glass-window design has no results
 * UI of its own, so the attachment cards below are what carries that over).
 */

const API = '/api/v1';

document.addEventListener('DOMContentLoaded', () => {
  // ── Small UI flourishes ───────────────────────────────────────────
  document.querySelectorAll('.btn-send').forEach((btn) => {
    btn.addEventListener('click', () => {
      btn.style.transform = 'scale(0.95)';
      setTimeout(() => (btn.style.transform = 'scale(1)'), 150);
    });
  });

  document.querySelectorAll('nav.main-nav a').forEach((link) => {
    link.addEventListener('mouseenter', () => (link.style.letterSpacing = '1.5px'));
    link.addEventListener('mouseleave', () => (link.style.letterSpacing = '1px'));
  });

  const sendBtn = document.querySelector('.btn-send');
  const searchInput = document.querySelector('.search-input');
  const chatGlassWindow = document.getElementById('chat-glass-window');
  const chatBody = document.getElementById('chat-body');

  // The landing page has no chat — nothing further to wire there.
  if (!sendBtn || !searchInput || !chatGlassWindow || !chatBody) return;

  let sessionId = null;
  let busy = false;

  // ── Chat plumbing ───────────────────────────────────────────────

  const ICON_USER =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
  const ICON_AI =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>';

  const escapeHtml = (s) =>
    String(s).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));

  const clockNow = () =>
    new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  function openGlassWindow() {
    chatGlassWindow.classList.add('active');
  }

  function appendMessage(role, innerHtml) {
    const msg = document.createElement('div');
    msg.className = `chat-msg ${role}`;
    msg.style.opacity = '0';
    msg.style.transform = 'translateY(10px)';
    msg.style.transition = 'all 0.3s ease';
    msg.innerHTML = `
      <div class="msg-icon">${role === 'user' ? ICON_USER : ICON_AI}</div>
      <div class="msg-content">${innerHtml}</div>
    `;
    chatBody.appendChild(msg);
    requestAnimationFrame(() => {
      msg.style.opacity = '1';
      msg.style.transform = 'translateY(0)';
      chatBody.scrollTop = chatBody.scrollHeight;
    });
    return msg.querySelector('.msg-content');
  }

  function setContent(node, innerHtml, withTime = true) {
    node.innerHTML = withTime
      ? `${innerHtml}<span class="time">${clockNow()}</span>`
      : innerHtml;
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  // ── API ─────────────────────────────────────────────────────────

  async function api(path, options = {}) {
    const res = await fetch(API + path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`${res.status} ${res.statusText}${detail ? ` — ${detail}` : ''}`);
    }
    return res.json();
  }

  async function ensureSession() {
    if (sessionId) return sessionId;
    const data = await api('/sessions', { method: 'POST' });
    sessionId = data.session_id;
    return sessionId;
  }

  // ── Step 1: understand the brief ────────────────────────────────

  async function handleSubmission() {
    const query = searchInput.value.trim();
    if (!query || busy) return;

    busy = true;
    searchInput.value = '';
    openGlassWindow();
    appendMessage('user', `${escapeHtml(query)}<span class="time">${clockNow()} &#10003;&#10003;</span>`);

    const reply = appendMessage('assistant', 'Reading your brief&hellip;');

    try {
      const sid = await ensureSession();
      const result = await api('/parse/text', {
        method: 'POST',
        body: JSON.stringify({ session_id: sid, text: query }),
      });

      switch (result.status) {
        case 'success':
          setContent(
            reply,
            escapeHtml(
              result.clarification_prompt ||
                'Requirements captured. Generating your floor plan now.',
            ),
          );
          await runPipeline(sid);
          break;

        // The parser still needs something before it can hand off.
        case 'interactive':
          setContent(reply, escapeHtml(result.question || 'Could you tell me a little more?'));
          break;

        case 'incomplete': {
          const missing = (result.missing_fields || []).join(', ');
          setContent(
            reply,
            escapeHtml(result.clarification_prompt || 'I need a few more details.') +
              (missing ? `<span class="time">Still needed: ${escapeHtml(missing)}</span>` : ''),
            !missing,
          );
          break;
        }

        default:
          setContent(reply, escapeHtml(result.message || 'Something went wrong reading that.'));
      }
    } catch (err) {
      setContent(reply, `Could not reach the PlanGen engine.<br><small>${escapeHtml(err.message)}</small>`);
    } finally {
      busy = false;
      searchInput.focus();
    }
  }

  // ── Steps 2-5: run the pipeline and stream progress ─────────────

  async function runPipeline(sid) {
    const progress = appendMessage('assistant', 'Starting the pipeline&hellip;');

    const { run_id: runId } = await api('/pipeline/run', {
      method: 'POST',
      body: JSON.stringify({ session_id: sid, options: {} }),
    });

    const started = Date.now();

    while (true) {
      await new Promise((r) => setTimeout(r, 1200));

      let state;
      try {
        state = await api(`/pipeline/status/${runId}`);
      } catch (err) {
        setContent(progress, `Lost contact with the run.<br><small>${escapeHtml(err.message)}</small>`);
        return;
      }

      const recent = (state.logs || []).slice(-4).map(escapeHtml).join('<br>');
      const elapsed = Math.round((Date.now() - started) / 1000);

      if (state.status === 'running') {
        setContent(
          progress,
          `<strong>Step ${state.step || 1} of 5</strong><br>${recent || 'Working&hellip;'}` +
            `<span class="time">${elapsed}s elapsed</span>`,
          false,
        );
        continue;
      }

      if (state.status === 'error') {
        setContent(progress, `The run failed.<br><small>${escapeHtml(state.error || 'unknown error')}</small>`);
        return;
      }

      setContent(progress, `<strong>Done in ${elapsed}s.</strong><br>${recent}`, false);
      renderResult(runId, state.result || {});
      return;
    }
  }

  function renderResult(runId, result) {
    const files = result.svg_files || [];
    const summary = result.step4?.summary || {};

    const facts = [
      summary.total_rooms_placed != null ? `${summary.total_rooms_placed} rooms` : null,
      summary.total_area_sqft != null ? `${summary.total_area_sqft} sq ft` : null,
      summary.floors != null ? `${summary.floors} floor` + (summary.floors === 1 ? '' : 's') : null,
    ].filter(Boolean);

    const sheets = files
      .map((name) => {
        const url = `${API}/runs/${runId}/svg/${encodeURIComponent(name)}`;
        return `
          <a class="attachment" href="${url}" target="_blank" rel="noopener">
            <img class="sheet-preview" src="${url}" alt="${escapeHtml(name)}">
            <div class="attachment-info">
              <span class="attachment-name">${escapeHtml(name)}</span>
              <span class="attachment-size">Open full sheet &rarr;</span>
            </div>
          </a>`;
      })
      .join('');

    appendMessage(
      'assistant',
      `Here is your floor plan${facts.length ? ` — ${escapeHtml(facts.join(' · '))}` : ''}.` +
        (sheets || '<br><small>The run finished but produced no sheets.</small>') +
        `<span class="time">Run ${escapeHtml(runId)}</span>`,
    );
  }

  // ── Events ──────────────────────────────────────────────────────

  sendBtn.addEventListener('click', (e) => {
    e.preventDefault();
    handleSubmission();
  });

  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmission();
    }
  });
});
