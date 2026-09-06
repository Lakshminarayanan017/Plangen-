/* Plangen UI — wired to the PlanGen API on the same origin.
 *
 * THE LOOP. A plan generator that can only answer once is a demo. The real
 * product is: describe the house, look at it, say what is wrong, look again.
 * Every endpoint that loop needs has existed on the server for a while and
 * nothing on screen ever called them — alternatives were rendered and thrown
 * away, the Vastu scorecard was computed and never shown, and the preference
 * log has been live and empty because nothing ever asked a user to choose.
 *
 * TWO MODES, because after a plan exists the same sentence is ambiguous.
 * "Add a study room" is a new brief in `brief` mode and an edit to the plan
 * on screen in `review` mode. Rather than guess, the UI states which one it
 * is in (the input placeholder says so) and gives an explicit way back.
 *
 * PREVIEW BEFORE SPENDING A RUN. An edit is parsed deterministically, so the
 * UI can say "I understood: kitchen 25% larger" before the engine does ~20
 * seconds of work — and can say which part of the sentence did NOT land,
 * instead of quietly generating something that ignores half the request.
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

  const PLACEHOLDER = {
    brief: 'Describe your plot and the rooms you want…',
    review: 'Change something — "make the kitchen bigger", "move the pooja room north-east"…',
  };

  const state = {
    sessionId: null,
    mode: 'brief',      // 'brief' | 'review'
    runId: null,        // the plan currently on screen
    pendingEdit: null,  // text awaiting confirmation
    busy: false,
  };

  function setMode(mode) {
    state.mode = mode;
    searchInput.placeholder = PLACEHOLDER[mode] || PLACEHOLDER.brief;
  }

  // ── Chat plumbing ───────────────────────────────────────────────

  const ICON_USER =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
  const ICON_AI =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>';

  const escapeHtml = (s) =>
    String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));

  const COMPASS = {
    N: 'north', S: 'south', E: 'east', W: 'west',
    NE: 'north-east', NW: 'north-west', SE: 'south-east', SW: 'south-west',
  };
  const compass = (d) => COMPASS[String(d).toUpperCase()] || String(d);

  const clockNow = () =>
    new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  /* The glass panel sits on top of the hero and the blueprint gallery, and
   * both stayed put — a plan sheet rendered over the Taj Mahal is unreadable.
   * `.hidden-up` was written for this and never applied by anything. */
  function openGlassWindow() {
    chatGlassWindow.classList.add('active');
    document.getElementById('chat-intro')?.classList.add('hidden-up');
    document.getElementById('blueprint-gallery')?.classList.add('hidden-up');
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
    if (state.sessionId) return state.sessionId;
    const data = await api('/sessions', { method: 'POST' });
    state.sessionId = data.session_id;
    return state.sessionId;
  }

  // ── Step 1: understand the brief ────────────────────────────────

  async function handleBrief(query) {
    const reply = appendMessage('assistant', 'Reading your brief&hellip;');
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
  }

  // ── Editing a plan in words ─────────────────────────────────────

  /* Parse first, generate second. The parser is deterministic and instant,
   * so there is no reason to make someone wait for a run to find out that
   * half their sentence was not understood. */
  async function handleEdit(text) {
    const reply = appendMessage('assistant', 'Reading that change&hellip;');
    const preview = await api(`/runs/${encodeURIComponent(state.runId)}/edit`, {
      method: 'POST',
      body: JSON.stringify({
        session_id: state.sessionId, text, preview: true,
      }),
    });

    if (!preview.understood) {
      state.pendingEdit = null;
      setContent(
        reply,
        `${escapeHtml(preview.message || 'I could not turn that into a change.')}` +
          `<div class="edit-actions">
             <button class="chip-btn" data-action="new-brief">Start a new brief instead</button>
           </div>`,
        false,
      );
      return;
    }

    state.pendingEdit = text;
    const changes = (preview.changes || [])
      .map((c) => `<li>${escapeHtml(c)}</li>`)
      .join('');
    const unparsed = (preview.unparsed || []).length
      ? `<div class="edit-unparsed">I could not act on:
           ${escapeHtml(preview.unparsed.join('; '))}</div>`
      : '';

    setContent(
      reply,
      `<div class="edit-preview">
         <div class="edit-understood">${escapeHtml(preview.summary)}</div>
         ${changes ? `<ul class="edit-changes">${changes}</ul>` : ''}
         ${unparsed}
         <div class="edit-actions">
           <button class="chip-btn primary" data-action="apply-edit">Apply this change</button>
           <button class="chip-btn" data-action="cancel-edit">Cancel</button>
         </div>
       </div>`,
      false,
    );
  }

  async function applyPendingEdit() {
    if (!state.pendingEdit || state.busy) return;
    const text = state.pendingEdit;
    state.pendingEdit = null;
    state.busy = true;
    try {
      const started = await api(`/runs/${encodeURIComponent(state.runId)}/edit`, {
        method: 'POST',
        body: JSON.stringify({ session_id: state.sessionId, text }),
      });
      if (!started.run_id) {
        appendMessage('assistant', 'The edit did not start a run.');
        return;
      }
      await pollRun(started.run_id, 'Applying your change&hellip;');
    } catch (err) {
      appendMessage('assistant',
        `The edit failed.<br><small>${escapeHtml(err.message)}</small>`);
    } finally {
      state.busy = false;
      searchInput.focus();
    }
  }

  // ── Steps 2-5: run the pipeline and stream progress ─────────────

  async function runPipeline(sid) {
    const { run_id: runId } = await api('/pipeline/run', {
      method: 'POST',
      body: JSON.stringify({ session_id: sid, options: {} }),
    });
    await pollRun(runId, 'Starting the pipeline&hellip;');
  }

  async function regenerate() {
    if (state.busy) return;
    state.busy = true;
    try {
      const { run_id: runId } = await api('/pipeline/regenerate', {
        method: 'POST',
        body: JSON.stringify({ session_id: state.sessionId, options: {} }),
      });
      await pollRun(runId, 'Exploring a different set of layouts&hellip;');
    } catch (err) {
      appendMessage('assistant',
        `Could not start another run.<br><small>${escapeHtml(err.message)}</small>`);
    } finally {
      state.busy = false;
    }
  }

  async function pollRun(runId, openingLine) {
    const progress = appendMessage('assistant', openingLine);
    const started = Date.now();

    while (true) {
      await new Promise((r) => setTimeout(r, 1200));

      let status;
      try {
        status = await api(`/pipeline/status/${encodeURIComponent(runId)}`);
      } catch (err) {
        setContent(progress,
          `Lost contact with the run.<br><small>${escapeHtml(err.message)}</small>`);
        return;
      }

      const recent = (status.logs || []).slice(-4).map(escapeHtml).join('<br>');
      const elapsed = Math.round((Date.now() - started) / 1000);

      if (status.status === 'running') {
        setContent(
          progress,
          `<strong>Step ${status.step || 1} of 5</strong><br>${recent || 'Working&hellip;'}` +
            `<span class="time">${elapsed}s elapsed</span>`,
          false,
        );
        continue;
      }

      if (status.status === 'error') {
        setContent(progress,
          `The run failed.<br><small>${escapeHtml(status.error || 'unknown error')}</small>`);
        return;
      }

      setContent(progress, `<strong>Done in ${elapsed}s.</strong><br>${recent}`, false);
      state.runId = runId;
      setMode('review');
      renderResult(runId, status.result || {});
      return;
    }
  }

  // ── The plan card ───────────────────────────────────────────────

  function sheetUrl(runId, name) {
    return `${API}/runs/${encodeURIComponent(runId)}/svg/${encodeURIComponent(name)}`;
  }

  function renderSheets(runId, files) {
    if (!files.length) return '<div class="plan-empty">The run finished but produced no sheets.</div>';
    return files.map((name) => {
      const url = sheetUrl(runId, name);
      return `
        <a class="attachment" href="${url}" target="_blank" rel="noopener">
          <img class="sheet-preview" src="${url}" alt="${escapeHtml(name)}">
          <div class="attachment-info">
            <span class="attachment-name">${escapeHtml(name)}</span>
            <span class="attachment-size">Open full sheet &rarr;</span>
          </div>
        </a>`;
    }).join('');
  }

  /* The Vastu scorecard. `vastu_enabled: true` on its own told a user
   * nothing; this is the per-room account behind the flag, and it is the
   * feature this product is meant to lead with.
   *
   * The engine works in a plot-relative frame and Vastu is absolute, so the
   * site line states which side was taken as north. A user whose plot is
   * rotated needs to see that assumption, not discover it in the result. */
  function renderVastu(floors) {
    if (!floors || !floors.length) return '';
    return floors.map((f, idx) => {
      const c = f.counts || {};
      const facts = [
        [c.ideal, 'ideal'], [c.near, 'near'],
        [c.off, 'off'], [c.barred, 'in barred sectors'],
      ].filter(([n]) => n).map(([n, label]) => `${n} ${label}`);

      if (f.brahmasthan && f.brahmasthan.clear != null) {
        facts.push(f.brahmasthan.clear
          ? 'Brahmasthan clear'
          : `Brahmasthan ${Math.round((f.brahmasthan.barred_use_share || 0) * 100)}% built over`);
      }
      if (f.entrance && f.entrance.pada) {
        facts.push(`entrance ${f.entrance.pada}` +
          (f.entrance.name ? ` (${f.entrance.name})` : '') +
          ` — ${String(f.entrance.status).replace(/_/g, ' ')}`);
      }

      // `site.orientation` is the engine's own frame description ("grid N=N,
      // grid E=E …") — true, and meaningless to a home builder. The two facts
      // that matter are which side is north and which way the door faces.
      const site = f.site && f.site.north_side
        ? `<div class="vastu-site">North is the ${compass(f.site.north_side)} side of your plot${
            f.site.entrance_faces
              ? `; the entrance faces ${compass(f.site.entrance_faces)}` : ''}.</div>`
        : '';

      const rows = (f.rooms || [])
        .filter((r) => r.status !== 'unconstrained')
        .map((r) => `
          <tr class="v-${escapeHtml(r.status)}">
            <td>${escapeHtml(r.room)}</td>
            <td>${escapeHtml(r.wanted || '—')}</td>
            <td>${escapeHtml(r.got)}</td>
            <td><span class="v-status">${escapeHtml(r.status)}</span></td>
            <td class="v-advice">${escapeHtml(r.advice || '')}</td>
          </tr>`).join('');

      // advice earned its place by being actionable — a grade with no reason
      // is just a number. `data_notes` is deliberately NOT here: the health
      // of data/vastuRules1.json belongs in diagnostics, not in front of
      // someone deciding where to put their kitchen.
      const advice = [
        f.entrance && f.entrance.advice,
        f.brahmasthan && f.brahmasthan.advice,
      ].filter(Boolean).concat(f.notes || []);
      const adviceList = advice.length
        ? `<ul class="vastu-notes">${
            advice.map((n) => `<li>${escapeHtml(n)}</li>`).join('')}</ul>`
        : '';

      return `
        <div class="vastu-panel">
          <div class="vastu-head">
            <span class="vastu-grade grade-${escapeHtml(f.grade)}">${escapeHtml(f.grade)}</span>
            <span class="vastu-title">Vastu ${Math.round((f.score || 0) * 100)}%${
              floors.length > 1 ? ` — ${escapeHtml(f.floor_label)}` : ''}</span>
            <button class="chip-btn ghost" data-action="toggle-vastu">
              Room by room
            </button>
          </div>
          <div class="vastu-counts">${
            facts.map((t) => `<span class="vastu-fact">${escapeHtml(t)}</span>`).join('')
            || '<span class="vastu-fact">no directional rooms</span>'}</div>
          ${site}
          <div class="vastu-detail" data-vastu="${idx}" hidden>
            ${rows ? `<table class="vastu-table">
              <thead><tr><th>Room</th><th>Wanted</th><th>Got</th><th></th><th></th></tr></thead>
              <tbody>${rows}</tbody></table>` : '<p>No room carries a Vastu direction.</p>'}
            ${adviceList}
          </div>
        </div>`;
    }).join('');
  }

  /* The engine keeps every candidate that survives the reviewer; only the
   * top one was ever shown. Offering the rest is also the only way the
   * learned critic ever sees taste — a pick among real options is not
   * something that can be collected retroactively. */
  function renderAlternatives(runId, alts, chosenRank) {
    if (!alts || alts.length < 2) return '';
    // every option carves the same program, so the room count is usually
    // identical across all of them — worth saying only when it is not
    const baseRooms = alts[0].rooms;
    const cards = alts.map((a) => {
      const picked = chosenRank === a.rank;
      const grade = a.vastu_grade
        ? `<span class="alt-vastu">Vastu ${escapeHtml(a.vastu_grade)}</span>` : '';
      const traits = (a.highlights || [])
        .map((h) => `<li>${escapeHtml(h)}</li>`).join('');
      return `
        <div class="alt-card${picked ? ' chosen' : ''}${a.is_best ? ' best' : ''}">
          <img class="alt-thumb" src="${sheetUrl(runId, a.svg)}"
               alt="Option ${a.rank + 1}">
          <div class="alt-meta">
            <span class="alt-name">Option ${a.rank + 1}${a.is_best ? ' · engine pick' : ''}</span>
            <span class="alt-score" title="reviewer score ${escapeHtml(a.score)}">${
              a.rooms && a.rooms !== baseRooms
                ? `<span class="alt-rooms">${a.rooms} rooms</span>` : ''}${grade}</span>
          </div>
          ${traits ? `<ul class="alt-traits">${traits}</ul>` : ''}
          <button class="chip-btn${picked ? '' : ' primary'}"
                  data-action="choose" data-run="${escapeHtml(runId)}"
                  data-rank="${a.rank}"${picked ? ' disabled' : ''}>
            ${picked ? 'Chosen' : 'Use this one'}
          </button>
        </div>`;
    }).join('');
    return `
      <div class="alt-block">
        <div class="alt-head">The engine also built these. Picking one teaches it
          what you like.</div>
        <div class="alt-strip">${cards}</div>
      </div>`;
  }

  function renderResult(runId, result) {
    const files = result.svg_files || [];
    const summary = (result.step4 && result.step4.summary) || {};

    const facts = [
      summary.total_rooms_placed != null ? `${summary.total_rooms_placed} rooms` : null,
      summary.total_area_sqft != null ? `${summary.total_area_sqft} sq ft` : null,
      summary.floors != null ? `${summary.floors} floor` + (summary.floors === 1 ? '' : 's') : null,
    ].filter(Boolean);

    const continuity = result.continuity
      ? `<div class="continuity-note">${escapeHtml(result.continuity.describes)}</div>`
      : '';
    const program = result.program && result.program.headline
      ? `<div class="program-note">${escapeHtml(result.program.headline)}</div>`
      : '';

    appendMessage(
      'assistant',
      `<div class="plan-card">
         <div class="plan-head">Here is your floor plan${
           facts.length ? ` — ${escapeHtml(facts.join(' · '))}` : ''}.</div>
         ${program}
         ${continuity}
         ${renderSheets(runId, files)}
         ${renderVastu(result.vastu)}
         ${renderAlternatives(runId, result.alternatives, result.chosen_rank)}
         <div class="plan-actions">
           <button class="chip-btn" data-action="regenerate">Try a different set</button>
           <button class="chip-btn" data-action="new-brief">Start a new brief</button>
         </div>
         <div class="plan-hint">Not quite right? Just say what to change —
           <em>"make the kitchen bigger"</em>, <em>"move the pooja room to the
           north east"</em>, <em>"add a study room"</em>.</div>
         <span class="time">Run ${escapeHtml(runId)}</span>
       </div>`,
    );
  }

  // ── Delegated actions on everything rendered above ──────────────

  chatBody.addEventListener('click', async (event) => {
    const btn = event.target.closest('[data-action]');
    if (!btn) return;
    event.preventDefault();

    switch (btn.dataset.action) {
      case 'apply-edit':
        btn.closest('.edit-actions').innerHTML = '<span class="edit-applied">Applying…</span>';
        await applyPendingEdit();
        break;

      case 'cancel-edit':
        state.pendingEdit = null;
        btn.closest('.edit-preview').innerHTML =
          '<div class="edit-understood">Cancelled — the plan is unchanged.</div>';
        break;

      case 'new-brief':
        // A fresh SESSION, not just a mode switch. `/parse/text` calls
        // `handle_followup`, which MERGES into the running conversation by
        // design — so keeping the session would fold the new brief into the
        // old requirements and "start a new brief" would be a lie.
        state.pendingEdit = null;
        state.sessionId = null;
        state.runId = null;
        setMode('brief');
        appendMessage('assistant',
          'Starting fresh. Describe the plot and the rooms you want.');
        searchInput.focus();
        break;

      case 'regenerate':
        await regenerate();
        break;

      case 'toggle-vastu': {
        const panel = btn.closest('.vastu-panel');
        const detail = panel && panel.querySelector('.vastu-detail');
        if (detail) {
          detail.hidden = !detail.hidden;
          btn.textContent = detail.hidden ? 'Room by room' : 'Hide detail';
        }
        break;
      }

      case 'choose': {
        const rank = Number(btn.dataset.rank);
        const runId = btn.dataset.run;
        btn.disabled = true;
        try {
          const picked = await api(`/runs/${encodeURIComponent(runId)}/choose`, {
            method: 'POST',
            body: JSON.stringify({ session_id: state.sessionId, rank }),
          });
          // "Use this one" has to actually use it: swap the sheet above, or
          // the button records a preference and shows the user the plan they
          // just rejected. (Alternatives are single-floor only, so there is
          // exactly one sheet to swap.)
          const card = btn.closest('.plan-card');
          const sheet = card && card.querySelector('.attachment');
          if (sheet && picked.svg) {
            const url = sheetUrl(runId, picked.svg);
            sheet.href = url;
            sheet.querySelector('.sheet-preview').src = url;
            sheet.querySelector('.attachment-name').textContent =
              `Option ${rank + 1}`;
          }
          const strip = btn.closest('.alt-strip');
          strip.querySelectorAll('.alt-card').forEach((other) => {
            other.classList.remove('chosen');
            const b = other.querySelector('[data-action="choose"]');
            if (b) { b.disabled = false; b.textContent = 'Use this one'; b.classList.add('primary'); }
          });
          btn.closest('.alt-card').classList.add('chosen');
          btn.textContent = 'Chosen';
          btn.classList.remove('primary');
          btn.disabled = true;
        } catch (err) {
          btn.disabled = false;
          appendMessage('assistant',
            `Could not record that choice.<br><small>${escapeHtml(err.message)}</small>`);
        }
        break;
      }
    }
  });

  // ── Events ──────────────────────────────────────────────────────

  async function handleSubmission() {
    const query = searchInput.value.trim();
    if (!query || state.busy) return;

    state.busy = true;
    searchInput.value = '';
    openGlassWindow();
    appendMessage('user',
      `${escapeHtml(query)}<span class="time">${clockNow()} &#10003;&#10003;</span>`);

    try {
      if (state.mode === 'review' && state.runId) {
        await handleEdit(query);
      } else {
        await handleBrief(query);
      }
    } catch (err) {
      appendMessage('assistant',
        `Could not reach the PlanGen engine.<br><small>${escapeHtml(err.message)}</small>`);
    } finally {
      state.busy = false;
      searchInput.focus();
    }
  }

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

  setMode('brief');
});
