/* ═══════════════════════════════════════════════════════════════
   PlanGen Frontend Application (Stitch UI Integration)
   Single-page app connected to FastAPI backend
   ═══════════════════════════════════════════════════════════════ */

const API = '/api/v1';
const app = document.getElementById('app');

// ── State ─────────────────────────────────────────────────────
let state = {
  screen: 'landing',      // landing | chat | pipeline | viewer
  sessionId: null,
  messages: [],
  requirements: {},
  pendingAction: null,
  runId: null,
  runResult: null,
  svgContent: null,
  activeFloor: 0,
  vastuEnabled: true,
};

// ── API helpers ───────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || err.message || 'API error');
  }
  return res.json();
}

// ── Render dispatcher ─────────────────────────────────────────
function render() {
  const tplId = `tpl_${state.screen}`;
  const template = document.getElementById(tplId);
  if (!template) {
    console.error(`Template not found: ${tplId}`);
    return;
  }
  
  app.innerHTML = '';
  app.appendChild(template.content.cloneNode(true));
  
  switch (state.screen) {
    case 'landing':  bindLanding(); break;
    case 'chat':     bindChat(); break;
    case 'pipeline': bindPipeline(); break;
    case 'viewer':   bindViewer(); break;
  }
}

// ═══════════════════════════════════════════════════════════════
// SCREEN 1: LANDING
// ═══════════════════════════════════════════════════════════════
function bindLanding() {
  const startBtns = document.querySelectorAll('button');
  startBtns.forEach(btn => {
    if (btn.textContent.includes('Start Designing')) {
      btn.onclick = startDesigning;
    }
  });
}

async function startDesigning() {
  try {
    const { session_id } = await api('POST', '/sessions');
    state.sessionId = session_id;
    state.messages = [];
    state.requirements = {};
    
    // Switch to chat
    state.screen = 'chat';
    render();
  } catch (e) {
    alert('Failed to start session: ' + e.message);
  }
}

// ═══════════════════════════════════════════════════════════════
// SCREEN 2: CHAT
// ═══════════════════════════════════════════════════════════════
function bindChat() {
  const inputEl = document.querySelector('input[type="text"]');
  
  const allBtns = document.querySelectorAll('button');
  let sendButton = null;
  allBtns.forEach(btn => {
    if (btn.innerHTML.includes('send')) sendButton = btn;
  });

  if (inputEl) {
    inputEl.onkeydown = (e) => {
      if (e.key === 'Enter') handleSend(inputEl.value);
    };
  }
  if (sendButton && inputEl) {
    sendButton.onclick = () => handleSend(inputEl.value);
  }
  
  const genBtn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('Generate Plan'));
  if (genBtn) {
    genBtn.onclick = generatePlan;
  }
  
  renderChatMessages();
  renderSidebar();
}

function handleSend(text) {
  if (!text) return;
  const inputEl = document.querySelector('input[type="text"]');
  if (inputEl) inputEl.value = '';
  
  state.messages.push({ role: 'user', text });
  renderChatMessages();
  sendToBackend(text);
}

// Interactively send answers from widgets
function sendAnswer(text) {
  state.messages.push({ role: 'user', text });
  renderChatMessages();
  sendToBackend(text);
}

async function sendToBackend(text) {
  state.messages.push({ role: 'ai', text: '...', typing: true });
  renderChatMessages();
  
  try {
    let result;
    if (state.pendingAction && state.pendingAction.action === 'ask') {
      result = await api('POST', '/parse/answer', {
        session_id: state.sessionId,
        answer: text,
      });
    } else {
      result = await api('POST', '/parse/text', {
        session_id: state.sessionId,
        text: text,
      });
    }

    state.messages = state.messages.filter(m => !m.typing);
    if (result.data) state.requirements = result.data;
    
    if (result.status === 'success') {
      state.pendingAction = { action: 'complete' };
      state.messages.push({
        role: 'ai',
        text: result.clarification_prompt || result.message || "All details gathered! Click Generate Plan.",
        status: 'success'
      });
    } else if (result.status === 'interactive' || result.status === 'incomplete') {
      let nextQ = {};
      try {
          nextQ = await api('GET', `/parse/next-question?session_id=${state.sessionId}`);
          state.pendingAction = nextQ;
      } catch (e) {
          console.error("Failed to get next question", e);
      }
      
      let widgetType = null;
      if (result.missing_fields) {
        if (result.missing_fields.some(m => m.toLowerCase().includes('facing'))) widgetType = 'compass';
      }
      if (result.suggestions) {
        if (result.suggestions.some(m => m.toLowerCase().includes('floors'))) widgetType = 'floors';
      }
      
      let textToShow = result.clarification_prompt || nextQ.question || result.message;
      
      state.messages.push({ 
        role: 'ai', 
        text: textToShow || "Could you clarify?",
        widget: widgetType
      });
    } else {
      state.messages.push({ role: 'ai', text: result.message || "An error occurred." });
    }
    
    renderChatMessages();
    renderSidebar();
  } catch (e) {
    state.messages = state.messages.filter(m => !m.typing);
    state.messages.push({ role: 'ai', text: `⚠️ Error: ${e.message}` });
    renderChatMessages();
  }
}

function getWidgetHTML(type) {
  if (type === 'compass') {
    return `
    <div class="flex justify-center py-4">
      <div class="relative w-48 h-48 rounded-full border border-[#1F1F1F] flex items-center justify-center bg-base-elevated">
        <div class="w-2 h-2 rounded-full bg-gray-500 absolute"></div>
        <button onclick="sendAnswer('North')" class="absolute top-2 w-8 h-8 rounded-full bg-base border border-[#333] text-gray-400 text-xs font-medium flex items-center justify-center hover:border-gold hover:text-gold transition-colors">N</button>
        <button onclick="sendAnswer('North-East')" class="absolute top-6 right-6 w-8 h-8 rounded-full bg-base border border-[#333] text-gray-400 text-xs font-medium flex items-center justify-center hover:border-gold hover:text-gold transition-colors">NE</button>
        <button onclick="sendAnswer('East')" class="absolute right-2 w-8 h-8 rounded-full bg-base border border-[#333] text-gray-400 text-xs font-medium flex items-center justify-center hover:border-gold hover:text-gold transition-colors">E</button>
        <button onclick="sendAnswer('South-East')" class="absolute bottom-6 right-6 w-8 h-8 rounded-full bg-base border border-[#333] text-gray-400 text-xs font-medium flex items-center justify-center hover:border-gold hover:text-gold transition-colors">SE</button>
        <button onclick="sendAnswer('South')" class="absolute bottom-2 w-8 h-8 rounded-full bg-base border border-[#333] text-gray-400 text-xs font-medium flex items-center justify-center hover:border-gold hover:text-gold transition-colors">S</button>
        <button onclick="sendAnswer('South-West')" class="absolute bottom-6 left-6 w-8 h-8 rounded-full bg-base border border-[#333] text-gray-400 text-xs font-medium flex items-center justify-center hover:border-gold hover:text-gold transition-colors">SW</button>
        <button onclick="sendAnswer('West')" class="absolute left-2 w-8 h-8 rounded-full bg-base border border-[#333] text-gray-400 text-xs font-medium flex items-center justify-center hover:border-gold hover:text-gold transition-colors">W</button>
        <button onclick="sendAnswer('North-West')" class="absolute top-6 left-6 w-8 h-8 rounded-full bg-base border border-[#333] text-gray-400 text-xs font-medium flex items-center justify-center hover:border-gold hover:text-gold transition-colors">NW</button>
      </div>
    </div>`;
  }
  if (type === 'floors') {
    return `
    <div class="grid grid-cols-3 gap-3">
      <button onclick="sendAnswer('1 floor')" class="flex flex-col items-center justify-center p-4 rounded-lg bg-base-elevated border border-[#1F1F1F] hover:bg-[#252525] transition-all hover:scale-[1.02]">
        <span class="material-symbols-outlined mb-2 text-gray-400">home</span>
        <span class="text-xs font-medium">1 Floor</span>
      </button>
      <button onclick="sendAnswer('2 floors')" class="flex flex-col items-center justify-center p-4 rounded-lg bg-base-elevated border border-[#1F1F1F] hover:bg-[#252525] transition-all hover:scale-[1.02]">
        <span class="material-symbols-outlined mb-2 text-gray-400">domain</span>
        <span class="text-xs font-medium">2 Floors</span>
      </button>
      <button onclick="sendAnswer('3 floors')" class="flex flex-col items-center justify-center p-4 rounded-lg bg-base-elevated border border-[#1F1F1F] hover:bg-[#252525] transition-all hover:scale-[1.02]">
        <span class="material-symbols-outlined mb-2 text-gray-400">location_city</span>
        <span class="text-xs font-medium">3 Floors</span>
      </button>
    </div>`;
  }
  return '';
}

function renderChatMessages() {
  const chatContainer = document.querySelector('main > div.w-full.max-w-\\[720px\\]');
  if (!chatContainer) return;
  
  const spacer = chatContainer.lastElementChild;
  chatContainer.innerHTML = '';
  
  // Default welcome
  chatContainer.innerHTML += `
    <div class="flex items-start gap-4 message-anim" style="animation-delay: 0.1s;">
      <div class="w-10 h-10 rounded-full bg-base-elevated border border-gold/30 flex items-center justify-center shrink-0">
        <span class="material-symbols-outlined text-gold" style="font-variation-settings: 'FILL' 1;">architecture</span>
      </div>
      <div class="bg-base-surface border border-[#1F1F1F] rounded-xl rounded-tl-sm p-4 max-w-[85%]">
        <p class="text-sm leading-relaxed text-gray-200">Welcome to PlanGen! I'm your AI architect. Describe your dream home.</p>
      </div>
    </div>
  `;
  
  state.messages.forEach((msg, idx) => {
    if (msg.role === 'ai') {
      let widgetHtml = '';
      if (msg.widget) widgetHtml = getWidgetHTML(msg.widget);
      
      chatContainer.innerHTML += `
        <div class="flex items-start gap-4 message-anim" style="animation-delay: 0.1s;">
          <div class="w-10 h-10 rounded-full bg-base-elevated border border-gold/30 flex items-center justify-center shrink-0">
            <span class="material-symbols-outlined text-gold" style="font-variation-settings: 'FILL' 1;">architecture</span>
          </div>
          <div class="bg-base-surface border border-[#1F1F1F] rounded-xl rounded-tl-sm p-4 max-w-[85%] space-y-4 w-full">
            <p class="text-sm leading-relaxed text-gray-200">${msg.text}</p>
            ${widgetHtml}
          </div>
        </div>
      `;
    } else {
      chatContainer.innerHTML += `
        <div class="flex items-start gap-4 justify-end message-anim" style="animation-delay: 0.1s;">
          <div class="bg-base-elevated border border-[#1F1F1F] rounded-xl rounded-tr-sm p-4 max-w-[85%] text-right">
            <p class="text-sm leading-relaxed text-gray-100">${msg.text}</p>
          </div>
        </div>
      `;
    }
  });
  
  if (spacer) chatContainer.appendChild(spacer);
  
  // Ensure DOM is updated before scrolling
  setTimeout(() => {
    const lastElement = chatContainer.lastElementChild;
    if (lastElement) {
      lastElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
    } else {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }, 50);
}

function renderSidebar() {
  const sidebar = document.querySelector('aside');
  if (!sidebar) return;
  
  // Make "Generate Plan" button active or disabled depending on pending action
  const genBtn = sidebar.querySelector('button:last-child');
  if (genBtn) {
    if (state.pendingAction && state.pendingAction.action === 'complete') {
      genBtn.disabled = false;
      genBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    } else {
      genBtn.disabled = true;
      genBtn.classList.add('opacity-50', 'cursor-not-allowed');
    }
  }

  // Update the requirements list
  const reqContainer = sidebar.querySelector('.flex-1.overflow-y-auto > .space-y-4');
  if (reqContainer && state.requirements) {
    let reqHtml = '';
    
    // Plot
    if (state.requirements.plot_dimensions && state.requirements.plot_dimensions.length) {
      reqHtml += `
      <div class="flex items-center justify-between group">
        <div class="flex items-center gap-3 text-gray-300">
          <span class="material-symbols-outlined text-gray-500 text-sm">square_foot</span>
          <span class="text-sm">Plot: <span class="font-mono text-gray-100">${state.requirements.plot_dimensions.width} × ${state.requirements.plot_dimensions.length} ${state.requirements.plot_dimensions.unit}</span></span>
        </div>
      </div>`;
    }
    
    // Facing
    if (state.requirements.plot_context && state.requirements.plot_context.road_facing_sides && state.requirements.plot_context.road_facing_sides.length) {
      reqHtml += `
      <div class="flex items-center justify-between group">
        <div class="flex items-center gap-3 text-gray-300">
          <span class="material-symbols-outlined text-gray-500 text-sm">explore</span>
          <span class="text-sm">Facing: <span class="text-gray-100 font-medium capitalize">${state.requirements.plot_context.road_facing_sides[0]}</span></span>
        </div>
      </div>`;
    }
    
    // Vastu
    if (state.requirements.vastu_compliant !== null && state.requirements.vastu_compliant !== undefined) {
      const isVastu = state.requirements.vastu_compliant;
      reqHtml += `
      <div class="flex items-center justify-between group">
        <div class="flex items-center gap-3 text-gray-300">
          <span class="material-symbols-outlined ${isVastu ? 'text-status-green' : 'text-gray-500'} text-sm" style="font-variation-settings: 'FILL' 1;">${isVastu ? 'check_circle' : 'cancel'}</span>
          <span class="text-sm">Vastu: <span class="${isVastu ? 'text-status-green' : 'text-gray-100'} font-medium">${isVastu ? 'Enabled' : 'Disabled'}</span></span>
        </div>
      </div>`;
    }
    
    // Floors
    if (state.requirements.number_of_floors) {
      reqHtml += `
      <div class="flex items-center justify-between group">
        <div class="flex items-center gap-3 text-gray-300">
          <span class="material-symbols-outlined text-gray-500 text-sm">domain</span>
          <span class="text-sm">Floors: <span class="text-gray-100 font-medium">${state.requirements.number_of_floors}</span></span>
        </div>
      </div>`;
    }
    
    reqContainer.innerHTML = reqHtml;
  }
  
  // Update the Rooms Chips list
  const roomsContainer = sidebar.querySelector('#sidebar-rooms-list');
  if (roomsContainer && state.requirements && state.requirements.rooms) {
    let roomsHtml = '';
    state.requirements.rooms.forEach(r => {
      const q = r.quantity > 1 ? `${r.quantity} ` : '';
      roomsHtml += `<span class="px-2 py-1 rounded text-xs bg-base-elevated border border-[#333] text-gray-300">${q}${r.room_type}</span>`;
    });
    if (roomsHtml) {
      roomsContainer.innerHTML = roomsHtml;
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// SCREEN 3: PIPELINE
// ═══════════════════════════════════════════════════════════════
async function generatePlan() {
  state.screen = 'pipeline';
  render();
  
  try {
    const initRes = await api('POST', '/pipeline/run', {
      session_id: state.sessionId,
      options: { use_gemini_enricher: true, prefer_cpsat: false, cpsat_timeout_s: 20 },
    });
    
    state.runId = initRes.run_id;
    
    // Start polling
    let isComplete = false;
    while (!isComplete) {
      await sleep(500);
      try {
        const statusData = await api('GET', `/pipeline/status/${state.runId}`);
        
        // Update UI
        updatePipelineUI(statusData);
        
        if (statusData.status === 'complete') {
          isComplete = true;
          state.runResult = statusData.result;
          await sleep(1000); // Let user see 5/5 completion
          state.screen = 'viewer';
          await loadSvg();
          render();
        } else if (statusData.status === 'error') {
          isComplete = true;
          alert("Pipeline error: " + statusData.error);
          state.screen = 'chat';
          render();
        }
      } catch (pollErr) {
        console.error("Poll error", pollErr);
      }
    }
  } catch (e) {
    alert("Pipeline start error: " + e.message);
    state.screen = 'chat';
    render();
  }
}

function updatePipelineUI(data) {
  const step = data.step || 1;
  const logs = data.logs || [];
  
  // Update progress text
  const textEl = document.getElementById('pipeline-progress-text');
  if (textEl) textEl.textContent = `${step}/5`;
  
  // Update circle
  const circleEl = document.getElementById('pipeline-progress-circle');
  if (circleEl) {
    const maxOffset = 251.2;
    const progress = step / 5;
    circleEl.style.strokeDashoffset = maxOffset - (maxOffset * progress);
  }
  
  // Update logs
  const logsEl = document.getElementById('pipeline-logs-container');
  if (logsEl) {
    let logHtml = '';
    logs.slice(-5).forEach(log => {
      if (log.includes('ERROR') || log.includes('WARN')) {
         logHtml += `<p class="opacity-90"><span class="text-[#D4A937]">${log}</span></p>`;
      } else if (log.includes('ENRICH')) {
         logHtml += `<div class="text-gold font-medium"><span class="typewriter-text">${log}</span></div>`;
      } else {
         logHtml += `<p class="opacity-80">${log}</p>`;
      }
    });
    if (data.status === 'running') {
       logHtml += `<p class="text-white/40 animate-pulse">_</p>`;
    }
    logsEl.innerHTML = logHtml;
  }
  
  // Update ETA
  const etaEl = document.getElementById('pipeline-eta');
  if (etaEl) {
     const remaining = 5 - step;
     if (remaining === 0) etaEl.textContent = "Done!";
     else etaEl.textContent = `~${remaining * 4}s remaining`;
  }
}

function bindPipeline() {
  // Let CSS animations run natively
}

// ═══════════════════════════════════════════════════════════════
// SCREEN 4: VIEWER
// ═══════════════════════════════════════════════════════════════
async function loadSvg() {
  if (!state.runId || !state.runResult) return;
  const svgFiles = state.runResult.svg_files || [];
  if (svgFiles.length > 0) {
    try {
      const res = await fetch(`${API}/runs/${state.runId}/svg/${svgFiles[state.activeFloor]}`);
      if (res.ok) {
        state.svgContent = await res.text();
      }
    } catch (e) {
      console.error('Failed to load SVG:', e);
    }
  }
}

function bindViewer() {
  // Try multiple possible container classes based on Stitch design
  const canvasEl = document.getElementById('svg-container') || 
                   document.querySelector('.canvas-bg') || 
                   document.querySelector('.viewer-canvas');
                   
  if (canvasEl && state.svgContent) {
    canvasEl.innerHTML = state.svgContent;
    const svg = canvasEl.querySelector('svg');
    if (svg) {
      svg.style.width = '100%';
      svg.style.height = '100%';
    }
  }
  
  updateViewerMetrics();
  
  // Wire up action buttons
  const allBtns = document.querySelectorAll('button, a');
  allBtns.forEach(el => {
    if (el.textContent.includes('BACK TO CHAT')) {
      el.onclick = (e) => { e.preventDefault(); state.screen = 'chat'; render(); };
    }
    if (el.textContent.includes('REGENERATE')) {
      el.onclick = generatePlan;
    }
    if (el.textContent.includes('EXPORT')) {
      el.onclick = exportSvg;
    }
  });
}

function exportSvg() {
  if (!state.svgContent) return;
  const blob = new Blob([state.svgContent], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `plangen_floor_${state.activeFloor}.svg`;
  a.click();
  URL.revokeObjectURL(url);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function updateViewerMetrics() {
  const res = state.runResult;
  if (!res || !res.layout_plan) return;
  
  const lp = res.layout_plan;
  
  // 1. Quality Score
  const elQuality = document.getElementById('metric-quality');
  if (elQuality) elQuality.textContent = (lp.layout_quality_score || 0).toFixed(2);
  
  const elAdj = document.getElementById('metric-adj');
  if (elAdj) elAdj.textContent = (lp.overall_adjacency_score || 0).toFixed(3);
  
  const elZone = document.getElementById('metric-zone');
  if (elZone) elZone.textContent = (lp.overall_zone_score || 0).toFixed(3);
  
  const elAdjBar = document.getElementById('metric-adj-bar');
  if (elAdjBar) elAdjBar.style.width = `${(lp.overall_adjacency_score || 0) * 100}%`;
  
  const elZoneBar = document.getElementById('metric-zone-bar');
  if (elZoneBar) elZoneBar.style.width = `${(lp.overall_zone_score || 0) * 100}%`;
  
  const elCircle = document.getElementById('metric-quality-circle');
  if (elCircle) {
     const maxOffset = 251.2;
     const q = lp.layout_quality_score || 0;
     elCircle.style.strokeDashoffset = maxOffset - (maxOffset * q);
  }
  
  // 2. Solver Details
  const elSolver = document.getElementById('metric-solver');
  if (elSolver) elSolver.textContent = (lp.solver_used || '').toUpperCase();
  
  const elStatus = document.getElementById('metric-status');
  if (elStatus) elStatus.textContent = (lp.solver_status || '').toUpperCase();
  
  const elTime = document.getElementById('metric-time');
  if (elTime) elTime.textContent = lp.solve_time_ms || 0;
  
  // 3. Project Parameters
  const elPlot = document.getElementById('metric-plot');
  if (elPlot) elPlot.textContent = `${lp.plot_width_ft} x ${lp.plot_length_ft} ft`;
  
  const elArea = document.getElementById('metric-area');
  if (elArea) elArea.textContent = (lp.plot_width_ft * lp.plot_length_ft).toLocaleString();
  
  const elRooms = document.getElementById('metric-rooms');
  if (elRooms) elRooms.textContent = lp.total_rooms_placed || 0;
  
  const elBuild = document.getElementById('metric-buildable');
  if (elBuild) elBuild.textContent = lp.total_area_placed_sqft || 0;
  
  // 4. Room List
  const rlContainer = document.getElementById('room-list-container');
  if (rlContainer && lp.floors && lp.floors[state.activeFloor]) {
    const floor = lp.floors[state.activeFloor];
    let html = '';
    
    const sortedRooms = [...floor.rooms].sort((a,b) => b.area_sqft - a.area_sqft);
    const colors = ['#A8C6FA', '#FACDA8', '#A8FAD2', '#E0A8FA', '#FADCA8', '#FCA8A8'];
    
    sortedRooms.forEach((r, idx) => {
      const color = colors[idx % colors.length];
      const name = r.room_id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      html += `
        <div class="bg-plangen-surface border border-transparent hover:border-plangen-border rounded-lg p-3 flex items-center justify-between cursor-pointer transition-all hover:bg-[#1C1C1C]">
          <div class="flex items-center space-x-3 ml-2">
            <div class="w-3 h-3 rounded-sm" style="background-color: ${color}"></div>
            <span class="text-sm text-white/80">${name}</span>
          </div>
          <div class="text-right">
            <p class="font-mono text-sm text-white/80">${Math.round(r.area_sqft)} <span class="text-[10px] text-white/40">sqft</span></p>
            <p class="font-mono text-[10px] text-white/40">${r.width_ft.toFixed(1)} × ${r.length_ft.toFixed(1)} ft</p>
          </div>
        </div>
      `;
    });
    rlContainer.innerHTML = html;
  }
}

render();
