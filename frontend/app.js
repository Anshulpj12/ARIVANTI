/**
 * SP 21:2005 RAG — Frontend Application
 */
const API = '';  // Same origin

// ─── DOM Elements ───────────────────────────────────────
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const sectionFilter = document.getElementById('section-filter');
const resultsSection = document.getElementById('results-section');
const loadingState = document.getElementById('loading-state');
const answerCard = document.getElementById('answer-card');
const answerContent = document.getElementById('answer-content');
const latencyBadge = document.getElementById('latency-badge');
const sourcesCard = document.getElementById('sources-card');
const sourcesList = document.getElementById('sources-list');
const sourcesCount = document.getElementById('sources-count');
const intentCard = document.getElementById('intent-card');
const intentDetails = document.getElementById('intent-details');
const historyList = document.getElementById('history-list');
const clearHistoryBtn = document.getElementById('clear-history');
const statusDot = document.getElementById('status-indicator');
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const closeSettings = document.getElementById('close-settings');
const saveConfigBtn = document.getElementById('save-config');
const configStatus = document.getElementById('config-status');
const hints = document.querySelectorAll('.hint');
const steps = {
    intent: document.getElementById('step-intent'),
    search: document.getElementById('step-search'),
    llm: document.getElementById('step-llm'),
};

// ─── State ──────────────────────────────────────────────
let history = JSON.parse(localStorage.getItem('rag_history') || '[]');
let isQuerying = false;

// ─── Init ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadSections();
    renderHistory();
    checkHealth();
    setInterval(checkHealth, 30000);
});

// ─── Events ─────────────────────────────────────────────
searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !isQuerying) submitQuery();
    if (e.key === 'Escape') { searchInput.value = ''; searchInput.blur(); }
});
searchBtn.addEventListener('click', () => { if (!isQuerying) submitQuery(); });
hints.forEach(h => h.addEventListener('click', () => {
    searchInput.value = h.dataset.query;
    submitQuery();
}));
clearHistoryBtn.addEventListener('click', () => {
    history = []; localStorage.removeItem('rag_history');
    renderHistory();
});
settingsBtn.addEventListener('click', () => { openSettings(); });
closeSettings.addEventListener('click', () => settingsModal.classList.add('hidden'));
document.querySelector('.modal-backdrop')?.addEventListener('click', () => settingsModal.classList.add('hidden'));
saveConfigBtn.addEventListener('click', saveConfig);

// ─── Core Query ─────────────────────────────────────────
async function submitQuery() {
    const question = searchInput.value.trim();
    if (!question) return;
    isQuerying = true;

    // Show loading
    resultsSection.classList.remove('hidden');
    loadingState.classList.remove('hidden');
    answerCard.classList.add('hidden');
    sourcesCard.classList.add('hidden');
    intentCard.classList.add('hidden');

    animateSteps('intent');

    try {
        setTimeout(() => animateSteps('search'), 400);
        setTimeout(() => animateSteps('llm'), 1200);

        const resp = await fetch(`${API}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
        const data = await resp.json();

        // Hide loading, show results
        loadingState.classList.add('hidden');
        renderAnswer(data);
        renderSources(data.sources);
        renderIntent(data.intent);

        // Save to history
        addToHistory(question);

    } catch (err) {
        loadingState.classList.add('hidden');
        answerCard.classList.remove('hidden');
        answerContent.textContent = `❌ Error: ${err.message}\n\nMake sure the backend server is running on port 8000.`;
        latencyBadge.textContent = '';
    }
    isQuerying = false;
}

// ─── Render Functions ───────────────────────────────────
function renderAnswer(data) {
    answerCard.classList.remove('hidden');
    answerContent.innerHTML = formatAnswer(data.answer);
    const lat = data.latency_ms;
    latencyBadge.textContent = `${lat.total}ms (search: ${lat.search}ms | llm: ${lat.llm}ms)`;
}

function formatAnswer(text) {
    // Basic formatting: bold IS codes, convert line breaks
    let html = escapeHtml(text);
    // Highlight IS codes
    html = html.replace(/(IS\s*\d{2,5}(?:\s*[\:\-]\s*\d{4})?)/g, '<strong>$1</strong>');
    // Convert markdown-like headers
    html = html.replace(/^#{1,3}\s+(.+)$/gm, '<strong style="font-size:1.05em;display:block;margin-top:12px;">$1</strong>');
    // Style section dividers
    html = html.replace(/^---\s*(.+?)\s*---$/gm, '<strong style="color:var(--accent);display:block;margin-top:14px;border-bottom:1px solid var(--border);padding-bottom:4px;">$1</strong>');
    // Convert bullet points
    html = html.replace(/^[\-\•]\s+/gm, '  &bull; ');
    // Convert numbered lists
    html = html.replace(/^(\d+)\.\s+/gm, '  $1. ');
    // Style label lines (e.g., "Section:", "Scope:")
    html = html.replace(/^([A-Z][A-Za-z\s]+):\n/gm, '<strong style="color:var(--text-secondary);">$1:</strong>\n');
    // Convert newlines to BR
    html = html.replace(/\n/g, '<br>');
    return html;
}

function escapeHtml(text) {
    const el = document.createElement('div');
    el.textContent = text;
    return el.innerHTML;
}

function renderSources(sources) {
    if (!sources || sources.length === 0) return;
    sourcesCard.classList.remove('hidden');
    sourcesCount.textContent = sources.length;

    sourcesList.innerHTML = sources.map(s => `
        <div class="source-item">
            <div class="source-left">
                <span class="source-code">${escapeHtml(s.is_code)}</span>
                <span class="source-title">${escapeHtml(s.title)}</span>
            </div>
            <div class="source-meta">
                <span>${escapeHtml(s.section)}</span>
                <span>p.${s.page}</span>
                <span class="source-score">${(s.score * 100).toFixed(1)}%</span>
            </div>
        </div>
    `).join('');
}

function renderIntent(intent) {
    if (!intent) return;
    intentCard.classList.remove('hidden');
    const tags = [];
    if (intent.section_name) tags.push(`<span class="intent-tag section">📂 ${intent.section_name}</span>`);
    else tags.push(`<span class="intent-tag low">📂 All Sections (fallback)</span>`);
    if (intent.content_type) tags.push(`<span class="intent-tag type">📋 ${intent.content_type}</span>`);
    if (intent.is_code_ref) tags.push(`<span class="intent-tag section">🔗 IS ${intent.is_code_ref}</span>`);
    const confClass = intent.confidence === 'low' ? 'low' : 'confidence';
    tags.push(`<span class="intent-tag ${confClass}">⚡ ${intent.confidence} confidence</span>`);
    intentDetails.innerHTML = tags.join('');
}

function animateSteps(active) {
    Object.entries(steps).forEach(([key, el]) => {
        el.classList.remove('active', 'done');
        if (key === active) el.classList.add('active');
        else if (
            (active === 'search' && key === 'intent') ||
            (active === 'llm' && (key === 'intent' || key === 'search'))
        ) el.classList.add('done');
    });
}

// ─── Sections Dropdown ──────────────────────────────────
async function loadSections() {
    try {
        const resp = await fetch(`${API}/sections`);
        const data = await resp.json();
        data.sections.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = `${s.id}. ${s.name}`;
            sectionFilter.appendChild(opt);
        });
    } catch (e) { /* silent */ }
}

// ─── History ────────────────────────────────────────────
function addToHistory(query) {
    history = [query, ...history.filter(q => q !== query)].slice(0, 15);
    localStorage.setItem('rag_history', JSON.stringify(history));
    renderHistory();
}

function renderHistory() {
    if (history.length === 0) {
        document.getElementById('history-section').style.display = 'none';
        return;
    }
    document.getElementById('history-section').style.display = 'block';
    historyList.innerHTML = history.map(q =>
        `<div class="history-item" onclick="replayQuery(this)">${escapeHtml(q)}</div>`
    ).join('');
}

function replayQuery(el) {
    searchInput.value = el.textContent;
    submitQuery();
}

// ─── Health Check ───────────────────────────────────────
async function checkHealth() {
    try {
        const resp = await fetch(`${API}/stats`);
        if (resp.ok) {
            statusDot.classList.remove('offline');
            statusDot.classList.add('online');
            statusDot.title = 'System Online';
        } else throw new Error();
    } catch {
        statusDot.classList.remove('online');
        statusDot.classList.add('offline');
        statusDot.title = 'Backend Offline';
    }
}

// ─── Settings ───────────────────────────────────────────
async function openSettings() {
    settingsModal.classList.remove('hidden');
    configStatus.textContent = '';
    try {
        const resp = await fetch(`${API}/config`);
        const cfg = await resp.json();

        // LLM Provider
        const provSelect = document.getElementById('cfg-llm-provider');
        provSelect.innerHTML = '';
        Object.entries(cfg.llm.available_providers).forEach(([key, val]) => {
            const opt = document.createElement('option');
            opt.value = key; opt.textContent = val.name;
            if (key === cfg.llm.active_provider) opt.selected = true;
            provSelect.appendChild(opt);
        });

        // LLM Model
        document.getElementById('cfg-llm-model').value = cfg.llm.active_model;

        // Embedding Model
        const embedSelect = document.getElementById('cfg-embed-model');
        embedSelect.innerHTML = '';
        Object.entries(cfg.embedding.available_models).forEach(([key, val]) => {
            const opt = document.createElement('option');
            opt.value = key; opt.textContent = `${val.name} (${val.dimensions}d, ${val.size_mb}MB)`;
            if (key === cfg.embedding.active_model) opt.selected = true;
            embedSelect.appendChild(opt);
        });
    } catch { configStatus.textContent = '⚠ Could not load config'; }
}

async function saveConfig() {
    try {
        const resp = await fetch(`${API}/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                llm_provider: document.getElementById('cfg-llm-provider').value,
                llm_model: document.getElementById('cfg-llm-model').value,
                embedding_model: document.getElementById('cfg-embed-model').value,
            }),
        });
        const data = await resp.json();
        configStatus.textContent = '✅ ' + (data.changes?.join(', ') || 'No changes');
    } catch { configStatus.textContent = '❌ Failed to save'; }
}
