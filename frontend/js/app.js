/**
 * app.js — Model Runner Frontend
 * Handles UI state, API communication, and WebSocket streaming.
 */

// ============================================================
// State
// ============================================================

const state = {
    apiBase: window.location.origin,
    models: [],
    currentModel: null,
    ws: null,
    chatHistory: [],
    settings: {
        n_ctx: 4096,
        n_gpu_layers: -1,
        device_map: 'auto',
    },
};

// ============================================================
// Utility Functions
// ============================================================

function $(sel) {
    return document.querySelector(sel);
}

function $$(sel) {
    return document.querySelectorAll(sel);
}

function formatSize(bytes) {
    if (!bytes) return 'Unknown';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function toast(message, type = 'info') {
    const container = $('#toastContainer');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

async function api(path, options = {}) {
    const url = `${state.apiBase}${path}`;
    const defaults = {
        headers: { 'Content-Type': 'application/json' },
    };
    const res = await fetch(url, { ...defaults, ...options });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
    }
    return res.json();
}

// ============================================================
// Navigation
// ============================================================

function switchPanel(panelId) {
    $$('.panel').forEach((p) => p.classList.remove('active'));
    $$('.nav-btn').forEach((b) => b.classList.remove('active'));

    const panel = $(`#panel-${panelId}`);
    if (panel) panel.classList.add('active');

    const btn = $(`.nav-btn[data-panel="${panelId}"]`);
    if (btn) btn.classList.add('active');

    // Refresh data on panel switch
    if (panelId === 'models') refreshModels();
    if (panelId === 'settings') refreshSystemInfo();
}

$$('.nav-btn').forEach((btn) => {
    btn.addEventListener('click', () => switchPanel(btn.dataset.panel));
});

// ============================================================
// Health Check
// ============================================================

async function checkHealth() {
    try {
        const status = await api('/api/health');
        $('#systemStatus').classList.add('connected');
        $('#systemStatus .status-text').textContent = 'Connected';
        updateCurrentModelDisplay();
    } catch {
        $('#systemStatus').classList.remove('connected');
        $('#systemStatus .status-text').textContent = 'Disconnected';
    }
}

setInterval(checkHealth, 5000);
checkHealth();

// ============================================================
// Models Management
// ============================================================

async function refreshModels() {
    try {
        state.models = await api('/api/models');
        renderModels();
        updateCurrentModelDisplay();
    } catch (err) {
        toast(`Failed to load models: ${err.message}`, 'error');
    }
}

function renderModels() {
    const container = $('#modelsList');

    if (state.models.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📦</span>
                <p>No models yet. Upload a model file or download from HuggingFace.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = state.models
        .map(
            (m) => `
        <div class="model-card ${m.loaded ? 'loaded' : ''}" data-id="${m.model_id}">
            <div class="model-info">
                <div class="model-name">${escapeHtml(m.name)}</div>
                <div class="model-meta">
                    <span class="model-badge ${m.format}">${m.format}</span>
                    <span>${formatSize(m.size_bytes)}</span>
                    ${m.loaded ? '<span style="color: var(--success)">● Active</span>' : ''}
                </div>
            </div>
            <div class="model-actions">
                ${
                    m.loaded
                        ? `<button class="btn btn-secondary btn-sm" onclick="unloadModel('${m.model_id}')">Unload</button>`
                        : `<button class="btn btn-primary btn-sm" onclick="loadModel('${m.model_id}')">Load</button>`
                }
                <button class="btn btn-danger btn-sm" onclick="deleteModel('${m.model_id}')">Delete</button>
            </div>
        </div>
    `
        )
        .join('');
}

async function loadModel(modelId) {
    try {
        toast(`Loading ${modelId}...`, 'info');
        await api('/api/models/load', {
            method: 'POST',
            body: JSON.stringify({
                model_id: modelId,
                n_ctx: state.settings.n_ctx,
                n_gpu_layers: state.settings.n_gpu_layers,
                device_map: state.settings.device_map,
            }),
        });
        toast(`Model loaded: ${modelId}`, 'success');
        await refreshModels();
    } catch (err) {
        toast(`Failed to load model: ${err.message}`, 'error');
    }
}

async function unloadModel(modelId) {
    try {
        await api(`/api/models/unload?model_id=${encodeURIComponent(modelId)}`, {
            method: 'POST',
        });
        toast(`Model unloaded: ${modelId}`, 'info');
        await refreshModels();
    } catch (err) {
        toast(`Failed to unload model: ${err.message}`, 'error');
    }
}

async function deleteModel(modelId) {
    if (!confirm(`Delete model "${modelId}"? This cannot be undone.`)) return;
    try {
        await api(`/api/models/${encodeURIComponent(modelId)}`, {
            method: 'DELETE',
        });
        toast(`Model deleted: ${modelId}`, 'success');
        await refreshModels();
    } catch (err) {
        toast(`Failed to delete model: ${err.message}`, 'error');
    }
}

function updateCurrentModelDisplay() {
    const loaded = state.models.find((m) => m.loaded);
    const el = $('#currentModel');
    if (loaded) {
        el.textContent = `Loaded: ${loaded.name}`;
        el.classList.add('active');
        state.currentModel = loaded.model_id;
        enableInference(true);
    } else {
        el.textContent = 'No model loaded';
        el.classList.remove('active');
        state.currentModel = null;
        enableInference(false);
    }
}

function enableInference(enabled) {
    $('#chatSendBtn').disabled = !enabled;
    $('#genBtn').disabled = !enabled;
}

$('#refreshModelsBtn').addEventListener('click', refreshModels);

// ============================================================
// Upload
// ============================================================

const dropZone = $('#dropZone');
const fileInput = $('#fileInput');
const uploadQueue = $('#uploadQueue');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
    fileInput.value = '';
});

async function handleFiles(files) {
    for (const file of files) {
        await uploadFile(file);
    }
}

async function uploadFile(file) {
    const id = Math.random().toString(36).slice(2, 10);
    const queueItem = document.createElement('div');
    queueItem.className = 'upload-item';
    queueItem.id = `upload-${id}`;
    queueItem.innerHTML = `
        <div class="upload-item-info">
            <div class="upload-item-name">${escapeHtml(file.name)}</div>
            <div class="upload-item-size">${formatSize(file.size)}</div>
        </div>
        <div class="upload-status uploading">Uploading...</div>
        <div class="progress-bar" style="width: 120px;">
            <div class="progress-fill" style="width: 0%"></div>
        </div>
    `;
    uploadQueue.prepend(queueItem);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${state.apiBase}/api/models/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail);
        }

        const result = await res.json();
        queueItem.querySelector('.upload-status').textContent = 'Complete';
        queueItem.querySelector('.upload-status').className = 'upload-status complete';
        queueItem.querySelector('.progress-fill').style.width = '100%';

        toast(`Model uploaded: ${result.name}`, 'success');
        await refreshModels();
    } catch (err) {
        queueItem.querySelector('.upload-status').textContent = `Error: ${err.message}`;
        queueItem.querySelector('.upload-status').className = 'upload-status error';
        toast(`Upload failed: ${err.message}`, 'error');
    }
}

// ============================================================
// HuggingFace
// ============================================================

let selectedHFModel = null;

$('#hfSearchBtn').addEventListener('click', searchHF);
$('#hfSearchInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') searchHF();
});

async function searchHF() {
    const query = $('#hfSearchInput').value.trim();
    if (!query) return;

    $('#hfResults').innerHTML =
        '<div class="empty-state"><p>Searching HuggingFace Hub...</p></div>';

    try {
        const data = await api(
            `/api/hf/search?q=${encodeURIComponent(query)}&limit=20`
        );

        if (data.results.length === 0) {
            $('#hfResults').innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">🔍</span>
                    <p>No models found for "${escapeHtml(query)}"</p>
                </div>
            `;
            return;
        }

        $('#hfResults').innerHTML = data.results
            .map(
                (m) => `
            <div class="hf-model-card" data-id="${escapeHtml(m.id)}">
                <div class="hf-model-header">
                    <span class="hf-model-name">${escapeHtml(m.name)}</span>
                </div>
                <div class="hf-model-author">${escapeHtml(m.author)}</div>
                <div class="hf-model-stats">
                    <span>⬇ ${formatNumber(m.downloads)}</span>
                    <span>❤ ${formatNumber(m.likes)}</span>
                    ${m.pipeline_tag ? `<span>📌 ${escapeHtml(m.pipeline_tag)}</span>` : ''}
                </div>
                <div class="hf-model-tags">
                    ${(m.tags || []).slice(0, 5).map((t) => `<span class="hf-tag">${escapeHtml(t)}</span>`).join('')}
                </div>
            </div>
        `
            )
            .join('');

        // Add click handlers
        $$('.hf-model-card').forEach((card) => {
            card.addEventListener('click', () => showHFDetail(card.dataset.id));
        });
    } catch (err) {
        toast(`Search failed: ${err.message}`, 'error');
    }
}

async function showHFDetail(repoId) {
    selectedHFModel = repoId;
    const detail = $('#hfModelDetail');
    detail.classList.remove('hidden');

    $('#hfDetailName').textContent = repoId;
    $('#hfDetailInfo').textContent = 'Loading model info...';

    try {
        const info = await api(`/api/hf/models/${encodeURIComponent(repoId)}`);

        // Populate format dropdown
        const formatSelect = $('#hfDownloadFormat');
        const fileSelect = $('#hfSpecificFile');

        if (info.files) {
            // Populate specific file dropdown
            fileSelect.innerHTML =
                '<option value="">Download all matching files</option>';
            const allFiles = info.files.gguf.length
                ? info.files.gguf
                : info.files.onnx.length
                ? info.files.onnx
                : [];
            allFiles.forEach((f) => {
                fileSelect.innerHTML += `<option value="${escapeHtml(f)}">${escapeHtml(f)}</option>`;
            });

            // Auto-select recommended format
            if (info.recommended_format) {
                formatSelect.value = info.recommended_format;
            }
        }

        $('#hfDetailInfo').innerHTML = `
            <p><strong>Downloads:</strong> ${formatNumber(info.downloads)} | <strong>Likes:</strong> ${formatNumber(info.likes)}</p>
            <p><strong>Library:</strong> ${info.library_name || 'Unknown'}</p>
            <p><strong>Recommended:</strong> ${info.recommended_format || 'Unknown'}</p>
            <p><strong>GGUF files:</strong> ${info.files?.gguf?.length || 0}</p>
            <p><strong>ONNX files:</strong> ${info.files?.onnx?.length || 0}</p>
        `;
    } catch (err) {
        $('#hfDetailInfo').textContent = `Failed to load: ${err.message}`;
    }
}

$('#hfDetailClose').addEventListener('click', () => {
    $('#hfModelDetail').classList.add('hidden');
    selectedHFModel = null;
});

$('#hfDownloadBtn').addEventListener('click', async () => {
    if (!selectedHFModel) return;

    const btn = $('#hfDownloadBtn');
    btn.disabled = true;
    btn.textContent = 'Downloading...';

    const progressEl = $('#hfDownloadProgress');
    progressEl.classList.remove('hidden');

    try {
        const result = await api('/api/hf/download', {
            method: 'POST',
            body: JSON.stringify({
                repo_id: selectedHFModel,
                format: $('#hfDownloadFormat').value,
                specific_file: $('#hfSpecificFile').value || null,
                token: $('#hfToken').value || null,
            }),
        });

        progressEl.querySelector('.progress-fill').style.width = '100%';
        progressEl.querySelector('.progress-text').textContent =
            'Download complete!';

        toast(`Downloaded: ${result.model_id}`, 'success');
        await refreshModels();

        // Reset after 3 seconds
        setTimeout(() => {
            $('#hfModelDetail').classList.add('hidden');
            progressEl.classList.add('hidden');
            btn.disabled = false;
            btn.textContent = 'Download Model';
        }, 2000);
    } catch (err) {
        toast(`Download failed: ${err.message}`, 'error');
        btn.disabled = false;
        btn.textContent = 'Download Model';
    }
});

// ============================================================
// Chat
// ============================================================

const chatInput = $('#chatInput');
const chatSendBtn = $('#chatSendBtn');
const chatMessages = $('#chatMessages');

chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
});

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
});

chatSendBtn.addEventListener('click', sendChatMessage);

$('#chatClearBtn').addEventListener('click', () => {
    state.chatHistory = [];
    chatMessages.innerHTML = `
        <div class="message system">
            <div class="message-content">Chat cleared. Start a new conversation!</div>
        </div>
    `;
});

// Update slider values
$('#chatTemp').addEventListener('input', (e) => {
    $('#chatTempVal').textContent = e.target.value;
});
$('#chatMaxTokens').addEventListener('input', (e) => {
    $('#chatMaxTokensVal').textContent = e.target.value;
});

async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text || !state.currentModel) return;

    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Add user message
    addChatMessage('user', text);
    state.chatHistory.push({ role: 'user', content: text });

    const isStreaming = $('#chatStream').checked;
    const maxTokens = parseInt($('#chatMaxTokens').value);
    const temperature = parseFloat($('#chatTemp').value);

    if (isStreaming) {
        await streamChatReply(text, maxTokens, temperature);
    } else {
        await getChatReply(text, maxTokens, temperature);
    }
}

function addChatMessage(role, content) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    msg.innerHTML = `
        <div class="message-avatar">${role === 'user' ? '👤' : '🦞'}</div>
        <div class="message-content">${escapeHtml(content)}</div>
    `;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg.querySelector('.message-content');
}

async function streamChatReply(prompt, maxTokens, temperature) {
    const replyEl = addChatMessage('assistant', '');
    let fullText = '';

    // Connect WebSocket
    const ws = new WebSocket(
        `ws://${window.location.host}/ws/generate`
    );

    ws.onopen = () => {
        ws.send(
            JSON.stringify({
                model_id: state.currentModel,
                prompt: buildChatPrompt(),
                max_tokens: maxTokens,
                temperature: temperature,
                stream: true,
            })
        );
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
            fullText += data.text;
            replyEl.textContent = fullText;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } else if (data.type === 'complete') {
            ws.close();
            state.chatHistory.push({ role: 'assistant', content: fullText });
        } else if (data.type === 'error') {
            replyEl.textContent += `\n[Error: ${data.error}]`;
            ws.close();
        }
    };

    ws.onerror = () => {
        replyEl.textContent += '\n[WebSocket error]';
    };
}

async function getChatReply(prompt, maxTokens, temperature) {
    const replyEl = addChatMessage('assistant', 'Generating...');

    try {
        const result = await api('/api/generate', {
            method: 'POST',
            body: JSON.stringify({
                model_id: state.currentModel,
                prompt: buildChatPrompt(),
                max_tokens: maxTokens,
                temperature: temperature,
            }),
        });

        replyEl.textContent = result.generated_text;
        state.chatHistory.push({
            role: 'assistant',
            content: result.generated_text,
        });
    } catch (err) {
        replyEl.textContent = `[Error: ${err.message}]`;
    }
}

function buildChatPrompt() {
    let prompt = '';
    for (const msg of state.chatHistory) {
        if (msg.role === 'system') prompt += `System: ${msg.content}\n`;
        else if (msg.role === 'user') prompt += `User: ${msg.content}\n`;
        else if (msg.role === 'assistant') prompt += `Assistant: ${msg.content}\n`;
    }
    prompt += 'Assistant: ';
    return prompt;
}

// ============================================================
// Generate Panel
// ============================================================

$('#genBtn').addEventListener('click', generateText);

async function generateText() {
    const prompt = $('#genPrompt').value.trim();
    if (!prompt || !state.currentModel) return;

    const outputEl = $('#genOutput');
    outputEl.innerHTML = '';

    const isStreaming = $('#genStream').checked;
    const maxTokens = parseInt($('#genMaxTokens').value);
    const temperature = parseFloat($('#genTemperature').value);
    const topP = parseFloat($('#genTopP').value);
    const topK = parseInt($('#genTopK').value);
    const stopStr = $('#genStop').value;
    const stop = stopStr ? stopStr.split(',').map((s) => s.trim()) : [];

    if (isStreaming) {
        await streamGenerate(prompt, maxTokens, temperature, topP, topK, stop, outputEl);
    } else {
        await singleGenerate(prompt, maxTokens, temperature, topP, topK, stop, outputEl);
    }
}

async function streamGenerate(prompt, maxTokens, temperature, topP, topK, stop, outputEl) {
    let fullText = '';

    const ws = new WebSocket(`ws://${window.location.host}/ws/generate`);

    ws.onopen = () => {
        outputEl.textContent = '';
        ws.send(
            JSON.stringify({
                model_id: state.currentModel,
                prompt: prompt,
                max_tokens: maxTokens,
                temperature: temperature,
                top_p: topP,
                top_k: topK,
                stop: stop,
                stream: true,
            })
        );
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
            fullText += data.text;
            outputEl.textContent = fullText;
            outputEl.scrollTop = outputEl.scrollHeight;
        } else if (data.type === 'complete') {
            ws.close();
            toast('Generation complete', 'success');
        } else if (data.type === 'error') {
            outputEl.textContent += `\n\n[Error: ${data.error}]`;
            ws.close();
            toast('Generation failed', 'error');
        }
    };

    ws.onerror = () => {
        outputEl.textContent = 'WebSocket error. Check server connection.';
        toast('WebSocket error', 'error');
    };
}

async function singleGenerate(prompt, maxTokens, temperature, topP, topK, stop, outputEl) {
    outputEl.textContent = 'Generating...';

    try {
        const result = await api('/api/generate', {
            method: 'POST',
            body: JSON.stringify({
                model_id: state.currentModel,
                prompt: prompt,
                max_tokens: maxTokens,
                temperature: temperature,
                top_p: topP,
                top_k: topK,
                stop: stop,
            }),
        });

        outputEl.textContent = result.generated_text;
        toast('Generation complete', 'success');
    } catch (err) {
        outputEl.textContent = `[Error: ${err.message}]`;
        toast(`Generation failed: ${err.message}`, 'error');
    }
}

// ============================================================
// Settings
// ============================================================

async function refreshSystemInfo() {
    try {
        const info = await api('/api/system/info');
        const el = $('#systemInfo');
        el.innerHTML = `
            <div class="system-info-item">
                <div class="system-info-label">CPU</div>
                <div class="system-info-value">${info.cpu_count} cores @ ${info.cpu_percent}%</div>
            </div>
            <div class="system-info-item">
                <div class="system-info-label">Memory</div>
                <div class="system-info-value">${info.memory_used_gb} / ${info.memory_total_gb} GB (${info.memory_percent}%)</div>
            </div>
            <div class="system-info-item">
                <div class="system-info-label">Current Model</div>
                <div class="system-info-value">${info.current_model || 'None loaded'}</div>
            </div>
            <div class="system-info-item">
                <div class="system-info-label">GPU</div>
                <div class="system-info-value">${
                    info.gpu_info.length > 0
                        ? info.gpu_info.map((g) => g.name || g.type).join(', ')
                        : 'None detected'
                }</div>
            </div>
        `;

        // Set API base URL
        $('#apiBaseUrl').value = state.apiBase;
    } catch (err) {
        $('#systemInfo').innerHTML = `<p style="color: var(--error)">Failed to load: ${err.message}</p>`;
    }
}

$('#saveSettingsBtn').addEventListener('click', () => {
    state.settings.n_ctx = parseInt($('#defaultNCtx').value);
    state.settings.n_gpu_layers = parseInt($('#defaultGpuLayers').value);
    state.settings.device_map = $('#defaultDeviceMap').value;
    state.apiBase = $('#apiBaseUrl').value.replace(/\/+$/, '');
    toast('Settings saved', 'success');
});

// ============================================================
// Helpers
// ============================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatNumber(n) {
    if (!n) return '0';
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toString();
}

// ============================================================
// Initialize
// ============================================================

refreshModels();
