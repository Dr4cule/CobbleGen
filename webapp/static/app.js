'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const api = async (path, opts) => {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
};

const state = {
  voices: [],
  assets: { stories: [], footage: [], music: [] },
  settings: [],
  storySource: 'paste',
  activeStreams: {},
};

function toast(msg, kind = '') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast ' + kind;
  setTimeout(() => el.classList.add('hidden'), 3200);
}

/* ---------------- Navigation ---------------- */
$$('.nav-item').forEach((btn) => {
  btn.addEventListener('click', () => {
    $$('.nav-item').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    const view = btn.dataset.view;
    if (view !== 'library') closeModal();
    stopDetachedMedia();
    $$('.view').forEach((v) => v.classList.remove('active'));
    $('#view-' + view).classList.add('active');
    if (view === 'library') loadLibrary();
    if (view === 'assets') renderAssets();
    if (view === 'create') refreshJobs();
  });
});

/* ---------------- Health ---------------- */
async function loadHealth() {
  try {
    const h = await api('/api/health');
    $('#health').innerHTML =
      `<span class="dot"></span><b>Online</b><br>TTS: ${h.tts_backend}<br>` +
      `Footage: ${h.footage_count} / Music: ${h.music_count}<br>` +
      `Ollama: ${h.ollama_hosts.length} host(s)`;
  } catch (e) {
    $('#health').innerHTML = `<span class="dot down"></span><b>Offline</b>`;
  }
}

/* ---------------- Story input ---------------- */
const storyText = $('#story-text');
storyText.addEventListener('input', () => {
  const words = storyText.value.trim().split(/\s+/).filter(Boolean).length;
  $('#word-count').textContent = words + ' words';
  $('#dur-est').textContent = Math.round((words / 145) * 60) + 's';
});

$$('#story-source-seg .seg-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    $$('#story-source-seg .seg-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    state.storySource = btn.dataset.src;
    $('#story-paste').classList.toggle('hidden', state.storySource !== 'paste');
    $('#story-existing').classList.toggle('hidden', state.storySource !== 'existing');
  });
});

$('#story-file').addEventListener('change', (e) => {
  const story = state.assets.stories.find((s) => s.name === e.target.value);
  $('#story-preview').textContent = story ? story.preview + '...' : '';
});

/* ---------------- Voices ---------------- */
function voiceSortKey(v) {
  // Surface English (US/GB) voices first, then everything else alphabetically.
  const loc = (v.locale || '') + ' ' + v.id;
  if (/EN-US|en-US/.test(loc)) return '0' + v.id;
  if (/EN-GB|en-GB|EN-/.test(loc)) return '1' + v.id;
  return '2' + v.id;
}

function renderVoiceOptions(filter) {
  const sel = $('#voice-select');
  const f = (filter || '').trim().toLowerCase();
  const matches = state.voices
    .filter((v) => !f || (v.id + ' ' + v.gender + ' ' + v.locale).toLowerCase().includes(f))
    .sort((a, b) => voiceSortKey(a).localeCompare(voiceSortKey(b)));
  const keep = sel.value;
  sel.innerHTML = matches.slice(0, 300).map((v) => {
    // ElevenLabs ids arrive as "id|Name"; show the friendly name + description.
    const label = v.id.includes('|') ? v.id.split('|')[1] : v.id;
    const extra = [v.gender, v.locale].filter(Boolean).join(' / ');
    return `<option value="${v.id}">${label}${extra ? ' / ' + extra : ''}</option>`;
  }).join('');
  if (matches.some((v) => v.id === keep)) sel.value = keep;
  else if (matches.some((v) => v.id === state.currentVoice)) sel.value = state.currentVoice;
}

async function loadEngines() {
  try {
    const data = await api('/api/engines');
    state.engines = data.engines || [];
    state.engine = data.current || (state.engines[0] && state.engines[0].id) || 'nvidia_magpie';
    const sel = $('#engine-select');
    sel.innerHTML = state.engines.map((e) => `<option value="${e.id}">${e.label}</option>`).join('');
    sel.value = state.engine;
    updateEngineNote();
    sel.addEventListener('change', async () => {
      state.engine = sel.value;
      updateEngineNote();
      $('#voice-filter').value = '';
      await loadVoices();
    });
  } catch (e) {
    $('#engine-select').innerHTML = '<option value="nvidia_magpie">NVIDIA Magpie</option>';
  }
}

function updateEngineNote() {
  const eng = (state.engines || []).find((e) => e.id === state.engine);
  $('#engine-note').textContent = eng ? eng.note : '';
}

async function loadVoices() {
  const sel = $('#voice-select');
  sel.innerHTML = '<option value="">Loading voices…</option>';
  try {
    const data = await api('/api/voices?engine=' + encodeURIComponent(state.engine || ''));
    state.voices = data.voices || [];
    state.currentVoice = data.current || '';
    if (data.error) toast('Voices: ' + data.error, 'bad');
    if (!state.voices.length) {
      sel.innerHTML = `<option value="">(default: ${data.current || 'backend voice'})</option>`;
      return;
    }
    renderVoiceOptions('');
    sel.value = state.currentVoice || sel.options[0]?.value || '';
  } catch (e) {
    sel.innerHTML = '<option value="">(voices unavailable)</option>';
  }
}

const voiceFilterEl = document.getElementById('voice-filter');
if (voiceFilterEl) voiceFilterEl.addEventListener('input', (e) => renderVoiceOptions(e.target.value));

$('#voice-preview-btn').addEventListener('click', async () => {
  const voice = $('#voice-select').value;
  const btn = $('#voice-preview-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  try {
    const params = new URLSearchParams();
    if (voice) params.set('voice', voice);
    if (state.engine) params.set('engine', state.engine);
    const res = await fetch('/api/voice-preview?' + params.toString());
    if (!res.ok) throw new Error('Preview failed');
    const blob = await res.blob();
    const audio = $('#voice-audio');
    audio.src = URL.createObjectURL(blob);
    audio.play();
  } catch (e) {
    toast('Voice preview failed: ' + e.message, 'bad');
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Play Preview';
  }
});

/* ---------------- Assets / selects ---------------- */
async function loadAssets() {
  state.assets = await api('/api/assets');
  // Story file dropdown
  const sf = $('#story-file');
  sf.innerHTML = state.assets.stories.map((s) => `<option value="${s.name}">${s.name} (${s.words}w)</option>`).join('') || '<option value="">No stories</option>';
  sf.dispatchEvent(new Event('change'));
  // Music dropdown
  const mu = $('#music-select');
  mu.innerHTML = '<option value="">Random</option>' + state.assets.music.map((m) => `<option value="${m.name}">${m.name}</option>`).join('');
  // Footage dropdown
  const fo = $('#footage-select');
  fo.innerHTML = '<option value="">Auto (least used)</option>' + state.assets.footage.map((f) => `<option value="${f.name}">${f.name} (${f.size_mb}MB)</option>`).join('');
}

function renderAssets() {
  const render = (kind, list) => {
    const ul = $('#asset-' + kind);
    if (!list.length) { ul.innerHTML = '<li style="color:var(--text-dim)">Empty</li>'; return; }
    ul.innerHTML = list.map((item) => `
      <li>
        <span class="a-name" title="${item.name}">${item.name}</span>
        <span class="a-size">${item.size_mb ? item.size_mb + 'MB' : (item.words || '') + 'w'}</span>
        <button class="x-btn" data-kind="${kind}" data-name="${item.name}">X</button>
      </li>`).join('');
  };
  render('footage', state.assets.footage);
  render('music', state.assets.music);
  render('stories', state.assets.stories);
  $$('.asset-list .x-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      if (!confirm(`Delete ${btn.dataset.name}?`)) return;
      try {
        await api(`/api/asset/${btn.dataset.kind}/${encodeURIComponent(btn.dataset.name)}`, { method: 'DELETE' });
        await loadAssets();
        renderAssets();
        toast('Deleted', 'good');
      } catch (e) { toast(e.message, 'bad'); }
    });
  });
}

/* Upload (drop + click) */
function wireUploads() {
  $$('.upload-card').forEach((card) => {
    const kind = card.dataset.kind;
    const dz = $('.dropzone', card);
    const input = $('.file-input', card);
    dz.addEventListener('click', () => input.click());
    input.addEventListener('change', () => { if (input.files[0]) uploadFile(kind, input.files[0]); });
    ['dragover', 'dragenter'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
    dz.addEventListener('drop', (e) => { if (e.dataTransfer.files[0]) uploadFile(kind, e.dataTransfer.files[0]); });
  });
}

async function uploadFile(kind, file) {
  const fd = new FormData();
  fd.append('file', file);
  toast(`Uploading ${file.name}...`);
  try {
    await api('/api/upload/' + kind, { method: 'POST', body: fd });
    await loadAssets();
    renderAssets();
    toast('Uploaded ' + file.name, 'good');
  } catch (e) { toast('Upload failed: ' + e.message, 'bad'); }
}

/* ---------------- Quick settings + Settings view ---------------- */
const QUICK_KEYS = ['STORY_TARGET_WPM', 'BACKGROUND_MUSIC_VOLUME', 'SUBTITLE_FONT_SIZE', 'SHOW_INTRO_CARD'];

async function loadSettings() {
  const data = await api('/api/settings');
  state.settings = data.fields;
  renderQuickSettings();
  renderSettingsView();
}

function settingControl(field, idPrefix) {
  const id = idPrefix + field.key;
  if (field.type === 'bool') {
    const checked = (field.value === true || field.value === 'true') ? 'checked' : '';
    return `<label class="switch"><input type="checkbox" id="${id}" data-key="${field.key}" data-type="bool" ${checked}><span></span></label>`;
  }
  if (field.type === 'int' || field.type === 'float') {
    return `<input type="number" id="${id}" data-key="${field.key}" data-type="${field.type}" value="${field.value}" ${field.min !== undefined ? `min="${field.min}"` : ''} ${field.max !== undefined ? `max="${field.max}"` : ''} step="${field.type === 'float' ? '0.01' : '1'}">`;
  }
  return `<input type="text" id="${id}" data-key="${field.key}" data-type="str" value="${field.value}">`;
}

function renderQuickSettings() {
  const wrap = $('#quick-settings');
  wrap.innerHTML = '';
  state.settings.filter((f) => QUICK_KEYS.includes(f.key)).forEach((field) => {
    const row = document.createElement('div');
    row.className = 'qs-row';
    row.innerHTML = `<label>${field.label}</label>` + settingControl(field, 'quick_');
    wrap.appendChild(row);
  });
}

function renderSettingsView() {
  const groups = {};
  state.settings.forEach((f) => { (groups[f.group] = groups[f.group] || []).push(f); });
  const wrap = $('#settings-form');
  wrap.innerHTML = '';
  Object.entries(groups).forEach(([group, fields]) => {
    const card = document.createElement('div');
    card.className = 'card settings-group';
    card.innerHTML = `<h3>${group}</h3>` + fields.map((f) =>
      `<div class="set-row"><label>${f.label}</label>${settingControl(f, 'set_')}</div>`).join('');
    wrap.appendChild(card);
  });
}

/* Collect overrides from both quick + settings inputs (settings view wins) */
function collectSettings() {
  const out = {};
  $$('[data-key]').forEach((el) => {
    const key = el.dataset.key;
    let val;
    if (el.dataset.type === 'bool') val = el.checked ? 'true' : 'false';
    else val = el.value;
    if (val !== '' && val !== null && val !== undefined) out[key] = val;
  });
  return out;
}

/* ---------------- Generate ---------------- */
$('#generate-btn').addEventListener('click', async () => {
  const btn = $('#generate-btn');
  const payload = { settings: collectSettings() };

  if (state.storySource === 'existing') {
    const file = $('#story-file').value;
    if (!file) { toast('Pick a story file', 'bad'); return; }
    payload.story_file = file;
  } else {
    const text = storyText.value.trim();
    if (text.length < 40) { toast('Story is too short', 'bad'); return; }
    payload.story_text = text;
    payload.story_name = $('#story-title').value.trim();
  }
  if (state.engine) payload.engine = state.engine;
  const voice = $('#voice-select').value; if (voice) payload.voice = voice;
  const music = $('#music-select').value; if (music) payload.music_file = music;
  const footage = $('#footage-select').value; if (footage) payload.footage_file = footage;

  btn.disabled = true;
  btn.textContent = 'Queuing...';
  try {
    const res = await api('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    toast('Reel queued!', 'good');
    $('#generate-msg').textContent = 'Job ' + res.job_id + ' started.';
    streamJob(res.job_id);
    if (state.storySource === 'paste') await loadAssets();
  } catch (e) {
    toast('Failed: ' + e.message, 'bad');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate Reel';
  }
});

/* ---------------- Jobs ---------------- */
const STEP_NAMES = ['Prep', 'AI story', 'Narration', 'Subtitles', 'Images', 'Footage', 'Render'];

function fmtElapsed(job) {
  const start = job.started_at || job.created_at;
  const end = job.finished_at || (Date.now() / 1000);
  if (!start) return '';
  const s = Math.max(0, Math.round(end - start));
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
}

function stepTimeline(job) {
  const done = ['done', 'error', 'cancelled'].includes(job.status);
  return `<div class="timeline">` + STEP_NAMES.map((name, i) => {
    const n = i + 1;
    let cls = 'pending';
    if (job.step > n || (done && job.status === 'done')) cls = 'complete';
    else if (job.step === n && job.status === 'running') cls = 'current';
    else if (job.step === n && job.status === 'error') cls = 'failed';
    return `<div class="tl-step ${cls}" title="${name}"><span class="tl-dot"></span><span class="tl-name">${name}</span></div>`;
  }).join('') + `</div>`;
}

function jobCard(job) {
  const pct = Math.round((job.step / job.step_count) * 100);
  let actions = '';
  if (job.status === 'running' || job.status === 'queued') {
    actions = `<button class="btn-danger" data-cancel="${job.id}">Cancel</button>`;
  } else if (job.status === 'done' && job.output_video) {
    actions = `<button class="btn-ghost" data-open="${job.output_video}">Open</button>
               <a class="btn-ghost" href="/api/video/${encodeURIComponent(job.output_video)}" download>Download</a>`;
  } else if (job.status === 'error') {
    actions = `<button class="btn-ghost" data-retry="${job.id}">Dismiss</button>`;
  }
  const detail = job.status === 'error' ? `<span style="color:var(--bad)">${job.error || 'Error'}</span>`
    : (job.title || job.step_label);
  const spin = job.status === 'running' ? '<span class="spinner"></span> ' : '';
  const logs = (job.logs || []).filter((l) => /warn|error|fail|fallback|salvage|unhealthy/i.test(l)).slice(-4);
  const logHtml = logs.length ? `<details class="job-logs"><summary>${logs.length} notice(s)</summary><pre>${logs.map(escapeHtml).join('\n')}</pre></details>` : '';
  return `
    <div class="job ${job.status}" id="job-${job.id}">
      <div class="job-top">
        <div class="job-title">${spin}${escapeHtml(job.title || job.story_name)}</div>
        <span class="job-status ${job.status}">${job.status}</span>
      </div>
      ${stepTimeline(job)}
      <div class="progress"><div class="progress-bar" style="width:${pct}%"></div></div>
      <div class="job-step"><span>${escapeHtml(job.step_label)}</span><span>${job.step}/${job.step_count} / ${fmtElapsed(job)}</span></div>
      <div class="job-detail hint">${detail}</div>
      ${logHtml}
      <div class="job-actions">${actions}</div>
    </div>`;
}

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function upsertJobCard(job) {
  const strip = $('#active-jobs');
  let el = $('#job-' + job.id);
  state.jobCache = state.jobCache || {};
  const prev = state.jobCache[job.id];
  state.jobCache[job.id] = job;
  // Preserve an open log <details> across re-renders to avoid flicker.
  const wasOpen = el && $('.job-logs', el)?.open;
  const html = jobCard(job);
  if (el) {
    el.outerHTML = html;
    const fresh = $('#job-' + job.id);
    if (wasOpen && fresh) { const d = $('.job-logs', fresh); if (d) d.open = true; }
  } else {
    strip.insertAdjacentHTML('afterbegin', html);
  }
  wireJobActions();
}

/* Tick elapsed timers for running jobs every second without a server round-trip. */
setInterval(() => {
  $$('.job.running').forEach((card) => {
    const id = card.id.replace('job-', '');
    const job = (state.jobCache || {})[id];
    if (!job) return;
    const stepEl = $('.job-step span:last-child', card);
    if (stepEl) stepEl.textContent = `${job.step}/${job.step_count} / ${fmtElapsed(job)}`;
  });
}, 1000);

function wireJobActions() {
  $$('[data-cancel]').forEach((b) => b.onclick = async () => {
    try { await api(`/api/jobs/${b.dataset.cancel}/cancel`, { method: 'POST' }); toast('Cancelled'); } catch (e) { toast(e.message, 'bad'); }
  });
  $$('[data-open]').forEach((b) => b.onclick = () => openReel(b.dataset.open));
  $$('[data-retry]').forEach((b) => b.onclick = () => { const el = $('#job-' + b.dataset.retry); if (el) el.remove(); });
}

function streamJob(jobId) {
  if (state.activeStreams[jobId]) return;
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  state.activeStreams[jobId] = es;
  es.onmessage = (e) => {
    const job = JSON.parse(e.data);
    upsertJobCard(job);
    if (['done', 'error', 'cancelled'].includes(job.status)) {
      es.close();
      delete state.activeStreams[jobId];
      if (job.status === 'done') {
        toast('Reel ready: ' + (job.title || job.story_name), 'good');
        if ($('#view-library').classList.contains('active')) loadLibrary();
      }
      if (job.status === 'error') { toast('Generation failed - see notices', 'bad'); }
    }
  };
  es.onerror = () => { es.close(); delete state.activeStreams[jobId]; setTimeout(() => refreshJobs(), 1500); };
}

async function refreshJobs() {
  try {
    const data = await api('/api/jobs');
    const strip = $('#active-jobs');
    strip.innerHTML = '';
    data.jobs.slice(0, 8).forEach((job) => {
      strip.insertAdjacentHTML('beforeend', jobCard(job));
      if (job.status === 'running' || job.status === 'queued') streamJob(job.id);
    });
    wireJobActions();
  } catch (e) {}
}

/* ---------------- Library ---------------- */
async function loadLibrary() {
  const data = await api('/api/library');
  const grid = $('#library-grid');
  $('#library-empty').classList.toggle('hidden', data.reels.length > 0);
  grid.innerHTML = data.reels.map((r) => `
    <div class="reel-card" data-name="${r.name}">
      <video class="reel-thumb" src="/api/video/${encodeURIComponent(r.name)}#t=2" preload="metadata" muted></video>
      <div class="reel-info">
        <h4>${r.title || r.name}</h4>
        <div class="reel-meta-line">${r.size_mb}MB / ${r.name}</div>
      </div>
    </div>`).join('');
  $$('.reel-card').forEach((c) => c.addEventListener('click', () => openReel(c.dataset.name)));
}

async function openReel(name) {
  const data = await api('/api/library');
  const reel = data.reels.find((r) => r.name === name) || { name, title: name };
  const tags = (reel.hashtags || '').trim();
  const content = `
    <div class="modal-grid">
      <div><video src="/api/video/${encodeURIComponent(name)}" controls autoplay></video></div>
      <div>
        <h2 style="margin-top:0">${reel.title || name}</h2>
        ${metaBlock('Hook', reel.hook)}
        ${metaBlock('Description', reel.description, true)}
        ${tags ? `<div class="meta-block"><div class="meta-key">Hashtags</div><div class="copy-row"><div class="meta-val">${tags}</div><button class="btn-ghost" data-copy="${encodeURIComponent(tags)}">Copy</button></div></div>` : ''}
        <div class="job-actions" style="margin-top:18px">
          <a class="btn-primary" href="/api/video/${encodeURIComponent(name)}" download>Download</a>
          <button class="btn-danger" data-del="${name}">Delete</button>
        </div>
      </div>
    </div>`;
  $('.modal-content').innerHTML = content;
  $('#modal').classList.remove('hidden');
  $$('[data-copy]').forEach((b) => b.onclick = () => { navigator.clipboard.writeText(decodeURIComponent(b.dataset.copy)); toast('Copied hashtags', 'good'); });
  $$('[data-del]').forEach((b) => b.onclick = async () => {
    if (!confirm('Delete this reel?')) return;
    try { await api('/api/library/' + encodeURIComponent(b.dataset.del), { method: 'DELETE' }); closeModal(); loadLibrary(); toast('Deleted', 'good'); } catch (e) { toast(e.message, 'bad'); }
  });
}

function closeModal() {
  const modal = $('#modal');
  if (modal.classList.contains('hidden')) return;
  // Hard-stop any playing media before removing it from the DOM, so audio can
  // never keep playing after the modal closes.
  $$('#modal video, #modal audio').forEach((m) => {
    try { m.pause(); m.currentTime = 0; m.removeAttribute('src'); m.load(); } catch (e) {}
  });
  modal.classList.add('hidden');
  $('.modal-content').innerHTML = '';
}

function stopDetachedMedia() {
  $$('video, audio').forEach((m) => {
    if (m.closest('#modal') || m.id === 'voice-audio') return;
    try {
      m.pause();
      if (!m.classList.contains('reel-thumb')) m.currentTime = 0;
    } catch (e) {}
  });
}

function metaBlock(key, val, copyable) {
  if (!val) return '';
  const copyBtn = copyable ? `<button class="btn-ghost" data-copy="${encodeURIComponent(val)}">Copy</button>` : '';
  return `<div class="meta-block"><div class="meta-key">${key}</div><div class="copy-row"><div class="meta-val">${val}</div>${copyBtn}</div></div>`;
}

$('.modal-close').addEventListener('click', closeModal);
$('.modal-backdrop').addEventListener('click', closeModal);

/* ---------------- Init ---------------- */
/* Ctrl/Cmd+Enter generates from anywhere on the Create view */
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && $('#view-create').classList.contains('active')) {
    e.preventDefault();
    $('#generate-btn').click();
  }
  if (e.key === 'Escape') closeModal();
});

async function init() {
  await loadHealth();
  await loadEngines();
  await Promise.all([loadVoices(), loadAssets(), loadSettings()]);
  await refreshJobs();
  wireUploads();
  setInterval(loadHealth, 20000);
}
init();

