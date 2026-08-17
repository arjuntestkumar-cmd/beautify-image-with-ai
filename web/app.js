/* Beautify — upload, poll, compare, download. Vanilla JS, no build step. */
(() => {
  'use strict';

  const MAX_BYTES = 20 * 1024 * 1024;
  const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp'];
  const POLL_MS = 900;

  const $ = (id) => document.getElementById(id);

  const views = {
    idle: $('view-idle'),
    selected: $('view-selected'),
    processing: $('view-processing'),
    done: $('view-done'),
    error: $('view-error'),
  };

  const state = {
    file: null,
    objectUrl: null,
    jobId: null,
    polling: null,
    shown: 0,        // the percentage currently painted on the bar
    target: 0,       // the percentage the server last reported
  };

  // ── view switching ──────────────────────────────────────────────────
  function show(name) {
    for (const [key, el] of Object.entries(views)) el.classList.toggle('hidden', key !== name);
  }

  function fmtBytes(n) {
    if (!n && n !== 0) return '—';
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  // ── service health ──────────────────────────────────────────────────
  async function checkHealth() {
    const pill = $('status-pill');
    try {
      const res = await fetch('/health');
      const h = await res.json();
      if (h.mockMode) {
        pill.textContent = 'mock mode — no AI';
        pill.className = 'pill warn';
      } else if (h.ready) {
        pill.textContent = `ready · ${h.device}`;
        pill.className = 'pill ok';
      } else {
        pill.textContent = 'models not loaded';
        pill.className = 'pill bad';
      }
    } catch {
      pill.textContent = 'offline';
      pill.className = 'pill bad';
    }
  }

  // ── choosing a file ─────────────────────────────────────────────────
  function selectFile(file) {
    if (!file) return;
    if (!ACCEPTED.includes(file.type)) {
      return fail('Unsupported file', 'Please choose a JPG, PNG or WebP image.');
    }
    if (file.size > MAX_BYTES) {
      return fail('That image is too large', `The limit is ${MAX_BYTES / 1024 / 1024} MB — this one is ${fmtBytes(file.size)}.`);
    }

    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.file = file;
    state.objectUrl = URL.createObjectURL(file);

    $('preview').src = state.objectUrl;
    $('processing-preview').src = state.objectUrl;
    $('file-name').textContent = file.name;

    const img = new Image();
    img.onload = () => {
      $('file-meta').textContent = `${img.naturalWidth} × ${img.naturalHeight} · ${fmtBytes(file.size)}`;
    };
    img.src = state.objectUrl;
    $('file-meta').textContent = fmtBytes(file.size);

    show('selected');
  }

  const drop = $('drop');
  const fileInput = $('file');

  drop.addEventListener('click', () => fileInput.click());
  drop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', (e) => selectFile(e.target.files[0]));

  ['dragenter', 'dragover'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => selectFile(e.dataTransfer.files[0]));

  // Dropping anywhere on the page works too, as long as we are not mid-job.
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    if (!views.processing.classList.contains('hidden')) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
  });

  window.addEventListener('paste', (e) => {
    if (!views.processing.classList.contains('hidden')) return;
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) selectFile(item.getAsFile());
  });

  $('btn-change').addEventListener('click', () => { fileInput.value = ''; show('idle'); });

  // ── enhancing ───────────────────────────────────────────────────────
  $('btn-enhance').addEventListener('click', enhance);

  async function enhance() {
    if (!state.file) return;
    state.shown = 0;
    state.target = 5;
    paintProgress(5, 'Uploading…', 'Sending your photo to the enhancer.');
    show('processing');

    const body = new FormData();
    body.append('image', state.file, state.file.name);

    let data;
    try {
      const res = await fetch('/api/enhance', { method: 'POST', body });
      data = await res.json();
      if (!res.ok || !data.success) {
        const err = data.error || {};
        return fail('Upload rejected', err.message || 'The server refused that image.');
      }
    } catch {
      return fail('Cannot reach the server', 'The backend is not responding. Is it still running?');
    }

    state.jobId = data.data.jobId;
    poll();
  }

  function poll() {
    clearInterval(state.polling);
    state.polling = setInterval(tick, POLL_MS);
    tick();
  }

  async function tick() {
    if (!state.jobId) return;
    let job;
    try {
      const res = await fetch(`/api/jobs/${state.jobId}`);
      const payload = await res.json();
      if (!res.ok || !payload.success) {
        clearInterval(state.polling);
        const err = payload.error || {};
        return fail('Lost the job', err.message || 'The job could not be found.');
      }
      job = payload.data;
    } catch {
      return; // a single dropped poll is not fatal — try again next tick
    }

    state.target = job.progress || 0;
    paintProgress(state.target, stageLabel(job.stage), job.message);

    if (job.status === 'completed') {
      clearInterval(state.polling);
      finish(job);
    } else if (job.status === 'failed') {
      clearInterval(state.polling);
      const err = job.error || {};
      fail('Could not enhance that photo', err.message || 'The enhancement failed.');
    }
  }

  function stageLabel(stage) {
    return ({
      queued: 'Queued…',
      starting: 'Starting…',
      analysing: 'Analysing the photo…',
      restoring_faces: 'Restoring faces…',
      blending: 'Blending faces…',
      enhancing: 'Enhancing detail…',
      finishing: 'Finishing…',
      encoding: 'Saving…',
      completed: 'Done',
    })[stage] || 'Working…';
  }

  // The server reports real stage boundaries; between them we creep a little so the bar never
  // looks stuck during the long inference step.
  function paintProgress(pct, label, hint) {
    // Creep forward between polls, but never past 99 until the server actually says 100.
    state.shown = pct >= 100 ? 100 : Math.min(99, Math.max(state.shown + 0.6, pct));
    $('bar').style.width = `${state.shown}%`;
    $('bar-outer').setAttribute('aria-valuenow', Math.round(state.shown));
    $('progress-pct').textContent = `${Math.round(state.shown)}%`;
    $('stage-text').textContent = label;
    if (hint) $('progress-hint').textContent = hint;
  }

  // ── result ──────────────────────────────────────────────────────────
  function finish(job) {
    paintProgress(100, 'Done', 'Finished.');
    const resultUrl = `/api/jobs/${job.jobId}/result`;

    const after = $('img-after');
    const before = $('img-before');
    before.src = state.objectUrl;

    after.onload = () => {
      setSplit(50);
      renderStats(job);
      const dl = $('btn-download');
      dl.href = resultUrl;
      const stem = (state.file?.name || 'image').replace(/\.[^.]+$/, '');
      const ext = (job.output?.format || 'image/jpeg').split('/')[1].replace('jpeg', 'jpg');
      dl.setAttribute('download', `${stem}-beautified.${ext}`);
      show('done');
    };
    after.onerror = () => fail('Could not load the result', 'The enhanced image could not be fetched.');
    after.src = resultUrl;
  }

  function renderStats(job) {
    const o = job.output || {};
    const i = job.input || {};
    const d = job.details || {};
    const chips = [];

    if (i.width && o.width) chips.push(`<span class="stat"><b>${i.width}×${i.height}</b> → <b>${o.width}×${o.height}</b></span>`);
    if (o.scale > 1) chips.push(`<span class="stat">upscaled <b>${o.scale}×</b></span>`);
    if (d.facesRestored > 0) chips.push(`<span class="stat"><b>${d.facesRestored}</b> face${d.facesRestored > 1 ? 's' : ''} restored</span>`);
    else if (d.faceCount > 0) chips.push(`<span class="stat"><b>${d.faceCount}</b> face${d.faceCount > 1 ? 's' : ''} detected</span>`);
    if (o.bytes) chips.push(`<span class="stat">${fmtBytes(o.bytes)}</span>`);
    if (d.processingTimeMs) chips.push(`<span class="stat">in <b>${(d.processingTimeMs / 1000).toFixed(1)}s</b></span>`);

    $('stats').innerHTML = chips.join('');
  }

  $('btn-again').addEventListener('click', reset);
  $('btn-retry').addEventListener('click', reset);

  function reset() {
    clearInterval(state.polling);
    state.jobId = null;
    state.file = null;
    fileInput.value = '';
    if (state.objectUrl) { URL.revokeObjectURL(state.objectUrl); state.objectUrl = null; }
    $('img-after').removeAttribute('src');
    $('img-before').removeAttribute('src');
    show('idle');
  }

  function fail(title, message) {
    clearInterval(state.polling);
    $('error-title').textContent = title;
    $('error-message').textContent = message;
    show('error');
  }

  // ── before / after slider ───────────────────────────────────────────
  const compare = $('compare');
  const handle = $('handle');
  let dragging = false;

  function setSplit(pct) {
    const v = Math.max(0, Math.min(100, pct));
    $('cmp-after').style.setProperty('--split', `${v}%`);
    handle.style.left = `${v}%`;
    handle.setAttribute('aria-valuenow', Math.round(v));
  }

  function splitFromEvent(e) {
    const rect = compare.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    setSplit((x / rect.width) * 100);
  }

  compare.addEventListener('pointerdown', (e) => {
    dragging = true;
    compare.setPointerCapture(e.pointerId);
    splitFromEvent(e);
  });
  compare.addEventListener('pointermove', (e) => { if (dragging) splitFromEvent(e); });
  ['pointerup', 'pointercancel'].forEach((ev) =>
    compare.addEventListener(ev, (e) => {
      dragging = false;
      try { compare.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    }));

  handle.addEventListener('keydown', (e) => {
    const now = Number(handle.getAttribute('aria-valuenow')) || 50;
    const step = e.shiftKey ? 10 : 2;
    if (e.key === 'ArrowLeft') { e.preventDefault(); setSplit(now - step); }
    if (e.key === 'ArrowRight') { e.preventDefault(); setSplit(now + step); }
    if (e.key === 'Home') { e.preventDefault(); setSplit(0); }
    if (e.key === 'End') { e.preventDefault(); setSplit(100); }
  });

  // ── boot ────────────────────────────────────────────────────────────
  checkHealth();
  setSplit(50);
})();
