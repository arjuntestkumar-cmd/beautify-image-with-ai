/* Beautify — upload, poll, compare, download. Vanilla JS, no build step. */
(() => {
  'use strict';

  // Kept in step with MAX_UPLOAD_BYTES on the server. A large photo is no longer a problem to
  // be refused: it is processed in chunks and simply takes longer.
  const MAX_BYTES = 64 * 1024 * 1024;
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

  const selectedMode = () =>
    (document.querySelector('input[name="mode"]:checked') || {}).value || 'beautify';

  const state = {
    file: null,
    objectUrl: null,
    jobId: null,
    polling: null,
    shown: 0,        // the percentage currently painted on the bar
    target: 0,       // the percentage the server last reported
    looks: [],       // the filter catalogue, fetched once from /api/filters
    look: null,      // the look chosen for the next run
    defaultLook: null,
    job: null,       // the last completed job payload, for re-styling
    restyling: false,
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

  // ── premium looks ───────────────────────────────────────────────────
  // The catalogue comes from the server so the two never disagree about what exists or which
  // one is the default.
  async function loadLooks() {
    try {
      const res = await fetch('/api/filters');
      const payload = await res.json();
      if (!payload.success) return;
      state.looks = payload.data.filters || [];
      state.defaultLook = payload.data.default;
      state.look = state.defaultLook;
      renderChips($('looks-chips'), pickLook, state.look);
      $('looks-pick').hidden = state.looks.length === 0;
    } catch {
      // No catalogue: the picker stays hidden and the server applies its own default.
    }
  }

  function renderChips(host, onPick, selected) {
    host.innerHTML = '';
    for (const look of state.looks) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip';
      chip.setAttribute('role', 'radio');
      chip.setAttribute('aria-checked', String(look.id === selected));
      chip.title = look.description;
      chip.innerHTML =
        `<span class="chip-name"></span><span class="chip-tag"></span>`;
      chip.querySelector('.chip-name').textContent = look.name;
      chip.querySelector('.chip-tag').textContent =
        look.id === 'none' ? 'no filter' : (look.id === state.defaultLook ? 'default' : '');
      chip.addEventListener('click', () => onPick(look.id));
      host.appendChild(chip);
    }
  }

  function markSelected(host, id) {
    [...host.children].forEach((chip, i) =>
      chip.setAttribute('aria-checked', String(state.looks[i] && state.looks[i].id === id)));
  }

  function pickLook(id) {
    state.look = id;
    markSelected($('looks-chips'), id);
  }

  // In the result view a look change is a server round-trip, but a cheap one: it re-renders
  // from the un-styled image the job kept, so nothing is analysed or restored a second time.
  async function pickDoneLook(id) {
    if (state.restyling || !state.job) return;
    const block = $('looks-done');
    const previous = state.job.look;
    const jobId = state.job.jobId;
    state.restyling = true;
    block.classList.add('busy');
    markSelected($('looks-done-chips'), id);
    try {
      const body = new FormData();
      body.append('filter', id);
      const res = await fetch(`/api/jobs/${jobId}/filter`, { method: 'POST', body });
      const payload = await res.json();
      const job = (res.ok && payload.success) ? await waitForJob(jobId) : null;
      // The user may have moved on while this was in flight; if they have, the result of a
      // job they are no longer looking at must not be painted over whatever is on screen now.
      if (state.job && state.job.jobId !== jobId) return;
      if (job) applyResult(job);
      else markSelected($('looks-done-chips'), previous);
    } catch {
      markSelected($('looks-done-chips'), previous);
    } finally {
      state.restyling = false;
      block.classList.remove('busy');
    }
  }

  // Poll until the re-style finishes. It is far cheaper than a full run, but it still goes
  // through the same one-at-a-time queue, so it can be waiting behind somebody's photo.
  async function waitForJob(id) {
    for (let i = 0; i < 1200; i++) {
      await new Promise((r) => setTimeout(r, 300));
      try {
        const res = await fetch(`/api/jobs/${id}`);
        const payload = await res.json();
        if (!res.ok || !payload.success) return null;
        if (payload.data.status === 'completed') return payload.data;
        if (payload.data.status === 'failed') return null;
      } catch { /* a dropped poll is not fatal */ }
    }
    return null;
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

  // Keep the action button honest about what it is going to do.
  document.querySelectorAll('input[name="mode"]').forEach((radio) =>
    radio.addEventListener('change', () => {
      $('btn-enhance').textContent = selectedMode() === 'clear' ? 'Clear image' : 'Beautify';
    }));

  // ── enhancing ───────────────────────────────────────────────────────
  $('btn-enhance').addEventListener('click', enhance);

  async function enhance() {
    if (!state.file) return;
    state.shown = 0;
    state.target = 5;
    paintProgress(5, 'Uploading…', 'Sending your photo to the enhancer.');
    show('processing');

    const mode = selectedMode();
    const body = new FormData();
    body.append('image', state.file, state.file.name);
    body.append('mode', mode);
    if (state.look) body.append('filter', state.look);

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
    // Being behind other people is not the same as being stuck, and the difference is worth
    // saying out loud when the bar has not moved for a while.
    const queued = job.status === 'queued' && job.queuePosition > 0;
    const ahead = job.queuePosition;
    const label = queued
      ? `Queued — ${ahead} photo${ahead > 1 ? 's' : ''} ahead of yours`
      : stageLabel(job.stage);
    paintProgress(state.target, label, queued
      ? 'The server finishes one photo at a time so every one of them completes.'
      : job.message);

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
      exposure: 'Correcting the exposure…',
      deblock: 'Cleaning compression…',
      chroma: 'Cleaning colour noise…',
      denoise: 'Removing grain…',
      face_clarity: 'Refining faces…',
      body_clarity: 'Refining detail…',
      hair: 'Refining hair…',
      skin_clean: 'Evening out skin…',
      finishing: 'Finishing…',
      styling: 'Applying the finish…',
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
    $('img-before').src = state.objectUrl;
    applyResult(job, () => {
      // The picker only appears if the server kept an un-styled copy; without one, changing
      // the look would mean re-running the whole pipeline, so it is not offered.
      const canRestyle = job.details?.canRestyle && state.looks.length > 0;
      $('looks-done').hidden = !canRestyle;
      if (canRestyle) renderChips($('looks-done-chips'), pickDoneLook, job.look);
      show('done');
    });
  }

  // Point the comparison and the download button at whatever the job's current result is.
  // `resultUrl` carries a version, so a re-styled image is a new address and the browser
  // fetches it instead of showing the one it already has.
  function applyResult(job, then) {
    state.job = job;
    const after = $('img-after');
    after.onload = () => {
      setSplit(50);
      renderStats(job);
      const dl = $('btn-download');
      dl.href = job.resultUrl;
      const stem = (state.file?.name || 'image').replace(/\.[^.]+$/, '');
      const ext = (job.output?.format || 'image/jpeg').split('/')[1].replace('jpeg', 'jpg');
      dl.setAttribute('download', `${stem}-beautified.${ext}`);
      if (then) then();
    };
    after.onerror = () => fail('Could not load the result', 'The enhanced image could not be fetched.');
    after.src = job.resultUrl;
  }

  function renderStats(job) {
    const o = job.output || {};
    const i = job.input || {};
    const d = job.details || {};
    const chips = [];

    if (i.width && o.width) chips.push(`<span class="stat"><b>${i.width}×${i.height}</b> → <b>${o.width}×${o.height}</b></span>`);
    if (job.mode) chips.push(`<span class="stat">${job.mode === 'clear' ? 'Cleared' : 'Beautified'}</span>`);
    const look = state.looks.find((l) => l.id === job.look);
    if (look && look.id !== 'none') chips.push(`<span class="stat"><b>${look.name}</b></span>`);
    if (d.chunked) chips.push(`<span class="stat">processed in chunks</span>`);
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
    state.job = null;
    state.file = null;
    state.look = state.defaultLook;
    if (state.looks.length) renderChips($('looks-chips'), pickLook, state.look);
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
  loadLooks();
  setSplit(50);
})();


