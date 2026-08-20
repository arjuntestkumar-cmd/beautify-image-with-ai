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
    (document.querySelector('input[name="mode"]:checked') || {}).value || 'portrait';

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
    // Bitmaps the look previews are drawn from. `pickSource` is a square of the photo the user
    // chose; `baseSource` and `basePreview` are squares and a bounded whole frame of the
    // ENHANCED, UN-STYLED result — the same image the server renders every look from, so a
    // preview and the render that replaces it start from identical pixels.
    pickSource: null,
    baseSource: null,
    basePreview: null,
    // Every look, rendered once in the browser and kept. A click is then an <img> src swap
    // and nothing else - no canvas work, no request, no wait.
    previewCache: new Map(),   // look id -> object URL of this photo under that look
    doneLook: null,     // the look the result view is SHOWING, the instant it is clicked
    renderedLook: null, // the look the server's current result file actually holds
    pump: null,         // the in-flight server-render pass, as a promise anyone can await
    renderNudge: null,  // debounce timer for the server render
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
      // Build every look's colour table now, before anyone can click one. The tables are 768
      // bytes each; the point is not the microseconds saved but that the FIRST pick is as fast
      // as the tenth, which is the whole difference between "instant" and "usually instant".
      if (window.Looks) window.Looks.preload(state.looks);
      renderChips($('looks-chips'), pickLook, state.look, state.pickSource);
      $('looks-pick').hidden = state.looks.length === 0;
    } catch {
      // No catalogue: the picker stays hidden and the server applies its own default.
    }
  }

  // `source`, when present, is a small square bitmap of the user's own photo (see Looks.
  // squareCrop). Each chip then shows that square with its own look already on it, rendered
  // here in the browser — so the strip answers "what will this one do to MY face" instead of
  // "what is this one called", and it answers it before the first click rather than after it.
  function renderChips(host, onPick, selected, source) {
    host.innerHTML = '';
    for (const look of state.looks) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip';
      chip.setAttribute('role', 'radio');
      chip.setAttribute('aria-checked', String(look.id === selected));
      chip.title = look.description;
      chip.innerHTML =
        `<span class="chip-thumb" aria-hidden="true"></span>` +
        `<span class="chip-text"><span class="chip-name"></span><span class="chip-tag"></span></span>`;
      chip.querySelector('.chip-name').textContent = look.name;
      chip.querySelector('.chip-tag').textContent =
        look.id === 'none' ? 'no filter' : (look.id === state.defaultLook ? 'default' : '');
      chip.addEventListener('click', () => onPick(look.id));
      host.appendChild(chip);
    }
    if (source) paintSwatches(host, source);
  }

  // Painted after the chips are in the document, one animation frame at a time. Thirteen 104 px
  // squares is a few milliseconds of work in total, but doing it in one synchronous burst is a
  // few milliseconds during which the click that opened this panel has not finished — and that
  // is exactly the moment the interface is being judged for responsiveness.
  function paintSwatches(host, source) {
    if (!window.Looks) return;
    const chips = [...host.children];
    let i = 0;
    const step = () => {
      if (i >= chips.length || !host.isConnected) return;
      const look = state.looks[i], chip = chips[i];
      i++;
      try {
        const slot = chip && chip.querySelector('.chip-thumb');
        if (slot && look) {
          slot.appendChild(window.Looks.paint(source, look));
          chip.classList.add('with-thumb');
        }
      } catch { /* a swatch is a nicety; the chip works without one */ }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // One square of the photo, at swatch resolution, shared by all thirteen renders.
  function buildSource(img, boxes) {
    if (!window.Looks) return null;
    try {
      return window.Looks.squareCrop(img, 104, boxes);
    } catch {
      return null;   // a tainted or undecodable canvas: fall back to text-only chips
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

  // ── changing the look, without the wait ─────────────────────────────
  //
  // A look change has to feel like a filter, not like submitting a job. Three things make that
  // true and all three matter:
  //
  //   1. EVERY look is rendered up front, in the browser, from the un-styled base the job kept
  //      (`prerenderLooks`). By the time anyone reaches for the strip the pictures already
  //      exist, so a click is an <img> src swap - no canvas work, no request, no wait.
  //   2. The chips are NEVER disabled. The old version locked the whole strip for the length of
  //      the server round trip, which is 1.2 to 1.6 seconds of real work plus polling; so the
  //      second click of a comparison - the one that decides it - did nothing at all for about
  //      two seconds. That lock, not the rendering, was what made this feel slow.
  //   3. The server render still happens, because the browser only draws the frame-wide half of
  //      a look and the file you download has to be the whole thing. It just happens BEHIND the
  //      picture, for whichever look you settled on, and swaps itself in when it arrives.
  function pickDoneLook(id) {
    if (!state.job || id === state.doneLook) return;
    state.doneLook = id;
    markSelected($('looks-done-chips'), id);
    showLook(id);
    nudgeRender();
  }

  // Put a look on screen NOW: the server's own render when we already have that one, the
  // browser's cached preview otherwise, and - only if it has not been cached yet - a render on
  // the spot, which is about a sixth of a second rather than the two seconds a round trip costs.
  function showLook(id) {
    if (state.job && id === state.renderedLook) return showImage(state.job.resultUrl);
    const cached = state.previewCache.get(id);
    if (cached) return showImage(cached);
    renderPreview(id).then((url) => {
      if (url && state.doneLook === id) showImage(url);
    });
  }

  function showImage(src) {
    const after = $('img-after');
    after.onload = null;
    after.onerror = null;
    after.src = src;
  }

  // One look, painted from the un-styled base and kept as an object URL.
  function renderPreview(id) {
    const hit = state.previewCache.get(id);
    if (hit) return Promise.resolve(hit);
    const look = state.looks.find((l) => l.id === id);
    if (!look || !state.basePreview || !window.Looks) return Promise.resolve(null);
    let canvas;
    try {
      canvas = window.Looks.paint(state.basePreview, look);
    } catch {
      return Promise.resolve(null);
    }
    return window.Looks.toBlobUrl(canvas).then((url) => {
      if (url) state.previewCache.set(id, url);
      return url;
    });
  }

  // Fill the cache in the background while the result is being looked at. One look per idle
  // slice, so nothing janks, and whatever is currently selected is always taken first.
  function prerenderLooks() {
    if (!state.basePreview || !window.Looks) return;
    const queue = state.looks.map((l) => l.id).filter((id) => !state.previewCache.has(id));
    const idle = window.requestIdleCallback
      ? (fn) => window.requestIdleCallback(fn, { timeout: 500 })
      : (fn) => setTimeout(fn, 16);
    const step = () => {
      if (!queue.length || !state.job) return;
      const i = queue.indexOf(state.doneLook);
      const id = queue.splice(i >= 0 ? i : 0, 1)[0];
      renderPreview(id).then(() => idle(step));
    };
    idle(step);
  }

  // ── the server render, behind the picture ───────────────────────────
  // Debounced, so browsing the strip does not queue thirteen renders on a server that does one
  // at a time.
  function nudgeRender() {
    clearTimeout(state.renderNudge);
    state.renderNudge = setTimeout(pumpRender, 400);
  }

  // Serialised, and it re-reads the selection every time round: a look picked while the previous
  // render was in flight is taken up next, and the ones clicked past in between are skipped
  // rather than queued. So the server renders what you settled on, not everything you touched.
  // Returns the IN-FLIGHT pass when there already is one, rather than returning immediately.
  // That distinction is the whole correctness of the Download button: it awaits this, and the
  // version that bailed out early handed back an already-resolved promise while the render it
  // was supposed to be waiting for was still running - so the click did nothing at all,
  // silently, in exactly the window where someone is most likely to press it.
  function pumpRender() {
    if (!state.job) return Promise.resolve();
    if (!state.pump) state.pump = runPump().finally(() => { state.pump = null; });
    return state.pump;
  }

  async function runPump() {
    setSyncing(true);
    try {
      while (state.job && state.doneLook && state.doneLook !== state.renderedLook) {
        if (!(await serverRender(state.doneLook))) break;
      }
    } finally {
      setSyncing(false);
    }
  }

  async function serverRender(want) {
    const jobId = state.job.jobId;
    try {
      const body = new FormData();
      body.append('filter', want);
      const res = await fetch(`/api/jobs/${jobId}/filter`, { method: 'POST', body });
      const payload = await res.json();
      if (!res.ok || !payload.success) return false;
      const job = await waitForJob(jobId);
      // The user may have started a different photo entirely while this was in flight.
      if (!job || !state.job || state.job.jobId !== jobId) return false;
      adoptJob(job);
      return true;
    } catch {
      return false;
    }
  }

  // A finished render. Always becomes what the Download button points at; goes on screen only
  // if it is still the look being looked at, because a render the user has already clicked past
  // must not paint over the one they are on.
  function adoptJob(job) {
    state.job = job;
    state.renderedLook = job.look;
    renderStats(job);
    setDownload(job);
    if (state.doneLook === job.look) showImage(job.resultUrl);
  }

  function setDownload(job) {
    const dl = $('btn-download');
    dl.href = job.resultUrl;
    const stem = (state.file?.name || 'image').replace(/\.[^.]+$/, '');
    const ext = (job.output?.format || 'image/jpeg').split('/')[1].replace('jpeg', 'jpg');
    dl.setAttribute('download', `${stem}-beautified.${ext}`);
  }

  function setSyncing(on) {
    const note = $('looks-done-note');
    if (!note) return;
    if (on) {
      if (!note.dataset.idleText) note.dataset.idleText = note.textContent;
      note.textContent = 'Preview shown \u2014 finishing the full-quality version\u2026';
    } else if (note.dataset.idleText) {
      note.textContent = note.dataset.idleText;
    }
  }

  // What is on screen may be the browser's preview while the server's own render is still
  // coming. The preview is the frame-wide half of the look only - the skin, lips and eyes are
  // the server's half - so the file that leaves has to be the real render, even if that means
  // waiting a moment for it. Waiting HERE is the right place: it is the one click where a
  // second of delay buys something the user actually receives.
  const DOWNLOAD_LABEL = $('btn-download').textContent;

  $('btn-download').addEventListener('click', async (e) => {
    if (!state.job || state.renderedLook === state.doneLook) return;   // already the real thing
    e.preventDefault();
    const dl = $('btn-download');
    dl.textContent = 'Preparing\u2026';
    dl.classList.add('waiting');
    clearTimeout(state.renderNudge);      // do not sit through the debounce as well
    await pumpRender();
    dl.classList.remove('waiting');
    dl.textContent = DOWNLOAD_LABEL;
    if (state.renderedLook === state.doneLook) {
      dl.click();          // re-entry is harmless: the guard above lets it straight through
    } else {
      // Say so rather than doing nothing. Handing over the previous look's file because this
      // one would not render is the one outcome worse than a click that failed.
      const note = $('looks-done-note');
      if (note) {
        note.textContent = 'That look could not be prepared \u2014 try it again.';
        setTimeout(() => setSyncing(false), 4000);
      }
    }
  });

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
      // The swatches for the "what finish do you want" strip come from the photo in front of
      // the user, not from a stock face. There is no face box yet — nothing has looked at the
      // photo — so the crop falls back to the upper middle of the frame.
      state.pickSource = buildSource(img, null);
      if (state.looks.length) renderChips($('looks-chips'), pickLook, state.look, state.pickSource);
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
      $('btn-enhance').textContent =
        ({ clear: 'Clear image', portrait: 'Enhance photo' })[selectedMode()] || 'Beautify';
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
      blemish: 'Clearing blemishes…',
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
    // Creep forward between polls, but never more than CREEP_LEAD points ahead of what the
    // server actually reported, and never past 99 until it says 100.
    //
    // The lead used to be unbounded: +0.6 every 900 ms poll regardless of the truth, so the bar
    // reached 99% after about two and a half minutes on ANY long job. A failure then always read
    // as "it failed at 99%", whatever stage it really died in - which is worse than useless when
    // the number is the only clue anyone has. Bounded, the percentage is evidence again.
    const CREEP_LEAD = 6;
    const ceiling = Math.min(99, pct + CREEP_LEAD);
    state.shown = pct >= 100
      ? 100
      : Math.max(state.shown, Math.min(ceiling, Math.max(state.shown + 0.6, pct)));
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
      if (canRestyle) renderChips($('looks-done-chips'), pickDoneLook, state.doneLook, state.baseSource);
      show('done');
      if (canRestyle) prepareBase(job);
    });
  }

  // Fetch the un-styled enhanced image once, and keep two bitmaps cut from it: a square for the
  // swatches and a bounded whole frame for the instant preview.
  //
  // It has to be the BASE and not the result on screen. The result already carries a look, so
  // previewing another one on top of it would show Amber-then-Aura — a grade nobody will ever be
  // sent — and every swatch in the strip would drift further from the truth the more the user
  // explored. The base is the exact image the server renders each look from.
  async function prepareBase(job) {
    const url = job.details?.baseUrl;
    if (!url || !window.Looks) return;
    const jobId = job.jobId;
    try {
      const img = await window.Looks.loadImage(url);
      if (!state.job || state.job.jobId !== jobId) return;   // the user moved on
      state.baseSource = buildSource(img, job.details?.faceBoxes);
      // 900 px on the long side. The compare box is never wider than about 900 CSS pixels, and
      // this size is rendered thirteen times over in the background - measured at roughly 135 ms
      // each in Chromium against 230 ms at 1400, which is the difference between filling the
      // cache during one glance at the result and still filling it during the next.
      state.basePreview = window.Looks.fitted(img, 900);
      prerenderLooks();
      if (state.baseSource && !$('looks-done').hidden) {
        // `doneLook` and not `job.look`: the user may well have picked something else during
        // the second this took to load, and re-checking the old chip under them would be a
        // small lie about what they are looking at.
        renderChips($('looks-done-chips'), pickDoneLook, state.doneLook, state.baseSource);
      }
    } catch {
      // No base, no swatches and no instant preview — every look still works, it just costs the
      // round trip it always did.
    }
  }

  // Point the comparison and the download button at whatever the job's current result is.
  // `resultUrl` carries a version, so a re-styled image is a new address and the browser
  // fetches it instead of showing the one it already has.
  // The FIRST result only. It waits for the image to decode before the panel is revealed, so
  // the done view never appears with an empty frame in it. Every look change after this goes
  // through `pickDoneLook`, which must not wait for anything.
  function applyResult(job, then) {
    state.job = job;
    state.doneLook = job.look;
    state.renderedLook = job.look;
    const after = $('img-after');
    after.onload = () => {
      setSplit(50);
      renderStats(job);
      setDownload(job);
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
    if (job.mode) chips.push(`<span class="stat">${
      ({ clear: 'Cleared', portrait: 'Portrait' })[job.mode] || 'Beautified'}</span>`);
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
    clearTimeout(state.renderNudge);
    for (const url of state.previewCache.values()) URL.revokeObjectURL(url);
    state.previewCache.clear();
    state.renderedLook = null;
    state.jobId = null;
    state.job = null;
    state.file = null;
    state.look = state.defaultLook;
    state.doneLook = null;
    state.pickSource = null;
    state.baseSource = null;
    state.basePreview = null;
    setSyncing(false);
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


