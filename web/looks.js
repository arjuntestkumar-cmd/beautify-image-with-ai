/* Looks, rendered in the browser.
 *
 * A port of the frame-wide half of app/pipeline/filters.py — `tone_lut` and `apply_global` —
 * so picking a look is instant instead of a round trip. It is a port and not a lookalike: the
 * grade parameters come from /api/filters, which serves the same numbers the server renders
 * from, and the curve below is the same arithmetic. There is one definition of "Golden Aura"
 * and both sides read it.
 *
 * What is here and what is not, deliberately:
 *
 *   HERE   the tone curve (exposure, contrast, matte lift, warmth, tint, split tone) — exact,
 *          because it collapses to a 256-entry per-channel table on both sides; hue-selective
 *          saturation with the same skin cap; monochrome; the vignette.
 *   NOT    the face half — skin evening, smoothing, lips, eyes, glow. Those need the face boxes
 *          and a skin mask the server already computed, and they are why the preview is a
 *          preview: the grade is what a look reads as at a glance, and it lands immediately;
 *          the face work arrives with the real render a moment later.
 *
 * Local contrast is CLAHE on both sides, cell for cell — see `clahe` below. What is left
 * between the two is that the server equalises LAB's L channel and this equalises Rec.601 luma,
 * because an sRGB -> LAB round trip per pixel would cost more than the round trip this exists to
 * beat. That is a few levels on the looks that lean hardest on clarity and nothing at all on the
 * rest, measured; it is not a different grade.
 */
(() => {
  'use strict';

  // ── the grade's colour half, straight out of filters.py ─────────────
  const BUMP_PEAK = 0.33551;                        // max of sin^2(pi y) * y^2
  const WARM_RGB = [0.052, 0.008, -0.046];
  const TINT_RGB = [0.015, -0.030, 0.015];
  const SPLIT_HI_RGB = [0.055, 0.018, -0.030];
  const SPLIT_LO_RGB = [-0.022, -0.004, 0.048];
  const MONO_W = [0.52, 0.36, 0.12];                // warm-filter monochrome, sums to 1
  const SKIN_SAT_MIN = 0.90, SKIN_SAT_MAX = 1.18;
  const WARM_BAND = [10.0, 30.0], COOL_BAND = [105.0, 50.0];   // OpenCV hue, 0..179

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  /* The whole grade — six passes — as one 256x3 table, exactly as filters.tone_lut builds it.
     Pointwise, so this is not an approximation of the server's answer, it is the same answer. */
  function toneLut(g) {
    const lut = new Uint8ClampedArray(256 * 3);
    const exposure = g.exposure || 0, contrast = g.contrast || 0, lift = g.lift || 0;
    const warmth = g.warmth || 0, tint = g.tint || 0, split = g.split || 0;
    const a = 0.55 * contrast;
    const lo = 0.075 * lift, hi = 0.030 * lift;
    for (let i = 0; i < 256; i++) {
      let y = i / 255;
      if (exposure) y = Math.pow(clamp(y, 0, 1), 1 / (1 + 0.85 * exposure));
      if (contrast) y = y - (a / (2 * Math.PI)) * Math.sin(2 * Math.PI * clamp(y, 0, 1));
      if (lift) y = lo + y * (1 - lo - hi);
      y = clamp(y, 0, 1);

      const s = Math.sin(Math.PI * y);
      const bell = s * s;                                  // 0 at both ends, always
      const upper = bell * y * y / BUMP_PEAK;
      const lower = bell * (1 - y) * (1 - y) / BUMP_PEAK;
      for (let c = 0; c < 3; c++) {
        let d = bell * warmth * WARM_RGB[c] + bell * tint * TINT_RGB[c];
        if (split) d += upper * split * SPLIT_HI_RGB[c] + lower * split * SPLIT_LO_RGB[c];
        lut[i * 3 + c] = clamp((y + d) * 255 + 0.5, 0, 255);
      }
    }
    return lut;
  }

  const lutCache = new Map();
  function lutFor(look) {
    if (!lutCache.has(look.id)) lutCache.set(look.id, toneLut(look.grade || {}));
    return lutCache.get(look.id);
  }

  /* Warm the LUTs before anyone clicks anything. Thirteen tables of 768 bytes: the point is not
     the microseconds, it is that the first pick is as fast as the tenth. */
  function preload(looks) {
    for (const look of looks || []) if (look && look.grade) lutFor(look);
  }

  // ── helpers shared by the passes below ──────────────────────────────
  const hueBand = (h, centre, halfwidth) =>
    clamp(1 - Math.abs(((h - centre + 90) % 180 + 180) % 180 - 90) / halfwidth, 0, 1);

  /* The server's skin test, minus its 7x7 feather: the same Cr/Cb window, taken per pixel.
     It gates a saturation CAP and a clarity brake, never a visible edge, so the missing blur
     costs nothing you can see and saves a full-frame convolution per preview. */
  function isSkin(r, g, b) {
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const cr = (r - y) * 0.713 + 128, cb = (b - y) * 0.564 + 128;
    return (cr > 135 && cr < 180 && cb > 85 && cb < 135) ? 1 : 0;
  }

  /* CLAHE — contrast-limited adaptive histogram equalisation — on one plane.
   *
   * The same algorithm the server runs, not an impression of it: an 8x8 grid of cells, one
   * clipped and integrated histogram per cell, and every pixel mapped through a bilinear blend
   * of the four curves nearest it. An unsharp mask stood here first and was the only part of
   * this file that could be called an approximation; on the two looks that lean hardest on
   * local contrast it drifted from the server by up to 54 levels, which is enough for a swatch
   * to promise a grade the render does not deliver.
   *
   * Cost is one pass to count and one to map — linear in pixels, independent of the grid — so
   * being exact here is cheaper than the blur it replaced.
   */
  function clahe(plane, w, h, clipLimit, grid) {
    const g = grid || 8;
    const luts = new Float32Array(g * g * 256);
    const edge = (i, n, ext) => Math.round(i * ext / n);
    const hist = new Float32Array(256);
    for (let cy = 0; cy < g; cy++) {
      const y0 = edge(cy, g, h), y1 = Math.max(edge(cy + 1, g, h), y0 + 1);
      for (let cx = 0; cx < g; cx++) {
        const x0 = edge(cx, g, w), x1 = Math.max(edge(cx + 1, g, w), x0 + 1);
        hist.fill(0);
        let total = 0;
        for (let y = y0; y < y1 && y < h; y++) {
          const row = y * w;
          for (let x = x0; x < x1 && x < w; x++) {
            hist[clamp(Math.round(plane[row + x]), 0, 255) | 0]++;
            total++;
          }
        }
        total = Math.max(1, total);
        // Clip the histogram and spread the excess evenly — the "contrast limited" half, and
        // the reason this lifts local detail without turning flat areas into noise.
        const limit = Math.max(1, clipLimit * total / 256);
        let excess = 0;
        for (let i = 0; i < 256; i++) if (hist[i] > limit) { excess += hist[i] - limit; hist[i] = limit; }
        const share = excess / 256;
        let acc = 0;
        const base = (cy * g + cx) * 256, k = 255 / total;
        for (let i = 0; i < 256; i++) { acc += hist[i] + share; luts[base + i] = acc * k; }
      }
    }
    // Which two cell curves a coordinate falls between, and how far across it sits. Tabulated
    // per axis rather than computed per pixel: the x answer is the same on every row, and
    // returning it as a fresh three-element array a million times over was the single most
    // expensive thing in this file — a megapixel preview spent most of a second in the
    // allocator rather than on arithmetic.
    const axis = (n, ext) => {
      const i0 = new Int32Array(n), i1 = new Int32Array(n), f = new Float32Array(n);
      for (let v = 0; v < n; v++) {
        const t = v * (g / Math.max(1, ext)) - 0.5;
        i0[v] = clamp(Math.floor(t), 0, g - 1);
        i1[v] = clamp(i0[v] + 1, 0, g - 1);
        f[v] = clamp(t - i0[v], 0, 1);
      }
      return { i0, i1, f };
    };
    const ax = axis(w, w), ay = axis(h, h);
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const iy0 = ay.i0[y] * g, iy1 = ay.i1[y] * g, fy = ay.f[y], gy = 1 - fy;
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const fx = ax.f[x], gx = 1 - fx;
        const a0 = (iy0 + ax.i0[x]) << 8, a1 = (iy0 + ax.i1[x]) << 8;
        const b0 = (iy1 + ax.i0[x]) << 8, b1 = (iy1 + ax.i1[x]) << 8;
        const v = clamp(Math.round(plane[row + x]), 0, 255) | 0;
        out[row + x] = gy * (gx * luts[a0 + v] + fx * luts[a1 + v])
                     + fy * (gx * luts[b0 + v] + fx * luts[b1 + v]);
      }
    }
    return out;
  }

  /* Render one look over an ImageData, in place. Returns the same ImageData.
     `full` optionally places this frame inside a larger one so the vignette is drawn from the
     real centre — the same argument filters.apply_global makes for a tile. */
  function render(imageData, look, full) {
    const grade = (look && look.grade) || {};
    const d = imageData.data, w = imageData.width, h = imageData.height;
    const n = w * h;

    // ---- 1. monochrome, first, so any colour term becomes a tone on top of it -------------
    const mono = grade.mono || 0;
    if (mono > 0.01) {
      for (let i = 0; i < n; i++) {
        const p = i * 4;
        const grey = MONO_W[0] * d[p] + MONO_W[1] * d[p + 1] + MONO_W[2] * d[p + 2];
        d[p] += (grey - d[p]) * mono;
        d[p + 1] += (grey - d[p + 1]) * mono;
        d[p + 2] += (grey - d[p + 2]) * mono;
      }
    }

    // ---- 2. the grade, as one table ------------------------------------------------------
    const lut = lutFor(look);
    for (let i = 0; i < n; i++) {
      const p = i * 4;
      d[p] = lut[d[p] * 3];
      d[p + 1] = lut[d[p + 1] * 3 + 1];
      d[p + 2] = lut[d[p + 2] * 3 + 2];
    }

    // ---- 3. saturation, by hue band, capped on skin ---------------------------------------
    const vib = grade.vibrance || 0, satSkin = grade.sat_skin || 0, satCool = grade.sat_cool || 0;
    if (Math.max(Math.abs(vib), Math.abs(satSkin), Math.abs(satCool)) > 0.01) {
      for (let i = 0; i < n; i++) {
        const p = i * 4;
        const r = d[p], g = d[p + 1], b = d[p + 2];
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        if (max === 0) continue;
        const s = (max - min) / max;                       // OpenCV HSV saturation, 0..1
        let hue = 0;                                       // OpenCV hue, 0..179
        if (max !== min) {
          const c = max - min;
          if (max === r) hue = 30 * ((g - b) / c);
          else if (max === g) hue = 30 * ((b - r) / c) + 60;
          else hue = 30 * ((r - g) / c) + 120;
          if (hue < 0) hue += 180;
        }
        let gain = 1 + 0.55 * vib * (1 - s);               // the duller colours move most
        if (Math.abs(satSkin) > 0.01) gain *= 1 + 0.45 * satSkin * hueBand(hue, WARM_BAND[0], WARM_BAND[1]);
        if (Math.abs(satCool) > 0.01) gain *= 1 + 0.50 * satCool * hueBand(hue, COOL_BAND[0], COOL_BAND[1]);
        // The cap is on the COMBINED gain and only on skin: a look may mute a wall or saturate
        // a sunset, and a complexion still cannot go orange or grey.
        if (isSkin(r, g, b)) gain = clamp(gain, SKIN_SAT_MIN, SKIN_SAT_MAX);
        if (gain === 1) continue;
        const ns = clamp(s * gain, 0, 1);
        // Rebuild at the same value with the new saturation: v stays max, min moves.
        const nmin = max * (1 - ns);
        const k = (max - min) === 0 ? 0 : (max - nmin) / (max - min);
        d[p] = max - (max - r) * k;
        d[p + 1] = max - (max - g) * k;
        d[p + 2] = max - (max - b) * k;
      }
    }

    // ---- 4. local contrast ----------------------------------------------------------------
    const clarity = grade.clarity || 0;
    if (clarity > 0.01) {
      const lum = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const p = i * 4;
        lum[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
      }
      // Same clip limit as filters.apply_global. If one of the two ever changes, a swatch stops
      // matching its own render, so they are quoted from the same sentence in both files.
      const boosted = clahe(lum, w, h, 1.0 + 2.2 * clarity, 8);
      for (let i = 0; i < n; i++) {
        const p = i * 4, l = lum[i];
        // The server's luminosity mask: local contrast is worth the most in the midtones and
        // costs the most at the ends — it turns a white shirt into paper and a shadow into mud.
        const top = clamp((l - 204) / 51, 0, 1);
        const keep = (1 - top * top) * (1 - 0.7 * clamp((26 - l) / 26, 0, 1));
        const k = Math.min(1, clarity) * keep * (1 - 0.35 * isSkin(d[p], d[p + 1], d[p + 2]));
        if (k <= 0.001) continue;
        const f = (l < 1 ? 1 : (l + (boosted[i] - l) * k) / l);
        d[p] = clamp(d[p] * f, 0, 255);
        d[p + 1] = clamp(d[p + 1] * f, 0, 255);
        d[p + 2] = clamp(d[p + 2] * f, 0, 255);
      }
    }

    // ---- 5. vignette ----------------------------------------------------------------------
    const vig = grade.vignette || 0;
    if (vig > 0.01) {
      const fy0 = (full && full.y0) || 0, fx0 = (full && full.x0) || 0;
      const fh = (full && full.height) || h, fw = (full && full.width) || w;
      for (let y = 0; y < h; y++) {
        const yy = (y + fy0) / Math.max(1, fh - 1) - 0.5;
        for (let x = 0; x < w; x++) {
          const xx = (x + fx0) / Math.max(1, fw - 1) - 0.5;
          const rr = Math.sqrt(yy * yy * 1.15 + xx * xx);
          const t = clamp((rr - 0.30) / 0.42, 0, 1);
          const fall = 1 - 0.34 * vig * t * t;
          if (fall >= 0.999) continue;
          const p = (y * w + x) * 4;
          d[p] *= fall; d[p + 1] *= fall; d[p + 2] *= fall;
        }
      }
    }
    return imageData;
  }

  // ── sources: one small bitmap, reused by every swatch ────────────────
  /* A square crop, drawn once and handed to all thirteen renders.
   *
   * Cropping to the FACE when we know where it is, rather than to the middle of the frame, is
   * most of what makes a strip of swatches readable: a look is a statement about skin, light and
   * eyes, and a 96 px square of someone's shoulder says nothing about any of them. `boxes` are
   * in the coordinates of `img`; without them the crop falls back to the upper-middle, where a
   * portrait's head almost always is.
   */
  function squareCrop(img, size, boxes) {
    const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
    let side, cx, cy;
    const box = boxes && boxes.length
      ? boxes.reduce((a, b) => (b[2] * b[3] > a[2] * a[3] ? b : a))
      : null;
    if (box) {
      side = clamp(Math.max(box[2], box[3]) * 2.1, 32, Math.min(iw, ih));
      cx = box[0] + box[2] / 2;
      cy = box[1] + box[3] / 2;
    } else {
      side = Math.min(iw, ih);
      cx = iw / 2;
      cy = ih * 0.42;                     // heads sit above the middle far more often than not
    }
    const sx = clamp(cx - side / 2, 0, Math.max(0, iw - side));
    const sy = clamp(cy - side / 2, 0, Math.max(0, ih - side));

    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, sx, sy, side, side, 0, 0, size, size);
    return { ctx, imageData: ctx.getImageData(0, 0, size, size) };
  }

  /* A whole-frame copy, bounded, for the full-size instant preview. The real render replaces it
     within a second or two, so there is nothing to gain from previewing at 24 megapixels. */
  function fitted(img, maxSide) {
    const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
    const s = Math.min(1, maxSide / Math.max(iw, ih));
    const w = Math.max(1, Math.round(iw * s)), h = Math.max(1, Math.round(ih * s));
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, w, h);
    return { canvas, ctx, imageData: ctx.getImageData(0, 0, w, h) };
  }

  /* Paint one look from a prepared source onto a canvas. The source ImageData is never mutated —
     each render works on a copy — so the same bitmap serves every swatch and every re-pick. */
  function paint(source, look, targetCanvas) {
    const src = source.imageData;
    const copy = new ImageData(new Uint8ClampedArray(src.data), src.width, src.height);
    render(copy, look);
    const canvas = targetCanvas || document.createElement('canvas');
    canvas.width = src.width; canvas.height = src.height;
    canvas.getContext('2d').putImageData(copy, 0, 0);
    return canvas;
  }

  function toBlobUrl(canvas) {
    return new Promise((resolve) => {
      if (canvas.toBlob) canvas.toBlob((b) => resolve(b ? URL.createObjectURL(b) : null), 'image/jpeg', 0.92);
      else resolve(canvas.toDataURL('image/jpeg', 0.92));
    });
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('image load failed'));
      img.src = src;
    });
  }

  window.Looks = { toneLut, render, preload, squareCrop, fitted, paint, toBlobUrl, loadImage };
})();
