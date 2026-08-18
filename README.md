# Beautify

One backend. One page. One thing: upload a photo, get a genuinely better version back.

This is a stripped-down descendant of a larger multi-mode enhancer. The restoration logic is
the same proven pipeline (Real-ESRGAN + GFPGAN, with the same adaptive tuning and the same
anti-waxy safeguards) — but there is a **single mode**, no options to pick, and none of the
heavy machinery that existed to serve the other modes.

| | |
| --- | --- |
| Backend | Python 3.11+ · FastAPI · **one process**, API + inference together |
| Frontend | one static page (vanilla HTML/CSS/JS), served by that same process |
| Models | Real-ESRGAN `general-x4v3` (+ `wdn` sibling) and GFPGAN v1.4 — ~420 MB total |
| Infrastructure | none. No database, no Redis, no queue service, no cloud storage, no Docker |

---

## Run it

```powershell
./run.ps1
```

Then open <http://127.0.0.1:8000>.

`run.ps1` pins the interpreter, refuses to start if it cannot import torch (that failure is
otherwise silent — the service would fall back to a plain resize and still report success),
frees the port if a stale copy holds it, and runs from the project root, which GFPGAN requires
in order to find its auxiliary weights.

Manual equivalent:

```powershell
cd D:\FrescoProjects\demo\latest\image-enhancer-py
C:\Users\Fresco-Arjun\aienv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Confirm it is really doing AI:

```bash
curl http://127.0.0.1:8000/health
# "mockMode" MUST be false and "models" must list realesrgan:general + gfpgan
```

If `mockMode` is `true` or `models` is empty, the output is a plain resize — wrong interpreter,
or the weights are missing from `models/`.

### First-time setup on another machine

```powershell
python -m venv C:\aienv                       # a SHORT path: deep paths break the torch install
C:\aienv\Scripts\python.exe -m pip install --upgrade pip
C:\aienv\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
C:\aienv\Scripts\python.exe -m pip install -r requirements.txt
./run.ps1 -VenvPython 'C:\aienv\Scripts\python.exe'
```

Weights go in `models/` (`realesr-general-x4v3.pth`, `realesr-general-wdn-x4v3.pth`,
`GFPGANv1.4.pth`) and `gfpgan/weights/` (`detection_Resnet50_Final.pth`,
`parsing_parsenet.pth`). They are already in place in this checkout.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | the web UI |
| `GET` | `/health` | readiness, device, which models are loaded |
| `POST` | `/api/enhance` | multipart `image=@photo.jpg`, optional `mode=beautify\|clear` → `202` + job id |
| `GET` | `/api/jobs/{id}` | status, stage, progress, and the result metadata when done |
| `GET` | `/api/jobs/{id}/result` | the beautified image |
| `GET` | `/api/jobs/{id}/original` | the original (the before/after slider uses it) |
| `DELETE` | `/api/jobs/{id}` | delete both files now |

```bash
curl -X POST http://127.0.0.1:8000/api/enhance -F "image=@photo.jpg"                 # beautify
curl -X POST http://127.0.0.1:8000/api/enhance -F "image=@photo.jpg" -F "mode=clear" # clear only
curl http://127.0.0.1:8000/api/jobs/<id>
curl -o out.jpg http://127.0.0.1:8000/api/jobs/<id>/result
```

Every response is wrapped: `{"success": true, "data": {...}}`, or
`{"success": false, "error": {"code": "...", "message": "..."}}`. Interactive docs at `/docs`.

Uploads are validated before a job is created (magic bytes, real decode, EXIF orientation,
animation rejected, 20 MB / 40 MP ceilings), so a bad file fails instantly rather than a minute
into processing.

---

## The two modes

Both run the same restoration - upscale, denoise, face restoration, face clarity. They differ
only in what happens to the skin and the colour afterwards.

| | Beautify (default) | Clear only |
| --- | --- | --- |
| Restore + denoise + sharpen | yes | yes |
| Face restoration + eye/lip clarity | yes | yes |
| Skin evened out | yes | **no** |
| Soft highlight glow on skin | yes | **no** |
| Tone curve, vibrance, white balance | full (0.46) | **none** |
| Original skin texture kept | less | **more** (+0.18) |

Measured on a grainy 240x240 portrait, beautify comes out ~11% more saturated and ~11% higher
contrast than clear, while clear retains more raw skin texture.

Pick **Clear only** when the photo should stay exactly itself, just cleaner - documents of
people, evidence, product shots, anything where a flattering grade would be wrong.

## What "Beautify" actually does

There is one mode, but it is not one fixed recipe — it reads the photo first and adapts. That
adaptation is exactly what the original service's mode presets did; it now happens
automatically instead of being a question put to the user.

```
decode → analyse → rescue exposure (dark photos only) → re-analyse → plan
  → de-block, if the input is heavily JPEG-compressed
  → GFPGAN face restoration on a Real-ESRGAN background   (photos with faces that need it)
    or plain Real-ESRGAN restoration                      (everything else)
  → chroma denoise, if the result is still noisy
  → hair / fine-texture refinement around detected heads
  → photographic finish: white balance, highlight recovery, shadow lift, midtone
    S-curve, skin-protected vibrance, a touch of local contrast
  → edge-aware detail + an over-sharpening safeguard
  → validate → encode
```

Decisions the pipeline makes on its own:

- **Scale.** Photos at or below 1600 px on the long side get a real 2x upscale; larger photos
  are beautified at native size. `AUTO_UPSCALE_MAX_SIDE` moves that line.
- **Exposure.** A photo shot indoors at night is restored faithfully as a *dark* photo unless
  something corrects the exposure first, so a badly under-exposed frame gets a bounded gamma
  lift, its black point reset, its local contrast restored and some of the chroma a sensor
  fails to record in the dark put back. It then gets **re-analysed**, because every measurement
  taken on a dark frame misleads: grain hides in the shadows and reads as clean, soft edges
  read as blur. Strictly gated on measured under-exposure - a normally-exposed photo is not
  touched at all. On a dark test frame: +73% brightness, +100% contrast.
- **Shadow noise.** Brightening multiplies whatever grain was hiding in the shadows, and the
  noise measurement understates it badly (that grain surfaces as slow colour mottling, which a
  3-pixel median residual barely registers - one test frame read 0.099 while the background
  visibly crawled). The lift applied is itself the evidence, so it sets a floor under the
  effective noise figure, and a dedicated chroma pass cleans the colour blotching at reduced
  scale, where the blobs are actually small enough to filter. Luminance is left alone.
- **Faces.** Face restoration is attempted on every photo, and the **face model's own detector**
  decides whether there is a face - not OpenCV's Haar cascade, which is unreliable enough to
  miss an obvious portrait entirely. Every face it finds is restored. If it finds none, the
  result is simply the Real-ESRGAN background, so trying costs only the detection.
- **How hard.** Denoise, detail, face clarity and how much original texture is preserved all
  scale with the measured blur, noise and compression of that specific photo.
- **Clarity inside the face.** Eyes, lashes, brows and lips get a dedicated sharpening pass with
  the skin protection turned off. Everywhere else that protection is right; on the face it left
  the part people actually look at as the softest thing in the frame.
- **Structure survives the clean-up.** Denoising runs *after* super-resolution, so at the
  strength heavy grain needs it also smears hair strands and outlines into blobs. The denoiser
  blends back toward the original wherever the luminance carries confident structure: flat skin
  is cleaned in full, a hair strand keeps most of itself. On a grainy test portrait that
  recovered ~60% more strand detail in the hair.
- **Hands, clothing and objects get their own clarity pass.** Only the face goes through the
  face model; everything else is whatever the upscaler produced, then softened by denoising, so
  it needs *more* sharpening than the face to belong in the same photo. Two things used to
  prevent that. Edge confidence was normalised against a single global level, so one very
  high-contrast region - dark hair against bright skin - set the bar for the whole frame and
  fabric folds or a fingernail rim scored near zero; it is now normalised locally. And the skin
  protection applied to anything skin-coloured anywhere, which quietly held back hands, arms and
  necks by 55%; it is now scoped to the face. On the same test portrait, detail rose 66% on the
  hand and 181% on the clothing.
- **Screenshots and documents** are detected and never get face restoration or heavy texture.

What holds the waxy, re-drawn "AI face" look back is **skin-texture re-injection**: after
restoration the source's own high-frequency detail is added back, in skin areas only, scaled by
how clean the source was. This scaling matters more than it sounds: a clean photo has real pores
worth keeping, but a degraded one has only blur, noise and JPEG blocks in those same
frequencies, and pushing those back over a restored face is what made results look barely
different from the input.

Note that GFPGAN v1.4's `clean` architecture ignores the `weight` argument entirely — it takes
`**kwargs` and never reads it. So texture re-injection is not merely *a* control over how much
the face is repainted, it is the **only** one. Any code that appears to throttle the face model
with a weight is not doing anything.

Beauty comes from tone, hair, eyes and a properly restored face — not from repainting it.

---

## What was removed, and why it is safe

Everything below existed to serve modes this build does not have. Nothing the Beautify path
runs was changed to remove them:

| Removed | It belonged to |
| --- | --- |
| 12 enhancement modes + `auto`, 21 filters, 21 manual sliders | the premium options layer |
| Old-photo colour restoration (fade / sepia / haze) | Old Photo Recovery |
| Scratch, tear and crack inpainting | Old Photo Recovery |
| CodeFormer (376 MB) | heavily-degraded old faces |
| SwinIR | artifact reduction |
| RealESRGAN_x4plus (67 MB) | `wow` mode only |
| Product relighting, document mode | Product / Document modes |
| Subject/background separation (bokeh) | portrait modes; it was off by default anyway |
| Postgres, Redis, BullMQ, Cloudinary, the Node API, two workers | the distributed deployment |

Also dropped: the identity-blend and eye-refinement modules. Those were **already dead code** in
the original — the live path uses GFPGAN's own face-parsing paste-back, which never calls them.

Result: ~420 MB of weights instead of ~860 MB, one process instead of five, and no
infrastructure to install.

---

## Configuration

Every value has a working default, so there is no required `.env`. Copy `.env.example` to
`.env` to change any of it. The ones that matter:

| Setting | Default | Effect |
| --- | --- | --- |
| `AUTO_UPSCALE_MAX_SIDE` | `1600` | photos up to this size get 2×; bigger ones stay native |
| `ENABLE_CUDA` | `true` | honoured only if torch reports a working CUDA device |
| `RESULT_RETENTION_MINUTES` | `30` | how long an original + result survive before deletion |
| `MAX_UPLOAD_BYTES` | `20971520` | 20 MB upload ceiling |
| `OUTPUT_QUALITY` | `92` | JPEG/WebP encoder quality |
| `MOCK_MODE` | `false` | `true` = plain resize, no AI. For wiring tests only |

Output keeps the input's format (JPEG in → JPEG out); PNG and WebP keep transparency.

---

## Storage and privacy

Uploads and results are written to `.data/` and deleted `RESULT_RETENTION_MINUTES` after the
job finishes, by a background sweeper. The directory is also wiped on startup, so a crash
cannot leave old photos behind. Nothing is uploaded anywhere — inference is local.

---

## Performance

Inference is serialised (`WORKER_CONCURRENCY=1`): it saturates the CPU or GPU, so running two
at once makes both slower.

On CPU (torch 2.2.2+cpu), roughly:

| Input | Path | Time |
| --- | --- | --- |
| 320×240, no faces | Real-ESRGAN 2× | ~2.5 s |
| 256×256, one face | de-block → Real-ESRGAN + GFPGAN 2× | ~9 s |

A CUDA build of torch is an order of magnitude faster; nothing else needs to change.

---

## Layout

```
image-enhancer-py/
├── app/
│   ├── main.py            FastAPI: API endpoints + serves the web UI
│   ├── config.py          all settings, all with defaults
│   ├── jobs.py            in-memory job store, single-slot worker, TTL sweeper
│   ├── analysis.py        classical-CV image analysis (what makes one mode adaptive)
│   ├── validation.py      magic bytes, safe decode, EXIF, pixel limits
│   ├── errors.py          typed errors → HTTP status + stable code
│   └── pipeline/
│       ├── beautify.py    THE pipeline: plan + orchestration
│       ├── registry.py    lazy model loading, DNI denoise blending
│       ├── ops.py         denoise, sharpen, hair, texture, finish, safeguards
│       └── encode.py      encode + output validation
├── web/                   index.html · styles.css · app.js  (the whole frontend)
├── models/                Real-ESRGAN + GFPGAN weights
├── gfpgan/weights/        face detection + parsing weights (GFPGAN loads these by relative path)
├── requirements.txt
└── run.ps1
```

## Third-party models

Real-ESRGAN (BSD-3-Clause) and GFPGAN (Apache-2.0) carry their own licenses and commercial-use
terms. Read them before deploying commercially.
