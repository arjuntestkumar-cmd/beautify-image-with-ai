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
| `POST` | `/api/enhance` | multipart `image=@photo.jpg` → `202` + job id |
| `GET` | `/api/jobs/{id}` | status, stage, progress, and the result metadata when done |
| `GET` | `/api/jobs/{id}/result` | the beautified image |
| `GET` | `/api/jobs/{id}/original` | the original (the before/after slider uses it) |
| `DELETE` | `/api/jobs/{id}` | delete both files now |

```bash
curl -X POST http://127.0.0.1:8000/api/enhance -F "image=@photo.jpg"
curl http://127.0.0.1:8000/api/jobs/<id>
curl -o out.jpg http://127.0.0.1:8000/api/jobs/<id>/result
```

Every response is wrapped: `{"success": true, "data": {...}}`, or
`{"success": false, "error": {"code": "...", "message": "..."}}`. Interactive docs at `/docs`.

Uploads are validated before a job is created (magic bytes, real decode, EXIF orientation,
animation rejected, 20 MB / 40 MP ceilings), so a bad file fails instantly rather than a minute
into processing.

---

## What "Beautify" actually does

There is one mode, but it is not one fixed recipe — it reads the photo first and adapts. That
adaptation is exactly what the original service's mode presets did; it now happens
automatically instead of being a question put to the user.

```
decode → analyse → plan
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

- **Scale.** Photos at or below 1600 px on the long side get a real 2× upscale; larger photos
  are beautified at native size. `AUTO_UPSCALE_MAX_SIDE` moves that line.
- **Whether to touch faces at all.** A large, sharp, clean face is left alone — running a face
  model over an already-good face makes it worse. Only degraded, blurry, small or noisy faces
  are restored.
- **How hard.** Face-model weight, denoise and detail all scale with the measured blur, noise
  and compression of that specific photo.
- **Which faces.** Only the dominant subject, unless it is a genuine group photo (a second face
  at least ~45% the area of the largest). This is what prevents ghost faces on busy backgrounds.
- **Screenshots and documents** are detected and never get face restoration or heavy texture.

Two things deliberately hold the "AI face" look back, because a high face-model weight is
exactly what produces waxy, re-drawn skin:

- the face-model weight is **capped at 0.58** — lower than the original's portrait mode;
- the source's real skin micro-texture is **re-injected** after restoration, in skin areas only.

Beauty comes from tone, hair and eyes, not from repainting the face.

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
