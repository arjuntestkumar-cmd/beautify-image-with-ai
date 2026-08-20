# Khushify AI

AI-powered image enhancement and beautification.

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
| `GET` | `/api/filters` | the premium looks, and which one applies by default |
| `POST` | `/api/enhance` | multipart `image=@photo.jpg`, optional `mode=portrait\|beautify\|clear` (default `portrait`), optional `filter=<id>` → `202` + job id |
| `GET` | `/api/jobs/{id}` | status, stage, progress, queue position, and the result metadata when done |
| `GET` | `/api/jobs/{id}/result` | the beautified image |
| `GET` | `/api/jobs/{id}/original` | the original (the before/after slider uses it) |
| `GET` | `/api/jobs/{id}/base` | the enhanced photo with **no look on it** — what the browser previews looks from |
| `POST` | `/api/jobs/{id}/filter` | `filter=<id>` → `202`; re-renders under a different look, or `none` |
| `DELETE` | `/api/jobs/{id}` | delete the files now |

```bash
curl http://127.0.0.1:8000/api/filters
curl -X POST http://127.0.0.1:8000/api/enhance -F "image=@photo.jpg"                    # portrait + default look
curl -X POST http://127.0.0.1:8000/api/enhance -F "image=@photo.jpg" -F "mode=beautify"
curl -X POST http://127.0.0.1:8000/api/enhance -F "image=@photo.jpg" -F "mode=clear"    # clear only
curl -X POST http://127.0.0.1:8000/api/enhance -F "image=@photo.jpg" -F "filter=silk"
curl http://127.0.0.1:8000/api/jobs/<id>
curl -o out.jpg http://127.0.0.1:8000/api/jobs/<id>/result
curl -X POST http://127.0.0.1:8000/api/jobs/<id>/filter -F "filter=none"   # take the look off
```

Every response is wrapped: `{"success": true, "data": {...}}`, or
`{"success": false, "error": {"code": "...", "message": "..."}}`. Interactive docs at `/docs`.

Uploads are validated before a job is created (magic bytes, real decode, EXIF orientation,
animation rejected, 64 MB / 80 MP ceilings), so a bad file fails instantly rather than a minute
into processing. A photo over the pixel budget is fitted to it and processed rather than
refused — being large is a reason to work differently, not a reason to hand the file back.

---

## The three modes

All three run the same restoration - upscale, denoise, face restoration, face clarity - over
every pixel of the frame. They differ only in what happens to the skin and the colour afterwards.

| | Portrait (default) | Beautify | Clear only |
| --- | --- | --- | --- |
| Restore + denoise + sharpen | yes | yes | yes |
| Face restoration + eye/lip clarity | yes | yes | yes |
| Spots and blemishes removed | yes | yes | **yes** |
| Soft-focus recovery on clothing and hair | **top of range** | standard | standard |
| Skin evened out | **half** | full | **no** |
| Soft highlight glow on skin | half | full | **no** |
| Tone curve, vibrance, white balance | 0.30 | 0.46 | **none** |
| Original skin texture kept | more | less | **most** (+0.18) |

**Portrait** is the default because it is the one that answers "clean up my photo" without a
follow-up question: the whole frame is restored, the recovery of a soft shirt or a soft hairline
runs at the top of its bounded range, and the cosmetic half is deliberately held back to about
half of Beautify so the result still reads as the person rather than as a filter.

Measured on a grainy 240x240 portrait, beautify comes out ~11% more saturated and ~11% higher
contrast than clear, while clear retains more raw skin texture.

Pick **Clear only** when the photo should stay exactly itself, just cleaner - documents of
people, evidence, product shots, anything where a flattering grade would be wrong.

Note the one row that is **yes** on both sides. Evening a complexion out is cosmetic and Clear
mode is right to refuse it; taking a spot off a cheek is repair, in the same sense that removing
a dust mark is, so it runs in both. Clear mode used to leave every mark in place *and* run an
unguarded sharpener over the face afterwards, which found each one and deepened it - a photo that
came back reading dirtier than the one that went in. See `ops.blemish_clean`.

---

## The premium looks

The pipeline above fixes what is *wrong* with a photo — grain, blur, compression, a face the
sensor never resolved. That is repair, and repair is not the same as flattery. The looks are the
flattery: the grade, the skin, the light. They run last, as a separable stage.

**One is applied by default** (Natural Radiance), because a photo that comes back looking merely
*correct* is not what anyone uploading to a beautifier wanted. It is a starting point and never
a commitment: the un-styled result is saved next to the finished one, so switching looks or
removing the filter re-renders from there and never touches the models again. Clear mode's
default is no look at all — its whole promise is that nothing was styled.

| Look | What it leans on |
| --- | --- |
| **Natural Radiance** *(default)* | even skin that keeps its texture, warm light, clear eyes |
| Soft Porcelain | softer skin, gentle light |
| Golden Aura | lit-from-within glow, golden highlights |
| Warm Amber | golden-hour warmth, richer colour |
| Crystal Clear | bright, cool, crisp — the cleanest of the set |
| Silk Premium | smooth skin over matte, editorial blacks |
| Sculpted Detail | depth and definition; eyes and brows forward, skin left alone |
| Even Tone | balances uneven skin tone and redness |
| Rose Bloom | fuller lips, sharper facial detail, soft warm base |
| Cinematic | teal shadows against warm skin — the film-trailer grade |
| Morning Dawn | barely there: soft light, open shadows, colour left as photographed |
| Studio Noir | black and white through a warm filter; deep blacks, luminous skin |
| Original | no look — the enhanced photo exactly as the pipeline produced it |

**They apply instantly in the browser.** `/api/filters` publishes each look's grade parameters,
not just its name, and `web/looks.js` is a port of the frame-wide half of `filters.py` - the same
tone curve, the same hue-band saturation, the same CLAHE, the same vignette. Measured against the
server's own output the preview is within about two levels out of 255 on most looks, three on the
one that leans hardest on local contrast.

Every look is rendered **once, up front**, from the un-styled base the job kept, while you are
still looking at the result. A click is then an `<img>` swap and nothing else - **14 to 32 ms**
from click to a decoded picture, measured in Chromium, against roughly two seconds before, when
the strip disabled itself for the length of the server round trip. The chips are never disabled.

The server render still happens, because the browser draws only the frame-wide half of a look and
the file you download has to carry the face work as well - skin, lips, eyes, glow. It runs behind
the picture, debounced and coalesced, so browsing seven looks costs **one** render rather than
seven, and it swaps itself in when it lands. Press Download before it has and the button waits
for it - about a second - rather than handing over the previous look's file.

Three rules keep them on the right side of natural:

* **Identity is not negotiable.** Nothing moves a feature, narrows a face, or lightens skin
  toward some other skin. Smoothing is frequency-separated — the low frequencies of the skin are
  evened out and the micro-texture is added straight back on top — because the plastic look is
  what happens when a filter takes the pores along with the blotches. Skin evening pulls chroma
  a fraction of the way toward the photo's **own** median skin tone, so a blotchy cheek is
  matched to the rest of that person's face rather than to a preset's idea of a face. Lips are
  found by intersecting the mouth band with pixels redder than that same measured skin tone,
  which is what keeps the effect off the chin and correct across complexions.
* **Every term is bounded.** The strongest look here still keeps more than 40% of the original
  skin, and the tone curve is the same bounded family the finish already uses.
* **They cost what a filter should cost.** The frame-wide half is a single per-channel LUT, a
  skin-protected vibrance and one local-contrast pass; the face half runs over the face region,
  and the lip and eye passes over their own bands rather than the whole photo. All of it is
  tile-safe on the same terms as everything else — a LUT is pointwise, so it is exactly
  identical on a tile, and the one whole-frame measurement is taken once for the frame.

## What "Beautify" actually does

There is one mode, but it is not one fixed recipe — it reads the photo first and adapts. That
adaptation is exactly what the original service's mode presets did; it now happens
automatically instead of being a question put to the user.

```
decode → analyse → rescue exposure (dark photos only) → re-analyse → plan
  → de-block, if the input is heavily JPEG-compressed
  → Real-ESRGAN restoration over the WHOLE frame, tile by tile if the photo is large
  → GFPGAN face restoration, blended onto it under this pipeline's own radial mask
  → chroma denoise, if the result is still noisy
  → blemish removal: spots and marks off the skin, in BOTH modes
  → clarity inside the face, then across everything that is not a face
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
- **Where the face is pasted back.** GFPGAN aligns each face to a 512x512 crop and pastes the
  result back through the inverse of that alignment, which carries a rotation - so anything
  square in aligned space returns as a *rotated square* in the photograph. facexlib builds that
  paste mask from a face-parsing network, and on a degraded face the network is unreliable:
  measured on the sample portrait blurred at sigma 3.4 it labelled 45% of the crop "lower lip",
  producing a mask that stood at 1.000 at the crop's own edge on three of four sides. That hard
  straight edge is the box users reported drawn across the head. The paste is therefore done
  here instead, under a **radial** mask that has no straight edge anywhere in it and reaches zero
  well before the crop border whatever the parser believes - and, as a bonus, one that restores
  the hair and jaw inside its circle instead of discarding them.
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
| `MAX_UPLOAD_BYTES` | `67108864` | 64 MB upload ceiling |
| `MAX_INPUT_PIXELS` | `80000000` | 80 MP; anything larger is fitted to it, not rejected |
| `CHUNK_THRESHOLD_PIXELS` | `4000000` | past 4 MP the heavy stages run in tiles |
| `CHUNK_TILE_SIZE` | `768` | tile edge for the classical stages |
| `MODEL_CHUNK_TILE_SIZE` | `384` | tile edge (input side) for the super-resolution model |
| `WORKER_CONCURRENCY` | `1` | heavy jobs running at once; the queue holds the rest |
| `MAX_QUEUED_JOBS` | `64` | how deep the queue goes before new uploads are turned away |
| `JOB_TIMEOUT_SECONDS_PER_MEGAPIXEL` | `90` | the deadline scales with the photo |
| `KEEP_UNFILTERED_BASE` | `true` | keep an un-styled copy so looks can be swapped for free |
| `OUTPUT_QUALITY` | `92` | JPEG/WebP encoder quality |
| `MOCK_MODE` | `false` | `true` = plain resize, no AI. For wiring tests only |

Output keeps the input's format (JPEG in → JPEG out); PNG and WebP keep transparency.

---

## Storage and privacy

Uploads and results are written to `.data/` and deleted `RESULT_RETENTION_MINUTES` after the
job finishes, by a background sweeper. The directory is also wiped on startup, so a crash
cannot leave old photos behind. Nothing is uploaded anywhere — inference is local.

---

## Large files, and more than one person at a time

Two things used to make this service fall over, and neither of them was the model.

**A big photo did not need more time, it needed less memory at once.** Every classical stage
was a whole-array operation, and at 40 MP one `cvtColor` to float LAB is 480 MB with five more
behind it. Worse, `RealESRGANer` always runs its 4× network before scaling back to the size you
asked for, so a 24 MP photo briefly became a 384 MP tensor — about 4.6 GB — to come back the
same size it went in. So the heavy stages now run over overlapping tiles and peak memory
follows the tile, not the photo (`app/pipeline/chunked.py`).

Two rules keep that invisible in the output:

* **Halo.** Each tile is cut with a margin of context that is discarded after the op runs. When
  the margin is wider than the filter's reach, every interior pixel sees exactly the
  neighbourhood it would have seen in the whole-image call — the tiled result is not close to
  the untiled one, it is *identical*.
* **Global constants are measured once.** White-balance means, the edge-magnitude percentile,
  the exposure gamma, the CLAHE curves: anything derived from the whole frame is measured up
  front and handed to every tile. This is the part that would otherwise show — each tile
  measuring its own would land a different correction on each side of a boundary.

CLAHE was the hard one, since a per-tile histogram prints its grid across the photo as steps in
brightness. It is already a grid algorithm, though, so `ClaheField` reads its 64 cell
histograms one cell at a time and applies the resulting curves to any tile: the same algorithm,
with its measurement separated from its application. Faces are handled the same way — the face
stages run over the face region rather than the frame, and on a large photo GFPGAN restores
each face from its own crop instead of upscaling the entire background to get at it.

Photos under `CHUNK_THRESHOLD_PIXELS` skip all of this and take the original path unchanged.

**Several people at once was a queue problem.** Inference is serialised
(`WORKER_CONCURRENCY=1`): it saturates the CPU or GPU, so running two at once makes both
slower and can exhaust memory. That single slot is the load control — ten simultaneous uploads
mean nine ordered, lossless waits (the UI shows how many are ahead) rather than ten jobs
racing each other into an out-of-memory kill. Jobs are submitted once, fail independently, and
hand their memory back before the next one starts. The deadline scales with megapixels, so a
large photo gets the minutes it needs and only a genuinely hung job is stopped.

The priority, in order: **stability → successful processing → output quality → speed.**

---

## Performance

On CPU (torch 2.2.2+cpu), roughly:

| Input | Path | Time |
| --- | --- | --- |
| 320×240, no faces | Real-ESRGAN 2× | ~2.5 s |
| 256×256, one face | de-block → Real-ESRGAN + GFPGAN 2× | ~9 s |

A CUDA build of torch is an order of magnitude faster; nothing else needs to change.

Peak memory of the classical stages on a 24 MP (6000×4000) photo, measured, whole-array against
tiled:

| | Peak allocation | Wall clock |
| --- | --- | --- |
| Whole-array | 3387 MB | 334 s |
| Tiled (`CHUNK_TILE_SIZE=768`) | **315 MB** | **143 s** |

The tiled run is also the faster one, which is not a coincidence: the whole-array version spends
most of its time moving half-gigabyte intermediates through a cache that cannot hold them.

Changing the look on a finished photo re-renders from the saved un-styled copy — no decode of
the original, no analysis, no models — so it costs a fraction of a full run rather than another
one. It still goes through the same queue, so it can be waiting behind somebody else's photo.

---

## Layout

```
image-enhancer-py/
├── app/
│   ├── main.py            FastAPI: API endpoints + serves the web UI
│   ├── config.py          all settings, all with defaults
│   ├── jobs.py            in-memory job store, single-slot queue, TTL sweeper
│   ├── analysis.py        classical-CV image analysis (what makes one mode adaptive)
│   ├── validation.py      magic bytes, safe decode, EXIF, pixel limits
│   ├── errors.py          typed errors → HTTP status + stable code
│   └── pipeline/
│       ├── beautify.py    THE pipeline: plan + orchestration
│       ├── chunked.py     tiled execution: what makes a very large photo finish
│       ├── filters.py     the premium looks applied after the restoration
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
