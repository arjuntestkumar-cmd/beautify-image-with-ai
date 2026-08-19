# Deploying to Hugging Face Spaces

Why Spaces: the pipeline needs roughly **960 MB of RAM** with the models loaded (measured, not
estimated). Render's free and Starter instances give 512 MB, which is why the deployed build was
OOM-bound even before the missing weights. A free CPU Space has enough headroom to run the whole
pipeline, face restoration included.

Check the current free-tier limits yourself before relying on them — providers change them.

---

## 1. Create the Space

On <https://huggingface.co/new-space>:

- **SDK:** Docker → *Blank*
- **Hardware:** CPU basic (free)
- Visibility: your choice

> **Already made a Static Space by mistake?** You do not have to delete it. A Space's type comes
> from the `sdk:` field in its `README.md` front-matter, so pushing this repo (whose README already
> declares `sdk: docker`) converts it on the next build. If the Space stubbornly stays static,
> delete it and create a new one with **Docker → Blank**.

## 2. The Space configuration

Already done — this repo's `README.md` starts with the front-matter Spaces reads:

```yaml
---
title: Beautify
emoji: ✨
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

`app_port: 7860` must match the port the container listens on — the `Dockerfile` defaults to it.
A **Static** Space serves files only and cannot run Python, which is why `sdk: docker` matters.

## 3. Push the code

From this project folder:

```bash
git init                                  # if it is not a repo yet
git add -A
git commit -m "Beautify"

git remote add space https://huggingface.co/spaces/<your-user>/<space-name>
git push space main                       # username = your HF user, password = an access token
```

Create the access token at <https://huggingface.co/settings/tokens> with **write** permission.

If the Space already has commits (a fresh Space ships a placeholder `index.html`), the first push
is rejected as non-fast-forward. Either force it, or pull first:

```bash
git push space main --force               # replaces the placeholder Space content
```

`.gitignore` already keeps the 528 MB of weights and the `.data/` scratch folder out of the push -
the Dockerfile downloads the weights during the build instead.

Everything needed is already in the repo: `Dockerfile`, `requirements.txt`,
`scripts/fetch_models.py`, `app/` and `web/`.

## 4. Watch the build

The build downloads ~528 MB of weights and installs CPU-only torch, so the first build takes a
while. In the build log you should see:

```
models/realesr-general-x4v3.pth        downloading 4.7 MB...
models/GFPGANv1.4.pth                  downloading 332.5 MB...
All weights ready (527.6 MB total, 5 downloaded).
```

Then, once it starts:

```
beautify.registry registry loaded: {'ready': True, ... 'models': {'realesrgan:general': True,
                                    'realesrgan:wdn': True, 'gfpgan': True}}
```

## 5. Verify it is really doing AI

```bash
curl https://<your-space>.hf.space/health
```

`"ready"` must be `true`, `"mockMode"` must be `false`, and `"models"` must list all three. If
`ready` is false the app now **refuses** to process anything rather than quietly returning a
resized copy — that is the failure this deployment guide exists to prevent.

---

## What to expect

- **Speed.** Shared CPU, so 30–60 s per photo is normal; your local machine is faster.
- **Sleeping.** Free Spaces idle out and take ~30 s to wake. The weights are inside the image, so
  a wake-up does not re-download them.
- **Storage.** Uploads and results live in the container's `.data/` and are deleted 30 minutes
  after each job, and on every restart. A Space restart wipes them regardless.

## Tuning

Set these as Space **Variables** if you need to:

| Variable | Default | Effect |
| --- | --- | --- |
| `AUTO_UPSCALE_MAX_SIDE` | `1600` | Lower it to make big photos faster (they skip the 2× upscale) |
| `RESULT_RETENTION_MINUTES` | `30` | How long a result survives |
| `MAX_UPLOAD_BYTES` | `67108864` | Upload ceiling (64 MB) |
| `CHUNK_TILE_SIZE` | `768` | Lower it first if the machine is memory-starved; peak memory scales with it |
| `MAX_QUEUED_JOBS` | `64` | How many people can be waiting before uploads are refused |
| `JOB_TIMEOUT_SECONDS` | `900` | Raise if slow hardware trips it |

Do **not** raise `WORKER_CONCURRENCY` or add uvicorn workers: each one loads its own ~960 MB copy
of the models.

## Other container hosts

The same `Dockerfile` works anywhere that runs a container with ~1.5 GB of RAM (Fly.io, Railway,
a VPS, Render Standard). Only the port convention differs, and `PORT` is honoured.
