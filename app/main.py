"""The one and only backend: HTTP API + the web UI, in a single FastAPI app.

    GET  /                      the web UI
    GET  /health                liveness + which models are loaded
    GET  /api/filters           the premium looks, and which one is applied by default
    POST /api/enhance           multipart upload -> 202 with a job id (mode=portrait|beautify|clear)
    GET  /api/jobs/{id}         status / progress
    GET  /api/jobs/{id}/result  the beautified image
    GET  /api/jobs/{id}/original  the original (used by the before/after slider)
    POST /api/jobs/{id}/filter  swap the look, or remove it -> 202, poll the same job
    DELETE /api/jobs/{id}       delete the files now

There is no second service to run: the model inference happens in this process.
"""
from __future__ import annotations

import mimetypes
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from fastapi.staticfiles import StaticFiles

from .config import BUILD_ID, ROOT, get_settings
from .errors import (AppError, CorruptedImage, FileTooLarge, ModelsUnavailable,
                     UnsupportedImageFormat)
from .jobs import Job, JobStore
from .logging_utils import configure, get_logger
from .pipeline import filters
from .pipeline.beautify import MODES, MODE_DEFAULT, beautify, restyle
from .pipeline.encode import FORMAT_MIME
from .pipeline.registry import ModelRegistry
from .validation import MIME_BY_FORMAT, decode_and_normalize, compress_heavy_image

settings = get_settings()
configure(settings.LOG_LEVEL)
log = get_logger("api")

registry = ModelRegistry(settings)
# `reclaim` runs between jobs: it hands the accelerator's cache back so the next job in the
# queue starts from a clean slate rather than on top of the last one's peak.
store = JobStore(settings, reclaim=lambda: registry.release())

WEB_DIR = os.path.join(ROOT, "web")


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.purge_all_files()
    store.start()
    log.info("loading models (first run can take a minute)...")
    registry.load()
    status = registry.public_status()
    if status["mockMode"]:
        log.warning("MOCK MODE — output is a plain resize, not AI enhancement")
    elif not status["ready"]:
        log.error("models did NOT load: %s — check models/ and requirements", status["modelErrors"])
    else:
        log.info("ready on %s | models: %s", status["device"], ", ".join(sorted(status["models"])))
    log.info("open http://%s:%s", settings.HOST, settings.PORT)
    yield
    store.shutdown()


# Python ships no mapping for .webp on some platforms (Windows among them), and StaticFiles
# then serves the brand assets as application/octet-stream. Browsers sniff an <img> and render
# it anyway, so the site looks fine — but link-preview crawlers and the PWA manifest reject a
# non-image type, which is exactly where the logo is supposed to earn its keep.
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("application/manifest+json", ".webmanifest")

app = FastAPI(title="Khushify AI", version="1.0.0", docs_url="/docs", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": {"code": exc.code, "message": exc.message}},
    )


# ---------------------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    status = registry.public_status()
    return {
        "status": "ok" if (status["ready"] or status["mockMode"]) else "degraded",
        # The code this process is running, not the code on disk. See config._build_id - this is
        # how you tell a deploy that landed from one that only looked like it did.
        "build": BUILD_ID,
        **status,
        "mode": "beautify",
    }


# ---------------------------------------------------------------------------------------
# enhance
# ---------------------------------------------------------------------------------------
def _work(job: Job) -> None:
    """Runs on the worker thread: decode -> beautify -> record the result.

    The upload was already decoded once during validation. Decoding again here (rather than
    carrying the array over) is deliberate: a queue of pending 40-megapixel arrays would sit in
    memory for as long as the queue is deep, and a re-decode costs a fraction of the inference.
    """
    decoded = decode_and_normalize(job.original_path, settings.MAX_INPUT_PIXELS,
                                   settings.DOWNSCALE_OVERSIZE_INPUT)
    out_template = os.path.join(settings.DATA_DIRECTORY, f"{job.id}-result.{{ext}}")
    path, result = beautify(
        registry, settings, decoded, out_template,
        progress=lambda stage, pct, msg: store.progress(job, stage, pct, msg),
        mode=job.mode, look_id=job.look, should_stop=job.timed_out,
    )
    job.result_path = path
    job.result = result


def _restyle_work(job: Job, look_id: str) -> None:
    """Runs on the worker thread: re-render the saved base under a different look.

    No decode of the original, no analysis, no models — just the look, over an image the job
    already produced. This is what a look change costs, and it is why the default look can be
    offered as a default rather than as a commitment.
    """
    out_template = os.path.join(settings.DATA_DIRECTORY,
                                f"{job.id}-result-{job.result_version + 1}.{{ext}}")
    previous = job.result_path
    path, result = restyle(
        settings, job.result, out_template, look_id,
        progress=lambda stage, pct, msg: store.progress(job, stage, pct, msg),
        should_stop=job.timed_out,
    )
    job.result_path = path
    job.result = result
    job.look = result.look
    job.result_version += 1
    if previous and previous != path:
        _safe_remove(previous)


@app.get("/api/filters")
def filter_catalogue(mode: str = MODE_DEFAULT) -> dict:
    """The looks on offer, and the one that applies when the caller does not choose."""
    return {"success": True, "data": filters.catalogue(mode if mode in MODES else MODE_DEFAULT)}


@app.post("/api/enhance", status_code=202)
async def enhance(image: UploadFile = File(...), mode: str = Form(MODE_DEFAULT),
                  filter: str = Form(None)) -> JSONResponse:
    # Refuse up front rather than queueing work that cannot produce a real result.
    if not settings.MOCK_MODE and not registry.status.ready:
        raise ModelsUnavailable(
            "The AI engine is not available on this server. Enhancement models are not loaded, "
            "so no photo can be processed."
        )

    # Land the upload on disk under a temporary name and fully validate it BEFORE a job exists,
    # so a rejected file never leaves a phantom job behind.
    upload_path = os.path.join(settings.DATA_DIRECTORY, f"upload-{uuid.uuid4().hex}")
    total = 0
    try:
        try:
            with open(upload_path, "wb") as fh:
                # A hard ceiling while streaming: a huge upload can never fill memory or the disk.
                while chunk := await image.read(1024 * 1024):
                    total += len(chunk)
                    if total > settings.MAX_UPLOAD_BYTES:
                        raise FileTooLarge(
                            f"That image is larger than {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                        )
                    fh.write(chunk)
        finally:
            await image.close()

        if total == 0:
            raise CorruptedImage("The uploaded file is empty.")

        try:
            # A photo bigger than the pixel budget is fitted to it and accepted. Needing more
            # room than the budget is a reason to work in chunks, not a reason to refuse the
            # file — see validation.decode_and_normalize and pipeline/chunked.py.
            decoded = decode_and_normalize(upload_path, settings.MAX_INPUT_PIXELS,
                                           settings.DOWNSCALE_OVERSIZE_INPUT)
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedImageFormat("That file could not be read as an image.") from exc

        # Compress heavy images before processing (> 10 MB → lighter load for models)
        compressed_path = os.path.join(settings.DATA_DIRECTORY, f"compressed-{uuid.uuid4().hex}")
        final_upload_path, was_compressed = compress_heavy_image(
            upload_path, compressed_path, size_threshold_bytes=10_000_000
        )
        if was_compressed:
            _safe_remove(upload_path)
            upload_path = final_upload_path
            total = os.path.getsize(upload_path)  # update size after compression
            log.info("image compressed: new size %s KB", total // 1024)
        else:
            _safe_remove(compressed_path)  # clean up unused path

    except Exception:
        _safe_remove(upload_path)
        raise

    # Accepted — now it gets a job (this is also where a full queue is reported).
    try:
        job = store.create()
    except AppError:
        _safe_remove(upload_path)
        raise

    original_path = os.path.join(settings.DATA_DIRECTORY, f"{job.id}-input")
    os.replace(upload_path, original_path)

    job.original_path = original_path
    job.mode = mode if mode in MODES else MODE_DEFAULT
    job.look = filters.resolve(filter, job.mode).id
    job.original_name = os.path.basename(image.filename or "image")
    job.original_mime = MIME_BY_FORMAT.get(decoded.detected_format, "image/jpeg")
    job.original_bytes = total
    # Drives the job's deadline: a big photo is given the time it needs instead of a timeout
    # sized for a small one.
    job.megapixels = round(decoded.width * decoded.height / 1_000_000.0, 3)

    store.submit(job, _work)
    log.info("job %s queued (%s, look=%s, %sx%s, %s MP, %s KB)", job.id, job.mode, job.look,
             decoded.width, decoded.height, job.megapixels, total // 1024)
    return JSONResponse(status_code=202, content={"success": True, "data": job.to_public()})


@app.post("/api/jobs/{job_id}/filter", status_code=202)
def job_filter(job_id: str, filter: str = Form(...)) -> JSONResponse:
    """Change this photo's look, or take it off — the default is never a one-way door.

    Re-renders from the un-styled base the job kept, so nothing expensive happens: no decode of
    the original, no analysis, no models. It still goes through the same queue as everything
    else, because it is still image work and unmetered work is how a server ends up overloaded.
    """
    job = store.get_for_file(job_id)
    if job.status != "completed" or job.result is None:
        return _conflict("NOT_READY", "That photo has not finished processing yet.")
    if not job.result.base_path or not os.path.exists(job.result.base_path):
        return _conflict("NO_BASE", "The un-styled version of this photo is no longer available.")
    if filter not in filters.BY_ID:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": {"code": "UNKNOWN_FILTER",
                                                 "message": "That filter does not exist."}},
        )
    if not store.reopen(job):
        return _conflict("IN_PROGRESS", "That job is still running.")

    store.submit(job, _restyle_work, filter)
    log.info("job %s re-styling to %s", job.id, filter)
    return JSONResponse(status_code=202, content={"success": True, "data": job.to_public()})


def _conflict(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=409,
                        content={"success": False, "error": {"code": code, "message": message}})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return {"success": True, "data": store.get(job_id).to_public()}


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = store.get_for_file(job_id)
    if job.status != "completed" or not job.result_path or not os.path.exists(job.result_path):
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": {"code": "NOT_READY", "message": "The result is not ready yet."}},
        )
    stem = os.path.splitext(job.original_name)[0] or "image"
    ext = os.path.splitext(job.result_path)[1]
    return FileResponse(
        job.result_path,
        media_type=job.result.mime_type if job.result else "application/octet-stream",
        headers={
            "Content-Disposition": f'inline; filename="{stem}-beautified{ext}"',
            "Cache-Control": "private, max-age=300",
        },
    )


@app.get("/api/jobs/{job_id}/base")
def job_base(job_id: str):
    """The enhanced photo with NO look on it - the image every look is rendered from.

    This is what makes the browser's filter previews honest. Without it the page can only
    preview a look on top of whatever result is currently on screen, which already carries a
    look: pick "Golden Aura" while "Warm Amber" is showing and the swatch is Amber-then-Aura, a
    grade nobody will ever be sent. Previewing from the base is previewing the real thing.

    It is the same file the server re-renders from (BeautifyResult.base_path), so a preview and
    the result that replaces it a second later started from identical pixels.
    """
    job = store.get_for_file(job_id)
    base = job.result.base_path if job.result else None
    if job.status != "completed" or not base or not os.path.exists(base):
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": {"code": "NO_BASE",
                                                 "message": "The un-styled version is not available."}},
        )
    return FileResponse(base, media_type=FORMAT_MIME.get(job.result.base_format, "image/jpeg"),
                        headers={"Cache-Control": "private, max-age=300"})


@app.get("/api/jobs/{job_id}/original")
def job_original(job_id: str):
    job = store.get_for_file(job_id)
    if not job.original_path or not os.path.exists(job.original_path):
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": {"code": "NOT_FOUND", "message": "The original is gone."}},
        )
    return FileResponse(job.original_path, media_type=job.original_mime,
                        headers={"Cache-Control": "private, max-age=300"})


@app.delete("/api/jobs/{job_id}")
def job_delete(job_id: str):
    job = store.get(job_id)
    if job.status in ("queued", "processing"):
        # Deleting the input from under the worker would fail the job in a confusing way.
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": {"code": "IN_PROGRESS",
                                                 "message": "That job is still running."}},
        )
    for path in store.files_of(job):
        _safe_remove(path)
    job.expires_at = time.time() - 1
    store.sweep()
    return {"success": True, "data": {"deleted": True}}


def _safe_remove(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------------------
# web UI (served by this same process — there is no separate frontend server)
# ---------------------------------------------------------------------------------------
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    _index_cache: dict = {}

    def _index_html() -> str:
        """index.html, re-read whenever it changes on disk.

        Keyed on mtime rather than cached once, because `web/` being live is a property this
        project leans on: editing the page and refreshing has to work without a restart.
        """
        path = os.path.join(WEB_DIR, "index.html")
        mtime = os.path.getmtime(path)
        if _index_cache.get("mtime") != mtime:
            with open(path, encoding="utf-8") as fh:
                _index_cache.update(mtime=mtime, html=fh.read())
        return _index_cache["html"]

    def _public_base(request: Request) -> str:
        """The absolute origin to advertise in the social-card tags.

        Open Graph will not accept a relative image URL - that is exactly why a shared link came
        back with a broken preview - and this app answers on a bare IP today and on a domain
        tomorrow, so the value cannot be baked into the HTML.

        PUBLIC_BASE_URL wins when set. Otherwise it is reconstructed from the request, honouring
        the X-Forwarded-* headers a reverse proxy sets, so a site fronted by Caddy advertises the
        https it is served over rather than the http it is proxied on. Each header is comma-split
        because a chain of proxies appends to them rather than replacing them.
        """
        configured = (settings.PUBLIC_BASE_URL or "").strip().rstrip("/")
        if configured:
            return configured
        def first(value: str) -> str:
            return value.split(",")[0].strip()
        proto = first(request.headers.get("x-forwarded-proto") or request.url.scheme)
        host = first(request.headers.get("x-forwarded-host")
                     or request.headers.get("host") or request.url.netloc)
        return f"{proto}://{host}"

    @app.get("/", include_in_schema=False)
    def index(request: Request):
        return HTMLResponse(_index_html().replace("{{BASE_URL}}", _public_base(request)))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return RedirectResponse("/static/favicon.svg")
