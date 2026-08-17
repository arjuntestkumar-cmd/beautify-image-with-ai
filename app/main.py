"""The one and only backend: HTTP API + the web UI, in a single FastAPI app.

    GET  /                      the web UI
    GET  /health                liveness + which models are loaded
    POST /api/enhance           multipart upload -> 202 with a job id
    GET  /api/jobs/{id}         status / progress
    GET  /api/jobs/{id}/result  the beautified image
    GET  /api/jobs/{id}/original  the original (used by the before/after slider)
    DELETE /api/jobs/{id}       delete both files now

There is no second service to run: the model inference happens in this process.
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, get_settings
from .errors import AppError, CorruptedImage, FileTooLarge, UnsupportedImageFormat
from .jobs import Job, JobStore
from .logging_utils import configure, get_logger
from .pipeline.beautify import beautify
from .pipeline.registry import ModelRegistry
from .validation import MIME_BY_FORMAT, decode_and_normalize

settings = get_settings()
configure(settings.LOG_LEVEL)
log = get_logger("api")

registry = ModelRegistry(settings)
store = JobStore(settings)

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


app = FastAPI(title="Beautify", version="1.0.0", docs_url="/docs", lifespan=lifespan)

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
    decoded = decode_and_normalize(job.original_path, settings.MAX_INPUT_PIXELS)
    out_template = os.path.join(settings.DATA_DIRECTORY, f"{job.id}-result.{{ext}}")
    path, result = beautify(
        registry, settings, decoded, out_template,
        progress=lambda stage, pct, msg: store.progress(job, stage, pct, msg),
    )
    job.result_path = path
    job.result = result


@app.post("/api/enhance", status_code=202)
async def enhance(image: UploadFile = File(...)) -> JSONResponse:
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
            decoded = decode_and_normalize(upload_path, settings.MAX_INPUT_PIXELS)
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedImageFormat("That file could not be read as an image.") from exc
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
    job.original_name = os.path.basename(image.filename or "image")
    job.original_mime = MIME_BY_FORMAT.get(decoded.detected_format, "image/jpeg")
    job.original_bytes = total

    store.submit(job, _work)
    log.info("job %s queued (%sx%s, %s KB)", job.id, decoded.width, decoded.height, total // 1024)
    return JSONResponse(status_code=202, content={"success": True, "data": job.to_public()})


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
    for path in (job.original_path, job.result_path):
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

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return RedirectResponse("/static/favicon.svg")
