"""In-memory job store + a single-slot worker pool.

There is no database, no Redis and no cloud storage. A job lives in a dict; the original and
the result live in DATA_DIRECTORY until they expire, then a sweeper deletes them. That is the
whole "infrastructure" — it is enough for one machine, and it is what makes this build light.

Inference is serialised (WORKER_CONCURRENCY=1) because it saturates the CPU/GPU: running two at
once makes both slower and can exhaust memory.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import Settings
from .errors import JobNotFound, QueueFull, ResultExpired
from .logging_utils import get_logger

log = get_logger("jobs")

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"


@dataclass
class Job:
    id: str
    status: str = QUEUED
    stage: str = "queued"
    progress: int = 5
    message: str = "Waiting to start"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    expires_at: float = 0.0
    original_path: Optional[str] = None
    original_name: str = "image"
    original_mime: str = "image/jpeg"
    original_bytes: int = 0
    result_path: Optional[str] = None
    result: Optional[Any] = None          # BeautifyResult
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_public(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "jobId": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "createdAt": _iso(self.created_at),
            "expiresAt": _iso(self.expires_at) if self.expires_at else None,
            "statusUrl": f"/api/jobs/{self.id}",
            "resultUrl": f"/api/jobs/{self.id}/result",
            "originalUrl": f"/api/jobs/{self.id}/original",
        }
        if self.status == COMPLETED and self.result is not None:
            r = self.result
            out["output"] = {
                "width": r.width,
                "height": r.height,
                "format": r.mime_type,
                "bytes": r.bytes,
                "scale": r.effective_scale,
            }
            out["input"] = {
                "width": r.original_width,
                "height": r.original_height,
                "bytes": self.original_bytes,
                "format": self.original_mime,
            }
            out["details"] = {
                "faceCount": r.face_count,
                "facesRestored": r.processed_faces,
                "strategy": r.strategy,
                "quality": r.quality_category,
                "models": r.models,
                "processingTimeMs": r.processing_time_ms,
                "stageTimingsMs": r.stage_timings_ms,
                "warnings": r.warnings,
                "fallbackUsed": r.fallback_used,
            }
        if self.status == FAILED:
            out["error"] = {"code": self.error_code, "message": self.error_message}
        return out


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class JobStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, settings.WORKER_CONCURRENCY), thread_name_prefix="beautify"
        )
        self._stop = threading.Event()
        self._sweeper: Optional[threading.Thread] = None
        os.makedirs(settings.DATA_DIRECTORY, exist_ok=True)

    # ---- lifecycle ----
    def start(self) -> None:
        self._sweeper = threading.Thread(target=self._sweep_loop, name="beautify-sweeper", daemon=True)
        self._sweeper.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- jobs ----
    def create(self) -> Job:
        with self._lock:
            active = sum(1 for j in self._jobs.values() if j.status in (QUEUED, PROCESSING))
            if active >= self._settings.MAX_QUEUED_JOBS:
                raise QueueFull("The server is busy. Please try again in a moment.")
            job = Job(id=uuid.uuid4().hex[:20])
            job.expires_at = job.created_at + self._settings.RESULT_RETENTION_MINUTES * 60
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound("That job does not exist, or it has expired.")
        return job

    def get_for_file(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.expires_at and time.time() > job.expires_at:
            raise ResultExpired("That result has expired and been deleted.")
        return job

    def submit(self, job: Job, fn, *args) -> None:
        """Queue the work. `fn(job, *args)` runs on the worker thread."""
        self._pool.submit(self._run, job, fn, *args)

    def _run(self, job: Job, fn, *args) -> None:
        job.started_at = time.time()
        job.status = PROCESSING
        job.stage = "starting"
        job.progress = 8
        job.message = "Starting"
        try:
            fn(job, *args)
            job.status = COMPLETED
            job.stage = "completed"
            job.progress = 100
            job.message = "Done"
        except Exception as exc:  # noqa: BLE001 - the boundary: every failure becomes a job error
            code = getattr(exc, "code", "PROCESSING_FAILED")
            message = getattr(exc, "message", None) or "The image could not be enhanced."
            job.status = FAILED
            job.stage = "failed"
            job.progress = 100
            job.message = message
            job.error_code = code
            job.error_message = message
            log.exception("job %s failed: %s", job.id, exc)
        finally:
            job.finished_at = time.time()
            # Results are short-lived; the clock starts when the job finishes.
            job.expires_at = job.finished_at + self._settings.RESULT_RETENTION_MINUTES * 60

    def progress(self, job: Job, stage: str, percent: int, message: str) -> None:
        job.stage = stage
        job.progress = max(job.progress, int(percent))
        job.message = message

    # ---- cleanup ----
    def _sweep_loop(self) -> None:
        interval = max(10, self._settings.CLEANUP_INTERVAL_SECONDS)
        while not self._stop.wait(interval):
            try:
                self.sweep()
            except Exception as exc:  # pragma: no cover
                log.warning("cleanup sweep failed: %s", exc)

    def sweep(self) -> int:
        """Delete expired jobs and their files. Returns how many were removed."""
        now = time.time()
        removed = 0
        with self._lock:
            expired = [j for j in self._jobs.values()
                       if j.expires_at and now > j.expires_at and j.status in (COMPLETED, FAILED)]
            for job in expired:
                self._jobs.pop(job.id, None)
        for job in expired:
            for path in (job.original_path, job.result_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:  # pragma: no cover
                        pass
            removed += 1
        if removed:
            log.info("cleanup removed %s expired job(s)", removed)
        return removed

    def purge_all_files(self) -> None:
        """Wipe the data directory (used on startup so a crash cannot leak old images)."""
        d = self._settings.DATA_DIRECTORY
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
