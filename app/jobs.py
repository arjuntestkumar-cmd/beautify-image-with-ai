"""In-memory job store + a single-slot worker pool.

There is no database, no Redis and no cloud storage. A job lives in a dict; the original, the
result and the un-styled base live in DATA_DIRECTORY until they expire, then a sweeper deletes
them. That is the whole "infrastructure" — it is enough for one machine, and it is what makes
this build light.

Inference is serialised (WORKER_CONCURRENCY=1) because it saturates the CPU/GPU: running two at
once makes both slower and can exhaust memory. That single slot IS the load control — when ten
people upload at the same moment, nine of them wait in an ordered queue and are told where they
are in it, rather than all ten starting and all ten running out of memory together. Waiting is
a worse experience than not waiting; it is a much better one than a server that has fallen over.

What the queue guarantees:
  * ordered and lossless — jobs run in the order they were accepted, and none is dropped
  * isolated — every job owns its own files, keyed by an id nothing else can guess or reuse
  * submitted once — a second submit for the same job is refused, not run twice
  * independently fallible — a job that fails, times out or runs out of memory is recorded as
    failed and the next one starts immediately; nothing it did leaks into its neighbours
  * bounded in time, not in size — the deadline scales with the photo, so a large file gets
    the minutes it needs and only a genuinely hung job is stopped
"""
from __future__ import annotations

import gc
import itertools
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

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
    seq: int = 0                          # accepted-at order; what the queue position is read from
    status: str = QUEUED
    stage: str = "queued"
    progress: int = 5
    message: str = "Waiting to start"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    expires_at: float = 0.0
    deadline: Optional[float] = None      # wall-clock stop point, scaled to the photo
    original_path: Optional[str] = None
    original_name: str = "image"
    mode: str = "beautify"
    look: Optional[str] = None
    original_mime: str = "image/jpeg"
    original_bytes: int = 0
    megapixels: float = 0.0
    result_path: Optional[str] = None
    result: Optional[Any] = None          # BeautifyResult
    result_version: int = 1               # bumped on restyle so a cached result is not shown
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    queue_position: int = 0
    _submitted: bool = False

    def timed_out(self) -> bool:
        return self.deadline is not None and time.time() > self.deadline

    def to_public(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "jobId": self.id,
            "mode": self.mode,
            "look": (self.result.look if self.result is not None else self.look),
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "queuePosition": self.queue_position,
            "createdAt": _iso(self.created_at),
            "expiresAt": _iso(self.expires_at) if self.expires_at else None,
            "statusUrl": f"/api/jobs/{self.id}",
            # The version makes a re-styled result a different URL, so the browser fetches the
            # new one instead of showing the cached image under the old address.
            "resultUrl": f"/api/jobs/{self.id}/result?v={self.result_version}",
            "originalUrl": f"/api/jobs/{self.id}/original",
            "filterUrl": f"/api/jobs/{self.id}/filter",
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
                "chunked": getattr(r, "chunked", False),
                # Whether a different look can still be applied without re-running the models.
                "canRestyle": bool(getattr(r, "base_path", None)),
            }
        if self.status == FAILED:
            out["error"] = {"code": self.error_code, "message": self.error_message}
        return out


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class JobStore:
    def __init__(self, settings: Settings, reclaim: Optional[Callable[[], None]] = None) -> None:
        self._settings = settings
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, settings.WORKER_CONCURRENCY), thread_name_prefix="beautify"
        )
        self._seq = itertools.count(1)
        self._reclaim = reclaim
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
            job = Job(id=uuid.uuid4().hex[:20], seq=next(self._seq))
            job.expires_at = job.created_at + self._settings.RESULT_RETENTION_MINUTES * 60
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status == QUEUED:
                # How many accepted jobs are still ahead of this one. Someone watching a
                # progress bar that has not moved deserves to know it is a queue and not a hang.
                job.queue_position = sum(
                    1 for j in self._jobs.values()
                    if j.status == PROCESSING or (j.status == QUEUED and j.seq < job.seq))
        if job is None:
            raise JobNotFound("That job does not exist, or it has expired.")
        return job

    def get_for_file(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.expires_at and time.time() > job.expires_at:
            raise ResultExpired("That result has expired and been deleted.")
        return job

    def submit(self, job: Job, fn, *args) -> None:
        """Queue the work. `fn(job, *args)` runs on the worker thread.

        The pool has one slot, so this is the admission point for the whole server: submitting
        is instant and cheap no matter how many people do it at once, and the actual heavy work
        starts only when the slot frees. Submitting the same job twice is refused rather than
        run twice - a retried POST must not double the load or race two writers onto one file.
        """
        with self._lock:
            if job._submitted:
                log.warning("job %s was already submitted; ignoring the duplicate", job.id)
                return
            job._submitted = True
        self._pool.submit(self._run, job, fn, *args)

    def _deadline_for(self, job: Job) -> float:
        """A stop point that scales with the photo.

        A fixed timeout is the wrong shape for this: it is either too short for a legitimately
        large photo - failing exactly the jobs chunking was added to make possible - or so long
        that a genuinely hung one holds the only worker for an hour. Tying it to megapixels
        gives a big file the minutes it actually needs and still catches a job that has stopped
        making progress.
        """
        s = self._settings
        budget = max(s.JOB_TIMEOUT_SECONDS,
                     int(job.megapixels * s.JOB_TIMEOUT_SECONDS_PER_MEGAPIXEL))
        return time.time() + min(budget, s.JOB_TIMEOUT_SECONDS_MAX)

    def _run(self, job: Job, fn, *args) -> None:
        job.started_at = time.time()
        job.deadline = self._deadline_for(job)
        job.status = PROCESSING
        job.stage = "starting"
        job.progress = 8
        job.message = "Starting"
        job.queue_position = 0
        try:
            fn(job, *args)
            job.status = COMPLETED
            job.stage = "completed"
            job.progress = 100
            job.message = "Done"
        except Exception as exc:  # noqa: BLE001 - the boundary: every failure becomes a job error
            # Deliberately the widest possible catch, MemoryError included. A worker thread that
            # dies with an exception takes the only processing slot with it and every job behind
            # this one waits forever; one bad photo must never be able to do that.
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
            job.deadline = None
            # Hand the memory back before the next job takes the slot. Without this the peak of
            # job N is still resident while job N+1 builds its own, and two jobs that each fit
            # comfortably can fail together.
            self._reclaim_memory(job)

    def _reclaim_memory(self, job: Job) -> None:
        try:
            gc.collect()
            if self._reclaim is not None:
                self._reclaim()
        except Exception as exc:  # pragma: no cover - never let cleanup fail a finished job
            log.warning("post-job cleanup for %s failed: %s", job.id, exc.__class__.__name__)

    def progress(self, job: Job, stage: str, percent: int, message: str) -> None:
        job.stage = stage
        job.progress = max(job.progress, int(percent))
        job.message = message

    def reopen(self, job: Job) -> bool:
        """Put a finished job back in the queue for a second, cheap pass (a look change).

        Returns False if the job is still running, so a second request cannot start a race with
        the first over the same output file. Re-styling goes through the same single-slot pool
        as everything else: it is far lighter than a full run, but it is still image work, and
        letting it jump the queue would be a way for load to arrive unmetered.
        """
        with self._lock:
            if job.status in (QUEUED, PROCESSING):
                return False
            job.status = QUEUED
            job.stage = "queued"
            job.progress = 0
            job.message = "Waiting to start"
            job.error_code = None
            job.error_message = None
            job.seq = next(self._seq)
            job._submitted = False
        return True

    def files_of(self, job: Job) -> tuple:
        base = getattr(job.result, "base_path", None) if job.result is not None else None
        return (job.original_path, job.result_path, base)

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
            for path in self.files_of(job):
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
