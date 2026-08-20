"""Configuration for the Beautify image enhancer.

Everything has a working default, so the service starts with no .env at all. Values can be
overridden with environment variables or a .env file next to this project's root.
"""
from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains app/, models/, web/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_id() -> str:
    """A short fingerprint of the application code this process is actually running.

    Computed once, at import, over the bytes of every module under `app/`. That timing is the
    whole point: it describes the code that was loaded into THIS process, not the code sitting on
    disk now. Editing the source of a running server changes the files and changes nothing else -
    Python holds what it imported at startup - and that failure looks exactly like success from
    the outside, right up until you wonder why a fix did not land.

    Note it before a deploy and after: same value means the process was never replaced.
    """
    h = hashlib.sha256()
    app_dir = os.path.dirname(os.path.abspath(__file__))
    for dirpath, dirnames, filenames in os.walk(app_dir):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(f for f in filenames if f.endswith(".py")):
            try:
                with open(os.path.join(dirpath, name), "rb") as fh:
                    h.update(fh.read())
            except OSError:  # pragma: no cover - unreadable file should never fail startup
                continue
    return h.hexdigest()[:8]


BUILD_ID = _build_id()


def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_ENV: str = "development"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = "*"

    # --- models -------------------------------------------------------------------------
    # Only two weight files are needed. The general (v3) Real-ESRGAN model is small (5 MB) and
    # gentler than x4plus; its "wdn" sibling is blended into it per request (DNI) so the denoise
    # strength adapts to the actual noise in the photo.
    MODEL_DIRECTORY: str = _p("models")
    REALESRGAN_GENERAL_MODEL_PATH: str = _p("models", "realesr-general-x4v3.pth")
    REALESRGAN_WDN_MODEL_PATH: str = _p("models", "realesr-general-wdn-x4v3.pth")
    GFPGAN_MODEL_PATH: str = _p("models", "GFPGANv1.4.pth")

    ENABLE_CUDA: bool = True          # honoured only if torch reports CUDA available
    ENABLE_HALF_PRECISION: bool = True  # GPU only; ignored on CPU
    GPU_DEVICE: str = "cuda:0"
    DEFAULT_TILE_SIZE: int = 400
    TILE_PADDING: int = 16
    GPU_OOM_RETRIES: int = 3

    # Set true to run without model weights (decode -> Lanczos resize -> gentle finish).
    # Useful only for wiring up the UI; it is NOT AI enhancement.
    MOCK_MODE: bool = False

    # --- limits -------------------------------------------------------------------------
    # These are higher than they were because the pipeline no longer holds a whole large photo
    # in memory at once (see pipeline/chunked.py). A big file is now a question of how long it
    # takes, not of whether the box survives it, so there is no longer a reason to refuse one.
    MAX_UPLOAD_BYTES: int = 67_108_864      # 64 MB
    MAX_INPUT_PIXELS: int = 80_000_000
    MAX_OUTPUT_PIXELS: int = 160_000_000
    MAX_OUTPUT_SIDE: int = 16384
    # An image past MAX_INPUT_PIXELS is fitted to that budget and processed, not rejected.
    # Only a file that cannot be decoded at all is turned away now.
    DOWNSCALE_OVERSIZE_INPUT: bool = True

    # Images whose longest side is at or below this get a genuine 2x upscale; larger images are
    # beautified at their native size (upscaling them is slow and rarely what the user wants).
    AUTO_UPSCALE_MAX_SIDE: int = 1600

    # --- storage ------------------------------------------------------------------------
    # Local, temporary. There is no database and no cloud storage: the original and the result
    # live in this folder until they expire, then they are deleted.
    DATA_DIRECTORY: str = _p(".data")
    RESULT_RETENTION_MINUTES: int = 30
    CLEANUP_INTERVAL_SECONDS: int = 120

    # --- processing ---------------------------------------------------------------------
    WORKER_CONCURRENCY: int = 1        # inference is CPU/GPU bound; one at a time is correct
    MAX_QUEUED_JOBS: int = 64          # waiting is fine; being turned away is not
    # A deadline, not a size limit. It scales with the photo (below), so a genuinely large job
    # is given the time it needs and only a job that has actually hung is stopped.
    JOB_TIMEOUT_SECONDS: int = 900
    JOB_TIMEOUT_SECONDS_PER_MEGAPIXEL: int = 90
    JOB_TIMEOUT_SECONDS_MAX: int = 5400

    # --- chunked processing ---------------------------------------------------------------
    # Past this many pixels the heavy stages run over overlapping tiles instead of the whole
    # array, so peak memory follows the tile size and not the photo. See pipeline/chunked.py.
    CHUNK_THRESHOLD_PIXELS: int = 4_000_000
    CHUNK_TILE_SIZE: int = 768          # tile edge for the classical stages
    CHUNK_OVERLAP: int = 48             # context carried around each tile, then discarded
    # The model tile is quoted on its INPUT side and is much smaller, because RealESRGANer runs
    # its 4x network before scaling back down: 384 px in is a 1536 px tensor, not a 384 px one.
    MODEL_CHUNK_TILE_SIZE: int = 384
    MODEL_CHUNK_OVERLAP: int = 24

    # --- tuning (bounded; see pipeline/beautify.py) ---------------------------------------
    SHARPNESS_SAFETY_LIMIT: float = 0.06  # max fraction of halo/over-shot pixels tolerated
    HAIR_REFINEMENT_ENABLED: bool = True
    OUTPUT_QUALITY: int = 92           # jpeg/webp encoder quality
    # The un-styled enhanced image is kept next to the result so a look can be changed or
    # removed later without re-running any of the models. Slightly higher quality than the
    # delivered file, because a look is applied on top of it.
    KEEP_UNFILTERED_BASE: bool = True
    BASE_QUALITY: int = 96

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() in ("production", "prod")

    @property
    def cors_origins(self) -> list[str]:
        raw = (self.CORS_ORIGINS or "*").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - startup failure path
        print(f"[config] invalid configuration: {exc}", file=sys.stderr)
        raise
