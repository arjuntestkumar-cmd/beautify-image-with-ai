"""Configuration for the Beautify image enhancer.

Everything has a working default, so the service starts with no .env at all. Values can be
overridden with environment variables or a .env file next to this project's root.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains app/, models/, web/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    MAX_UPLOAD_BYTES: int = 20_971_520      # 20 MB
    MAX_INPUT_PIXELS: int = 40_000_000
    MAX_OUTPUT_PIXELS: int = 80_000_000
    MAX_OUTPUT_SIDE: int = 8192

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
    MAX_QUEUED_JOBS: int = 32
    JOB_TIMEOUT_SECONDS: int = 900

    # --- tuning (bounded; see pipeline/beautify.py) ---------------------------------------
    SHARPNESS_SAFETY_LIMIT: float = 0.06  # max fraction of halo/over-shot pixels tolerated
    HAIR_REFINEMENT_ENABLED: bool = True
    OUTPUT_QUALITY: int = 92           # jpeg/webp encoder quality

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
