"""Automatic image-quality analysis — the thing that makes one mode adaptive.

Ported from the multi-mode service, trimmed to exactly the measurements the Beautify pipeline
consumes. Everything here is classical CV (OpenCV/NumPy): cheap, no model weights. Face
detection uses OpenCV's bundled Haar cascade, so analysis needs no extra download (GFPGAN uses
its own, better detector during restoration).

Dropped from the original because only the removed modes used them: colourfulness, sepia, fade
and haze scores (old-photo colour restoration), and the old-photo classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

# Quality categories drive how hard the pipeline works.
SEVERELY_DEGRADED = "SEVERELY_DEGRADED"
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Analysis:
    width: int
    height: int
    blur_score: float           # 0 sharp .. 1 very blurry
    noise_score: float          # 0 clean .. 1 very noisy
    jpeg_artifact_score: float  # 0 none .. 1 heavy blocking
    low_light_score: float      # 0 bright .. 1 very dark
    edge_density: float         # 0 flat .. 1 busy
    dynamic_range: float        # 0 low .. 1 full
    brightness: float = 0.5
    contrast: float = 0.0
    highlight_clip: float = 0.0
    shadow_clip: float = 0.0
    is_grayscale: bool = False
    is_portrait: bool = False   # a face is the dominant subject
    has_alpha: bool = False
    quality_category: str = MEDIUM
    faces: List[FaceBox] = field(default_factory=list)
    small_faces: bool = False
    already_high_quality: bool = False
    screenshot_like: bool = False

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def megapixels(self) -> float:
        return round(self.width * self.height / 1_000_000.0, 3)

    @property
    def largest_face(self) -> Optional[FaceBox]:
        return max(self.faces, key=lambda f: f.w * f.h) if self.faces else None


_FACE_CASCADE: "cv2.CascadeClassifier | None" = None


def _face_cascade() -> "cv2.CascadeClassifier":
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(path)
    return _FACE_CASCADE


def _detect_faces(gray: np.ndarray) -> List[FaceBox]:
    cascade = _face_cascade()
    if cascade.empty():
        return []
    # Detect on a downscaled copy for speed on large images, then map boxes back.
    h, w = gray.shape[:2]
    scale = 1.0
    max_side = 1024
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray
    faces = cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return [FaceBox(int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)) for (x, y, fw, fh) in faces]


def _blur_score(gray: np.ndarray) -> float:
    # Variance of Laplacian; higher = sharper. ~1000+ is very sharp, <50 is very blurry.
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(1.0, lap_var / 1000.0)
    return float(round(1.0 - sharpness, 4))


def _noise_score(gray: np.ndarray) -> float:
    # Noise estimated as the residual after a median blur.
    denoised = cv2.medianBlur(gray, 3)
    residual = cv2.absdiff(gray, denoised)
    return float(round(min(1.0, float(residual.mean()) / 32.0), 4))


def _jpeg_artifact_score(gray: np.ndarray) -> float:
    # Blockiness: gradients across 8x8 block boundaries vs gradients within blocks.
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return 0.0
    g = gray.astype(np.float32)
    col_diff = np.abs(np.diff(g, axis=1))
    row_diff = np.abs(np.diff(g, axis=0))
    boundary = (col_diff[:, 7::8].mean() + row_diff[7::8, :].mean()) / 2.0
    overall = (col_diff.mean() + row_diff.mean()) / 2.0 + 1e-6
    ratio = boundary / overall
    # ratio ~1 = no blocking; >1.3 = visible blocking.
    return float(round(min(1.0, max(0.0, (ratio - 1.0) / 0.6)), 4))


def _low_light_score(gray: np.ndarray) -> float:
    mean = float(gray.mean()) / 255.0
    return float(round(min(1.0, max(0.0, (0.35 - mean) / 0.35)), 4))


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float(round(float((edges > 0).mean()), 4))


def _dynamic_range(gray: np.ndarray) -> float:
    lo, hi = np.percentile(gray, (2, 98))
    return float(round((hi - lo) / 255.0, 4))


def _screenshot_like(gray: np.ndarray, edge_density: float) -> bool:
    # Screenshots/documents: many flat regions + lots of sharp text edges + limited tones.
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    peak_fraction = float(hist.max() / (hist.sum() + 1e-6))
    return peak_fraction > 0.25 and edge_density > 0.06


def _is_grayscale(rgb: np.ndarray) -> bool:
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA).astype(np.int16)
    diff = np.abs(small[..., 0] - small[..., 1]) + np.abs(small[..., 1] - small[..., 2])
    return float(diff.mean()) < 6.0


def classify_quality(blur: float, noise: float, jpeg: float, drange: float, longest_side: int) -> str:
    """Degradation metrics + resolution -> category. Not resolution-only: a large image can
    still be blurry or heavily compressed."""
    degradation = 0.45 * blur + 0.25 * noise + 0.30 * jpeg
    tiny = longest_side < 320
    small = longest_side < 640
    if degradation > 0.62 or (tiny and degradation > 0.4):
        return SEVERELY_DEGRADED
    if degradation > 0.4 or (small and degradation > 0.28):
        return LOW
    if blur < 0.16 and noise < 0.16 and jpeg < 0.16 and drange > 0.55 and longest_side >= 900:
        return HIGH
    return MEDIUM


def analyse(rgb: np.ndarray, has_alpha: bool = False) -> Analysis:
    """Run the full analysis on an RGB uint8 image."""
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    blur = _blur_score(gray)
    noise = _noise_score(gray)
    jpeg = _jpeg_artifact_score(gray)
    low_light = _low_light_score(gray)
    edges = _edge_density(gray)
    drange = _dynamic_range(gray)
    faces = _detect_faces(gray)
    screenshot = _screenshot_like(gray, edges)
    category = classify_quality(blur, noise, jpeg, drange, max(width, height))

    # A face is "small" (weaker restoration) if its largest side is under ~64 px in the source.
    small_faces = bool(faces) and all(max(f.w, f.h) < 64 for f in faces)

    largest_frac = 0.0
    if faces:
        largest_frac = max(f.w * f.h for f in faces) / float(max(1, width * height))
    is_portrait = bool(faces) and largest_frac > 0.03 and not screenshot

    return Analysis(
        width=width,
        height=height,
        blur_score=blur,
        noise_score=noise,
        jpeg_artifact_score=jpeg,
        low_light_score=low_light,
        edge_density=edges,
        dynamic_range=drange,
        brightness=float(round(float(gray.mean()) / 255.0, 4)),
        contrast=float(round(float(gray.std()) / 128.0, 4)),
        highlight_clip=float(round(float((gray >= 250).mean()), 4)),
        shadow_clip=float(round(float((gray <= 5).mean()), 4)),
        is_grayscale=_is_grayscale(rgb),
        is_portrait=is_portrait,
        has_alpha=has_alpha,
        quality_category=category,
        faces=faces,
        small_faces=small_faces,
        # The original also required low fade/haze here; those scores existed only for the
        # old-photo colour restoration that this build drops, so the category decides.
        already_high_quality=(category == HIGH),
        screenshot_like=screenshot,
    )
