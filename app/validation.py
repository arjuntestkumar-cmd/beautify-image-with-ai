"""Upload validation: magic bytes, safe decode, EXIF orientation, animation and pixel limits."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import CorruptedImage, InputPixelLimitExceeded, UnsupportedImageFormat
from .logging_utils import get_logger

log = get_logger("validation")

# Pillow guard against decompression bombs.
Image.MAX_IMAGE_PIXELS = 200_000_000

MIME_BY_FORMAT = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


@dataclass
class DecodedImage:
    rgb: np.ndarray            # HxWx3 uint8, RGB
    width: int
    height: int
    detected_format: str       # jpeg | png | webp
    had_alpha: bool
    alpha: Optional[np.ndarray] = None  # HxW uint8, if the source had transparency
    source_width: int = 0      # before any fit-to-budget downscale
    source_height: int = 0
    downscaled: bool = False


def detect_format(path: str) -> str:
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    raise UnsupportedImageFormat("That file is not a JPEG, PNG or WebP image.")


def decode_and_normalize(path: str, max_input_pixels: int,
                         downscale_oversize: bool = True) -> DecodedImage:
    """Decode, honour EXIF orientation, reject animation, fit the photo to the pixel budget.

    A photo over `max_input_pixels` is now fitted to that budget rather than refused: needing
    more room than the budget is a reason to work differently, not a reason to hand the file
    back. Everything the pipeline does from here is chunked, so what actually gets processed is
    a very large image that the box can hold rather than an error message.

    For JPEG the shrink happens inside libjpeg via `draft()`, which decodes at 1/2, 1/4 or 1/8
    scale directly. The oversized array is never allocated at all, so an enormous JPEG costs
    roughly what a normal one costs.
    """
    fmt = detect_format(path)
    source_size = (0, 0)
    downscaled = False

    try:
        img = Image.open(path)
        n_frames = getattr(img, "n_frames", 1)
        if n_frames and n_frames > 1:
            raise UnsupportedImageFormat("Animated images are not supported.")

        source_size = img.size
        if downscale_oversize and (img.size[0] * img.size[1]) > max_input_pixels > 0:
            ratio = (max_input_pixels / float(img.size[0] * img.size[1])) ** 0.5
            target = (max(1, int(img.size[0] * ratio)), max(1, int(img.size[1] * ratio)))
            if fmt == "jpeg":
                img.draft("RGB", target)          # decodes smaller; never allocates the original
            if img.size[0] * img.size[1] > max_input_pixels:
                img = img.resize(target, Image.LANCZOS)
            downscaled = True

        img = ImageOps.exif_transpose(img)
        had_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
        alpha_channel = None
        if had_alpha:
            alpha_channel = np.asarray(img.convert("RGBA").split()[-1], dtype=np.uint8)
        rgb_img = img.convert("RGB")
    except UnsupportedImageFormat:
        raise
    except Image.DecompressionBombError as exc:
        raise InputPixelLimitExceeded("That image is too large to decode safely.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise CorruptedImage("That image could not be decoded.") from exc

    width, height = rgb_img.size
    if width <= 0 or height <= 0:
        raise CorruptedImage("That image has malformed dimensions.")
    if width * height > max_input_pixels:
        # Only reachable with the fit-to-budget behaviour switched off.
        raise InputPixelLimitExceeded("That image exceeds the maximum pixel count.")

    arr = np.asarray(rgb_img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise CorruptedImage("Unexpected image channel layout.")

    return DecodedImage(
        rgb=arr, width=width, height=height, detected_format=fmt,
        had_alpha=bool(had_alpha), alpha=alpha_channel,
        source_width=source_size[0] or width, source_height=source_size[1] or height,
        downscaled=downscaled,
    )


def compress_heavy_image(path: str, output_path: str, size_threshold_bytes: int = 10_000_000) -> tuple[str, bool]:
    """Compress heavy image files to reduce processing load.

    If the file size exceeds the threshold, intelligently compress it:
    - JPEG/WebP: reduce quality to 80% to maintain visual quality while reducing file size
    - PNG: re-encode with optimization
    - All: keep original dimensions but reduce file size

    Args:
        path: Path to the uploaded image file
        output_path: Path where compressed image should be saved
        size_threshold_bytes: Files larger than this get compressed (default: 10 MB)

    Returns:
        Tuple of (final_path, was_compressed)
        - final_path: Path to the compressed file if compression happened, else original path
        - was_compressed: Boolean indicating if compression was performed
    """
    file_size = os.path.getsize(path)

    # Skip compression if file is small enough
    if file_size <= size_threshold_bytes:
        return path, False

    try:
        fmt = detect_format(path)
        img = Image.open(path)

        # Preserve EXIF orientation during compression
        img = ImageOps.exif_transpose(img)

        if fmt in ("jpeg", "webp"):
            # For lossy formats, reduce quality to ~80% for lighter processing
            # This balances file size reduction with visual quality
            img.save(
                output_path,
                format=fmt.upper(),
                quality=80,
                optimize=True
            )
            compressed_size = os.path.getsize(output_path)
            reduction = (1 - compressed_size / file_size) * 100
            log.info(
                "compressed %s: %s → %s KB (%.1f%% reduction)",
                fmt.upper(), file_size // 1024, compressed_size // 1024, reduction
            )
            return output_path, True
        elif fmt == "png":
            # For PNG, optimize without losing any quality (lossless)
            img.save(output_path, format="PNG", optimize=True)
            compressed_size = os.path.getsize(output_path)
            reduction = (1 - compressed_size / file_size) * 100
            log.info(
                "optimized PNG: %s → %s KB (%.1f%% reduction)",
                file_size // 1024, compressed_size // 1024, reduction
            )
            return output_path, True

    except Exception as exc:
        log.warning("compression failed, proceeding with original: %s", exc)
        return path, False

    return path, False
