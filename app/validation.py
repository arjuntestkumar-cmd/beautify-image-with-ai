"""Upload validation: magic bytes, safe decode, EXIF orientation, animation and pixel limits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import CorruptedImage, InputPixelLimitExceeded, UnsupportedImageFormat

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


def decode_and_normalize(path: str, max_input_pixels: int) -> DecodedImage:
    """Decode, honour EXIF orientation, reject animation, enforce the pixel limit."""
    fmt = detect_format(path)

    try:
        img = Image.open(path)
        n_frames = getattr(img, "n_frames", 1)
        if n_frames and n_frames > 1:
            raise UnsupportedImageFormat("Animated images are not supported.")
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
        raise InputPixelLimitExceeded("That image exceeds the maximum pixel count.")

    arr = np.asarray(rgb_img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise CorruptedImage("Unexpected image channel layout.")

    return DecodedImage(
        rgb=arr, width=width, height=height, detected_format=fmt,
        had_alpha=bool(had_alpha), alpha=alpha_channel,
    )
