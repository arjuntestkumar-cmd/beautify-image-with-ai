"""Encode the final RGB(A) image, and validate it before it is served."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..errors import OutputPixelLimitExceeded, ProcessingFailed

FORMAT_MIME = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
FORMAT_EXT = {"jpeg": "jpg", "png": "png", "webp": "webp"}


@dataclass
class EncodedOutput:
    path: str
    width: int
    height: int
    bytes: int
    mime_type: str


def check_array(rgb: np.ndarray, max_output_pixels: int, max_output_side: int) -> None:
    """Reject an output that is malformed, oversized, blank or fully clipped."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ProcessingFailed("Output has an unexpected shape.")
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ProcessingFailed("Output has invalid dimensions.")
    if max(h, w) > max_output_side:
        raise OutputPixelLimitExceeded("Output side exceeds the maximum.")
    if h * w > max_output_pixels:
        raise OutputPixelLimitExceeded("Output exceeds the maximum pixel count.")
    if not np.isfinite(rgb).all():
        raise ProcessingFailed("Output contains non-finite values.")
    if float(rgb.std()) < 1.0:
        raise ProcessingFailed("Output appears to be blank.")
    if float(np.mean((rgb <= 2) | (rgb >= 253))) > 0.98:
        raise ProcessingFailed("Output is almost entirely clipped.")


def encode(
    rgb: np.ndarray,
    fmt: str,
    quality: int,
    dest_path: str,
    alpha: Optional[np.ndarray] = None,
    jpeg_background: Tuple[int, int, int] = (255, 255, 255),
) -> EncodedOutput:
    fmt = fmt.lower()
    if fmt not in FORMAT_MIME:
        fmt = "jpeg"

    height, width = rgb.shape[:2]
    has_alpha = alpha is not None and fmt in ("png", "webp")

    if has_alpha:
        image = Image.fromarray(np.dstack([rgb, alpha.astype(np.uint8)]), mode="RGBA")
    else:
        image = Image.fromarray(rgb, mode="RGB")

    if fmt == "jpeg":
        if image.mode == "RGBA":  # JPEG cannot hold alpha — flatten onto white
            bg = Image.new("RGB", image.size, jpeg_background)
            bg.paste(image, mask=image.split()[-1])
            image = bg
        image.save(
            dest_path, format="JPEG", quality=int(quality), optimize=True, progressive=True,
            subsampling=1,  # 4:2:2 — kinder to skin and text than 4:2:0
        )
    elif fmt == "png":
        image.save(dest_path, format="PNG", optimize=True)
    else:  # webp
        image.save(dest_path, format="WEBP", quality=int(quality), method=6, lossless=quality >= 100)

    return EncodedOutput(
        path=dest_path, width=width, height=height,
        bytes=os.path.getsize(dest_path), mime_type=FORMAT_MIME[fmt],
    )


def verify_encoded(path: str, expected_mime: str) -> None:
    """Re-open the encoded file to confirm it decodes and matches the requested type."""
    fmt_to_mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img2:
            actual = fmt_to_mime.get(img2.format or "", "")
    except (UnidentifiedImageError, OSError) as exc:
        raise ProcessingFailed("Encoded output is not decodable.") from exc

    if actual and expected_mime and actual != expected_mime:
        raise ProcessingFailed("Encoded output type does not match the requested format.")
