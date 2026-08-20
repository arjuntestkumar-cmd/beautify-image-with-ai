"""Encode the final RGB(A) image, and validate it before it is served."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

from ..errors import OutputPixelLimitExceeded, ProcessingFailed
from ..logging_utils import get_logger

log = get_logger("encode")

# Pillow hands libjpeg ONE fixed output buffer, `ImageFile.MAXBLOCK`, and its default is 64 KB.
# Optimised and progressive JPEG both have to build a whole scan before they can flush, so a scan
# that does not fit makes libjpeg raise "Suspension not allowed here", which Pillow surfaces as
# OSError: broken data stream when writing image file.
#
# It depends on how COMPRESSIBLE the photograph is, not merely how big: measured, a smooth
# 2000x2500 portrait encodes fine at the default while incompressible content of exactly the same
# dimensions fails. That is why this presented as "anything over about 100 KB fails" and why it
# was intermittent - and because encoding is the last step, every one of those jobs ran the entire
# pipeline, reached 98%, and only then threw the error away.
#
# Three bytes per pixel is the uncompressed size, and so the only bound a JPEG scan is guaranteed
# to fit inside. The cap stops a 40 MP frame asking for 120 MB on a 2 GB box; past it, baseline
# encoding takes over, which streams and needs no buffer at all.
_JPEG_BUFFER_CAP = 64 * 1024 * 1024
_maxblock_lock = threading.Lock()

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
        save_jpeg(image, dest_path, quality)
    elif fmt == "png":
        image.save(dest_path, format="PNG", optimize=True)
    else:  # webp
        image.save(dest_path, format="WEBP", quality=int(quality), method=6, lossless=quality >= 100)

    return EncodedOutput(
        path=dest_path, width=width, height=height,
        bytes=os.path.getsize(dest_path), mime_type=FORMAT_MIME[fmt],
    )


def save_jpeg(image: Image.Image, dest_path: str, quality: int) -> None:
    """Progressive, optimised JPEG — degrading to baseline rather than failing.

    See the note on `_JPEG_BUFFER_CAP`. Baseline is the safety net because it streams its output
    instead of buffering a scan, so it cannot hit this limit at all: measured good on a 48 MP
    frame of pure noise, the worst case there is. The file comes out a few percent larger and is
    not progressive, which beats handing back an error by a very long way.
    """
    common = dict(format="JPEG", quality=int(quality), subsampling=1)  # 4:2:2 — kind to skin
    want = min(3 * image.width * image.height + (1 << 16), _JPEG_BUFFER_CAP)
    # The buffer size is a Pillow-wide global, so the swap is serialised. With one worker this is
    # never contended; it is what keeps raising WORKER_CONCURRENCY from corrupting an encode.
    with _maxblock_lock:
        previous = ImageFile.MAXBLOCK
        ImageFile.MAXBLOCK = max(previous, want)
        try:
            image.save(dest_path, optimize=True, progressive=True, **common)
            return
        except OSError as exc:
            log.warning("progressive JPEG needed more than %s MB of buffer (%s) — "
                        "falling back to baseline", want // (1024 * 1024), exc)
        finally:
            ImageFile.MAXBLOCK = previous
    image.save(dest_path, **common)


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
