"""Upload validation: magic bytes, safe decode, EXIF orientation, animation and pixel limits."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import CorruptedImage, InputPixelLimitExceeded, UnsupportedImageFormat
# Shared so there is ONE JPEG writer: the encoder buffer that fixes has bitten this path too.
from .pipeline.encode import save_jpeg
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
        # A multi-FRAME still is not an animation, and conflating the two rejected a large share
        # of ordinary iPhone photographs.
        #
        # Apple's dual-camera and portrait modes write MPO files - several complete JPEG images
        # in one container, used for depth and for the alternate exposure. Pillow reports
        # `n_frames > 1` for those exactly as it does for a GIF, so testing that number on its own
        # answered "Animated images are not supported" to a perfectly ordinary photo.
        #
        # Of the three formats accepted here, JPEG cannot animate at all: extra frames in a JPEG
        # container are alternate stills, and the first is the photograph the user took. PNG
        # (APNG) and WebP genuinely can, and are still refused - returning one frame of an
        # animation silently would be worse than saying so.
        n_frames = getattr(img, "n_frames", 1) or 1
        if n_frames > 1:
            if fmt in ("png", "webp"):
                raise UnsupportedImageFormat("Animated images are not supported.")
            img.seek(0)          # MPO and friends: frame 0 is the photograph

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


def compress_heavy_image(path: str, output_path: str,
                         size_threshold_bytes: int = 3 * 1024 * 1024,
                         quality: int = 85) -> tuple[str, bool]:
    """Re-encode an upload larger than the threshold — and keep the result only if it got smaller.

    What this buys and what it does not: the pipeline decodes to a pixel array, so a 3 MB JPEG
    and a 12 MB JPEG of the same dimensions cost exactly the same to process. Re-encoding saves
    disk and the bytes held per queued job; it is not a speed control. The dial that changes
    processing cost is the number of PIXELS — `AUTO_UPSCALE_MAX_SIDE` and `MAX_INPUT_PIXELS`.

    Keeping the file only when it shrank is not a nicety. A photographic PNG re-encoded with
    `optimize=True` measured 13234 KB -> 13491 KB: lossless re-compression of data that is
    already packed efficiently simply grows it, and the previous version returned that larger
    file and reported success.

    Returns (path_to_use, was_compressed). On any failure the original is returned untouched —
    a compression step must never be able to cost someone their upload.
    """
    try:
        file_size = os.path.getsize(path)
    except OSError:
        return path, False
    if size_threshold_bytes <= 0 or file_size <= size_threshold_bytes:
        return path, False

    try:
        fmt = detect_format(path)
        with Image.open(path) as opened:
            # Bake the EXIF rotation in here, because the tag does not survive the re-encode.
            # `decode_and_normalize` transposes again later; that is a no-op once the tag is
            # gone, which is what keeps this from rotating a photo twice. Verified: a portrait
            # carrying Orientation=6 decodes to the same dimensions with and without this step.
            img = ImageOps.exif_transpose(opened)
            if fmt == "jpeg":
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")        # CMYK and P JPEGs exist
                save_jpeg(img, output_path, quality)
            elif fmt == "webp":
                img.save(output_path, format="WEBP", quality=quality, method=4)
            elif fmt == "png":
                img.save(output_path, format="PNG", optimize=True)
            else:
                return path, False
    except Exception as exc:  # noqa: BLE001 - never fail an upload over an optimisation
        log.warning("compression skipped (%s); using the original", exc.__class__.__name__)
        return path, False

    try:
        new_size = os.path.getsize(output_path)
    except OSError:
        return path, False

    if new_size >= file_size:
        log.info("re-encoding %s did not help (%s KB -> %s KB); keeping the original",
                 fmt.upper(), file_size // 1024, new_size // 1024)
        return path, False

    log.info("compressed %s: %s KB -> %s KB (%.0f%% smaller)", fmt.upper(),
             file_size // 1024, new_size // 1024, (1.0 - new_size / file_size) * 100.0)
    return output_path, True
