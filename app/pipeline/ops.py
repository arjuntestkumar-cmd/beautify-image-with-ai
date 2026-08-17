"""Classical image operations used around the models.

Ported verbatim (same math, same constants) from the multi-mode service:
  * denoise / deblock  — chroma-priority noise cleanup and 8x8 JPEG de-blocking
  * edge_aware_sharpen — luminance sharpening weighted by edge confidence, halo-clamped,
                         skin-protected. Global sharpening is deliberately avoided.
  * refine_hair        — the same sharpening, concentrated on the head/hair band
  * premium_finish     — the photographic "beautify" finish: neutral white balance, highlight
                         recovery, shadow lift, midtone S-curve, skin-protected vibrance and a
                         whisper of local contrast
  * check_result       — over-sharpening / colour-shift safeguard for one bounded fallback
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------
# shared masks
# ---------------------------------------------------------------------------------------
def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    mask = ((cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)).astype(np.float32)
    return cv2.GaussianBlur(mask, (7, 7), 0)


def _skin_mask_wide(rgb: np.ndarray) -> np.ndarray:
    """Slightly wider skin range used when re-injecting face texture."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    m = ((cr > 133) & (cr < 180) & (cb > 77) & (cb < 130)).astype(np.float32)
    return cv2.GaussianBlur(m, (9, 9), 0)


# ---------------------------------------------------------------------------------------
# denoise / deblock
# ---------------------------------------------------------------------------------------
def deblock_jpeg(rgb: np.ndarray, jpeg_score: float) -> np.ndarray:
    """Light edge-preserving smoothing that targets 8x8 blocking without blurring real edges."""
    if jpeg_score < 0.35:
        return rgb
    amount = min(1.0, (jpeg_score - 0.3) / 0.6)
    smoothed = cv2.bilateralFilter(rgb, d=5, sigmaColor=int(20 + 30 * amount), sigmaSpace=5)
    return cv2.addWeighted(rgb, 1 - 0.6 * amount, smoothed, 0.6 * amount, 0)


def denoise(rgb: np.ndarray, strength: float, noise_score: float, jpeg_score: float) -> np.ndarray:
    """Chroma-priority denoise + optional deblock. Preserves luminance detail."""
    s = float(max(0.0, min(1.0, strength)))
    # Blend the requested strength with the measured noise, so a clean image is barely touched.
    amount = min(1.0, 0.35 * s + 0.65 * s * (0.4 + noise_score))
    out = rgb

    if amount > 0.05:
        ycrcb = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        # Colour noise carries no useful detail — smooth Cr/Cb harder.
        chroma_d = int(round(3 + 6 * amount))
        cr = cv2.bilateralFilter(cr, d=chroma_d, sigmaColor=25, sigmaSpace=chroma_d)
        cb = cv2.bilateralFilter(cb, d=chroma_d, sigmaColor=25, sigmaSpace=chroma_d)
        # Luminance: gentle and capped so fine texture survives.
        luma_h = int(round(2 + 5 * min(0.6, amount)))
        y = cv2.fastNlMeansDenoising(y, None, h=luma_h, templateWindowSize=7, searchWindowSize=15)
        out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
        # Blend back so even strength=1 keeps ~15% of the original micro-texture.
        keep = 0.15 + 0.25 * (1 - amount)
        out = cv2.addWeighted(rgb, keep, out, 1 - keep, 0)

    return deblock_jpeg(out, jpeg_score)


# ---------------------------------------------------------------------------------------
# detail
# ---------------------------------------------------------------------------------------
def _edge_confidence(l: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(l, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(l, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    hi = np.percentile(mag, 92) + 1e-6
    conf = np.clip(mag / hi, 0.0, 1.0)
    conf = np.power(conf, 0.7)  # emphasise clear structure over faint gradients
    return cv2.GaussianBlur(conf, (0, 0), sigmaX=1.0)


def edge_aware_sharpen(
    rgb: np.ndarray,
    strength: float,
    protect_skin: bool = True,
    region_mask: Optional[np.ndarray] = None,
    halo_limit: float = 16.0,
) -> np.ndarray:
    """Sharpen luminance where structure is confident. `strength` 0..1."""
    if strength <= 0.02:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    blur = cv2.GaussianBlur(l, (0, 0), sigmaX=1.1)
    high = np.clip(l - blur, -halo_limit, halo_limit)

    weight = _edge_confidence(l) * float(min(1.2, strength * 1.4))
    if protect_skin:
        weight *= (1.0 - 0.55 * _skin_mask(rgb))
    if region_mask is not None:
        weight = weight * np.clip(region_mask, 0.0, 1.0)

    lab[:, :, 0] = np.clip(l + weight * high, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def hair_region_mask(shape: tuple, face_boxes: List) -> np.ndarray:
    """Approximate head/hair band around detected faces (expanded up + sideways)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.float32)
    for b in face_boxes:
        ex, ey = int(b.w * 0.55), int(b.h * 0.9)
        x0 = max(0, b.x - ex)
        y0 = max(0, b.y - ey)          # extend well above the box for hair
        x1 = min(w, b.x + b.w + ex)
        y1 = min(h, b.y + b.h + int(b.h * 0.2))
        mask[y0:y1, x0:x1] = 1.0
    if mask.max() > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3.0, w / 60.0))
    return mask


def refine_hair(rgb: np.ndarray, face_boxes: List, strength: float, out_scale: float = 1.0) -> np.ndarray:
    """Edge-aware detail concentrated on the head/hair region (strand separation, not fake strands)."""
    if strength <= 0.02 or not face_boxes:
        return rgb
    scaled = [type(b)(int(b.x * out_scale), int(b.y * out_scale), int(b.w * out_scale), int(b.h * out_scale))
              for b in face_boxes]
    mask = hair_region_mask(rgb.shape, scaled)
    if mask.max() <= 0:
        return rgb
    return edge_aware_sharpen(rgb, strength=strength, protect_skin=True, region_mask=mask, halo_limit=12.0)


def overshoot_ratio(before: np.ndarray, after: np.ndarray, threshold: float = 60.0) -> float:
    """Fraction of luminance pixels whose change exceeds `threshold` (halo/ringing indicator)."""
    lb = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    la = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if lb.shape != la.shape:
        la = cv2.resize(la, (lb.shape[1], lb.shape[0]))
    return float((np.abs(la - lb) > threshold).mean())


# ---------------------------------------------------------------------------------------
# face texture re-injection (anti-waxy)
# ---------------------------------------------------------------------------------------
def reinject_texture(blended: np.ndarray, original: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Add the ORIGINAL image's real skin micro-texture back in skin areas — kills the waxy look.

    GFPGAN smooths skin; the source usually still has real pores/texture. We add the source's
    bounded high-frequency detail only where skin is detected, so faces look natural, not painted.
    """
    b = blended.astype(np.float32)
    o = original.astype(np.float32)
    orig_high = np.clip(o - cv2.GaussianBlur(o, (0, 0), sigmaX=1.4), -22, 22)
    skin = _skin_mask_wide(original)[..., None]
    return np.clip(b + amount * skin * orig_high, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------
# photographic finish — this is what makes the result read as "beautified"
# ---------------------------------------------------------------------------------------
def _neutral_white_balance(rgb: np.ndarray, amount: float) -> np.ndarray:
    f = rgb.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    for c in range(3):
        if means[c] > 1e-3:
            target = f[:, :, c] * (gray / means[c])
            f[:, :, c] = f[:, :, c] * (1 - amount) + target * amount
    return np.clip(f, 0, 255).astype(np.uint8)


def premium_finish(rgb: np.ndarray, tone_depth: float, is_grayscale: bool = False) -> np.ndarray:
    """Subtle DSLR-style tone finish. `tone_depth` 0..1 scales intensity. Every term is bounded."""
    t = float(max(0.0, min(1.0, tone_depth)))
    out = rgb
    if not is_grayscale:
        out = _neutral_white_balance(out, amount=0.25 * t)

    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0

    # Highlight recovery: soft rolloff near the top.
    hi = np.clip((l - 0.82) / 0.18, 0, 1)
    l = l - hi * hi * (0.06 * t)
    # Shadow lift: gentle gamma in the low range.
    l = np.where(l < 0.5, np.power(np.clip(l * 2, 0, 1), 1.0 - 0.12 * t) / 2.0, l)
    # Midtone S-curve for controlled contrast/depth.
    l = l + (0.10 * t) * np.sin((l - 0.5) * np.pi)
    lab[:, :, 0] = np.clip(l, 0, 1) * 255.0
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Controlled vibrance (skin-protected, so faces never go orange).
    if not is_grayscale and t > 0.05:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        s = hsv[:, :, 1] / 255.0
        gain = 1.0 + (0.18 * t) * (1.0 - s)
        skin = _skin_mask(out)
        gain = gain * (1.0 - 0.5 * skin) + 1.0 * (0.5 * skin)
        hsv[:, :, 1] = np.clip(s * gain, 0, 1) * 255.0
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # A whisper of local contrast (lens-like microcontrast).
    lab2 = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    l2, a2, b2 = cv2.split(lab2)
    clahe = cv2.createCLAHE(clipLimit=1.0 + 0.6 * t, tileGridSize=(8, 8))
    l2 = clahe.apply(l2)
    return cv2.cvtColor(cv2.merge([l2, a2, b2]), cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------------------
# safeguards
# ---------------------------------------------------------------------------------------
@dataclass
class SafetyReport:
    ok: bool
    overshoot: float
    color_shift: float
    warnings: List[str] = field(default_factory=list)


def _color_shift(reference: np.ndarray, output: np.ndarray) -> float:
    ref = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = output
    if out.shape[:2] != reference.shape[:2]:
        out = cv2.resize(out, (reference.shape[1], reference.shape[0]))
    out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    da = np.abs(ref[:, :, 1] - out[:, :, 1]).mean()
    db = np.abs(ref[:, :, 2] - out[:, :, 2]).mean()
    return float((da + db) / 2.0 / 128.0)


def check_result(reference_rgb: np.ndarray, output_rgb: np.ndarray, sharpness_limit: float) -> SafetyReport:
    """Compare the final output to the pre-detail reference."""
    warnings: List[str] = []
    over = overshoot_ratio(reference_rgb, output_rgb)
    shift = _color_shift(reference_rgb, output_rgb)

    ok = True
    if over > sharpness_limit:
        ok = False
        warnings.append("oversharpened")
    if shift > 0.10:
        ok = False
        warnings.append("color_shift")
    return SafetyReport(ok=ok, overshoot=round(over, 4), color_shift=round(shift, 4), warnings=warnings)
