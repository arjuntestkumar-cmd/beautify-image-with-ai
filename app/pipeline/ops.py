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


def denoise(rgb: np.ndarray, strength: float, noise_score: float, jpeg_score: float,
            protect_edges: bool = True) -> np.ndarray:
    """Chroma-priority denoise + optional deblock.

    `protect_edges` blends the cleaned result back toward the original wherever the luminance
    carries confident structure. Without it, denoising strong enough to clear heavy grain also
    smears hair strands, lashes and outlines into blobs - it runs AFTER super-resolution, so it
    is erasing exactly the detail the model just reconstructed. Flat skin has almost no gradient
    and is cleaned in full; a hair strand keeps most of itself.
    """
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

        if protect_edges:
            lab_l = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
            structure = (_edge_confidence(lab_l) * 0.85)[..., None]
            out = np.clip(rgb.astype(np.float32) * structure
                          + out.astype(np.float32) * (1.0 - structure), 0, 255).astype(np.uint8)

    return deblock_jpeg(out, jpeg_score)


# ---------------------------------------------------------------------------------------
# detail
# ---------------------------------------------------------------------------------------
def _edge_confidence(l: np.ndarray, local: bool = True) -> np.ndarray:
    """Where does this image carry real structure? 0 = flat, 1 = a confident edge.

    Normalisation is LOCAL by default, and that matters more than it sounds. Against a single
    global scale, one very high-contrast region - dark hair against bright skin - sets the bar
    for the entire frame, and everything gentler scores near zero: fabric folds, the edge of a
    hand, the rim of a fingernail. They then receive almost no sharpening, which is exactly why
    everything except the face came back mushy. Comparing each pixel against its own
    neighbourhood gives quiet regions their structure back, while the floor stops a flat, empty
    patch from being scaled up into amplified noise.
    """
    gx = cv2.Scharr(l, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(l, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    hi = float(np.percentile(mag, 92)) + 1e-6
    if local:
        ref = np.maximum(cv2.blur(mag, (31, 31)) * 2.0, 0.12 * hi)
        conf = np.clip(mag / ref, 0.0, 1.0)
    else:
        conf = np.clip(mag / hi, 0.0, 1.0)
    conf = np.power(conf, 0.7)  # emphasise clear structure over faint gradients
    return cv2.GaussianBlur(conf, (0, 0), sigmaX=1.0)


def edge_aware_sharpen(
    rgb: np.ndarray,
    strength: float,
    protect_skin: bool = True,
    region_mask: Optional[np.ndarray] = None,
    halo_limit: float = 16.0,
    skin_guard_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sharpen luminance where structure is confident. `strength` 0..1.

    `skin_guard_mask` scopes the skin protection. Unscoped, that protection applies to anything
    skin-coloured anywhere in the frame - including a hand, an arm or a neck, which are then
    held back by 55% and come out soft while the rest of the photo sharpens around them. Pass
    the face region to protect the face and leave the rest of the body alone.
    """
    if strength <= 0.02:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    blur = cv2.GaussianBlur(l, (0, 0), sigmaX=1.1)
    high = np.clip(l - blur, -halo_limit, halo_limit)

    weight = _edge_confidence(l) * float(min(1.2, strength * 1.4))
    if protect_skin:
        guard = _skin_mask(rgb)
        if skin_guard_mask is not None:
            guard = guard * np.clip(skin_guard_mask, 0.0, 1.0)
        weight *= (1.0 - 0.55 * guard)
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


def face_region_mask(shape: tuple, face_boxes: List) -> np.ndarray:
    """Soft elliptical mask over each detected face, feathered so no edge shows."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.float32)
    for b in face_boxes:
        cx, cy = b.x + b.w // 2, b.y + b.h // 2
        cv2.ellipse(mask, (cx, cy), (max(1, int(b.w * 0.62)), max(1, int(b.h * 0.68))),
                    0, 0, 360, 1.0, -1)
    if mask.max() > 0:
        # Feather relative to the FACE, not the frame: a fixed blur leaves a visible edge to the
        # effect on a big face and washes across a small one.
        span = max(b.w for b in face_boxes)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(6.0, span / 9.0))
    return np.clip(mask, 0.0, 1.0)


def face_clarity(rgb: np.ndarray, face_boxes: List, strength: float) -> np.ndarray:
    """Sharpen structure inside the face: eyes, lashes, brows, lips, stubble.

    Everywhere else the sharpener protects skin, which is right for a photo as a whole but
    leaves the face the softest thing in the frame - the opposite of what someone looking at a
    portrait wants. Here that protection is off. Edge confidence still does the discriminating:
    flat cheeks carry almost no gradient and stay smooth, while the eyes and lash line - which
    carry the most - get the full amount. The halo clamp is tighter than the global pass because
    the eye sockets are exactly where ringing would be visible.
    """
    if strength <= 0.02 or not face_boxes:
        return rgb
    mask = face_region_mask(rgb.shape, face_boxes)
    if mask.max() <= 0:
        return rgb
    return edge_aware_sharpen(rgb, strength=strength, protect_skin=False,
                              region_mask=mask, halo_limit=10.0)


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


def skin_clean(rgb: np.ndarray, face_boxes: List, strength: float) -> np.ndarray:
    """Even out skin without erasing it.

    Edge-preserving smoothing, blended back only under a skin-AND-face mask, so the grain and
    blotches on a cheek go while lashes, lips, nostrils and hair keep their edges. The blend is
    capped at 0.7 so even at full strength almost a third of the original skin survives - the
    difference between "clear skin" and the plastic look.
    """
    if strength <= 0.02 or not face_boxes:
        return rgb
    face = face_region_mask(rgb.shape, face_boxes)
    if face.max() <= 0:
        return rgb
    s = float(max(0.0, min(1.0, strength)))
    mask = (np.clip(face * _skin_mask(rgb), 0.0, 1.0) * (0.70 * s))[..., None]
    smooth = cv2.bilateralFilter(rgb, d=9, sigmaColor=int(22 + 34 * s), sigmaSpace=9)
    out = rgb.astype(np.float32) * (1.0 - mask) + smooth.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def chroma_cleanup(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Remove low-frequency colour blotching, leaving luminance untouched.

    Shadow noise - the kind that only becomes visible once a dark photo is brightened - surfaces
    as slow colour mottling across flat areas: a wall that crawls green and magenta. Ordinary
    denoising barely touches it, because at full resolution those blobs are far larger than any
    sensible filter window. Cleaning the chroma channels at a reduced scale removes the blobs
    themselves, and costs nothing visible: the eye takes almost all of its detail from
    luminance, which is why colour is the first thing every image codec throws away.
    """
    if amount <= 0.02:
        return rgb
    k = float(max(0.0, min(1.0, amount)))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    h, w = a.shape[:2]
    f = max(2, int(round(2 + 4 * k)))
    sw, sh = max(1, w // f), max(1, h // f)

    def clean(ch):
        small = cv2.resize(ch, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.medianBlur(small, 3)
        small = cv2.bilateralFilter(small, d=5, sigmaColor=30, sigmaSpace=5)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    a2 = (a.astype(np.float32) * (1 - k) + clean(a).astype(np.float32) * k)
    b2 = (b.astype(np.float32) * (1 - k) + clean(b).astype(np.float32) * k)
    merged = cv2.merge([l, np.clip(a2, 0, 255).astype(np.uint8), np.clip(b2, 0, 255).astype(np.uint8)])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def auto_exposure(rgb: np.ndarray, target: float = 0.46,
                  max_gain: float = 3.2) -> "tuple[np.ndarray, float]":
    """Rescue a badly under-exposed photo, before anything else looks at it.

    An indoor phone photo at night comes back muddy and grey no matter how well the face is
    restored, because nothing in the pipeline ever corrected the exposure - the models faithfully
    reconstruct a dark image as a dark image. This lifts luminance toward a normal mid-tone with
    a bounded gamma, resets the black point so the result is not veiled, restores the local
    contrast that under-exposure flattens, and puts back some of the chroma a sensor fails to
    record in the dark (lifting luminance alone leaves everything grey).

    Returns the corrected image and how much lift was applied (0.0 = untouched).
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0
    mean = float(l.mean())
    if mean >= target - 0.02 or mean <= 0.0:
        return rgb, 0.0

    # gamma < 1 brightens; the floor caps how much we are willing to invent.
    gamma = float(np.log(max(target, 1e-3)) / np.log(max(mean, 0.02)))
    gamma = min(1.0, max(1.0 / max_gain, gamma))
    lifted = np.power(np.clip(l, 0.0, 1.0), gamma)

    # Black point: under-exposed frames are usually veiled as well as dark.
    lo = float(np.percentile(lifted, 0.5))
    lifted = np.clip((lifted - lo) / max(1e-3, 1.0 - lo), 0.0, 1.0)

    lab[:, :, 0] = lifted * 255.0
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    l2, a2, b2 = cv2.split(cv2.cvtColor(out, cv2.COLOR_RGB2LAB))
    out = cv2.cvtColor(cv2.merge([cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l2),
                                  a2, b2]), cv2.COLOR_LAB2RGB)

    lift = 1.0 - gamma
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.55 * lift), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out, round(float(lift), 3)


def body_clarity(rgb: np.ndarray, exclude_mask: Optional[np.ndarray], strength: float) -> np.ndarray:
    """Give shape back to everything that is NOT a face: hands, clothing, objects, background.

    Only the face goes through the face model; the rest of the frame is whatever the upscaler
    produced, then softened by denoising. Without this it stays a smear next to a finely
    restored face - fingers with no separation, fabric with no folds. Skin protection is off
    here on purpose, because out here "skin" means a hand, and a hand needs its edges.
    """
    if strength <= 0.02:
        return rgb
    mask = None
    if exclude_mask is not None:
        mask = np.clip(1.0 - np.clip(exclude_mask, 0.0, 1.0), 0.0, 1.0)
        if float(mask.max()) <= 0.01:
            return rgb
    return edge_aware_sharpen(rgb, strength=strength, protect_skin=False,
                              region_mask=mask, halo_limit=12.0)


def soft_glow(rgb: np.ndarray, face_boxes: List, amount: float) -> np.ndarray:
    """Diffused highlight bloom on skin - the "lit from within" look of a retouched portrait.

    A screen blend against a blurred copy, held to skin inside the face and kept low, so it
    reads as light rather than as haze. This is beautify-only: it is a flattering lie, and
    Clear mode must not tell it.
    """
    if amount <= 0.02 or not face_boxes:
        return rgb
    face = face_region_mask(rgb.shape, face_boxes)
    if face.max() <= 0:
        return rgb
    a = float(max(0.0, min(1.0, amount)))
    mask = (np.clip(face * _skin_mask(rgb), 0.0, 1.0) * a)[..., None]
    span = max(b.w for b in face_boxes)
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=max(2.0, span / 22.0)).astype(np.float32) / 255.0
    base = rgb.astype(np.float32) / 255.0
    screen = 1.0 - (1.0 - base) * (1.0 - blur)
    out = base * (1.0 - mask) + screen * mask
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


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
