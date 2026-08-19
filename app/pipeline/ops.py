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
from typing import Callable, List, Optional

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
# mask-exact compositing
# ---------------------------------------------------------------------------------------
# OpenCV sizes a Gaussian kernel at cvRound(sigma * (depth == CV_8U ? 3 : 4) * 2 + 1) | 1. Every
# mask in this file is float32, so its feather reaches FOUR sigma, not three. Anything that crops
# for one of these masks has to pad by that much, or the mask is still non-zero where the crop
# stops - and a non-zero mask on a crop edge is a straight line between processed and untouched
# pixels, which is the rectangle this pipeline was reported to draw around restored faces.
GAUSS_SUPPORT = 4.0


def mask_gate(mask: np.ndarray, floor: float = 0.10) -> np.ndarray:
    """1 wherever `mask` carries at least `floor` of its strength, ramping to EXACTLY 0 at 0.

    A stage's own mask cannot be reused directly as a composite weight: the stage has already
    scaled its effect by that mask, so multiplying by it a second time would attenuate the very
    thing it was asked to do. The gate reaches 1 while the mask is still weak, so everything the
    mask actually asked for passes through unchanged; the only pixels it holds back are the ones
    where the effect was already below a tenth of full strength. It reaches a true zero exactly
    where the mask does, which is the entire point.
    """
    return np.clip(np.asarray(mask, np.float32) * (1.0 / max(1e-6, float(floor))), 0.0, 1.0)


def composite_masked(base: np.ndarray, out: np.ndarray, mask: Optional[np.ndarray],
                     floor: float = 0.10) -> np.ndarray:
    """Put `out` back over `base` under `mask`, so a masked op is BIT-IDENTICAL to its input
    wherever the mask is zero.

    Every stage here round-trips its piece through LAB or YCrCb, and those conversions are lossy:
    RGB -> LAB -> RGB moves a pixel by up to four levels even where the mask says to leave it
    alone. Measured on a smooth frame, outside the stage's own mask: face_clarity max 3 mean
    0.889, refine_hair max 3 mean 0.926, body_clarity max 1, skin_clean max 1, soft_glow max 1,
    filters.apply_face max 4 mean 0.774. Inside a cropped region that dither is present and
    outside it is not, and the boundary between them is a straight line - which on smooth content,
    a blurred background most of all, is exactly the square edge users see.

    At g = 0 this returns `base` unchanged; at g = 1 it returns `out` unchanged, because a uint8
    promoted to float32 and multiplied by 1.0 is still an exact integer. So it is a genuine
    identity at both ends, not an approximation of one.

    Rounds rather than truncates on the way back to uint8. The tail of a Gaussian feather is full
    of values like 1e-8, and truncation turns the resulting 136.9999 into 136 - a one-level error
    along the entire fringe of every mask in this pipeline.

    Pointwise, so a tiled call still matches the whole-image call exactly: the chunking
    guarantees in chunked.py are untouched by this.
    """
    if mask is None:
        return out
    g = mask_gate(mask, floor)[..., None]
    return np.clip(base.astype(np.float32) * (1.0 - g) + out.astype(np.float32) * g,
                   0, 255).round().astype(np.uint8)


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
            protect_edges: bool = True, edge_hi: Optional[float] = None) -> np.ndarray:
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
            structure = (_edge_confidence(lab_l, hi=edge_hi) * 0.85)[..., None]
            out = np.clip(rgb.astype(np.float32) * structure
                          + out.astype(np.float32) * (1.0 - structure), 0, 255).astype(np.uint8)

    return deblock_jpeg(out, jpeg_score)


# ---------------------------------------------------------------------------------------
# detail
# ---------------------------------------------------------------------------------------
def _edge_confidence(l: np.ndarray, local: bool = True, hi: Optional[float] = None) -> np.ndarray:
    """Where does this image carry real structure? 0 = flat, 1 = a confident edge.

    Normalisation is LOCAL by default, and that matters more than it sounds. Against a single
    global scale, one very high-contrast region - dark hair against bright skin - sets the bar
    for the entire frame, and everything gentler scores near zero: fabric folds, the edge of a
    hand, the rim of a fingernail. They then receive almost no sharpening, which is exactly why
    everything except the face came back mushy. Comparing each pixel against its own
    neighbourhood gives quiet regions their structure back, while the floor stops a flat, empty
    patch from being scaled up into amplified noise.

    `hi` is that floor's reference, and it is the one number here measured over the WHOLE frame.
    When this runs on a tile of a large photo it must be passed in (chunked.sample_edge_hi),
    because a tile of flat sky and a tile of hair would otherwise pick wildly different floors
    and sharpen by visibly different amounts on either side of the join.
    """
    gx = cv2.Scharr(l, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(l, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    hi = float(np.percentile(mag, 92)) + 1e-6 if hi is None else float(hi)
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
    edge_hi: Optional[float] = None,
    soft_radius: float = 0.0,
    soft_gain: float = 0.0,
) -> np.ndarray:
    """Sharpen luminance where structure is confident. `strength` 0..1.

    `skin_guard_mask` scopes the skin protection. Unscoped, that protection applies to anything
    skin-coloured anywhere in the frame - including a hand, an arm or a neck, which are then
    held back by 55% and come out soft while the rest of the photo sharpens around them. Pass
    the face region to protect the face and leave the rest of the body alone.

    `soft_radius` / `soft_gain` add a SECOND, wider scale, and they are what makes this pass
    able to do anything at all for a soft-focus photograph. The 1.1 px unsharp below measures
    the detail an image carries between one and two pixels. A frame defocused at sigma ~4 has
    almost none there: the residual it leaves on a 60-level edge is about 0.5 of a level, so no
    value of `strength` can make it visible - the radius is wrong, not the gain. At sigma 4 the
    same edge leaves ~4.5 levels, and a gain near 2 is what inverts a Gaussian of that width at
    its half-power point. Both default to off, so every existing caller is unchanged.

    What keeps the wide scale safe is that its clamp is ABSOLUTE. A genuinely sharp edge leaves
    a residual of ~30% of its contrast at this radius, so proportional gain there would be a
    halo machine; clamping the wide term to `halo_limit * 0.5` LEVELS means a hard edge can
    never receive more than that times `soft_gain` of overshoot, however contrasty it is.
    """
    if strength <= 0.02:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0]
    blur = cv2.GaussianBlur(l, (0, 0), sigmaX=1.1)
    high = np.clip(l - blur, -halo_limit, halo_limit)

    conf = _edge_confidence(l, hi=edge_hi)
    weight = conf * float(min(1.2, strength * 1.4))
    guard_mul = None
    if protect_skin:
        guard = _skin_mask(rgb)
        if skin_guard_mask is not None:
            guard = guard * np.clip(skin_guard_mask, 0.0, 1.0)
        guard_mul = (1.0 - 0.55 * guard)
        weight *= guard_mul
    region = None if region_mask is None else np.clip(region_mask, 0.0, 1.0)
    if region is not None:
        weight = weight * region

    if soft_radius > 1.0 and soft_gain > 0.02:
        # Same confidence map, same guards, same region - only the scale differs.
        wide_weight = conf * float(min(2.0, soft_gain))
        if guard_mul is not None:
            wide_weight *= guard_mul
        if region is not None:
            wide_weight = wide_weight * region
        wide_limit = halo_limit * 0.5
        wide_high = np.clip(l - cv2.GaussianBlur(l, (0, 0), sigmaX=float(soft_radius)),
                            -wide_limit, wide_limit)
        lab[:, :, 0] = np.clip(l + weight * high + wide_weight * wide_high, 0, 255)
    else:
        lab[:, :, 0] = np.clip(l + weight * high, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    # Where the mask is zero this op has already decided to do nothing to L - but the LAB round
    # trip does something anyway, by up to four levels, over the whole piece it was handed. On a
    # cropped region that dither stops dead at the crop edge. Compositing the result back under
    # the mask makes "nothing" mean nothing. Unmasked callers (the final detail pass, _mock_restore)
    # pass region_mask=None and are returned untouched.
    return composite_masked(rgb, out, region_mask)


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
        # Feather relative to the HEAD, not the frame. A frame-relative sigma made this band
        # depend on how big the photo happens to be - 100 px of blur on a 6000 px image, which
        # smears the band far past the hair - and it made the mask impossible to reproduce from
        # a crop, which is what the chunked path needs it to be.
        span = max(b.w for b in face_boxes)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3.0, span / 9.0))
    return mask


def refine_hair(rgb: np.ndarray, face_boxes: List, strength: float, out_scale: float = 1.0,
                mask: Optional[np.ndarray] = None, edge_hi: Optional[float] = None) -> np.ndarray:
    """Edge-aware detail concentrated on the head/hair region (strand separation, not fake strands)."""
    if strength <= 0.02 or not face_boxes:
        return rgb
    if mask is None:
        scaled = [type(b)(int(b.x * out_scale), int(b.y * out_scale),
                          int(b.w * out_scale), int(b.h * out_scale)) for b in face_boxes]
        mask = hair_region_mask(rgb.shape, scaled)
    if mask.max() <= 0:
        return rgb
    return edge_aware_sharpen(rgb, strength=strength, protect_skin=True, region_mask=mask,
                              halo_limit=12.0, edge_hi=edge_hi)


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


# What a stage cropping for one of these masks has to know, attached to the mask function itself
# so the crop can ask instead of guessing and the two can never drift apart:
#   .reach  how far the mask's SHAPE extends past the face box, as a fraction of (box w, box h)
#   .sigma  the feather, as (floor in pixels, fraction of the box WIDTH)
# face: an ellipse of half-axes 0.62w / 0.68h about the box centre reaches 0.12w past the box
#       sides and 0.18h past its top and bottom.
# hair: the band extends 0.55w sideways, 0.9h above the box and 0.2h below it.
# The two reach numbers differ per axis while both sigmas are driven by the WIDTH, which is why
# chunked.union_rect must pad per axis rather than with one scalar. See chunked.union_rect.
face_region_mask.reach = (0.12, 0.18)
face_region_mask.sigma = (6.0, 1.0 / 9.0)
hair_region_mask.reach = (0.55, 0.90)
hair_region_mask.sigma = (3.0, 1.0 / 9.0)


def face_clarity(rgb: np.ndarray, face_boxes: List, strength: float,
                 mask: Optional[np.ndarray] = None, edge_hi: Optional[float] = None) -> np.ndarray:
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
    if mask is None:
        mask = face_region_mask(rgb.shape, face_boxes)
    if mask.max() <= 0:
        return rgb
    return edge_aware_sharpen(rgb, strength=strength, protect_skin=False,
                              region_mask=mask, halo_limit=10.0, edge_hi=edge_hi)


def _stride_for(shape, budget: int = 4_000_000) -> int:
    """Sub-sampling step that keeps a whole-image comparison inside `budget` pixels.

    The safeguards below are statistics over the frame, so they can be read off a regular
    sample of it rather than off every pixel. Taking a STRIDE and not a resize is the point: a
    downscale averages a halo away with its neighbours, which is exactly the thing being
    measured, while a stride keeps the pixels intact and just looks at fewer of them.
    """
    h, w = shape[:2]
    return max(1, int(np.ceil(np.sqrt((h * w) / float(max(1, budget))))))


def acutance_ratio(face_rgb: np.ndarray, frame_rgb: np.ndarray, face_mask: np.ndarray,
                   sigma: float = 1.6) -> float:
    """How much sharper is a restored face than the photograph it is being pasted into?

    Mean high-pass magnitude under the face mask, over the same measured in the band outside it.
    Both readings come from the SAME crop at the same scale, so exposure, subject and resolution
    cancel and only sharpness is left. Returns 1.0 when there is not enough of either region to
    measure, so the caller's cap can never fire on a degenerate crop.
    """
    def _hp(img: np.ndarray) -> np.ndarray:
        l = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        return np.abs(l - cv2.GaussianBlur(l, (0, 0), sigmaX=float(sigma)))

    m = np.clip(face_mask, 0.0, 1.0)
    inner = (m > 0.6).astype(np.float32)
    outer = (m < 0.05).astype(np.float32)
    if float(inner.sum()) < 64.0 or float(outer.sum()) < 64.0:
        return 1.0
    face_a = float((_hp(face_rgb) * inner).sum() / float(inner.sum()))
    frame_a = float((_hp(frame_rgb) * outer).sum() / float(outer.sum()))
    return float(face_a / max(frame_a, 0.5))


def overshoot_ratio(before: np.ndarray, after: np.ndarray, threshold: float = 60.0) -> float:
    """Fraction of luminance pixels whose change exceeds `threshold` (halo/ringing indicator)."""
    k = _stride_for(before.shape)
    lb = cv2.cvtColor(before[::k, ::k], cv2.COLOR_RGB2GRAY).astype(np.float32)
    la = cv2.cvtColor(after[::k, ::k], cv2.COLOR_RGB2GRAY).astype(np.float32)
    if lb.shape != la.shape:                      # only when the stages changed the size
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


def skin_clean(rgb: np.ndarray, face_boxes: List, strength: float,
               mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Even out skin without erasing it.

    Edge-preserving smoothing, blended back only under a skin-AND-face mask, so the grain and
    blotches on a cheek go while lashes, lips, nostrils and hair keep their edges. The blend is
    capped at 0.7 so even at full strength almost a third of the original skin survives - the
    difference between "clear skin" and the plastic look.
    """
    if strength <= 0.02 or not face_boxes:
        return rgb
    face = face_region_mask(rgb.shape, face_boxes) if mask is None else mask
    if face.max() <= 0:
        return rgb
    s = float(max(0.0, min(1.0, strength)))
    blend = (np.clip(face * _skin_mask(rgb), 0.0, 1.0) * (0.70 * s))[..., None]
    smooth = cv2.bilateralFilter(rgb, d=9, sigmaColor=int(22 + 34 * s), sigmaSpace=9)
    out = rgb.astype(np.float32) * (1.0 - blend) + smooth.astype(np.float32) * blend
    # This one already composites, and where `blend` is exactly zero it is already exact. The
    # measured max-1 leak is the truncation: the fringe of the Gaussian feather carries values
    # around 1e-8, and rgb * (1 - 1e-8) truncates one level down. One level, along the whole
    # fringe, ending at the crop edge. composite_masked rounds, which removes it. The bilateral
    # above is still evaluated over the whole piece, but only for cost - its result is discarded
    # wherever blend is zero, so it cannot leak.
    return composite_masked(rgb, np.clip(out, 0, 255).astype(np.uint8), face)


def chroma_cleanup(rgb: np.ndarray, amount: float, factor: Optional[int] = None) -> np.ndarray:
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
    # `factor` is passed in on the chunked path so every tile blurs the chroma at the same
    # scale. Derived per tile it would depend on the tile's size, and the last, narrower column
    # of tiles would be cleaned harder than the rest.
    f = max(2, int(round(2 + 4 * k))) if factor is None else max(2, int(factor))
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


def plan_exposure(rgb: np.ndarray, target: float = 0.46,
                  max_gain: float = 3.2) -> "Optional[tuple[float, float]]":
    """Measure the exposure correction this photo needs: `(gamma, black_point)`, or None.

    Split out of `auto_exposure` so a large photo can measure once and correct per tile. Both
    numbers survive being taken from a downscaled proxy: a mean is preserved by area
    downscaling, and the black point is a percentile of `L ** gamma`, which - raising to a power
    being monotonic - is exactly `percentile(L) ** gamma`. So this stays a measurement of the
    whole frame no matter how small a copy of it is handed in.
    """
    l = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32) / 255.0
    mean = float(l.mean())
    if mean >= target - 0.02 or mean <= 0.0:
        return None
    # gamma < 1 brightens; the floor caps how much we are willing to invent.
    gamma = float(np.log(max(target, 1e-3)) / np.log(max(mean, 0.02)))
    gamma = min(1.0, max(1.0 / max_gain, gamma))
    # Black point: under-exposed frames are usually veiled as well as dark.
    lo = float(np.percentile(np.power(np.clip(l, 0.0, 1.0), gamma), 0.5))
    return gamma, lo


def auto_exposure(rgb: np.ndarray, target: float = 0.46, max_gain: float = 3.2,
                  plan: "Optional[tuple[float, float]]" = None,
                  local_contrast: Optional[Callable] = None) -> "tuple[np.ndarray, float]":
    """Rescue a badly under-exposed photo, before anything else looks at it.

    An indoor phone photo at night comes back muddy and grey no matter how well the face is
    restored, because nothing in the pipeline ever corrected the exposure - the models faithfully
    reconstruct a dark image as a dark image. This lifts luminance toward a normal mid-tone with
    a bounded gamma, resets the black point so the result is not veiled, restores the local
    contrast that under-exposure flattens, and puts back some of the chroma a sensor fails to
    record in the dark (lifting luminance alone leaves everything grey).

    Pass `plan` (and `local_contrast` in place of the CLAHE) to apply one whole-frame correction to
    a tile. Returns the corrected image and how much lift was applied (0.0 = untouched).
    """
    if plan is None:
        plan = plan_exposure(rgb, target, max_gain)
    if plan is None:
        return rgb, 0.0
    gamma, lo = plan

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lifted = np.power(np.clip(lab[:, :, 0] / 255.0, 0.0, 1.0), gamma)
    lifted = np.clip((lifted - lo) / max(1e-3, 1.0 - lo), 0.0, 1.0)

    lab[:, :, 0] = lifted * 255.0
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    l2, a2, b2 = cv2.split(cv2.cvtColor(out, cv2.COLOR_RGB2LAB))
    l2 = (cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l2)
          if local_contrast is None else local_contrast(l2))
    out = cv2.cvtColor(cv2.merge([l2, a2, b2]), cv2.COLOR_LAB2RGB)

    lift = 1.0 - gamma
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.55 * lift), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out, round(float(lift), 3)


def body_clarity(rgb: np.ndarray, exclude_mask: Optional[np.ndarray], strength: float,
                 edge_hi: Optional[float] = None, soft_radius: float = 0.0,
                 soft_gain: float = 0.0) -> np.ndarray:
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
                              region_mask=mask, halo_limit=12.0, edge_hi=edge_hi,
                              soft_radius=soft_radius, soft_gain=soft_gain)


def soft_glow(rgb: np.ndarray, face_boxes: List, amount: float,
              mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Diffused highlight bloom on skin - the "lit from within" look of a retouched portrait.

    A screen blend against a blurred copy, held to skin inside the face and kept low, so it
    reads as light rather than as haze. This is beautify-only: it is a flattering lie, and
    Clear mode must not tell it.
    """
    if amount <= 0.02 or not face_boxes:
        return rgb
    face = face_region_mask(rgb.shape, face_boxes) if mask is None else mask
    if face.max() <= 0:
        return rgb
    a = float(max(0.0, min(1.0, amount)))
    blend = (np.clip(face * _skin_mask(rgb), 0.0, 1.0) * a)[..., None]
    span = max(b.w for b in face_boxes)
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=max(2.0, span / 22.0)).astype(np.float32) / 255.0
    base = rgb.astype(np.float32) / 255.0
    screen = 1.0 - (1.0 - base) * (1.0 - blur)
    out = base * (1.0 - blend) + screen * blend
    # /255 then *255 in float32 does not round-trip: at blend = 0 a pixel of 137 comes back as
    # 136.99998 and truncates to 136. One level, over the whole piece, stopping at the crop edge.
    return composite_masked(rgb, np.clip(out * 255.0, 0, 255).astype(np.uint8), face)


# ---------------------------------------------------------------------------------------
# photographic finish — this is what makes the result read as "beautified"
# ---------------------------------------------------------------------------------------
def _neutral_white_balance(rgb: np.ndarray, amount: float,
                           means: Optional[np.ndarray] = None) -> np.ndarray:
    f = rgb.astype(np.float32)
    # The means MUST come from the whole frame. A tile of sky and a tile of skin have different
    # colour casts, so per-tile means would white-balance each one to a different place and put
    # a hard colour step down the join. On the chunked path they are measured once and passed.
    means = f.reshape(-1, 3).mean(axis=0) if means is None else np.asarray(means, np.float32)
    gray = float(means.mean())
    for c in range(3):
        if means[c] > 1e-3:
            target = f[:, :, c] * (gray / means[c])
            f[:, :, c] = f[:, :, c] * (1 - amount) + target * amount
    return np.clip(f, 0, 255).astype(np.uint8)


def tone_curve_lut(tone_depth: float) -> np.ndarray:
    """The finish's luminance curve as a 256-entry table: highlights, shadows, midtone S.

    Every term is pointwise on an 8-bit L channel, so a table is not an approximation of the
    arithmetic below - it is the same arithmetic, evaluated 256 times instead of forty million.
    Pulling it out also lets the local-contrast pass measure its histograms on the L this curve
    produces, which is the L it will actually be applied to.
    """
    t = float(max(0.0, min(1.0, tone_depth)))
    l = np.arange(256, dtype=np.float32) / 255.0
    hi = np.clip((l - 0.82) / 0.18, 0, 1)          # highlight recovery: soft rolloff near the top
    l = l - hi * hi * (0.06 * t)
    l = np.where(l < 0.5, np.power(np.clip(l * 2, 0, 1), 1.0 - 0.12 * t) / 2.0, l)  # shadow lift
    l = l + (0.10 * t) * np.sin((l - 0.5) * np.pi)  # midtone S-curve for depth
    return np.clip(l, 0, 1).astype(np.float32) * 255.0


def premium_finish(rgb: np.ndarray, tone_depth: float, is_grayscale: bool = False,
                   wb_means: Optional[np.ndarray] = None,
                   local_contrast: Optional[Callable] = None) -> np.ndarray:
    """Subtle DSLR-style tone finish. `tone_depth` 0..1 scales intensity. Every term is bounded.

    `wb_means` and `local_contrast` are the two whole-frame measurements this stage makes,
    supplied from outside when it runs per tile: the white-balance means, and a CLAHE whose
    histograms were read from the whole frame (chunked.ClaheField). Everything else here is a
    pointwise curve, which is identical on a tile by construction and needs nothing.
    """
    t = float(max(0.0, min(1.0, tone_depth)))
    out = rgb
    if not is_grayscale:
        out = _neutral_white_balance(out, amount=0.25 * t, means=wb_means)

    # Highlight recovery, shadow lift and the midtone S-curve, in one table (tone_curve_lut).
    l, a, b = cv2.split(cv2.cvtColor(out, cv2.COLOR_RGB2LAB))
    l = cv2.LUT(l, tone_curve_lut(t)).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

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
    # Per tile, CLAHE would build a fresh histogram for each one and print its grid across the
    # photo as brightness steps. `local_contrast` is the same algorithm with its histograms read
    # once from the whole frame, so no two tiles can disagree. See chunked.ClaheField.
    l2 = (cv2.createCLAHE(clipLimit=1.0 + 0.6 * t, tileGridSize=(8, 8)).apply(l2)
          if local_contrast is None else local_contrast(l2))
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
    k = _stride_for(reference.shape)
    ref = cv2.cvtColor(reference[::k, ::k], cv2.COLOR_RGB2LAB).astype(np.float32)
    out = output[::k, ::k]
    if out.shape[:2] != ref.shape[:2]:
        out = cv2.resize(out, (ref.shape[1], ref.shape[0]))
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
