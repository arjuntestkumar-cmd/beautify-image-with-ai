"""Premium beautification looks - the layer that runs after the photo has been restored.

The pipeline before this fixes what is *wrong* with a photo: grain, blur, compression, a face
the sensor never resolved. That is repair, and repair is not the same as flattery. These looks
are the flattery: the grade, the skin, the light. They are deliberately the last stage and a
separable one, so a look can be swapped or taken off later without any of the expensive work
being redone.

Design rules, in the order they matter:

  * IDENTITY IS NOT NEGOTIABLE. Nothing here moves a feature, narrows a face or lightens skin
    toward some other skin. Smoothing is frequency-separated - the low frequencies of the skin
    are evened out and the micro-texture is put straight back on top - because the plastic look
    is what happens when a filter takes the pores with the blotches. Skin evening pulls chroma
    a fraction of the way toward the photo's OWN median skin tone, so a blotchy cheek matches
    the rest of that person's face rather than a preset's idea of a face.
  * EVERY TERM IS BOUNDED, AND THE BOUNDS ARE STRUCTURAL. The strongest look still keeps more
    than 40% of the original skin. Saturation on skin is clamped to [0.90, 1.18] of what it
    was, whatever the parameters ask for. `skin_tone` writes Cr and Cb and never Y, so no look
    can lighten a complexion even in principle. `skin_bright` is capped in absolute levels and
    weighted by remaining headroom. And every colour term in the grade is weighted by a bell
    that is identically zero at full black and full white, so no look can stain a white shirt
    or tint a black one - which is the tell that separates a grade from a filter.
  * TILE-SAFE BY CONSTRUCTION. The whole-frame stages are a per-channel LUT (grade, split tone,
    tint - all pointwise), a hue-selective saturation whose only spatial term is a 7x7 skin
    blur, and a local contrast pass whose histograms are read once from the whole frame. The
    face stages run over the face region and use fixed curves, never a per-piece measurement.
    Neither can produce a different answer on either side of a chunk boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import ops


@dataclass(frozen=True)
class Look:
    """One look. Fields are 0..1 unless marked -1..1, and each scales a bounded effect.

    The grade fields split into three groups on purpose. `exposure`/`contrast`/`lift` shape
    LUMINANCE and fix both endpoints, so no look can crush a black or blow a white. `warmth`/
    `tint`/`split` shape COLOUR, as offsets weighted by a bell that is exactly zero at 0 and at
    255 - which is the reason a grade here can be strong without staining a white shirt or
    tinting a black. `vibrance`/`sat_skin`/`sat_cool` shape SATURATION by hue band, so the sky
    and the complexion can move in opposite directions instead of together.
    """
    id: str
    name: str
    description: str
    warmth: float = 0.0        # -1 cool .. +1 warm; midtone-weighted R/B offset
    tint: float = 0.0          # -1 green .. +1 magenta; the other white-balance axis
    split: float = 0.0         # -1..1 split tone: +1 warm highlights + cool shadows
    exposure: float = 0.0      # -1..1 midtone gamma; endpoints fixed
    contrast: float = 0.0      # bounded S-curve; endpoints fixed
    lift: float = 0.0          # matte blacks - the "premium print" feel
    vibrance: float = 0.0      # saturation, weighted toward the duller colours
    sat_skin: float = 0.0      # -1..1 extra saturation on the red/orange/yellow band
    sat_cool: float = 0.0      # -1..1 extra saturation on the green/cyan/blue band
    clarity: float = 0.0       # local contrast
    skin_smooth: float = 0.0   # frequency-separated skin evening
    skin_even: float = 0.0     # pull blotches toward this face's own median tone
    skin_tone: float = 0.0     # -1 cooler .. +1 warmer complexion, CHROMA ONLY
    skin_bright: float = 0.0   # fill light on shadowed skin, not a whitening
    glow: float = 0.0          # diffused highlight bloom on skin
    lips: float = 0.0          # definition and colour in the lips only
    eyes: float = 0.0          # clarity across the eye band only
    eye_light: float = 0.0     # luminance shaping of the eye, skin excluded
    vignette: float = 0.0      # a barely-there corner falloff
    mono: float = 0.0          # 0 colour .. 1 black and white, with a warm-filter response

    # Exactly the fields `apply_global` reads. The browser renders its instant preview from
    # these and from nothing else, so a look cannot drift between the two: add a frame-wide term
    # here and the preview gets it, add one that is NOT here and the preview will visibly not
    # have it, which is the failure you want rather than the one you do not.
    GRADE_FIELDS = ("warmth", "tint", "split", "exposure", "contrast", "lift", "vibrance",
                    "sat_skin", "sat_cool", "clarity", "vignette", "mono")

    def public(self) -> dict:
        """What /api/filters publishes.

        `grade` is the frame-wide half of the look - the half that is a pointwise curve plus two
        cheap spatial terms, and therefore the half a browser can reproduce exactly on a canvas
        the moment someone clicks a chip. The face half (skin, lips, eyes, glow) needs the face
        boxes and the model output, so it stays on the server and arrives when the re-render
        does. Sending the numbers rather than a rendered swatch is what makes the preview instant
        AND correct: there is one definition of "Golden Aura" and both sides read it.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "grade": {f: round(float(getattr(self, f)), 4) for f in self.GRADE_FIELDS},
        }


# The default. Chosen to be the one that flatters the widest range of photos without ever
# announcing itself: warm by a whisper, clean skin that still has pores, light in the eyes.
DEFAULT_LOOK = "radiance"
NO_LOOK = "none"

LOOKS: Tuple[Look, ...] = (
    Look(NO_LOOK, "Original",
         "No look at all - the enhanced photo exactly as the pipeline produced it."),
    Look("radiance", "Natural Radiance",
         "The default. Warm light, clean even skin that still has its pores, bright eyes.",
         warmth=.32, split=.30, exposure=.12, contrast=.38, lift=.10,
         vibrance=.40, sat_skin=.10, sat_cool=.18, clarity=.26,
         skin_smooth=.36, skin_even=.42, skin_tone=.12, skin_bright=.24,
         glow=.20, lips=.34, eyes=.34, eye_light=.42, vignette=.05),
    Look("porcelain", "Soft Porcelain",
         "Airy and cool, over lifted blacks. The softest skin in the set.",
         warmth=.10, tint=.18, split=-.26, exposure=.16, contrast=.10, lift=.42,
         vibrance=.28, sat_skin=.12, sat_cool=-.24, clarity=.05,
         skin_smooth=.72, skin_even=.58, skin_tone=.0, skin_bright=.24,
         glow=.52, lips=.26, eyes=.22, eye_light=.30, vignette=.0),
    Look("aura", "Golden Aura",
         "Sun through a window: golden highlights and a real bloom in the light.",
         warmth=.60, split=.66, exposure=.06, contrast=.32, lift=.30,
         vibrance=.38, sat_skin=.26, sat_cool=-.30, clarity=.10,
         skin_smooth=.46, skin_even=.42, skin_tone=.26, skin_bright=.18,
         glow=.70, lips=.32, eyes=.24, eye_light=.32, vignette=.10),
    Look("amber", "Warm Amber",
         "Golden hour. Deep warm shadows, rich colour, real contrast.",
         warmth=.68, split=-.42, exposure=.02, contrast=.54, lift=.16,
         vibrance=.42, sat_skin=.18, sat_cool=-.42, clarity=.22,
         skin_smooth=.32, skin_even=.36, skin_tone=.24, skin_bright=.12,
         glow=.24, lips=.36, eyes=.26, eye_light=.30, vignette=.14),
    Look("crystal", "Crystal Clear",
         "Cool, bright and sharp. Blue skies, clear eyes, the least smoothing here.",
         warmth=-.34, tint=-.08, split=.44, exposure=.18, contrast=.48, lift=.04,
         vibrance=.38, sat_skin=-.02, sat_cool=.68, clarity=.50,
         skin_smooth=.24, skin_even=.34, skin_tone=-.12, skin_bright=.20,
         glow=.12, lips=.26, eyes=.42, eye_light=.52, vignette=.0),
    Look("silk", "Silk Premium",
         "Matte editorial film: muted colour, raised blacks, silk-smooth skin.",
         warmth=.34, split=.50, exposure=.04, contrast=.12, lift=.60,
         vibrance=.24, sat_skin=.10, sat_cool=-.38, clarity=.06,
         skin_smooth=.66, skin_even=.48, skin_tone=.16, skin_bright=.14,
         glow=.34, lips=.26, eyes=.20, eye_light=.26, vignette=.08),
    Look("sculpt", "Sculpted Detail",
         "Contrast and depth. Eyes, brows and lashes forward, skin left textured.",
         warmth=.04, tint=-.04, split=.24, exposure=-.06, contrast=.64, lift=.02,
         vibrance=.30, sat_skin=-.06, sat_cool=.26, clarity=.58,
         skin_smooth=.20, skin_even=.30, skin_tone=.0, skin_bright=.08,
         glow=.06, lips=.34, eyes=.58, eye_light=.58, vignette=.12),
    Look("even", "Even Tone",
         "Corrective. Evens out blotches and redness and keeps the grade quiet.",
         warmth=.14, tint=-.18, split=.16, exposure=.12, contrast=.32, lift=.10,
         vibrance=.26, sat_skin=-.06, sat_cool=.10, clarity=.22,
         skin_smooth=.46, skin_even=.82, skin_tone=-.10, skin_bright=.20,
         glow=.18, lips=.26, eyes=.28, eye_light=.32, vignette=.03),
    Look("bloom", "Rose Bloom",
         "Rose-tinted and soft, with the fullest lips in the set.",
         warmth=.22, tint=.46, split=.12, exposure=.14, contrast=.24, lift=.28,
         vibrance=.34, sat_skin=.30, sat_cool=-.22, clarity=.08,
         skin_smooth=.54, skin_even=.44, skin_tone=.18, skin_bright=.20,
         glow=.48, lips=.72, eyes=.34, eye_light=.40, vignette=.06),
    # ---- added: the three grade families the set had no way to say -------------------------
    # Teal-and-orange is the most recognisable colour grade there is, and it is not reachable by
    # turning any of the looks above up: it needs the warm and the cool bands moving in OPPOSITE
    # directions, which is exactly what `split` plus the two hue-band saturations are for.
    Look("cinema", "Cinematic",
         "Teal shadows against warm skin - the grade every film trailer is cut with.",
         warmth=.30, tint=-.06, split=-.62, exposure=.0, contrast=.52, lift=.26,
         vibrance=.30, sat_skin=.26, sat_cool=.42, clarity=.34,
         skin_smooth=.34, skin_even=.38, skin_tone=.20, skin_bright=.12,
         glow=.14, lips=.34, eyes=.44, eye_light=.46, vignette=.20),
    # Low contrast, high lift, almost no saturation movement: the quiet, expensive-looking
    # portrait that every editorial preset pack sells and that a "premium" set has to have.
    Look("dawn", "Morning Dawn",
         "Barely there. Soft light, open shadows, colour left almost exactly as photographed.",
         warmth=.18, tint=.06, split=.34, exposure=.18, contrast=.06, lift=.36,
         vibrance=.18, sat_skin=.06, sat_cool=-.10, clarity=.02,
         skin_smooth=.44, skin_even=.40, skin_tone=.08, skin_bright=.26,
         glow=.36, lips=.22, eyes=.20, eye_light=.28, vignette=.0),
    # Black and white belongs in any portrait set and could not be expressed at all before:
    # `mono` is the one term here that is not a colour move. The warm-filter weighting is the
    # classic portrait choice - it lightens skin's reds and darkens a blue sky, which is what
    # separates a photographed monochrome from a desaturated colour photo.
    Look("noir", "Studio Noir",
         "Black and white, shot as if through a warm filter. Deep blacks, luminous skin.",
         exposure=.04, contrast=.58, lift=.14, clarity=.42, mono=1.0,
         skin_smooth=.34, skin_even=.30, skin_bright=.16,
         glow=.16, eyes=.50, eye_light=.54, vignette=.22),
)

BY_ID = {lk.id: lk for lk in LOOKS}


def resolve(look_id: Optional[str], mode: str = "beautify") -> Look:
    """Look for `look_id`, falling back to the default this mode should start from.

    Clear mode's whole promise is that nothing was styled, so its default is no look at all.
    An explicit choice is always honoured either way - the default is a starting point, never
    something the caller is stuck with.
    """
    if look_id in BY_ID:
        return BY_ID[look_id]
    return BY_ID[NO_LOOK] if mode == "clear" else BY_ID[DEFAULT_LOOK]


def catalogue(mode: str = "beautify") -> dict:
    return {"default": resolve(None, mode).id, "filters": [lk.public() for lk in LOOKS]}


# ---------------------------------------------------------------------------------------
# whole-frame measurements (taken once; see chunked.py for why that matters)
# ---------------------------------------------------------------------------------------
def skin_reference(rgb: np.ndarray) -> Optional[Tuple[float, float]]:
    """The median chroma of this photo's skin: the target `skin_even` pulls blotches toward.

    Measured from the image itself and never from a constant, which is the whole reason skin
    evening here cannot drift someone's complexion - the destination IS their complexion.
    """
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    sel = (cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)
    if int(sel.sum()) < 64:
        return None
    return float(np.median(cr[sel])), float(np.median(cb[sel]))


# The colour half of the grade is carried by three weighting bells over the tone curve. Every
# one of them is IDENTICALLY ZERO at y=0 and y=1, and that single property is what lets these
# looks be three times stronger than the old ones without a white shirt going yellow or a black
# suit going blue - the classic tell of a cheap filter. `_BUMP_PEAK` normalises the two skewed
# bells so their peak is 1.0 and the coefficients below read directly as "fraction of full
# scale at the strongest point".
_GRADE_X = np.linspace(0.0, 1.0, 256, dtype=np.float64)
_BUMP_PEAK = 0.33551          # max of sin^2(pi*y) * y^2, at y ~ 0.646

# Per-channel offsets in fractions of full scale, at the peak of their bell.
WARM_RGB = np.array([0.052, 0.008, -0.046])    # warmth: +13 R / -12 B at midtone, warmth=1
TINT_RGB = np.array([0.015, -0.030, 0.015])    # tint:   green <-> magenta, +/- 7.6 G
SPLIT_HI_RGB = np.array([0.055, 0.018, -0.030])  # split>0: warm upper mids
SPLIT_LO_RGB = np.array([-0.022, -0.004, 0.048])  # split>0: cool lower mids


def _grade_bells(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(midtone, upper-mid, lower-mid) weights over a tone value, each 0 at both ends."""
    bell = np.sin(np.pi * y) ** 2
    return bell, bell * y * y / _BUMP_PEAK, bell * (1.0 - y) ** 2 / _BUMP_PEAK


def tone_lut(look: Look) -> Optional[np.ndarray]:
    """The whole colour grade - exposure, contrast, matte lift, warmth, tint, split tone -
    collapsed into one 256-entry table.

    Six grading passes become a single `cv2.LUT`, which matters twice over: it is the fastest
    thing in the pipeline, and being pointwise it gives a tile byte-for-byte the same answer it
    would have got as part of the whole frame. No global measurement, no seam, no cost.

    Every term fixes both ends of the range, and that is the design, not a detail:

      * exposure is a GAMMA. `y ** (1/(1+0.85E))` moves the midtone hard - E=0.16 is +11 levels
        at y=0.5 - while 0 stays 0 and 1 stays 1. The old affine lift blew highlights.
      * contrast is `y - (A/2pi) sin(2pi y)`, A = 0.55*contrast. Slope is 1-A at the ends and
        1+A at the midtone: a real S-curve whose toe and shoulder compress instead of clipping.
        The old half-sine was -1 at black and +1 at white, so it clipped both ends and could
        not be pushed. Monotone for A <= 1, i.e. contrast <= 1.
      * lift raises the blacks and pulls the top down a hair: film, not haze.
      * warmth, tint and split are ADDITIVE offsets weighted by `_grade_bells`. A colour cast
        that is zero at full white cannot stain a white; a cast that is zero at full black
        cannot tint a black. Split toning - warm highlights against cool shadows - is the term
        that makes a grade read as photographed rather than as brightened, and it is the one
        thing the old parameter set had no way at all to say.

    The table is asserted monotone per channel by the look tests; a non-monotone table would
    posterise. With the shipped LOOKS the worst-case offset slope is about half the curve's own
    slope at the same point, so there is real margin.
    """
    if not any((look.exposure, look.contrast, look.lift, look.warmth, look.tint, look.split)):
        return None
    y = _GRADE_X.copy()
    if look.exposure:
        y = np.power(np.clip(y, 0.0, 1.0), 1.0 / (1.0 + 0.85 * look.exposure))
    if look.contrast:
        a = 0.55 * float(look.contrast)
        y = y - (a / (2.0 * np.pi)) * np.sin(2.0 * np.pi * np.clip(y, 0.0, 1.0))
    if look.lift:
        lo, hi = 0.075 * look.lift, 0.030 * look.lift
        y = lo + y * (1.0 - lo - hi)
    y = np.clip(y, 0.0, 1.0)

    mid, upper, lower = _grade_bells(y)
    d = np.zeros((256, 3), np.float64)
    if look.warmth:
        d += np.outer(mid * look.warmth, WARM_RGB)
    if look.tint:
        d += np.outer(mid * look.tint, TINT_RGB)
    if look.split:
        d += np.outer(upper * look.split, SPLIT_HI_RGB)
        d += np.outer(lower * look.split, SPLIT_LO_RGB)

    lut = np.empty((1, 256, 3), np.uint8)
    for c in range(3):
        lut[0, :, c] = np.clip((y + d[:, c]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return lut


def _hue_band(hue: np.ndarray, centre: float, halfwidth: float) -> np.ndarray:
    """A soft, circular hue selector on OpenCV's 0..179 hue. Pointwise, so tile-exact."""
    d = np.abs(((hue - centre + 90.0) % 180.0) - 90.0)
    return np.clip(1.0 - d / halfwidth, 0.0, 1.0)


# The warm-filter monochrome mix. Sums to 1.0, so it cannot change overall exposure; red is
# weighted well above its 0.299 luminance share and blue well below its 0.114.
MONO_WEIGHTS = np.array([[0.52, 0.36, 0.12]], np.float32)

# Hue bands the looks are allowed to move independently. WARM is red-orange-yellow: skin, lips,
# wood, sand, gold. COOL is green-cyan-blue-violet: sky, foliage, shade, denim.
WARM_BAND = (10.0, 30.0)
COOL_BAND = (105.0, 50.0)
# Whatever a look asks for, saturation on skin may not leave this window. This is a hard cap on
# the one failure mode that ruins a portrait in both directions: an orange face, or a drained
# grey one. It is enforced on the combined gain, so it survives any future parameter.
SKIN_SAT_MIN, SKIN_SAT_MAX = 0.90, 1.18


def luma_curve_lut(look: Look) -> Optional[np.ndarray]:
    """What this look's grade does to luminance alone, as an L -> L table.

    The local-contrast pass runs after the grade, so its histograms have to be measured through
    the grade to be calibrated for the pixels it will meet. Derived by pushing a neutral ramp
    through the look's own channel tables and reading the L that comes back, so it stays correct
    automatically if the grade changes.
    """
    lut = tone_lut(look)
    if lut is None:
        return None
    ramp = np.repeat(np.arange(256, dtype=np.uint8).reshape(1, 256, 1), 3, axis=2)
    l_in = cv2.cvtColor(ramp, cv2.COLOR_RGB2LAB)[0, :, 0].astype(np.float32)
    l_out = cv2.cvtColor(cv2.LUT(ramp, lut), cv2.COLOR_RGB2LAB)[0, :, 0].astype(np.float32)
    return np.interp(np.arange(256, dtype=np.float32), l_in, l_out).astype(np.float32)


# ---------------------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------------------
def _band_mask(shape: Sequence[int], boxes: List, top: float, bottom: float,
               inset: float) -> np.ndarray:
    """A soft horizontal band across each face box - the eye line, or the mouth."""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), np.float32)
    for b in boxes:
        x0 = max(0, int(b.x + b.w * inset))
        x1 = min(w, int(b.x + b.w * (1.0 - inset)))
        y0 = max(0, int(b.y + b.h * top))
        y1 = min(h, int(b.y + b.h * bottom))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1.0
    if mask.max() > 0:
        span = max(b.w for b in boxes)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, span / 14.0))
    return np.clip(mask, 0.0, 1.0)


def _band_rect(shape: Sequence[int], boxes: List, top: float, bottom: float,
               inset: float) -> Tuple[int, int, int, int]:
    """The rectangle a band can reach: the band itself plus three sigma of its feather.

    Lips occupy about two percent of a portrait, and eyes not much more. Running a sharpener or
    a colour pass over the whole frame to change them costs the same as processing the entire
    photo again - the mask throws all of it away. Cropping to this rect first is the same
    argument the face stages make against the frame, one level down, and it is most of the
    difference between a look that renders in two seconds and one that renders in twelve.
    """
    h, w = int(shape[0]), int(shape[1])
    # Four sigma, not three: the band mask is float32, and OpenCV sizes a Gaussian kernel at four
    # sigma for anything wider than 8-bit (ops.GAUSS_SUPPORT). At three the mask was still
    # non-zero where this rect stops, so the lips pass and the eyes pass each left a small
    # rectangle of their own around the mouth and the eye line - the same defect as the face
    # crop, one level down.
    pad = int(np.ceil(4.0 * max(2.0, max(b.w for b in boxes) / 14.0))) + 3
    y0 = max(0, int(min(b.y + b.h * top for b in boxes)) - pad)
    y1 = min(h, int(max(b.y + b.h * bottom for b in boxes)) + pad)
    x0 = max(0, int(min(b.x + b.w * inset for b in boxes)) - pad)
    x1 = min(w, int(max(b.x + b.w * (1.0 - inset) for b in boxes)) + pad)
    return y0, x0, y1, x1


def _in_band(rgb: np.ndarray, boxes: List, face: Optional[np.ndarray], band: Tuple[float, float, float],
             run) -> np.ndarray:
    """Run `run(piece, mask, boxes)` over just the band's neighbourhood.

    Skipped when `face` is supplied, because that only happens on the path where the mask was
    sampled from a global map and the boxes are in the frame's coordinates rather than the
    piece's - there the rect would be meaningless, so the stage runs over the piece as given.
    """
    top, bottom, inset = band
    if face is not None:
        mask = _band_mask(rgb.shape, boxes, top, bottom, inset) * face
        if mask.max() <= 0.01:
            return rgb
        # `run` is handed the whole piece and round-trips it through YCrCb and LAB, both lossy.
        # Without this the lips and eyes passes dither every pixel of the piece by up to four
        # levels, including the ones the band mask says to leave alone.
        return ops.composite_masked(rgb, run(rgb, mask, boxes), mask)

    y0, x0, y1, x1 = _band_rect(rgb.shape, boxes, top, bottom, inset)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return rgb
    local = [type(b)(b.x - x0, b.y - y0, b.w, b.h) for b in boxes]
    # A real copy, not a view. The composite below needs the untouched pixels as its base, and
    # this slice IS contiguous - so ascontiguousarray does not copy it - whenever the band happens
    # to span the full width of the piece.
    piece = np.array(rgb[y0:y1, x0:x1])
    mask = _band_mask(piece.shape, local, top, bottom, inset) * ops.face_region_mask(
        piece.shape, local)
    if mask.max() <= 0.01:
        return rgb
    # Mask-exact: `run` round-trips the whole band rect through YCrCb (lips) and LAB (the
    # sharpener), and both are lossy, so without this the rect comes back dithered edge to edge
    # and stops dead at its own boundary. With it, and with the 4-sigma pad above, the write is
    # bit-identical to `piece` everywhere the band mask is zero - so the rect leaves no trace.
    rgb[y0:y1, x0:x1] = ops.composite_masked(piece, run(piece, mask, local), mask)
    return rgb


LIPS_BAND = (0.58, 0.95, 0.22)
EYES_BAND = (0.20, 0.55, 0.04)


def _lip_mask(rgb: np.ndarray, band_mask: np.ndarray,
              skin_ref: Optional[Tuple[float, float]]) -> np.ndarray:
    """Lips = the mouth band AND pixels redder than this face's own skin.

    The band alone would take the chin and the philtrum with it, which is how lip filters end up
    looking like a smear of lipstick. Requiring the pixel to also be redder than the person's
    measured skin tone confines the effect to the lips themselves, and because the threshold
    comes from their skin it travels correctly across complexions.
    """
    cr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 1].astype(np.float32)
    floor = (skin_ref[0] if skin_ref else 150.0) + 4.0
    redness = np.clip((cr - floor) / 12.0, 0.0, 1.0)
    return band_mask * cv2.GaussianBlur(redness, (0, 0), sigmaX=1.5)


# ---------------------------------------------------------------------------------------
# the two halves of a look
# ---------------------------------------------------------------------------------------
def apply_global(rgb: np.ndarray, look: Look, lut: Optional[np.ndarray] = None,
                 local_contrast: Optional[Callable] = None,
                 rect: Optional[Sequence[int]] = None,
                 full_shape: Optional[Sequence[int]] = None) -> np.ndarray:
    """The frame-wide half: grade, vibrance, local contrast, vignette.

    `rect`/`full_shape` place this piece inside the whole photo, so the vignette is drawn from
    the real centre of the image rather than the centre of whatever tile it happens to be in.
    """
    out = rgb
    if look.mono > 0.01:
        # Monochrome through a WARM filter, not a desaturation. Weighting red above its
        # luminance share is the choice a portrait photographer makes with an orange filter on
        # the lens: skin carries most of its signal in red, so it comes up luminous instead of
        # grey, while a blue sky and cool shadows drop away. Straight desaturation gives skin the
        # same 0.30/0.59/0.11 treatment as everything else and is the reason a "B&W filter"
        # usually reads as a colour photo with the colour switched off.
        #
        # Runs first, so anything the grade says about colour afterwards is a TONE on the
        # monochrome (split toning a black and white is a real look) rather than a cast fighting
        # the original colour. The face half of the look ran before this on the colour image, so
        # skin, lips and eyes were all found while there was still chroma to find them by.
        m = float(np.clip(look.mono, 0.0, 1.0))
        grey = cv2.cvtColor(cv2.transform(out, MONO_WEIGHTS), cv2.COLOR_GRAY2RGB)
        out = grey if m >= 0.999 else cv2.addWeighted(out, 1.0 - m, grey, m, 0.0)
    lut = tone_lut(look) if lut is None else lut
    if lut is not None:
        out = cv2.LUT(np.ascontiguousarray(out), lut)

    if max(abs(look.vibrance), abs(look.sat_skin), abs(look.sat_cool)) > 0.01:
        # Saturation, selectively by hue. A single saturation slider moves the sky and the face
        # together, which is why one filter can never be both "rich" and "flattering"; moving
        # the warm and the cool bands independently is what a colour grade actually is, and it
        # is where most of the difference between these looks lives.
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue = hsv[:, :, 0]
        s = hsv[:, :, 1] / 255.0
        gain = 1.0 + (0.55 * look.vibrance) * (1.0 - s)   # duller colours move most
        if abs(look.sat_skin) > 0.01:
            gain = gain * (1.0 + 0.45 * look.sat_skin * _hue_band(hue, *WARM_BAND))
        if abs(look.sat_cool) > 0.01:
            gain = gain * (1.0 + 0.50 * look.sat_cool * _hue_band(hue, *COOL_BAND))
        # The cap is on the COMBINED gain and only on skin, so a look can mute a tan wall or
        # saturate a sunset without the complexion following it anywhere it should not go.
        skin = ops._skin_mask(out)
        gain = gain * (1.0 - skin) + np.clip(gain, SKIN_SAT_MIN, SKIN_SAT_MAX) * skin
        hsv[:, :, 1] = np.clip(s * gain, 0.0, 1.0) * 255.0
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if look.clarity > 0.01:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # NOTE: this clip limit is duplicated in beautify._apply_look, which builds the
        # ClaheField for the tiled path. The two must stay equal or a large photo gets a
        # different amount of local contrast than a small one.
        boosted = (cv2.createCLAHE(clipLimit=1.0 + 2.2 * look.clarity,
                                   tileGridSize=(8, 8)).apply(l)
                   if local_contrast is None else local_contrast(l))
        lf = l.astype(np.float32)
        # Luminosity mask. Local contrast is worth the most in the midtones and costs the most
        # at the ends - it is what turns a white shirt into paper and a shadow into mud - so the
        # blend rolls off above L=204 and is held back in the deepest shadow. The skin brake is
        # separate and for a separate reason: crunching local contrast into a cheek is what
        # makes a face read as ruddy and over-processed, and at this strength it showed.
        keep = ((1.0 - np.clip((lf - 204.0) / 51.0, 0.0, 1.0) ** 2)
                * (1.0 - 0.7 * np.clip((26.0 - lf) / 26.0, 0.0, 1.0)))
        k = min(1.0, look.clarity) * keep * (1.0 - 0.35 * ops._skin_mask(out))
        l = np.clip(lf * (1.0 - k) + boosted.astype(np.float32) * k, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    if look.vignette > 0.01:
        out = _vignette(out, look.vignette, rect, full_shape)
    return out


def _vignette(rgb: np.ndarray, amount: float, rect, full_shape) -> np.ndarray:
    h, w = rgb.shape[:2]
    fy0, fx0 = (rect[0], rect[1]) if rect is not None else (0, 0)
    fh, fw = (full_shape[0], full_shape[1]) if full_shape is not None else (h, w)
    yy = (np.arange(h, dtype=np.float32) + fy0) / max(1.0, fh - 1.0) - 0.5
    xx = (np.arange(w, dtype=np.float32) + fx0) / max(1.0, fw - 1.0) - 0.5
    r = np.sqrt(yy[:, None] ** 2 * 1.15 + xx[None, :] ** 2)
    fall = 1.0 - (0.34 * amount) * np.clip((r - 0.30) / 0.42, 0.0, 1.0) ** 2
    return np.clip(rgb.astype(np.float32) * fall[..., None], 0, 255).astype(np.uint8)


def apply_face(rgb: np.ndarray, look: Look, boxes: List,
               skin_ref: Optional[Tuple[float, float]] = None,
               face_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """The face half: skin evening, smoothing, light, lips, eyes.

    Everything is held inside a feathered face mask, so it stops well before the edge of the
    region it was cropped from - which is what makes running this over a crop of a 40 MP photo
    identical to running it over the whole thing.

    `rgb` may be written to in place; use the return value.
    """
    if not boxes or not any((look.skin_smooth, look.skin_even, look.skin_tone,
                             look.skin_bright, look.glow, look.lips, look.eyes,
                             look.eye_light)):
        return rgb
    face = ops.face_region_mask(rgb.shape, boxes) if face_mask is None else face_mask
    if face.max() <= 0:
        return rgb
    skin = np.clip(face * ops._skin_mask(rgb), 0.0, 1.0)
    # Kept for the mask-exact composite at the end of this function. It has to be taken here and
    # not later: `_in_band` writes into the array it is handed, and on the region path that array
    # is a view of the caller's image, so by the time the last stage has run there is nothing
    # untouched left to fade back to.
    base = rgb.copy()
    out = rgb

    # ---- even out the tone, then even out the texture -----------------------------------
    if (look.skin_even > 0.01 and skin_ref is not None) or abs(look.skin_tone) > 0.01:
        ycrcb = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        if look.skin_even > 0.01 and skin_ref is not None:
            k = (0.42 * look.skin_even) * skin
            for ch, ref in ((1, skin_ref[0]), (2, skin_ref[1])):
                ycrcb[:, :, ch] = ycrcb[:, :, ch] * (1.0 - k) + float(ref) * k
        if abs(look.skin_tone) > 0.01:
            # Warm or cool the complexion - and NOTHING else about it. This writes Cr and Cb and
            # never touches Y, so it is arithmetically incapable of lightening skin: whatever a
            # look asks for, the luminance of every skin pixel comes out of here unchanged. That
            # is the difference between a tone control and a whitening filter, and it is a
            # property of the code rather than of the numbers in the table. Bounded to +/-5 of
            # Cr and +/-3.5 of Cb inside the face-and-skin mask: a visible warmth, nowhere near
            # enough to move anyone toward a different complexion. Evening runs first, so this
            # moves a face that has already been made consistent with itself.
            t = float(np.clip(look.skin_tone, -1.0, 1.0))
            ycrcb[:, :, 1] += (5.0 * t) * skin
            ycrcb[:, :, 2] -= (3.5 * t) * skin
        out = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    if look.skin_smooth > 0.01:
        s = float(min(1.0, look.skin_smooth))
        # NOT `base` — that name holds the pristine input for the mask-exact composite at the
        # end of this function, and shadowing it here silently defeated that composite: skin_even
        # runs first and dirties `out` through a whole-frame YCrCb round trip, so re-binding
        # `base` to `out` left nothing clean to fall back to. Each stage measured exact on its
        # own and the two together leaked across the entire frame.
        low = out.astype(np.float32)
        # Frequency separation. The blotches live in the low frequencies and are what gets
        # smoothed; the pores live in the high frequencies and are added straight back. Blend
        # the smooth layer alone and you get the plastic look this is written to avoid.
        smooth = cv2.bilateralFilter(out, d=7, sigmaColor=int(18 + 30 * s), sigmaSpace=7)
        detail = low - cv2.GaussianBlur(low, (0, 0), sigmaX=1.1)
        blend = (0.55 * s * skin)[..., None]
        out = np.clip(low * (1.0 - blend) + smooth.astype(np.float32) * blend
                      + np.clip(detail, -20, 20) * blend * 0.72, 0, 255).astype(np.uint8)

    if look.skin_bright > 0.01:
        # Fill light, not a lift. Capped in absolute levels rather than as a ratio - a lift that
        # scales with brightness is how a "brightening" filter turns into a whitening one - and
        # additionally weighted by how much headroom a pixel still has, so the light lands in
        # the shadowed side of a face and gives already-bright skin nothing at all. A flat lift
        # here measured +22 levels of mean skin luminance while tuning; this stays under +10.
        yl = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
        room = 1.0 - np.clip((yl - 140.0) / 115.0, 0.0, 1.0)
        lift = (8.0 * look.skin_bright) * skin * room
        out = np.clip(out.astype(np.float32) + lift[..., None], 0, 255).astype(np.uint8)

    # ---- light ---------------------------------------------------------------------------
    if look.glow > 0.01:
        # Bloom the LIGHT, not the face. soft_glow is a screen blend and can only brighten, so
        # across a flat face mask it is the largest whitening vector in this file - at the glow
        # levels these looks want it measured +28 levels of mean skin luminance. Weighting the
        # mask by luminance puts the bloom where a highlight already is and leaves the shadowed
        # side of the face alone, which is both the honest version of the effect and the one
        # that reads as light rather than as haze.
        yg = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
        highlight = np.clip((yg - 120.0) / 90.0, 0.0, 1.0)
        out = ops.soft_glow(out, boxes, min(0.55, look.glow * 0.62), mask=face * highlight)

    # ---- features ------------------------------------------------------------------------
    # Both of these run over the band's own rectangle rather than the frame; see _in_band.
    if look.lips > 0.01:
        def _lips(piece, band, _bx):
            lips = _lip_mask(piece, band, skin_ref)
            if lips.max() <= 0.01:
                return piece
            ycrcb = cv2.cvtColor(piece, cv2.COLOR_RGB2YCrCb).astype(np.float32)
            # Colour, warmth and depth, in that order of visibility. The old amounts worked out
            # to about two levels of Cr at the strongest look, which is nothing. Depth is what
            # separates "fuller lips" from "a smear of lipstick": the luma contrast about the
            # mouth's own midpoint and the small darkening give the lip line back, and it is all
            # still gated on pixels redder than this person's measured skin.
            m = (0.55 * look.lips) * lips
            ycrcb[:, :, 1] = np.clip(ycrcb[:, :, 1] + 22.0 * m, 0, 255)   # more colour
            ycrcb[:, :, 2] = np.clip(ycrcb[:, :, 2] - 6.0 * m, 0, 255)    # a shade warmer
            ycrcb[:, :, 0] = np.clip((ycrcb[:, :, 0] - 132.0) * (1.0 + 0.34 * m) + 132.0
                                     - 6.0 * m, 0, 255)                   # depth, not gloss
            piece = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
            return ops.edge_aware_sharpen(piece, strength=min(0.55, look.lips * 0.5),
                                          protect_skin=False, region_mask=lips, halo_limit=8.0)

        out = _in_band(out, boxes, face_mask, LIPS_BAND, _lips)

    if look.eyes > 0.01 or look.eye_light > 0.01:
        def _eyes(piece, band, _bx):
            if look.eye_light > 0.01:
                # Luminous eyes - the one thing that flatters a portrait more reliably than
                # anything else here, and the thing sharpening alone cannot do. Restricted to
                # the non-skin pixels of the eye band, so the lid, the brow and the cheek under
                # it are excluded by construction rather than by a number. The sclera and the
                # catchlight come up, the lash line and the pupil go down, and the chroma is
                # expanded a fraction so the iris keeps its colour instead of being greyed by
                # the lift. Fixed curves, no measurement: a band-local mean would give a tile a
                # different answer than the region, and this must not.
                e = band * (1.0 - 0.85 * ops._skin_mask(piece))
                lab = cv2.cvtColor(piece, cv2.COLOR_RGB2LAB).astype(np.float32)
                l = lab[:, :, 0]
                dl = (22.0 * np.clip((l - 150.0) / 105.0, 0.0, 1.0)
                      - 16.0 * np.clip((70.0 - l) / 70.0, 0.0, 1.0)) * (look.eye_light * e)
                lab[:, :, 0] = np.clip(l + dl, 0, 255)
                cs = 1.0 + 0.18 * look.eye_light * e
                lab[:, :, 1] = np.clip(128.0 + (lab[:, :, 1] - 128.0) * cs, 0, 255)
                lab[:, :, 2] = np.clip(128.0 + (lab[:, :, 2] - 128.0) * cs, 0, 255)
                piece = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            if look.eyes > 0.01:
                piece = ops.edge_aware_sharpen(piece, strength=min(0.6, look.eyes * 0.6),
                                               protect_skin=False, region_mask=band,
                                               halo_limit=9.0)
            return piece

        out = _in_band(out, boxes, face_mask, EYES_BAND, _eyes)
    # Every stage above is already scaled by a mask that is a subset of `face`, so this takes
    # nothing away that the look asked for - but skin_even round-trips the whole piece through
    # YCrCb and skin_smooth truncates its blend, and neither is exact where the mask is zero.
    # Measured at max 4 / mean 0.774 outside the face mask: a dither that covers the crop this
    # ran on and stops at its edge. See ops.composite_masked.
    return ops.composite_masked(base, out, face)
