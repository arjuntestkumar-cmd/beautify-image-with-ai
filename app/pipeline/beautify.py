"""The single enhancement pipeline: Beautify.

One mode, no manual sliders. The pipeline *adapts* to each photo — that adaptation is what the
original service's mode presets were for, and it is kept here in full — but the user never has
to think about it. A premium look is applied at the end and can be changed or removed; it is a
finish on top of the restoration, never a substitute for it.

    decode -> analyse -> plan
      -> de-block heavily compressed input
      -> GFPGAN face restoration on a Real-ESRGAN background   (photos with faces)
         or plain Real-ESRGAN restoration                      (everything else)
      -> chroma denoise if the result is still noisy
      -> hair / fine-texture refinement around detected heads
      -> photographic finish (white balance, highlights, shadows, midtones, vibrance)
      -> edge-aware detail + over-sharpening safeguard
      -> quality checks -> save the un-styled base -> apply the look -> encode

Everything heavy above goes through a `chunked.Runner`. On an ordinary photo it is a straight
passthrough and this is exactly the pipeline it always was; on a large one it splits each stage
into overlapping tiles so peak memory follows the tile rather than the photo. What the pipeline
computes does not change with the size of the upload — only how much of it is resident at once.
See pipeline/chunked.py.

Tuning comes from the original service's dedicated `beautify` preset, whose intent, in priority
order, is identity first and beauty second:

  * every detected face is restored — a face is the thing people look at, so "beautify" that
    silently skipped clean faces was the whole feature not firing.
  * the source's real skin micro-texture is re-injected afterwards, in skin areas only, scaled
    by how clean the source is. This is what keeps a restored face from looking waxy or
    re-drawn, and it is the dial that genuinely controls identity (see `skin_texture`).
  * hair refinement is pushed high (0.62) because it is safe to sharpen: no identity risk.
  * the photographic finish is bounded (tone depth 0.32) — richer than neutral, nowhere near a
    heavy grade.

Removed relative to the multi-mode original (they served modes this build does not have):
old-photo colour restoration, scratch/tear removal, CodeFormer, SwinIR, product relighting,
document mode and the manual adjustment layer. The original's 21 filters are not back either;
pipeline/filters.py is a small set of looks built for this pipeline, applied after it.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..analysis import Analysis, FaceBox, HIGH, LOW, MEDIUM, SEVERELY_DEGRADED, analyse
from ..config import Settings
from ..errors import ModelsUnavailable, OutOfMemory, ProcessingFailed
from ..logging_utils import get_logger
from ..validation import DecodedImage
from . import chunked, filters, ops
from .encode import FORMAT_EXT, FORMAT_MIME, check_array, encode, verify_encoded
from .registry import ModelRegistry

log = get_logger("beautify")

# Progress callback: (stage, percent, human message)
ProgressFn = Callable[[str, int, str], None]

# --------------------------------------------------------------------------------------
# The one preset. Ported from the original service's `beautify` ModePreset + ModeSpec.
# --------------------------------------------------------------------------------------
FACE_STRENGTH_CAP = 0.58    # upper bound on the GFPGAN weight (ignored by the v1.4 clean arch)
# How far the restored face is allowed to run ahead of the photograph around it, as a ratio of
# high-frequency energy. GFPGAN rebuilds a face from a learned prior; a shirt has no such prior
# and can only ever be deconvolved, so some gap is physics and not a bug. Past this multiple the
# gap stops reading as "a well restored portrait" and starts reading as a sharp face pasted into
# a blurry picture, which is the defect. Measured BEFORE body_clarity runs, so it is calibrated
# against the softer, pre-clarity body. Set it very high to disable the cap entirely.
FACE_FRAME_ACUTANCE_MAX = 4.0
DETAIL_CAP = 0.58           # upper bound on the final edge-aware sharpen
HAIR_REFINE_BASE = 0.62     # base hair/edge refinement strength
TONE_DEPTH = 0.32           # photographic finish intensity
TONE_DEPTH_HIGH_QUALITY = 0.50  # already-great photo: skip heavy SR, lean on the finish
BASE_DENOISE = 0.45         # deliberately moderate — more erases pores and looks plastic
BASE_DETAIL = 0.58
MAX_UPSCALE = 2

# The two things a user can ask for.
#   BEAUTIFY - clean it up AND make it look good: skin evened out, tone and colour lifted.
#   CLEAR    - clean it up and nothing else: same restoration and denoising, but no skin work
#              and no grading, so the result stays faithful to the original photo.
MODE_BEAUTIFY = "beautify"
MODE_CLEAR = "clear"
MODES = (MODE_BEAUTIFY, MODE_CLEAR)
SKIN_CLEAN_BASE = 0.62      # beautify only
GLOW_BASE = 0.22            # beautify only
TONE_DEPTH_BEAUTIFY = 0.46  # beautify grade; clear mode uses 0.0
# The post-restoration denoise runs on the already-upscaled image, where it is far more
# destructive than the model's own denoising. Keep the model blend strong and this pass light.
POST_DENOISE_SCALE = 0.45
# Under this much measured under-exposure, leave the photo's tone alone entirely.
LOW_LIGHT_THRESHOLD = 0.20

# Strategies (kept so the log/metadata explains what happened to a given photo).
SEVERE = "severe-restoration"
BALANCED = "balanced-restoration"
PORTRAIT = "portrait-restoration"
HIGH_QUALITY = "high-quality-refinement"
GENERAL = "general-restoration"
DOCUMENT = "document-safe"


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class Params:
    """Everything the pipeline needs, fully resolved and bounded."""
    mode: str
    strategy: str
    effective_scale: int
    denoise_strength: float
    detail_strength: float
    face_restore: bool
    face_strength: float
    skin_texture: float
    only_center_face: bool
    face_clarity: float
    body_clarity: float
    chroma_clean: float
    skin_clean: float
    glow: float
    hair_refine: float
    source_noise: float
    tone_depth: float
    # Soft-focus recovery for everything that is NOT a face. Zero on any photo that has a sharp
    # region anywhere in it; see build_params.
    soft_radius: float = 0.0
    soft_gain: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class BeautifyResult:
    mode: str
    look: str
    width: int
    height: int
    original_width: int
    original_height: int
    mime_type: str
    bytes: int
    face_count: int
    processed_faces: int
    models: List[str]
    strategy: str
    quality_category: str
    effective_scale: int
    processing_time_ms: int
    stage_timings_ms: Dict[str, int]
    warnings: List[str]
    fallback_used: bool
    chunked: bool = False
    # The un-styled enhanced image, kept so a look can be changed or taken off without any of
    # the model work being repeated, together with the face boxes that look would need.
    base_path: Optional[str] = None
    base_format: str = "jpeg"
    face_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)


class _Timer:
    def __init__(self) -> None:
        self.stages: Dict[str, int] = {}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0) + int((time.perf_counter() - t) * 1000)


def _safe_scale(width: int, height: int, requested: int, max_side: int, max_pixels: int) -> int:
    """Largest scale in {1,2} <= requested that stays within the side/pixel limits."""
    for scale in (2, 1):
        if scale > requested:
            continue
        if width * scale <= max_side and height * scale <= max_side and (width * scale) * (height * scale) <= max_pixels:
            return scale
    return 1


def _select_strategy(analysis: Analysis) -> str:
    """Deterministic strategy selection from the analysis alone (no model calls)."""
    if analysis.screenshot_like:
        # Documents / screenshots / line art: text-safe, never hallucinate faces.
        return DOCUMENT
    has_usable_face = analysis.is_portrait
    if analysis.quality_category == HIGH and not has_usable_face:
        return HIGH_QUALITY
    if analysis.quality_category == HIGH and has_usable_face and analysis.blur_score < 0.16:
        return HIGH_QUALITY  # already-sharp portrait: premium finish, no face model
    if analysis.quality_category == SEVERELY_DEGRADED:
        return SEVERE
    if has_usable_face or analysis.face_count > 0:
        return PORTRAIT
    if analysis.quality_category in (LOW, MEDIUM):
        return BALANCED
    return GENERAL


def build_params(analysis: Analysis, settings: Settings, mode: str = MODE_BEAUTIFY,
                 exposure_lift: float = 0.0, cheap_faces: bool = False) -> Params:
    """Combine the preset with the measured condition of THIS photo."""
    mode = mode if mode in MODES else MODE_BEAUTIFY
    warnings: List[str] = []

    # Effective noise, which is not the same as measured noise.
    #
    # Brightening a dark photo multiplies whatever grain was hiding in its shadows, and the
    # measurement understates what comes back: shadow noise surfaces as low-frequency colour
    # mottling, which a 3-pixel median residual barely registers. On a real under-exposed frame
    # that read as 0.099 - under the denoise threshold - while the background visibly crawled.
    # The lift we applied is itself the evidence, so it sets a floor.
    noise = max(analysis.noise_score, exposure_lift * 0.55)
    if exposure_lift > 0 and noise > analysis.noise_score:
        warnings.append("shadow_noise_lifted")

    # ---- scale -------------------------------------------------------------------------
    # Small photos get a genuine 2x; photos that are already big are beautified at native size
    # (upscaling them is slow, makes huge files, and is rarely what was wanted).
    wanted = 2 if max(analysis.width, analysis.height) <= settings.AUTO_UPSCALE_MAX_SIDE else 1
    effective_scale = _safe_scale(
        analysis.width, analysis.height, min(wanted, MAX_UPSCALE),
        settings.MAX_OUTPUT_SIDE, settings.MAX_OUTPUT_PIXELS,
    )

    # ---- denoise / detail base ---------------------------------------------------------
    # Grain is the single most common reason a result still looks rough, and the measured
    # noise scale is compressed (heavy visible grain lands around 0.25), so it needs a strong
    # multiplier here to move the model's denoise blend at all.
    denoise = _clamp(BASE_DENOISE + 1.10 * noise)
    # Colour blotching is worth attacking separately from grain, and only when there is enough
    # of it to see - otherwise this is a no-op.
    chroma_clean = _clamp((noise - 0.18) * 2.2) if noise > 0.18 else 0.0
    detail = _clamp((BASE_DETAIL + BASE_DETAIL) / 2.0)

    if analysis.already_high_quality:
        # Do not hallucinate detail into an already-clean image.
        effective_scale = min(effective_scale, 2)
        denoise = _clamp(denoise * 0.4)
        detail = _clamp(detail * 0.5)

    strategy = _select_strategy(analysis)

    # ---- faces -------------------------------------------------------------------------
    degradation = 0.5 * analysis.blur_score + 0.5 * analysis.jpeg_artifact_score

    # Beautify attempts face restoration on every photo and lets the FACE MODEL'S OWN detector
    # decide whether there is a face.
    #
    # This used to be gated twice, and both gates threw the feature away. It only ran on faces
    # that MEASURED as degraded, so a clean portrait came back untouched; and it only ran when
    # OpenCV's Haar cascade found a face, which is wildly unreliable - on a real test photo Haar
    # found nothing at all while GFPGAN's RetinaFace found the same face at 0.999 confidence.
    # Haar is still used for cheap hints (is this a portrait, are the faces tiny). It is no
    # longer allowed to veto restoration.
    #
    # If there really is no face, GFPGAN returns its Real-ESRGAN background untouched, so the
    # only cost of trying is the detector. The one guard is cost: an image too big to upscale
    # would make GFPGAN's internal 2x background pass wasteful, so those take the face path
    # only when Haar did see something.
    #
    # `cheap_faces` retires that guard for large photos. It is set on the chunked path, where
    # faces are restored from their own crops and the wasteful 2x background pass never happens
    # - so the cost the guard was protecting against is gone, and with it the reason a big
    # portrait had to get past Haar (which the paragraph above explains cannot be trusted) to
    # have its face restored at all.
    face_restore = not analysis.screenshot_like and (
        effective_scale == 2 or analysis.face_count > 0 or cheap_faces)

    # NOTE: GFPGANv1Clean.forward() accepts **kwargs and ignores `weight`, so for the v1.4
    # model this value does NOT throttle the model. It is still passed (other GFPGAN
    # architectures do honour it) and it still records intent, but the dial that genuinely
    # controls how much the face gets repainted is `skin_texture`, below.
    face_strength = _clamp(min(FACE_STRENGTH_CAP, 0.38 + 0.35 * degradation))

    # How much of the SOURCE's real skin micro-texture is put back after restoration — the real
    # identity/naturalness control, and the thing that stops a restored face looking waxy.
    #
    # It scales with how clean the source is, NOT with how hard the model worked: a clean photo
    # has genuine pores and lashes worth keeping, so most of them go back; a degraded photo has
    # noise and JPEG blocks in those same high frequencies, and putting those back would simply
    # undo the restoration.
    # How much of the source's high frequency is REAL detail rather than grain or compression.
    #
    # This must be driven by NOISE above all, and that is what was wrong before: grain inflates
    # the Laplacian variance, so a heavily grainy photo scores blur=0.0 - "perfectly sharp" -
    # and the old formula happily re-injected 56% of pure noise back onto the restored face.
    # That is exactly the dotted, speckled skin left on an otherwise well restored portrait.
    texture_trust = _clamp(1.0 - (2.2 * noise
                                  + 1.2 * analysis.jpeg_artifact_score
                                  + 0.5 * analysis.blur_score))
    skin_texture = _clamp(0.10 + 0.50 * texture_trust, 0.05, 0.55)
    if mode == MODE_CLEAR:
        # Fidelity first: keep noticeably more of the real face than the beautified version.
        skin_texture = _clamp(skin_texture + 0.18, 0.05, 0.70)
    if analysis.small_faces:
        warnings.append("small_faces")

    # Restore every face the model detects, not just the dominant one, so nobody in a group
    # shot is left blurry next to a restored face. RetinaFace is confident enough to trust.
    only_center_face = False

    # How hard to sharpen structure INSIDE the face - eyes, lashes, brows, lips. Damped hard by
    # noise: sharpening a grainy face just makes the grain crisper.
    # Damped by noise so it does not just make grain crisper - but only mildly, because the
    # denoise pass has already run by then and the face needs its structure re-asserted.
    face_clarity = _clamp((0.45 + 0.30 * degradation) * (1.0 - 0.7 * noise), 0.25, 1.0)

    # Hands, clothing and objects get no generative restoration at all - only the upscaler - so
    # they need MORE sharpening than the face, not less, to end up looking like they belong in
    # the same photo.
    # Damped hard by noise. Local edge confidence is what lets fabric and fingers be sharpened
    # at all, but it will just as happily treat colour mottling in a flat wall as structure.
    body_clarity = _clamp((0.60 + 0.25 * degradation) * (1.0 - 1.1 * noise), 0.25, 1.0)

    # ---- soft-focus recovery: the rest of the frame, not the face ------------------------
    # A soft photograph comes back with a razor-sharp face on a body that is as soft as it went
    # in, and raising `body_clarity` does not fix it. On a real 6.3 MP soft portrait that value
    # is already 0.685; the problem is that the pass it feeds measures detail with a 1.1 px
    # unsharp, and a frame defocused at sigma ~4 leaves about 0.5 of a level of residual there
    # on a 60-level edge. The radius is wrong, not the gain. Matching the radius to the blur is
    # the only classical operation that can lift a shirt at all, and it is bounded in levels by
    # edge_aware_sharpen's clamp rather than in proportion.
    #
    # Gated on blur_floor and NOT on blur_score. blur_score is the median over sampled windows,
    # so an ordinary portrait with a defocused background reads "blurry" and would be handed a
    # 4 px unsharp it does not need - a halo generator. blur_floor is the sharpest fifth of the
    # frame: it is only high when nothing in the photo is sharp.
    softness = _clamp((analysis.blur_floor - 0.72) / 0.20)
    if analysis.screenshot_like or analysis.already_high_quality:
        # Text does not want a large-radius clarity pass, and an image the analysis called HIGH
        # measured blur < 0.16 - there is nothing here to recover.
        softness = 0.0
    # Grain and blocking live in exactly the band this amplifies, so they cancel it: a soft AND
    # grainy photo would get a crisper-looking grain, which is worse than leaving it soft.
    soft_gain = _clamp(2.0 * softness * (1.0 - 1.6 * noise)
                       * (1.0 - 0.8 * analysis.jpeg_artifact_score), 0.0, 2.0)
    # In OUTPUT pixels: this stage runs after super-resolution, so the radius follows the scale.
    soft_radius = round((1.6 + 2.6 * softness) * effective_scale, 2) if soft_gain > 0.02 else 0.0
    if soft_gain > 0.02:
        warnings.append("soft_focus_recovered")

    # Skin cleanup is what "beautify" means beyond restoration - and it is the whole difference
    # between the two modes. Clear mode does no skin work at all.
    skin_clean = 0.0
    glow = 0.0
    if mode == MODE_BEAUTIFY:
        glow = _clamp(GLOW_BASE * (0.8 + 0.6 * (1.0 - degradation)))
        # More smoothing for a grainy source, less for a clean one that has real skin to keep.
        skin_clean = _clamp(SKIN_CLEAN_BASE * (0.55 + 1.8 * noise), 0.0, 1.0)

    # ---- hair --------------------------------------------------------------------------
    hair_on = settings.HAIR_REFINEMENT_ENABLED and not analysis.screenshot_like
    cat_factor = {SEVERELY_DEGRADED: 0.7, LOW: 0.9, MEDIUM: 1.0, HIGH: 0.55}.get(analysis.quality_category, 1.0)
    # Scaled up with noise: the grainier the source, the harder the denoise hit the hair, and
    # the more of its structure has to be put back.
    hair_refine = _clamp(HAIR_REFINE_BASE * cat_factor * (0.6 + 0.4 * BASE_DETAIL)
                         * (1.0 + 0.7 * noise)) if hair_on else 0.0

    # ---- whole-image detail ------------------------------------------------------------
    detail_strength = _clamp(min(DETAIL_CAP, detail))
    if analysis.quality_category == HIGH:
        detail_strength = _clamp(detail_strength * 0.55)
    elif analysis.quality_category == SEVERELY_DEGRADED:
        detail_strength = _clamp(detail_strength * 0.85)  # do not amplify artifacts

    # Clear mode is a faithful clean-up: no tone curve, no vibrance, no white-balance shift.
    tone_depth = TONE_DEPTH_BEAUTIFY if mode == MODE_BEAUTIFY else 0.0

    # ---- strategy overrides ------------------------------------------------------------
    if strategy == HIGH_QUALITY:
        # An already-excellent photo: skip aggressive super-resolution and lean on the finish.
        # Faces are still restored — that is the whole product — but with the maximum amount of
        # original texture put back, so a great face is refined rather than repainted.
        effective_scale = min(effective_scale, 2)
        skin_texture = max(skin_texture, 0.60)
        if mode == MODE_BEAUTIFY:
            tone_depth = TONE_DEPTH_HIGH_QUALITY
    if strategy == DOCUMENT:
        face_restore = False
        hair_refine = 0.0

    return Params(
        mode=mode,
        strategy=strategy,
        effective_scale=effective_scale,
        denoise_strength=round(denoise, 3),
        detail_strength=round(detail_strength, 3),
        face_restore=face_restore,
        face_strength=round(face_strength, 3),
        skin_texture=round(skin_texture, 3),
        only_center_face=only_center_face,
        face_clarity=round(face_clarity, 3),
        body_clarity=round(body_clarity, 3),
        chroma_clean=round(chroma_clean, 3),
        skin_clean=round(skin_clean, 3),
        glow=round(glow, 3),
        hair_refine=round(hair_refine, 3),
        source_noise=round(noise, 3),
        tone_depth=tone_depth,
        soft_radius=soft_radius,
        soft_gain=round(soft_gain, 3),
        warnings=warnings,
    )


# --------------------------------------------------------------------------------------
# model stages
# --------------------------------------------------------------------------------------
def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


def _restore_faces(
    registry: ModelRegistry, params: Params, working_rgb: np.ndarray
) -> Tuple[Optional[np.ndarray], int, List[FaceBox]]:
    """GFPGAN restoration with its native face-parsing paste-back onto a Real-ESRGAN background.

    One pass, no halos, no ghost faces. Returns the 2x restored RGB image, the number of faces
    actually processed, and where those faces ended up IN THE RETURNED IMAGE - the model's own
    detections, which are far more reliable than the Haar boxes from analysis, and which the
    clarity and hair stages then target.
    """
    gfp = registry.gfpgan
    if gfp is None:
        return None, 0, []

    # The background upsampler is the same Real-ESRGAN model, so DNI applies here too.
    if registry.has_dni:
        registry.set_dni(params.denoise_strength)

    bgr = cv2.cvtColor(working_rgb, cv2.COLOR_RGB2BGR)
    weight = _clamp(params.face_strength)
    attempts = 0

    with registry.lock:
        while True:
            try:
                _cropped, restored_faces, restored_bgr = gfp.enhance(
                    bgr, has_aligned=False, only_center_face=params.only_center_face,
                    paste_back=True, weight=weight,
                )
                break
            except (RuntimeError, MemoryError) as exc:
                if _is_oom(exc) and attempts < 2:
                    attempts += 1
                    if registry.torch is not None:
                        try:
                            registry.torch.cuda.empty_cache()
                        except Exception:  # pragma: no cover
                            pass
                    log.warning("GFPGAN out of memory — retry %s", attempts)
                    continue
                if _is_oom(exc):
                    raise OutOfMemory("Ran out of memory while restoring faces.") from exc
                raise ProcessingFailed(
                    f"Face restoration failed: {exc.__class__.__name__}") from exc

    if restored_bgr is None:
        return None, 0, []

    result = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

    # Map the model's detections (input coordinates) onto the restored image.
    boxes: List[FaceBox] = []
    ratio = result.shape[1] / float(max(1, working_rgb.shape[1]))
    for det in getattr(gfp.face_helper, "det_faces", []) or []:
        x1, y1, x2, y2 = (float(v) * ratio for v in det[:4])
        w, h = int(x2 - x1), int(y2 - y1)
        if w > 1 and h > 1:
            boxes.append(FaceBox(int(x1), int(y1), w, h))

    # Anti-waxy: add the source's real skin micro-texture back where skin is detected.
    if params.skin_texture > 0.02:
        src_up = cv2.resize(working_rgb, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        if params.source_noise > 0.12:
            # Borrow texture from a CLEANED copy. Otherwise "the source's real texture" is the
            # source's grain, and it lands straight back on the face we just restored.
            src_up = cv2.bilateralFilter(src_up, d=7, sigmaColor=30, sigmaSpace=7)
        result = ops.reinject_texture(result, src_up, amount=params.skin_texture)

    return result, len(restored_faces or []), boxes


def _sr(registry: ModelRegistry, settings: Settings, bgr: np.ndarray, scale: int) -> np.ndarray:
    """One Real-ESRGAN call, with tile reduction on out-of-memory. Unchanged behaviour."""
    upsampler = registry.realesrgan
    attempts = 0
    while True:
        try:
            output, _ = upsampler.enhance(bgr, outscale=scale)
            return output
        except (RuntimeError, MemoryError) as exc:
            if _is_oom(exc) and attempts < settings.GPU_OOM_RETRIES:
                attempts += 1
                if registry.torch is not None:
                    try:
                        registry.torch.cuda.empty_cache()
                    except Exception:  # pragma: no cover
                        pass
                current_tile = getattr(upsampler, "tile", 0) or 512
                upsampler.tile = max(64, current_tile // 2)
                log.warning("out of memory — tile reduced to %s (retry %s)", upsampler.tile, attempts)
                continue
            if _is_oom(exc):
                raise OutOfMemory("Ran out of memory while enhancing the image.") from exc
            raise ProcessingFailed(f"Enhancement failed: {exc.__class__.__name__}") from exc


def _restore_whole(
    registry: ModelRegistry, settings: Settings, rgb: np.ndarray, params: Params,
    runner: "chunked.Runner",
) -> np.ndarray:
    """Real-ESRGAN restoration, in one call for an ordinary photo and tile by tile for a big one.

    This is the stage that actually took servers down. RealESRGANer runs its 4x network and
    only then scales the result to `outscale`, so a 24 MP photo becomes a 384 MP float tensor
    on the way through - about 4.6 GB - no matter that the answer is meant to come back the
    same size it went in. Its internal `tile` setting bounds the network's working set but not
    that output array, so it cannot save this on its own.

    Feeding it 384 px at a time bounds both: the largest thing that ever exists is one tile's
    4x tensor, and the result is written straight into the output buffer. The tiles carry a
    halo and are cross-faded, because a network's receptive field is not a number we can point
    at the way we can for a blur.
    """
    if registry.realesrgan is None:
        raise ModelsUnavailable("The enhancement model is not loaded.")

    # ALWAYS take the model's output at 2x and area-average it down to the scale we want.
    #
    # RealESRGANer runs a 4x network and then resizes its result to `outscale` with
    # INTER_LANCZOS4, whose kernel spans a fixed 8 source taps whatever the ratio - OpenCV does
    # not widen an interpolating kernel for minification; INTER_AREA is the only mode that does.
    # At outscale=2 those 8 taps are an adequate filter for a 2:1 reduction. At outscale=1 the
    # same 8 taps have to perform a 4:1 reduction: they under-integrate and fold everything
    # above the new Nyquist back into the picture, so the restoration the network just performed
    # returns as aliased mush. That is a large part of why a big photo's clothing and background
    # come back as soft as they went in while the face - restored separately from its own crop
    # by a model with a face prior - comes back sharp.
    #
    # This costs nothing. The 4x tensor is built either way; the 2x tile is a few megabytes and
    # is discarded immediately. And it removes an inconsistency rather than adding one: on the
    # un-chunked face path GFPGAN already asks this same model for outscale=2 and beautify()
    # area-downscales the result, so small photos have always had the good resampling and only
    # large ones were being handed the 4:1 collapse.
    sr_scale = max(2, int(params.effective_scale))

    def tile_fn(piece: np.ndarray, _rect) -> np.ndarray:
        bgr = cv2.cvtColor(np.ascontiguousarray(piece), cv2.COLOR_RGB2BGR)
        out = cv2.cvtColor(_sr(registry, settings, bgr, sr_scale), cv2.COLOR_BGR2RGB)
        if sr_scale != params.effective_scale:
            ph, pw = piece.shape[:2]
            out = cv2.resize(out, (int(round(pw * params.effective_scale)),
                                   int(round(ph * params.effective_scale))),
                             interpolation=cv2.INTER_AREA)
        return out

    # One lock for the whole tiled run: the DNI blend below rewrites the shared model's weights,
    # so a second job settling its own denoise strength midway through this loop would finish
    # someone else's photo with the wrong ones.
    with registry.lock:
        if registry.has_dni:
            registry.set_dni(params.denoise_strength)
        return runner.map(rgb, tile_fn, halo=runner.model_halo, feather=runner.model_halo,
                          scale=params.effective_scale, tile=runner.model_tile)


def _detect_face_boxes(registry: ModelRegistry, rgb: np.ndarray, only_center: bool,
                       max_side: int = 1280) -> List[FaceBox]:
    """Where are the faces, in full-resolution coordinates?

    GFPGAN's RetinaFace detector, run on a bounded copy of the photo. Detection does not need
    the pixels a 40 MP frame has - faces are large, structural things - so this costs the same
    on any photo and gives the boxes the crop-based restoration below works from.
    """
    gfp = registry.gfpgan
    if gfp is None:
        return []
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    small = rgb if scale >= 1.0 else cv2.resize(
        rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    try:
        with registry.lock:
            helper = gfp.face_helper
            helper.clean_all()
            helper.read_image(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
            helper.get_face_landmarks_5(only_center_face=only_center, eye_dist_threshold=5)
            dets = list(getattr(helper, "det_faces", []) or [])
    except Exception as exc:  # noqa: BLE001 - detection is a hint; never fail the job over it
        log.warning("face detection skipped: %s", exc.__class__.__name__)
        return []
    finally:
        try:
            gfp.face_helper.clean_all()
        except Exception:  # pragma: no cover
            pass

    boxes: List[FaceBox] = []
    for det in dets:
        x1, y1, x2, y2 = (float(v) / max(scale, 1e-6) for v in det[:4])
        bw, bh = int(x2 - x1), int(y2 - y1)
        if bw > 1 and bh > 1:
            boxes.append(FaceBox(max(0, int(x1)), max(0, int(y1)), bw, bh))
    return boxes


# The face-paste mask's geometry, in one place, because the crop pad and the mask built inside
# that crop have to come from the same numbers. When they do not, the mask is clipped by its own
# crop and the result is a razor-sharp restored face fading into blurry untouched pixels along a
# straight line - the rectangle users see around the restored area.
#
#   `wide`  = the face box offset up and left by 0.12 and scaled 1.24 x 1.30
#   ops.face_region_mask then draws an ellipse of half-axes 0.62 x 0.68 of THAT, feathered with a
#   Gaussian of sigma = max(6, wide.w / 9) whose float32 kernel reaches four sigma.
#
# How far the finished ellipse reaches past the FACE BOX, as a fraction of the box:
#   x  0.62*1.24 - (1.24/2 - 0.12)         = 0.2688, the same on both sides
#   y  0.68*1.30 + (1.30/2 - 0.12) - 1     = 0.4140 below the box (0.3540 above; pad symmetrically
#                                            with the larger of the two)
_WIDE_OFF, _WIDE_SX, _WIDE_SY = 0.12, 1.24, 1.30
_ELL_AX, _ELL_AY = 0.62, 0.68
_PASTE_REACH_X = _ELL_AX * _WIDE_SX - (_WIDE_SX / 2.0 - _WIDE_OFF)
_PASTE_REACH_Y = _ELL_AY * _WIDE_SY + (_WIDE_SY / 2.0 - _WIDE_OFF) - 1.0
_PASTE_SIGMA_FRAC = _WIDE_SX / 9.0      # sigma = max(6, wide.w / 9), and wide.w = 1.24 * box.w


def _restore_faces_cropped(
    registry: ModelRegistry, params: Params, background: np.ndarray, source: np.ndarray,
    boxes: List[FaceBox], out_scale: float, runner: "chunked.Runner",
) -> Tuple[int, List[FaceBox]]:
    """Restore each face from its own crop and blend it onto an already-restored background.

    On a large photo `gfp.enhance(whole_image)` is unaffordable for a reason that has nothing to
    do with faces: it upscales the entire background to 2x internally before pasting the faces
    on. The faces themselves are cheap - GFPGAN aligns each one to 512x512 regardless of how big
    the photo is - so the fix is to hand it only the faces.

    Each crop is restored on its own, put back through the same texture re-injection the
    whole-image path uses, and blended in under a feathered face mask, so what crosses the
    boundary is a soft ramp between two versions of the same skin. `background` is modified in
    place. Returns how many faces were restored and where they ended up.
    """
    gfp = registry.gfpgan
    if gfp is None or not boxes:
        return 0, []

    bh, bw = background.shape[:2]
    sh, sw = source.shape[:2]
    restored_count = 0
    placed: List[FaceBox] = []

    with registry.lock:
        saved_bg = gfp.bg_upsampler
        try:
            # No background upsampler for a crop: the crop's background is discarded anyway, and
            # running super-resolution over it would put back the cost this exists to avoid.
            gfp.bg_upsampler = None
            for box in boxes:
                runner.check()
                # Enough room that the feathered blend mask below has faded to nothing before it
                # reaches the edge of the crop, so no part of the crop's own filler background
                # can cross into the result.
                # Derived from the paste mask's own geometry, not from a guessed fraction, and
                # PER AXIS: sigma is driven by the box WIDTH while the ellipse is taller than it
                # is wide, so one scalar cannot satisfy both. The old `0.60 * max(w, h) + 16`
                # gave a 400x520 face a 1056x1176 crop while the mask reached [-21, 1229] - the
                # mask was still at about 0.12 where the crop stopped, i.e. twelve percent of a
                # razor-sharp GFPGAN crop pasted against untouched blurry pixels along a straight
                # line, on every face size measured. This is the crop that makes that impossible.
                #
                # sigma is taken at SOURCE scale, where max(6/out_scale, ...) is bounded above by
                # max(6, ...), so the pad stays conservative for any out_scale >= 1. The +2 covers
                # the int()/round() slack in `local` and `wide` below.
                sigma = max(6.0, _PASTE_SIGMA_FRAC * box.w)
                pad_x = int(np.ceil(_PASTE_REACH_X * box.w + 4.0 * sigma)) + 2
                pad_y = int(np.ceil(_PASTE_REACH_Y * box.h + 4.0 * sigma)) + 2
                sx0, sy0 = max(0, box.x - pad_x), max(0, box.y - pad_y)
                sx1, sy1 = min(sw, box.x + box.w + pad_x), min(sh, box.y + box.h + pad_y)
                if sx1 - sx0 < 32 or sy1 - sy0 < 32:
                    continue

                crop = np.ascontiguousarray(source[sy0:sy1, sx0:sx1])
                try:
                    _c, faces, out_bgr = gfp.enhance(
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), has_aligned=False,
                        only_center_face=False, paste_back=True,
                        weight=_clamp(params.face_strength),
                    )
                except (RuntimeError, MemoryError) as exc:
                    # One face failing is not the photo failing: skip it and keep the rest.
                    if _is_oom(exc):
                        log.warning("skipping one face: out of memory on its crop")
                        continue
                    raise ProcessingFailed(
                        f"Face restoration failed: {exc.__class__.__name__}") from exc
                if out_bgr is None or not faces:
                    continue

                fixed = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                # Where this crop lands on the background, at the background's scale.
                tx0, ty0 = int(round(sx0 * out_scale)), int(round(sy0 * out_scale))
                tx1 = min(bw, int(round(sx1 * out_scale)))
                ty1 = min(bh, int(round(sy1 * out_scale)))
                if tx1 - tx0 < 8 or ty1 - ty0 < 8:
                    continue
                if (fixed.shape[1], fixed.shape[0]) != (tx1 - tx0, ty1 - ty0):
                    interp = (cv2.INTER_AREA if (tx1 - tx0) < fixed.shape[1]
                              else cv2.INTER_LANCZOS4)
                    fixed = cv2.resize(fixed, (tx1 - tx0, ty1 - ty0), interpolation=interp)

                local = FaceBox(int(round((box.x - sx0) * out_scale)),
                                int(round((box.y - sy0) * out_scale)),
                                max(1, int(round(box.w * out_scale))),
                                max(1, int(round(box.h * out_scale))))

                if params.skin_texture > 0.02:
                    src_up = cv2.resize(crop, (fixed.shape[1], fixed.shape[0]),
                                        interpolation=cv2.INTER_LANCZOS4)
                    if params.source_noise > 0.12:
                        src_up = cv2.bilateralFilter(src_up, d=7, sigmaColor=30, sigmaSpace=7)
                    fixed = ops.reinject_texture(fixed, src_up, amount=params.skin_texture)

                # A little wider than the mask the later stages use, so the blend sits on the
                # jaw and the hairline rather than across the middle of a cheek.
                wide = FaceBox(int(local.x - local.w * _WIDE_OFF),
                               int(local.y - local.h * _WIDE_OFF),
                               int(local.w * _WIDE_SX), int(local.h * _WIDE_SY))
                mask = np.clip(ops.face_region_mask(fixed.shape, [wide]), 0.0, 1.0)
                # Second guarantee, independent of the pad above. Fade the write to nothing at any
                # crop edge INTERIOR to the frame; an edge that coincides with the image border
                # needs nothing, because there is no untouched pixel on the far side of it. Width
                # is 1.5 feather sigma, floored at 32 px and capped at a quarter of the crop - the
                # pad is reach + 4 sigma, so this only ever touches pixels where the mask is below
                # 0.6% of full strength.
                #
                # The rect is given in SOURCE coordinates on purpose: that is where the crop was
                # actually clipped against the frame. Mixing scales is harmless because rect and
                # full_shape drive only the four "is this edge the border" comparisons.
                n = int(min(max(32.0, 1.5 * max(6.0, wide.w / 9.0)),
                            (ty1 - ty0) // 4, (tx1 - tx0) // 4))
                m = (mask * chunked.border_taper(fixed.shape, (sy0, sx0, sy1, sx1),
                                                 (sh, sw), n))[..., None]
                target = background[ty0:ty1, tx0:tx1]
                np.copyto(target, np.clip(target.astype(np.float32) * (1.0 - m)
                                          + fixed.astype(np.float32) * m,
                                          0, 255).round().astype(np.uint8))

                restored_count += 1
                placed.append(FaceBox(tx0 + local.x, ty0 + local.y, local.w, local.h))
        finally:
            gfp.bg_upsampler = saved_bg
            try:
                gfp.face_helper.clean_all()
            except Exception:  # pragma: no cover
                pass
    return restored_count, placed


def _mock_restore(rgb: np.ndarray, params: Params, runner: "chunked.Runner") -> np.ndarray:
    """No models: Lanczos upscale + a subtle finish. NOT AI — wiring/demo only."""
    s = params.effective_scale
    # Measured on a sample scaled the way the tiles will be, since gradients change with scale.
    probe = np.vstack([cv2.resize(w, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
                       for w in chunked.windows(rgb)]) if s > 1 else rgb
    hi = chunked.sample_edge_hi(probe) if runner.chunked else None

    def tile_fn(piece: np.ndarray, _rect) -> np.ndarray:
        ph, pw = piece.shape[:2]
        up = (cv2.resize(piece, (pw * s, ph * s), interpolation=cv2.INTER_LANCZOS4)
              if s > 1 else piece.copy())
        return ops.edge_aware_sharpen(up, strength=min(0.35, params.detail_strength), edge_hi=hi)

    return runner.map(rgb, tile_fn, halo=32, scale=s)


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
def _boxes_from(raw: List[Tuple[int, int, int, int]]) -> List[FaceBox]:
    return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in raw]


def _apply_look(rgb: np.ndarray, look: filters.Look, boxes: List[FaceBox],
                runner: "chunked.Runner") -> np.ndarray:
    """Run one look over the image, whole or tiled, with its whole-frame numbers measured once.

    The face half runs over the face region and the frame-wide half over the frame, and both
    are handed the same measurements no matter how many pieces they are executed in - the LUT
    is built once, the skin reference is this photo's own median skin tone, and the local
    contrast comes from a single sampled gain map. That is what keeps skin tone, brightness and
    filter strength identical across a boundary rather than merely close.
    """
    if look.id == filters.NO_LOOK:
        return rgb
    skin_ref = filters.skin_reference(chunked.proxy(rgb))

    if boxes:
        rgb = runner.region(
            rgb, boxes,
            lambda piece, bx, mask: filters.apply_face(piece, look, bx, skin_ref, mask),
            pad_frac=0.75, mask_fn=ops.face_region_mask, halo=64)

    lut = filters.tone_lut(look)
    clahe = (chunked.ClaheField(rgb, 1.0 + 2.2 * look.clarity,
                                pre_curve=filters.luma_curve_lut(look))
             if runner.chunked and look.clarity > 0.01 else None)
    shape = rgb.shape
    return runner.map(
        rgb,
        lambda piece, rect: filters.apply_global(
            piece, look, lut, (lambda l: clahe.apply(l, rect)) if clahe else None,
            rect, shape),
        halo=8,
    )


def beautify(
    registry: ModelRegistry,
    settings: Settings,
    decoded: DecodedImage,
    out_path_template: str,
    progress: Optional[ProgressFn] = None,
    mode: str = MODE_BEAUTIFY,
    look_id: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[str, BeautifyResult]:
    """Enhance one decoded image in `beautify` or `clear` mode, then apply a look.

    Returns (output path, result metadata).

    Every heavy stage below goes through `runner`, which is a straight passthrough for an
    ordinary photo and splits the work into tiles for a large one. Nothing about what the
    pipeline computes changes with the size of the photo; only how much of it is resident at
    any moment does.
    """
    started = time.perf_counter()
    timer = _Timer()
    report: ProgressFn = progress or (lambda *_: None)
    # MOCK_MODE is an explicit developer opt-in, and the ONLY way to get the resize path.
    #
    # Missing models used to fall through to it silently. That is how a deployment with no
    # weights on disk kept accepting photos, showing progress and reporting success while
    # returning nothing but a Lanczos upscale - the user is told their photo was enhanced when
    # it was not. Refusing the job is the honest outcome, and the failure is loud and specific.
    is_mock = settings.MOCK_MODE
    if not is_mock and not registry.status.ready:
        raise ModelsUnavailable(
            "The AI engine is not available on this server, so the photo was not changed. "
            "The enhancement models are not loaded."
        )
    models: List[str] = []
    warnings: List[str] = []
    processed_faces = 0
    fallback_used = False
    face_boxes: List[FaceBox] = []

    rgb = decoded.rgb
    runner = chunked.Runner(settings, rgb.shape, progress=report, should_stop=should_stop)
    look = filters.resolve(look_id, mode)
    if decoded.downscaled:
        warnings.append("input_resized_to_budget")
    if runner.chunked:
        log.info("chunked processing: %.1f MP, %s px tiles", rgb.shape[0] * rgb.shape[1] / 1e6,
                 runner.tile)

    # ---- analyse -----------------------------------------------------------------------
    report("analysing", 12, "Analysing the photo")
    with timer.stage("analyse"):
        # On a large photo the measurements come from a sample rather than the whole frame: a
        # Laplacian variance over 40 MP in float64 is 320 MB to learn a number that a megapixel
        # already tells us. See chunked.analysis_sample for why it is two kinds of sample.
        sample = chunked.analysis_sample(rgb) if runner.chunked else None
        analysis = analyse(rgb, has_alpha=decoded.had_alpha, sampled=sample)

    # ---- exposure rescue, before anything else looks at the photo ----------------------
    # Strictly gated on measured under-exposure, so a normally-exposed photo is not touched.
    exposure_lift = 0.0
    if not is_mock and analysis.low_light_score > LOW_LIGHT_THRESHOLD:
        runner.stage("exposure", 14, 20, "Correcting the exposure")
        with timer.stage("exposure"):
            # Measured once over the whole frame; applied identically to every tile. Left to
            # measure for itself, a tile of shadow and a tile of window would each be dragged
            # to the same mid-tone and the photo would come back in patches.
            plan = ops.plan_exposure(chunked.proxy(rgb)) if runner.chunked else None
            clahe = chunked.ClaheField(rgb, 1.8) if runner.chunked else None
            lifts: List[float] = []

            def _expose(piece, rect):
                out, lift = ops.auto_exposure(
                    piece, plan=plan,
                    local_contrast=(lambda l: clahe.apply(l, rect)) if clahe else None)
                lifts.append(lift)
                return out

            if runner.chunked and plan is None:
                exposure_lift = 0.0            # nothing to correct; skip the pass entirely
            else:
                rgb = runner.map(rgb, _expose, halo=8)
                exposure_lift = max(lifts) if lifts else 0.0
        if exposure_lift > 0:
            models.append("exposure")
            # Every measurement taken on a dark frame is misleading: grain hides in the
            # shadows and reads as clean, soft edges read as blur. Measure again now that the
            # photo is properly exposed, so the restoration is planned on what is really there.
            with timer.stage("analyse"):
                sample = chunked.analysis_sample(rgb) if runner.chunked else None
                analysis = analyse(rgb, has_alpha=decoded.had_alpha, sampled=sample)

    params = build_params(analysis, settings, mode, exposure_lift, cheap_faces=runner.chunked)
    warnings.extend(params.warnings)
    log.info(
        "plan: mode=%s look=%s strategy=%s scale=%sx faces=%s face_restore=%s quality=%s "
        "noise=%.2f skin_tex=%.2f skin_clean=%.2f chunked=%s",
        params.mode, look.id, params.strategy, params.effective_scale, analysis.face_count,
        params.face_restore, analysis.quality_category, analysis.noise_score,
        params.skin_texture, params.skin_clean, runner.chunked,
    )

    if is_mock:
        report("enhancing", 45, "Enhancing")
        with timer.stage("mock"):
            restored = _mock_restore(rgb, params, runner)
        models.append("mock-resize")
    else:
        working = rgb

        # De-block a heavily compressed input BEFORE super-resolution, so the model does not
        # amplify the 8x8 blocks into permanent structure.
        if analysis.jpeg_artifact_score > 0.45:
            runner.stage("deblock", 20, 26, "Cleaning compression artefacts")
            with timer.stage("deblock"):
                hi = chunked.sample_edge_hi(working) if runner.chunked else None
                working = runner.map(
                    working,
                    lambda piece, _r: ops.denoise(piece, strength=0.2, noise_score=0.0,
                                                  jpeg_score=analysis.jpeg_artifact_score,
                                                  edge_hi=hi),
                    halo=32)
            models.append("deblock")

        target_w = int(round(working.shape[1] * params.effective_scale))
        target_h = int(round(working.shape[0] * params.effective_scale))
        restored = None

        if runner.chunked:
            # Large photo: the background is restored tile by tile and the faces are restored
            # from their own crops, which is the same two jobs GFPGAN does internally in one
            # call - just without ever holding the whole 2x background it would build to do it.
            runner.stage("enhancing", 26, 66, "Enhancing detail")
            with timer.stage("realesrgan"):
                restored = _restore_whole(registry, settings, working, params, runner)
            models.append("realesrgan:general")

            if params.face_restore and registry.gfpgan is not None:
                runner.stage("restoring_faces", 66, 76, "Restoring faces")
                with timer.stage("gfpgan"):
                    detected = _detect_face_boxes(registry, working, params.only_center_face)
                    if detected:
                        processed_faces, face_boxes = _restore_faces_cropped(
                            registry, params, restored, working, detected,
                            params.effective_scale, runner)
                if processed_faces:
                    models.append("gfpgan")
                elif detected:
                    # Restoration produced nothing, but RetinaFace still knows where the faces
                    # are, and that is better information than Haar for the stages that follow.
                    k = params.effective_scale
                    face_boxes = [FaceBox(b.x * k, b.y * k, b.w * k, b.h * k) for b in detected]
        else:
            # Face path: GFPGAN restores faces AND upscales the background in one pass.
            if params.face_restore and registry.gfpgan is not None:
                report("restoring_faces", 45, "Restoring faces")
                with timer.stage("gfpgan"):
                    restored2x, processed_faces, model_faces = _restore_faces(
                        registry, params, working)
                if restored2x is not None:
                    models += ["realesrgan:general", "gfpgan"]
                    report("blending", 70, "Blending faces")
                    # GFPGAN always outputs 2x - resize to the scale we actually want.
                    if (restored2x.shape[1], restored2x.shape[0]) != (target_w, target_h):
                        interp = (cv2.INTER_AREA if target_w < restored2x.shape[1]
                                  else cv2.INTER_LANCZOS4)
                        k = target_w / float(max(1, restored2x.shape[1]))
                        restored = cv2.resize(restored2x, (target_w, target_h), interpolation=interp)
                        model_faces = [FaceBox(int(b.x * k), int(b.y * k), int(b.w * k), int(b.h * k))
                                       for b in model_faces]
                    else:
                        restored = restored2x
                    face_boxes = model_faces or face_boxes

            # Everything else (and the fallback when GFPGAN produced nothing).
            if restored is None:
                report("enhancing", 50, "Enhancing detail")
                with timer.stage("realesrgan"):
                    restored = _restore_whole(registry, settings, working, params, runner)
                models.append("realesrgan:general")

        # Denoise a result that is still grainy. The old threshold of 0.4 was far above what
        # heavy visible grain actually measures (~0.25), so the noisiest photos - the ones that
        # need it most - were sailing straight past this.
        if params.chroma_clean > 0:
            runner.stage("chroma", 76, 78, "Cleaning colour noise")
            with timer.stage("chroma"):
                # The reduction factor is fixed from the whole frame: derived per tile, the
                # narrow tiles at the right edge would be cleaned harder than the rest.
                factor = max(2, int(round(2 + 4 * min(1.0, params.chroma_clean))))
                restored = runner.map(
                    restored,
                    lambda piece, _r: ops.chroma_cleanup(piece, params.chroma_clean, factor=factor),
                    halo=48)   # the chroma blur runs at 1/factor scale, so its reach scales too
            models.append("chroma-clean")

        if params.source_noise > 0.15:
            runner.stage("denoise", 78, 82, "Removing grain")
            with timer.stage("denoise"):
                hi = chunked.sample_edge_hi(restored) if runner.chunked else None
                restored = runner.map(
                    restored,
                    lambda piece, _r: ops.denoise(
                        piece, strength=params.denoise_strength * POST_DENOISE_SCALE,
                        noise_score=params.source_noise, jpeg_score=0.0, protect_edges=True,
                        edge_hi=hi),
                    halo=32)
            models.append("denoise")

    out_scale = restored.shape[1] / float(max(1, rgb.shape[1]))
    # Prefer the face model's own boxes (already in output coordinates); fall back to scaling
    # the Haar boxes when the face path did not run.
    if not face_boxes and analysis.faces:
        face_boxes = [FaceBox(int(b.x * out_scale), int(b.y * out_scale),
                              int(b.w * out_scale), int(b.h * out_scale)) for b in analysis.faces]

    # ---- clarity inside the face: eyes, lashes, brows, lips -----------------------------
    # The final detail pass protects skin, which is right for the photo as a whole but leaves
    # the face - the part everyone actually looks at - the softest thing in the frame. This
    # pass sharpens the face WITHOUT that protection; edge confidence still keeps flat skin
    # from being crunched, so what gets crisper is structure, not pores.
    face_field = (chunked.mask_field(ops.face_region_mask, restored.shape, face_boxes)
                 if runner.chunked and face_boxes else None)
    face_mask = (ops.face_region_mask(restored.shape, face_boxes)
                 if face_boxes and not runner.chunked else None)

    if not is_mock and params.face_clarity > 0 and face_boxes:
        runner.stage("face_clarity", 82, 84, "Refining faces")
        with timer.stage("face_clarity"):
            hi = chunked.sample_edge_hi(restored) if runner.chunked else None
            restored = runner.region(
                restored, face_boxes,
                lambda piece, bx, mask: ops.face_clarity(piece, bx, params.face_clarity,
                                                         mask=mask, edge_hi=hi),
                pad_frac=0.75, mask_fn=ops.face_region_mask)
        models.append("face-clarity")

    # ---- clarity for everything else: hands, clothing, objects, background --------------
    if not is_mock and params.body_clarity > 0:
        runner.stage("body_clarity", 84, 87, "Refining detail")
        with timer.stage("body_clarity"):
            hi = chunked.sample_edge_hi(restored) if runner.chunked else None
            source = restored
            # The halo must still cover every filter this stage reaches with. The wide Gaussian
            # needs 3*sigma and the edge-confidence map needs its 31x31 window plus a 1 px blur
            # (18 px); 32 covers both at soft_radius 4.2, and the formula follows the radius up
            # if effective_scale ever doubles it. At soft_radius 0 it is exactly 32, as before,
            # so the seam guarantee is unchanged for every photo that does not use this.
            body_halo = max(32, int(3.0 * params.soft_radius) + 20)

            def _body(gain: float) -> np.ndarray:
                return runner.map(
                    source,
                    lambda piece, rect: ops.body_clarity(
                        piece, face_field.crop(rect) if face_field else face_mask,
                        params.body_clarity, edge_hi=hi,
                        soft_radius=params.soft_radius, soft_gain=gain),
                    halo=body_halo)

            lifted = _body(params.soft_gain)
            # The safeguard at the end of the pipeline cannot see this stage. ops.check_result
            # counts pixels that moved by more than 60 levels, and both terms here are clamped
            # far below that (12 * 0.959 narrow + 6 * soft_gain wide, about 22 levels at the
            # gain a soft portrait produces) - so it would pass this pass whatever it did. The
            # bound that actually holds is the clamp; this check is the same test taken at the
            # scale this stage can reach, and it exists so the one photo that saturates both
            # terms across the frame gets pulled back instead of shipping a halo.
            if params.soft_gain > 0.02 and ops.overshoot_ratio(
                    source, lifted, threshold=18.0) > settings.SHARPNESS_SAFETY_LIMIT:
                warnings.append("soft_recovery_reduced")
                del lifted
                lifted = _body(params.soft_gain * 0.4)
            restored = lifted
            del source
        models.append("body-clarity")

    # ---- hair / fine texture around heads ----------------------------------------------
    if not is_mock and params.hair_refine > 0 and face_boxes:
        runner.stage("hair", 87, 89, "Refining hair")
        with timer.stage("hair"):
            hi = chunked.sample_edge_hi(restored) if runner.chunked else None
            restored = runner.region(
                restored, face_boxes,
                lambda piece, bx, mask: ops.refine_hair(piece, bx, params.hair_refine,
                                                        out_scale=1.0, mask=mask, edge_hi=hi),
                pad_frac=1.4, mask_fn=ops.hair_region_mask)
        models.append("hair-refine")

    # ---- skin cleanup (beautify only) ---------------------------------------------------
    if not is_mock and params.skin_clean > 0 and face_boxes:
        runner.stage("skin_clean", 89, 90, "Evening out skin")
        with timer.stage("skin_clean"):
            restored = runner.region(
                restored, face_boxes,
                lambda piece, bx, mask: ops.skin_clean(piece, bx, params.skin_clean, mask=mask),
                pad_frac=0.75, mask_fn=ops.face_region_mask)
        models.append("skin-clean")

    if not is_mock and params.glow > 0 and face_boxes:
        with timer.stage("glow"):
            restored = runner.region(
                restored, face_boxes,
                lambda piece, bx, mask: ops.soft_glow(piece, bx, params.glow, mask=mask),
                pad_frac=0.9, mask_fn=ops.face_region_mask)
        models.append("glow")

    # ---- the photographic finish (this is the "beautify") -------------------------------
    runner.stage("finishing", 90, 93, "Finishing")
    with timer.stage("finish"):
        # White balance and the closing CLAHE are the two whole-frame measurements in here;
        # both are taken once and handed to every tile. See ops.premium_finish.
        wb = chunked.channel_means(restored) if runner.chunked else None
        clahe = (chunked.ClaheField(restored, 1.0 + 0.6 * params.tone_depth,
                                    pre_curve=ops.tone_curve_lut(params.tone_depth))
                 if runner.chunked else None)
        restored = runner.map(
            restored,
            lambda piece, rect: ops.premium_finish(
                piece, tone_depth=params.tone_depth, is_grayscale=analysis.is_grayscale,
                wb_means=wb,
                local_contrast=(lambda l: clahe.apply(l, rect)) if clahe else None),
            halo=8)
    models.append("photo-finish")

    # ---- final edge-aware detail + one bounded fallback ---------------------------------
    reference = restored
    runner.stage("detail", 93, 95, "Sharpening")
    with timer.stage("detail"):
        hi = chunked.sample_edge_hi(restored) if runner.chunked else None
        detailed = runner.map(
            restored,
            lambda piece, rect: ops.edge_aware_sharpen(
                piece, strength=params.detail_strength,
                skin_guard_mask=face_field.crop(rect) if face_field else face_mask, edge_hi=hi),
            halo=32)
    safety = ops.check_result(reference, detailed, settings.SHARPNESS_SAFETY_LIMIT)
    if safety.ok:
        restored = detailed
    else:
        # One controlled retry at reduced strength — never a loop.
        fallback_used = True
        warnings.extend(safety.warnings)
        del detailed
        with timer.stage("detail_fallback"):
            restored = runner.map(
                restored,
                lambda piece, _r: ops.edge_aware_sharpen(
                    piece, strength=params.detail_strength * 0.4, edge_hi=hi),
                halo=32)

    # ---- keep the un-styled result, then apply the look ---------------------------------
    # The base is written before the look so the look stays reversible: switching to another
    # one, or removing it, re-renders from here and never touches the models again. Without it
    # "remove the filter" would mean re-running the whole pipeline.
    check_array(restored, settings.MAX_OUTPUT_PIXELS, settings.MAX_OUTPUT_SIDE)
    out_fmt = decoded.detected_format if decoded.detected_format in FORMAT_MIME else "jpeg"
    out_path = out_path_template.format(ext=FORMAT_EXT[out_fmt])
    alpha = None
    if decoded.alpha is not None and out_fmt in ("png", "webp"):
        oh, ow = restored.shape[:2]
        alpha = cv2.resize(decoded.alpha, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

    base_path = None
    if settings.KEEP_UNFILTERED_BASE:
        report("encoding", 95, "Saving")
        with timer.stage("encode_base"):
            # Derived by suffixing the output path, so it can never collide with it whatever
            # template the caller passed - overwriting the result with the base would be a
            # silent, total loss of the look.
            root, ext = os.path.splitext(out_path)
            base_path = f"{root}-base{ext}"
            encode(restored, out_fmt, settings.BASE_QUALITY, base_path, alpha=alpha)

    if look.id != filters.NO_LOOK:
        runner.stage("styling", 96, 98, f"Applying {look.name}")
        with timer.stage("look"):
            restored = _apply_look(restored, look, face_boxes, runner)
        models.append(f"look:{look.id}")

    # ---- validate + encode -------------------------------------------------------------
    report("encoding", 98, "Saving")
    check_array(restored, settings.MAX_OUTPUT_PIXELS, settings.MAX_OUTPUT_SIDE)

    # Keep the user's format. Alpha survives for PNG/WebP.
    with timer.stage("encode"):
        encoded = encode(restored, out_fmt, settings.OUTPUT_QUALITY, out_path, alpha=alpha)
        verify_encoded(encoded.path, FORMAT_MIME[out_fmt])

    report("completed", 100, "Done")
    return encoded.path, BeautifyResult(
        mode=params.mode,
        look=look.id,
        width=encoded.width,
        height=encoded.height,
        original_width=decoded.width,
        original_height=decoded.height,
        mime_type=encoded.mime_type,
        bytes=encoded.bytes,
        face_count=analysis.face_count,
        processed_faces=processed_faces,
        models=models,
        strategy=("mock" if is_mock else params.strategy),
        quality_category=analysis.quality_category,
        effective_scale=params.effective_scale,
        processing_time_ms=int((time.perf_counter() - started) * 1000),
        stage_timings_ms=timer.stages,
        warnings=list(dict.fromkeys(warnings)),
        fallback_used=fallback_used,
        chunked=runner.chunked,
        base_path=base_path,
        base_format=out_fmt,
        face_boxes=[(b.x, b.y, b.w, b.h) for b in face_boxes],
    )


# --------------------------------------------------------------------------------------
# re-styling an already-enhanced photo
# --------------------------------------------------------------------------------------
def restyle(
    settings: Settings,
    result: BeautifyResult,
    out_path_template: str,
    look_id: Optional[str],
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[str, BeautifyResult]:
    """Put a different look on a finished photo - or take the look off entirely.

    This is what makes the default look something the user is offered rather than something
    they are stuck with. It reads the un-styled base saved during the job, applies the
    requested look to it and re-encodes; the models, the analysis and the restoration are all
    skipped, so changing your mind costs a fraction of a second rather than another full run.

    It is chunked on the same terms as everything else, so switching looks on a 40 MP photo is
    no more of a load than switching them on a small one.
    """
    if not result.base_path or not os.path.exists(result.base_path):
        raise ProcessingFailed("The un-styled version of this photo is no longer available.")

    started = time.perf_counter()
    report: ProgressFn = progress or (lambda *_: None)
    look = filters.resolve(look_id, result.mode)
    report("styling", 20, f"Applying {look.name}")

    fmt = result.base_format if result.base_format in FORMAT_MIME else "jpeg"
    with Image.open(result.base_path) as img:
        alpha = (np.asarray(img.convert("RGBA").split()[-1], dtype=np.uint8)
                 if img.mode in ("RGBA", "LA") and fmt in ("png", "webp") else None)
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

    runner = chunked.Runner(settings, rgb.shape, progress=report, should_stop=should_stop)
    runner.stage("styling", 25, 90, f"Applying {look.name}")
    rgb = _apply_look(rgb, look, _boxes_from(result.face_boxes), runner)

    report("encoding", 92, "Saving")
    check_array(rgb, settings.MAX_OUTPUT_PIXELS, settings.MAX_OUTPUT_SIDE)
    out_path = out_path_template.format(ext=FORMAT_EXT[fmt])
    encoded = encode(rgb, fmt, settings.OUTPUT_QUALITY, out_path, alpha=alpha)
    verify_encoded(encoded.path, FORMAT_MIME[fmt])

    report("completed", 100, "Done")
    updated = replace(
        result,
        look=look.id,
        width=encoded.width,
        height=encoded.height,
        mime_type=encoded.mime_type,
        bytes=encoded.bytes,
        processing_time_ms=int((time.perf_counter() - started) * 1000),
        stage_timings_ms={"look": int((time.perf_counter() - started) * 1000)},
        models=[m for m in result.models if not m.startswith("look:")]
               + ([f"look:{look.id}"] if look.id != filters.NO_LOOK else []),
    )
    return encoded.path, updated
