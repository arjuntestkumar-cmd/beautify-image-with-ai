"""The single enhancement pipeline: Beautify.

One mode. No presets to choose, no filters, no manual sliders. The pipeline still *adapts* to
each photo — that adaptation is what the original service's mode presets were for, and it is
kept here in full — but the user never has to think about it.

    decode -> analyse -> plan
      -> de-block heavily compressed input
      -> GFPGAN face restoration on a Real-ESRGAN background   (photos with faces)
         or plain Real-ESRGAN restoration                      (everything else)
      -> chroma denoise if the result is still noisy
      -> hair / fine-texture refinement around detected heads
      -> photographic finish (white balance, highlights, shadows, midtones, vibrance)
      -> edge-aware detail + over-sharpening safeguard
      -> quality checks -> encode

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
document mode, the 21 filters and the manual adjustment layer.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..analysis import Analysis, FaceBox, HIGH, LOW, MEDIUM, SEVERELY_DEGRADED, analyse
from ..config import Settings
from ..errors import ModelsUnavailable, OutOfMemory, ProcessingFailed
from ..logging_utils import get_logger
from ..validation import DecodedImage
from . import ops
from .encode import FORMAT_MIME, check_array, encode, verify_encoded
from .registry import ModelRegistry

log = get_logger("beautify")

# Progress callback: (stage, percent, human message)
ProgressFn = Callable[[str, int, str], None]

# --------------------------------------------------------------------------------------
# The one preset. Ported from the original service's `beautify` ModePreset + ModeSpec.
# --------------------------------------------------------------------------------------
FACE_STRENGTH_CAP = 0.58    # upper bound on the GFPGAN weight (ignored by the v1.4 clean arch)
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
    warnings: List[str] = field(default_factory=list)


@dataclass
class BeautifyResult:
    mode: str
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
                 exposure_lift: float = 0.0) -> Params:
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
    face_restore = not analysis.screenshot_like and (effective_scale == 2 or analysis.face_count > 0)

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
        warnings=warnings,
    )


# --------------------------------------------------------------------------------------
# model stages
# --------------------------------------------------------------------------------------
def _is_oom(exc: Exception) -> bool:
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

    while True:
        try:
            _cropped, restored_faces, restored_bgr = gfp.enhance(
                bgr, has_aligned=False, only_center_face=params.only_center_face,
                paste_back=True, weight=weight,
            )
            break
        except RuntimeError as exc:
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
            raise ProcessingFailed(f"Face restoration failed: {exc.__class__.__name__}") from exc

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


def _restore_whole(
    registry: ModelRegistry, settings: Settings, rgb: np.ndarray, params: Params
) -> np.ndarray:
    """Whole-image Real-ESRGAN restoration, with tile reduction on out-of-memory."""
    upsampler = registry.realesrgan
    if upsampler is None:
        raise ModelsUnavailable("The enhancement model is not loaded.")

    if registry.has_dni:
        registry.set_dni(params.denoise_strength)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    attempts = 0

    while True:
        try:
            output, _ = upsampler.enhance(bgr, outscale=params.effective_scale)
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        except RuntimeError as exc:
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


def _mock_restore(rgb: np.ndarray, params: Params) -> np.ndarray:
    """No models: Lanczos upscale + a subtle finish. NOT AI — wiring/demo only."""
    h, w = rgb.shape[:2]
    s = params.effective_scale
    up = cv2.resize(rgb, (w * s, h * s), interpolation=cv2.INTER_LANCZOS4) if s > 1 else rgb.copy()
    return ops.edge_aware_sharpen(up, strength=min(0.35, params.detail_strength))


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
def beautify(
    registry: ModelRegistry,
    settings: Settings,
    decoded: DecodedImage,
    out_path_template: str,
    progress: Optional[ProgressFn] = None,
    mode: str = MODE_BEAUTIFY,
) -> Tuple[str, BeautifyResult]:
    """Enhance one decoded image in `beautify` or `clear` mode.

    Returns (output path, result metadata).
    """
    started = time.perf_counter()
    timer = _Timer()
    report: ProgressFn = progress or (lambda *_: None)
    is_mock = settings.MOCK_MODE or not registry.status.ready
    models: List[str] = []
    warnings: List[str] = []
    processed_faces = 0
    fallback_used = False
    face_boxes: List[FaceBox] = []

    rgb = decoded.rgb

    # ---- analyse -----------------------------------------------------------------------
    report("analysing", 15, "Analysing the photo")
    with timer.stage("analyse"):
        analysis = analyse(rgb, has_alpha=decoded.had_alpha)

    # ---- exposure rescue, before anything else looks at the photo ----------------------
    # Strictly gated on measured under-exposure, so a normally-exposed photo is not touched.
    exposure_lift = 0.0
    if not is_mock and analysis.low_light_score > LOW_LIGHT_THRESHOLD:
        with timer.stage("exposure"):
            rgb, exposure_lift = ops.auto_exposure(rgb)
        if exposure_lift > 0:
            models.append("exposure")
            # Every measurement taken on a dark frame is misleading: grain hides in the
            # shadows and reads as clean, soft edges read as blur. Measure again now that the
            # photo is properly exposed, so the restoration is planned on what is really there.
            with timer.stage("analyse"):
                analysis = analyse(rgb, has_alpha=decoded.had_alpha)

    params = build_params(analysis, settings, mode, exposure_lift)
    warnings.extend(params.warnings)
    log.info(
        "plan: mode=%s strategy=%s scale=%sx faces=%s face_restore=%s quality=%s noise=%.2f "
        "skin_tex=%.2f skin_clean=%.2f",
        params.mode, params.strategy, params.effective_scale, analysis.face_count,
        params.face_restore, analysis.quality_category, analysis.noise_score,
        params.skin_texture, params.skin_clean,
    )

    if is_mock:
        report("enhancing", 45, "Enhancing")
        with timer.stage("mock"):
            restored = _mock_restore(rgb, params)
        models.append("mock-resize")
    else:
        working = rgb

        # De-block a heavily compressed input BEFORE super-resolution, so the model does not
        # amplify the 8x8 blocks into permanent structure.
        if analysis.jpeg_artifact_score > 0.45:
            with timer.stage("deblock"):
                working = ops.denoise(
                    working, strength=0.2, noise_score=0.0, jpeg_score=analysis.jpeg_artifact_score
                )
            models.append("deblock")

        target_w = int(round(working.shape[1] * params.effective_scale))
        target_h = int(round(working.shape[0] * params.effective_scale))
        restored = None

        # Face path: GFPGAN restores faces AND upscales the background in one pass.
        if params.face_restore and registry.gfpgan is not None:
            report("restoring_faces", 45, "Restoring faces")
            with timer.stage("gfpgan"):
                restored2x, processed_faces, model_faces = _restore_faces(registry, params, working)
            if restored2x is not None:
                models += ["realesrgan:general", "gfpgan"]
                report("blending", 70, "Blending faces")
                # GFPGAN always outputs 2x - resize to the scale we actually want.
                if (restored2x.shape[1], restored2x.shape[0]) != (target_w, target_h):
                    interp = cv2.INTER_AREA if target_w < restored2x.shape[1] else cv2.INTER_LANCZOS4
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
                restored = _restore_whole(registry, settings, working, params)
            models.append("realesrgan:general")

        # Denoise a result that is still grainy. The old threshold of 0.4 was far above what
        # heavy visible grain actually measures (~0.25), so the noisiest photos - the ones that
        # need it most - were sailing straight past this.
        if params.chroma_clean > 0:
            with timer.stage("chroma"):
                restored = ops.chroma_cleanup(restored, params.chroma_clean)
            models.append("chroma-clean")

        if params.source_noise > 0.15:
            with timer.stage("denoise"):
                restored = ops.denoise(
                    restored, strength=params.denoise_strength * POST_DENOISE_SCALE,
                    noise_score=params.source_noise, jpeg_score=0.0, protect_edges=True,
                )
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
    face_mask = ops.face_region_mask(restored.shape, face_boxes) if face_boxes else None

    if not is_mock and params.face_clarity > 0 and face_boxes:
        with timer.stage("face_clarity"):
            restored = ops.face_clarity(restored, face_boxes, params.face_clarity)
        models.append("face-clarity")

    # ---- clarity for everything else: hands, clothing, objects, background --------------
    if not is_mock and params.body_clarity > 0:
        with timer.stage("body_clarity"):
            restored = ops.body_clarity(restored, face_mask, params.body_clarity)
        models.append("body-clarity")

    # ---- hair / fine texture around heads ----------------------------------------------
    if not is_mock and params.hair_refine > 0 and face_boxes:
        with timer.stage("hair"):
            restored = ops.refine_hair(restored, face_boxes, params.hair_refine, out_scale=1.0)
        models.append("hair-refine")

    # ---- skin cleanup (beautify only) ---------------------------------------------------
    if not is_mock and params.skin_clean > 0 and face_boxes:
        with timer.stage("skin_clean"):
            restored = ops.skin_clean(restored, face_boxes, params.skin_clean)
        models.append("skin-clean")

    if not is_mock and params.glow > 0 and face_boxes:
        with timer.stage("glow"):
            restored = ops.soft_glow(restored, face_boxes, params.glow)
        models.append("glow")

    # ---- the photographic finish (this is the "beautify") -------------------------------
    report("finishing", 82, "Finishing")
    with timer.stage("finish"):
        restored = ops.premium_finish(restored, tone_depth=params.tone_depth, is_grayscale=analysis.is_grayscale)
    models.append("photo-finish")

    # ---- final edge-aware detail + one bounded fallback ---------------------------------
    reference = restored
    with timer.stage("detail"):
        detailed = ops.edge_aware_sharpen(restored, strength=params.detail_strength,
                                          skin_guard_mask=face_mask)
    safety = ops.check_result(reference, detailed, settings.SHARPNESS_SAFETY_LIMIT)
    if safety.ok:
        restored = detailed
    else:
        # One controlled retry at reduced strength — never a loop.
        fallback_used = True
        warnings.extend(safety.warnings)
        with timer.stage("detail_fallback"):
            restored = ops.edge_aware_sharpen(restored, strength=params.detail_strength * 0.4)

    # ---- validate + encode -------------------------------------------------------------
    report("encoding", 92, "Saving")
    check_array(restored, settings.MAX_OUTPUT_PIXELS, settings.MAX_OUTPUT_SIDE)

    # Keep the user's format. Alpha survives for PNG/WebP.
    out_fmt = decoded.detected_format if decoded.detected_format in FORMAT_MIME else "jpeg"
    out_path = out_path_template.format(ext={"jpeg": "jpg", "png": "png", "webp": "webp"}[out_fmt])
    alpha = None
    if decoded.alpha is not None and out_fmt in ("png", "webp"):
        oh, ow = restored.shape[:2]
        alpha = cv2.resize(decoded.alpha, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
    with timer.stage("encode"):
        encoded = encode(restored, out_fmt, settings.OUTPUT_QUALITY, out_path, alpha=alpha)
        verify_encoded(encoded.path, FORMAT_MIME[out_fmt])

    report("completed", 100, "Done")
    return encoded.path, BeautifyResult(
        mode=params.mode,
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
    )
