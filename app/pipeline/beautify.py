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

  * face strength is capped LOW (0.58) — high face-model weight is exactly what produces the
    waxy, re-drawn "AI face". Beauty comes from tone, hair and eyes, not from repainting a face.
  * the source's real skin micro-texture is re-injected after restoration (anti-waxy).
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

from ..analysis import Analysis, HIGH, LOW, MEDIUM, SEVERELY_DEGRADED, analyse
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
FACE_STRENGTH_CAP = 0.58    # upper bound on the GFPGAN weight
DETAIL_CAP = 0.58           # upper bound on the final edge-aware sharpen
HAIR_REFINE_BASE = 0.62     # base hair/edge refinement strength
TONE_DEPTH = 0.32           # photographic finish intensity
TONE_DEPTH_HIGH_QUALITY = 0.50  # already-great photo: skip heavy SR, lean on the finish
BASE_DENOISE = 0.45         # deliberately moderate — more erases pores and looks plastic
BASE_DETAIL = 0.58
MAX_UPSCALE = 2

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
    strategy: str
    effective_scale: int
    denoise_strength: float
    detail_strength: float
    face_restore: bool
    face_strength: float
    skin_texture: float
    only_center_face: bool
    hair_refine: float
    tone_depth: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class BeautifyResult:
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


def build_params(analysis: Analysis, settings: Settings) -> Params:
    """Combine the preset with the measured condition of THIS photo."""
    warnings: List[str] = []

    # ---- scale -------------------------------------------------------------------------
    # Small photos get a genuine 2x; photos that are already big are beautified at native size
    # (upscaling them is slow, makes huge files, and is rarely what was wanted).
    wanted = 2 if max(analysis.width, analysis.height) <= settings.AUTO_UPSCALE_MAX_SIDE else 1
    effective_scale = _safe_scale(
        analysis.width, analysis.height, min(wanted, MAX_UPSCALE),
        settings.MAX_OUTPUT_SIDE, settings.MAX_OUTPUT_PIXELS,
    )

    # ---- denoise / detail base ---------------------------------------------------------
    denoise = _clamp((BASE_DENOISE + BASE_DENOISE) / 2.0 + analysis.noise_score * 0.3)
    detail = _clamp((BASE_DETAIL + BASE_DETAIL) / 2.0)

    if analysis.already_high_quality:
        # Do not hallucinate detail into an already-clean image.
        effective_scale = min(effective_scale, 2)
        denoise = _clamp(denoise * 0.4)
        detail = _clamp(detail * 0.5)

    strategy = _select_strategy(analysis)

    # ---- faces -------------------------------------------------------------------------
    # A large, sharp, clean face is left alone: running a face model over an already-good face
    # changes it for the worse. Only degraded / blurry / small / noisy faces get restored.
    degradation = 0.5 * analysis.blur_score + 0.5 * analysis.jpeg_artifact_score
    face_needs_restoration = (
        analysis.quality_category in (SEVERELY_DEGRADED, LOW)
        or analysis.blur_score > 0.28
        or analysis.jpeg_artifact_score > 0.35
        or analysis.small_faces
        or analysis.noise_score > 0.4
    )
    face_restore = (
        analysis.face_count > 0 and not analysis.screenshot_like and face_needs_restoration
    )

    face_strength = _clamp(min(FACE_STRENGTH_CAP, 0.3 + 0.5 * degradation))
    if analysis.small_faces:
        face_strength = _clamp(face_strength * 0.55)  # tiny faces: avoid a hallucinated identity
        warnings.append("small_faces_reduced_strength")
    if analysis.blur_score < 0.15:
        face_strength = _clamp(face_strength * 0.75)  # already-sharp face: be conservative
    # Re-inject more original texture the more the face model smooths.
    skin_texture = _clamp(0.35 + 0.35 * face_strength, 0.0, 0.65)

    # Restore only the dominant face unless this is a genuine group photo (a second face at
    # least ~half the area of the largest). This is what kills spurious extra-face artifacts.
    areas = sorted((f.w * f.h for f in analysis.faces), reverse=True)
    genuine_group = len(areas) >= 2 and areas[1] >= 0.45 * areas[0]
    only_center_face = not genuine_group

    # ---- hair --------------------------------------------------------------------------
    hair_on = settings.HAIR_REFINEMENT_ENABLED and analysis.face_count > 0 and not analysis.screenshot_like
    cat_factor = {SEVERELY_DEGRADED: 0.7, LOW: 0.9, MEDIUM: 1.0, HIGH: 0.55}.get(analysis.quality_category, 1.0)
    hair_refine = _clamp(HAIR_REFINE_BASE * cat_factor * (0.6 + 0.4 * BASE_DETAIL)) if hair_on else 0.0

    # ---- whole-image detail ------------------------------------------------------------
    detail_strength = _clamp(min(DETAIL_CAP, detail))
    if analysis.quality_category == HIGH:
        detail_strength = _clamp(detail_strength * 0.55)
    elif analysis.quality_category == SEVERELY_DEGRADED:
        detail_strength = _clamp(detail_strength * 0.85)  # do not amplify artifacts

    tone_depth = TONE_DEPTH

    # ---- strategy overrides ------------------------------------------------------------
    if strategy == HIGH_QUALITY:
        # A good photo only degrades under a face model — finish it photographically instead.
        effective_scale = min(effective_scale, 2)
        face_restore = False
        hair_refine = 0.0
        tone_depth = TONE_DEPTH_HIGH_QUALITY
    if strategy == DOCUMENT:
        face_restore = False
        hair_refine = 0.0

    return Params(
        strategy=strategy,
        effective_scale=effective_scale,
        denoise_strength=round(denoise, 3),
        detail_strength=round(detail_strength, 3),
        face_restore=face_restore,
        face_strength=round(face_strength, 3),
        skin_texture=round(skin_texture, 3),
        only_center_face=only_center_face,
        hair_refine=round(hair_refine, 3),
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
) -> Tuple[Optional[np.ndarray], int]:
    """GFPGAN restoration with its native face-parsing paste-back onto a Real-ESRGAN background.

    One pass, no halos, no ghost faces. Returns the 2x restored RGB image and the number of
    faces actually processed, or (None, 0) when GFPGAN found nothing (the caller then falls
    back to plain Real-ESRGAN).
    """
    gfp = registry.gfpgan
    if gfp is None:
        return None, 0

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
        return None, 0

    result = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

    # Anti-waxy: add the source's real skin micro-texture back where skin is detected.
    if params.skin_texture > 0.02:
        src_up = cv2.resize(working_rgb, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        result = ops.reinject_texture(result, src_up, amount=params.skin_texture)

    return result, len(restored_faces or [])


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
) -> Tuple[str, BeautifyResult]:
    """Beautify one decoded image. Returns (output path, result metadata)."""
    started = time.perf_counter()
    timer = _Timer()
    report: ProgressFn = progress or (lambda *_: None)
    is_mock = settings.MOCK_MODE or not registry.status.ready
    models: List[str] = []
    warnings: List[str] = []
    processed_faces = 0
    fallback_used = False

    rgb = decoded.rgb

    # ---- analyse -----------------------------------------------------------------------
    report("analysing", 15, "Analysing the photo")
    with timer.stage("analyse"):
        analysis = analyse(rgb, has_alpha=decoded.had_alpha)
    params = build_params(analysis, settings)
    warnings.extend(params.warnings)
    log.info(
        "plan: strategy=%s scale=%sx faces=%s face_restore=%s quality=%s",
        params.strategy, params.effective_scale, analysis.face_count, params.face_restore,
        analysis.quality_category,
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
                restored2x, processed_faces = _restore_faces(registry, params, working)
            if restored2x is not None:
                models += ["realesrgan:general", "gfpgan"]
                report("blending", 70, "Blending faces")
                # GFPGAN always outputs 2x — resize to the scale we actually want.
                if (restored2x.shape[1], restored2x.shape[0]) != (target_w, target_h):
                    interp = cv2.INTER_AREA if target_w < restored2x.shape[1] else cv2.INTER_LANCZOS4
                    restored = cv2.resize(restored2x, (target_w, target_h), interpolation=interp)
                else:
                    restored = restored2x

        # Everything else (and the fallback when GFPGAN produced nothing).
        if restored is None:
            report("enhancing", 50, "Enhancing detail")
            with timer.stage("realesrgan"):
                restored = _restore_whole(registry, settings, working, params)
            models.append("realesrgan:general")

        # Chroma denoise for a result that is still noisy (luminance detail preserved).
        if analysis.noise_score > 0.4:
            with timer.stage("denoise"):
                restored = ops.denoise(
                    restored, strength=params.denoise_strength,
                    noise_score=analysis.noise_score, jpeg_score=0.0,
                )
            models.append("denoise")

    out_scale = restored.shape[1] / float(max(1, rgb.shape[1]))

    # ---- hair / fine texture around heads ----------------------------------------------
    if not is_mock and params.hair_refine > 0 and analysis.faces:
        with timer.stage("hair"):
            restored = ops.refine_hair(restored, analysis.faces, params.hair_refine, out_scale=out_scale)
        models.append("hair-refine")

    # ---- the photographic finish (this is the "beautify") -------------------------------
    report("finishing", 82, "Finishing")
    with timer.stage("finish"):
        restored = ops.premium_finish(restored, tone_depth=params.tone_depth, is_grayscale=analysis.is_grayscale)
    models.append("photo-finish")

    # ---- final edge-aware detail + one bounded fallback ---------------------------------
    reference = restored
    with timer.stage("detail"):
        detailed = ops.edge_aware_sharpen(restored, strength=params.detail_strength)
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
