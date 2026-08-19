"""Model registry — loads the two models this build needs, once, at startup.

Heavy imports (torch, realesrgan, gfpgan) happen inside `load()` so the module imports cleanly
without weights (mock mode / tests).

What is loaded:
  * Real-ESRGAN "general" (realesr-general-x4v3, ~5 MB) — the gentle, denoise-aware restorer.
    Its "wdn" sibling is blended into it per request (DNI) so denoise strength tracks the
    measured noise instead of being a fixed setting.
  * GFPGAN v1.4 — face restoration, using the Real-ESRGAN model above as its background
    upsampler so faces and background are produced in ONE pass and blended with GFPGAN's own
    face-parsing mask (no rectangular seams, no ghost faces).

Deliberately NOT loaded (they belong to the modes this build drops): RealESRGAN_x4plus,
CodeFormer, SwinIR.
"""
from __future__ import annotations

import os
import sys
import threading
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..config import Settings
from ..logging_utils import get_logger

log = get_logger("registry")


def _install_torchvision_shim() -> None:
    """basicsr imports `torchvision.transforms.functional_tensor`, removed in torchvision>=0.17.

    Alias it to `functional` so basicsr/gfpgan/realesrgan import cleanly on any torchvision.
    Without this the whole model load fails on a modern torchvision.
    """
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except Exception:
        pass
    try:
        import torchvision.transforms.functional as _F

        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = _F.rgb_to_grayscale  # the only symbol basicsr needs
        sys.modules["torchvision.transforms.functional_tensor"] = shim
    except Exception as exc:  # pragma: no cover
        log.warning("could not install torchvision shim: %s", exc.__class__.__name__)


@dataclass
class RegistryStatus:
    ready: bool = False
    mock_mode: bool = False
    cuda_available: bool = False
    device: str = "cpu"
    half_precision: bool = False
    loaded: Dict[str, bool] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._status = RegistryStatus(mock_mode=settings.MOCK_MODE)
        self._realesrgan: Optional[Any] = None
        self._gfpgan: Optional[Any] = None
        self._torch: Any = None
        # DNI (denoise interpolation) state.
        self._general_state: Optional[dict] = None
        self._wdn_state: Optional[dict] = None
        self._general_dni: Optional[float] = None
        self._dni_lock = threading.Lock()
        # Held for the whole model phase of a job. Both models carry per-request state that
        # lives on the shared object - DNI rewrites the Real-ESRGAN weights, and the crop-based
        # face path swaps GFPGAN's background upsampler out and back - so two jobs in the model
        # stage at once would read each other's settings and mix their output. With
        # WORKER_CONCURRENCY=1 this is never contended; it is what makes raising it safe.
        self.lock = threading.RLock()

    # ---- accessors ----
    @property
    def status(self) -> RegistryStatus:
        return self._status

    @property
    def torch(self) -> Any:
        return self._torch

    @property
    def realesrgan(self) -> Optional[Any]:
        return self._realesrgan

    @property
    def gfpgan(self) -> Optional[Any]:
        return self._gfpgan

    @property
    def has_dni(self) -> bool:
        return self._general_state is not None and self._wdn_state is not None

    def public_status(self) -> Dict[str, Any]:
        s = self._status
        return {
            "ready": s.ready,
            "mockMode": s.mock_mode,
            "cudaAvailable": s.cuda_available,
            "device": s.device,
            "halfPrecision": s.half_precision,
            "models": {k: bool(v) for k, v in s.loaded.items()},
            "modelErrors": {k: "load_failed" for k in s.errors},  # generic — never leak paths
        }

    def release(self) -> None:
        """Hand cached accelerator memory back between jobs.

        Torch keeps its allocator's blocks after a job finishes, so the next one in the queue
        starts on top of the last one's peak instead of on a clean slate. Two photos that each
        fit comfortably can then fail together. Cheap to call and safe on CPU-only builds.
        """
        torch = self._torch
        if torch is None or not self._status.cuda_available:
            return
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - reclaiming memory must never raise
            pass

    # ---- lifecycle ----
    def load(self) -> None:
        if self._settings.MOCK_MODE:
            self._status.ready = True
            log.warning("MOCK_MODE is on — no models loaded, output is a plain resize")
            return

        try:
            import torch
        except Exception as exc:  # pragma: no cover
            self._status.errors["torch"] = str(exc)
            log.error("failed to import torch: %s", exc.__class__.__name__)
            return

        _install_torchvision_shim()
        self._torch = torch

        cuda = bool(self._settings.ENABLE_CUDA and torch.cuda.is_available())
        self._status.cuda_available = cuda
        self._status.device = self._settings.GPU_DEVICE if cuda else "cpu"
        self._status.half_precision = bool(cuda and self._settings.ENABLE_HALF_PRECISION)

        self._load_realesrgan()
        self._load_gfpgan()

        self._status.ready = bool(self._status.loaded.get("realesrgan:general"))
        log.info("registry loaded: %s", self.public_status())

    def _load_realesrgan(self) -> None:
        try:
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        except Exception as exc:  # pragma: no cover
            self._status.errors["realesrgan"] = str(exc)
            log.error("failed to import realesrgan: %s", exc.__class__.__name__)
            return

        path = self._settings.REALESRGAN_GENERAL_MODEL_PATH
        if not os.path.isfile(path):
            self._status.errors["realesrgan:general"] = "missing weight file"
            log.error("Real-ESRGAN weights not found at %s", path)
            return
        try:
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
            )
            self._realesrgan = RealESRGANer(
                scale=4, model_path=path, model=model,
                tile=self._settings.DEFAULT_TILE_SIZE, tile_pad=self._settings.TILE_PADDING,
                pre_pad=0, half=self._status.half_precision, device=self._status.device,
            )
            self._status.loaded["realesrgan:general"] = True
            self._prepare_dni(path)
        except Exception as exc:  # pragma: no cover
            self._status.errors["realesrgan:general"] = str(exc)
            log.error("failed to load Real-ESRGAN: %s", exc)

    def _prepare_dni(self, general_path: str) -> None:
        """Load general + wdn state dicts so denoise strength can interpolate them per request."""
        wdn_path = self._settings.REALESRGAN_WDN_MODEL_PATH
        if not wdn_path or not os.path.isfile(wdn_path) or wdn_path == general_path:
            return
        try:
            torch = self._torch
            g = torch.load(general_path, map_location="cpu")
            w = torch.load(wdn_path, map_location="cpu")
            key = "params_ema" if "params_ema" in g else "params"
            self._general_state = g[key]
            self._wdn_state = w[key] if key in w else w[list(w.keys())[0]]
            self._status.loaded["realesrgan:wdn"] = True
        except Exception as exc:  # pragma: no cover
            self._status.errors["realesrgan:wdn"] = str(exc)
            self._general_state = None
            self._wdn_state = None

    def set_dni(self, denoise_strength: float) -> None:
        """Blend general+wdn weights by denoise strength (0 weak .. 1 strong) and load them.

        Thread-safe and cached — only recomputes when the strength changes meaningfully.
        """
        if not self.has_dni or self._realesrgan is None:
            return
        w = float(max(0.0, min(1.0, denoise_strength)))
        with self._dni_lock:
            if self._general_dni is not None and abs(self._general_dni - w) < 0.02:
                return
            try:
                blended = {k: self._general_state[k] * w + self._wdn_state[k] * (1 - w)  # type: ignore[index]
                           for k in self._general_state}  # type: ignore[union-attr]
                self._realesrgan.model.load_state_dict(blended, strict=True)
                self._general_dni = w
            except Exception as exc:  # pragma: no cover
                log.warning("DNI blend failed; using base weights: %s", exc.__class__.__name__)

    def _load_gfpgan(self) -> None:
        path = self._settings.GFPGAN_MODEL_PATH
        if not os.path.isfile(path):
            self._status.errors["gfpgan"] = "missing weight file"
            log.warning("GFPGAN weights not found at %s — faces will not be restored", path)
            return
        try:
            from gfpgan import GFPGANer
        except Exception as exc:  # pragma: no cover
            self._status.errors["gfpgan"] = str(exc)
            return
        try:
            # bg_upsampler = the Real-ESRGAN model above: GFPGAN restores faces and blends them
            # onto a Real-ESRGAN background using its OWN face-parsing mask. One SR pass total.
            self._gfpgan = GFPGANer(
                model_path=path, upscale=2, arch="clean", channel_multiplier=2,
                bg_upsampler=self._realesrgan, device=self._status.device,
            )
            self._status.loaded["gfpgan"] = True
        except Exception as exc:  # pragma: no cover
            self._status.errors["gfpgan"] = str(exc)
            log.warning("failed to load GFPGAN: %s", exc)
