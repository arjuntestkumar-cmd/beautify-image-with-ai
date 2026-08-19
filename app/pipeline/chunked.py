"""Chunked execution - what lets a very large photo finish instead of taking the box with it.

The pipeline's classical stages are written as whole-array operations, and every one of them
allocates several float32 copies of the frame. At 4 MP that is invisible. At 40 MP a single
`cv2.cvtColor(rgb, RGB2LAB).astype(float32)` is 480 MB, `_edge_confidence` adds five more
arrays of the same size, and the super-resolution model is worse still: RealESRGANer always
runs its 4x network before scaling back down, so a 24 MP input briefly becomes a 384 MP
tensor. That is the actual reason a big upload kills the server, not the upload itself.

Nothing here changes what the pipeline computes. It changes *how much of it exists at once*:

    whole image  ->  overlapping tiles  ->  op per tile  ->  written into one output buffer

Two rules make the seams impossible to see rather than merely hard to see:

  1. HALO. Every tile is cut with a margin of context around it, and the margin is thrown away
     after the op runs. For a filter whose support radius is smaller than the halo, each
     interior output pixel is computed from exactly the neighbourhood it would have had in the
     whole-image call - so the tiled result is not "close to" the untiled one, it is identical.
     A short feather is available on top for the model stage, whose receptive field is fuzzy.

  2. GLOBAL CONSTANTS ARE MEASURED ONCE. Anything an op derives from the *whole* frame - white
     balance means, an edge-magnitude percentile, a CLAHE curve, an exposure gamma - is
     computed here, up front, and injected into every tile. Left alone, each tile would measure
     its own and land a visibly different correction on each side of a boundary. That is the
     part that would actually show, and it is removed by construction rather than by blending.

Small images skip all of it: `Runner.chunked` is False and every call goes straight through to
the original function, so the existing fast path for an ordinary photo is untouched.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Rect = Tuple[int, int, int, int]          # (y0, x0, y1, x1), in source pixels
TileFn = Callable[[np.ndarray, Rect], np.ndarray]

PROXY_SIDE = 512          # longest side of the low-resolution copy used to measure globals


# ---------------------------------------------------------------------------------------
# proxies and globally-measured constants
# ---------------------------------------------------------------------------------------
def proxy(rgb: np.ndarray, max_side: int = PROXY_SIDE) -> np.ndarray:
    """A cheap area-downscaled copy. Means, medians and percentiles taken here match the
    full-resolution ones closely enough to drive a correction, for 1/1000th of the memory."""
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    s = max_side / float(max(h, w))
    return cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _magnitude(l: np.ndarray) -> np.ndarray:
    return cv2.magnitude(cv2.Scharr(l, cv2.CV_32F, 1, 0), cv2.Scharr(l, cv2.CV_32F, 0, 1))


def sample_edge_hi(rgb: np.ndarray, window: int = 256, grid: int = 4) -> float:
    """The 92nd-percentile edge magnitude that `_edge_confidence` normalises against.

    Deliberately NOT taken from the proxy: gradients steepen when an image is downscaled, so a
    proxy reading would be wrong by a large factor and every tile would sharpen too little or
    too much. Instead a fixed grid of full-resolution windows is sampled - real gradients, at
    the real scale, out of about a megapixel of memory.
    """
    h, w = rgb.shape[:2]
    if h <= window * 2 and w <= window * 2:
        l = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
        return float(np.percentile(_magnitude(l), 92)) + 1e-6

    vals: List[np.ndarray] = []
    for y in np.linspace(0, max(0, h - window), grid, dtype=int):
        for x in np.linspace(0, max(0, w - window), grid, dtype=int):
            patch = rgb[y:y + window, x:x + window]
            l = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
            vals.append(_magnitude(l).ravel())
    return float(np.percentile(np.concatenate(vals), 92)) + 1e-6


def windows(rgb: np.ndarray, window: int = 256, grid: int = 4) -> List[np.ndarray]:
    """Full-resolution sample windows spread across the frame.

    Sharpness, grain and JPEG blocking are properties of the pixels at their real scale - a
    downscaled copy of a grainy photo measures clean and a blurry one measures sharp - so the
    analysis cannot read them off a proxy. It does not need every pixel, though: sixteen
    full-resolution windows are a fair sample of the frame for a megapixel instead of forty.

    They are returned as separate windows and NOT stitched into one image on purpose. Stitching
    puts a hard discontinuity at every join, and those fake edges land squarely in the two
    measurements that matter most here - they raise the Laplacian variance the sharpness score
    reads, and (origins being aligned to the 8-pixel grid, as the blockiness metric requires)
    they land exactly on the block boundaries the blockiness score samples. Measuring each
    window on its own and pooling the results has no such artefact.
    """
    h, w = rgb.shape[:2]
    if h <= window * grid and w <= window * grid:
        return [rgb]
    win_h, win_w = min(window, h), min(window, w)
    ys = [int(v) & ~7 for v in np.linspace(0, max(0, h - win_h), grid)]
    xs = [int(v) & ~7 for v in np.linspace(0, max(0, w - win_w), grid)]
    return [np.ascontiguousarray(rgb[y:y + win_h, x:x + win_w]) for y in ys for x in xs]


def stride_sample(rgb: np.ndarray, budget: int = 1_000_000) -> np.ndarray:
    """Every n-th pixel of the whole frame, for statistics that must not be biased.

    A stride and not a downscale, because the two differ exactly where it matters here.
    Averaging pulls the extremes in, so an area-downscaled copy reports a narrower dynamic
    range and fewer clipped pixels than the photo really has - enough to move a photo across
    the threshold that decides how hard the pipeline works on it. A stride keeps whatever
    values it lands on intact and simply looks at fewer of them, which is what an unbiased
    sample of a distribution is.
    """
    h, w = rgb.shape[:2]
    k = max(1, int(np.ceil(np.sqrt((h * w) / float(max(1, budget))))))
    return rgb[::k, ::k]


def analysis_sample(rgb: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """What `analyse` needs from a photo too large to measure whole.

    Two different things, because its measurements split into two kinds. Degradation - blur,
    grain, blocking, edge density - has to be read at full resolution, so it gets the windows
    above. Intensity statistics - brightness, contrast, dynamic range, how dark the photo is -
    only need an unbiased sample of the frame's pixels, so they get a strided one and stay a
    property of the whole photo rather than of wherever the windows happened to land.
    """
    return windows(rgb), stride_sample(rgb)


def channel_means(rgb: np.ndarray) -> np.ndarray:
    """Per-channel means for white balance. Area-downscaling preserves a mean, so taking this
    from the proxy is not an approximation - it is the same number."""
    return proxy(rgb).reshape(-1, 3).mean(axis=0).astype(np.float32)


class Field:
    """A low-frequency map measured once over the whole frame and sampled back per tile.

    CLAHE is the reason this exists. Run per tile it would give every tile its own histogram
    and stamp a grid of brightness steps across the photo. Its *effect*, though, is a smooth
    local tone remap, so `clahe(L) - L` measured on a 512 px proxy and stretched back over the
    frame is the same curve to the eye - and, being one global map, it cannot disagree with
    itself at a boundary. Face and hair masks are sampled the same way.

    The field carries a DIFFERENCE and not a ratio. `clahe(L)/L` looks equivalent and is not:
    in the deep shadows the denominator approaches zero, so a proxy pixel at L=2 lifted to 20
    yields a gain of 7, and applying that to a full-resolution pixel at L=40 blows it to white.
    An offset behaves the same everywhere in the range.
    """

    def __init__(self, small: np.ndarray, full_shape: Sequence[int]) -> None:
        self.small = small.astype(np.float32)
        self.h, self.w = int(full_shape[0]), int(full_shape[1])

    def crop(self, rect: Rect) -> np.ndarray:
        y0, x0, y1, x1 = rect
        sh, sw = self.small.shape[:2]
        sy0, sy1 = y0 * sh / self.h, y1 * sh / self.h
        sx0, sx1 = x0 * sw / self.w, x1 * sw / self.w
        iy0, ix0 = int(np.floor(sy0)), int(np.floor(sx0))
        iy1, ix1 = min(sh, int(np.ceil(sy1)) + 1), min(sw, int(np.ceil(sx1)) + 1)
        patch = self.small[iy0:max(iy1, iy0 + 1), ix0:max(ix1, ix0 + 1)]
        return cv2.resize(patch, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)


class ClaheField:
    """CLAHE whose histograms are measured over the whole frame, applied one tile at a time.

    The pipeline closes several stages with CLAHE, and it is the single hardest thing here to
    chunk. Called per tile it builds a fresh histogram for each one and prints its grid across
    the photo as steps in brightness - the most visible artefact chunking can produce.

    The way out is that CLAHE is already a grid algorithm. It divides the frame into an 8x8
    grid, derives one tone curve per cell from that cell's histogram, and interpolates between
    the four nearest curves for every pixel. None of that needs the whole image resident: the
    64 histograms are read one cell at a time, and after that the curves are 64 tables of 256
    bytes that any tile can be mapped through.

    So this is not an approximation of CLAHE that avoids the seams - it is CLAHE, with its
    measurement separated from its application. Measuring on a downscaled proxy instead would
    have been far simpler and quietly wrong: downscaling averages away exactly the local
    variance the histograms are built from, so the curves come back too aggressive and a large
    photo ends up with visibly more local contrast than a small one.
    """

    def __init__(self, rgb: np.ndarray, clip_limit: float, grid: int = 8,
                 pre_curve: Optional[np.ndarray] = None) -> None:
        self.h, self.w = rgb.shape[:2]
        self.g = max(1, int(grid))
        ys = np.linspace(0, self.h, self.g + 1).astype(int)
        xs = np.linspace(0, self.w, self.g + 1).astype(int)
        self.luts = np.empty((self.g, self.g, 256), np.float32)
        for i in range(self.g):
            for j in range(self.g):
                cell = rgb[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)]
                l = cv2.cvtColor(np.ascontiguousarray(cell), cv2.COLOR_RGB2LAB)[:, :, 0]
                # `pre_curve` is any pointwise tone curve the caller will have applied to L by
                # the time this runs. Measuring the histogram without it would equalise the
                # distribution the photo had BEFORE the grade rather than the one it has when
                # the curve lands, and the local contrast comes out mis-calibrated.
                if pre_curve is not None:
                    l = cv2.LUT(l, pre_curve).astype(np.uint8)
                self.luts[i, j] = _cell_lut(l, clip_limit)

    def apply(self, l: np.ndarray, rect: Rect) -> np.ndarray:
        """Map this tile's L channel through the four curves nearest each of its pixels."""
        y0, x0, y1, x1 = rect
        iy0, iy1, fy = _cell_weights(np.arange(y0, y1), self.g, self.h)
        ix0, ix1, fx = _cell_weights(np.arange(x0, x1), self.g, self.w)
        out = np.zeros((y1 - y0, x1 - x0), np.float32)
        for iy, wy in ((iy0, 1.0 - fy), (iy1, fy)):
            for ix, wx in ((ix0, 1.0 - fx), (ix1, fx)):
                out += (wy[:, None] * wx[None, :]) * self.luts[iy[:, None], ix[None, :], l]
        return np.clip(out, 0, 255).astype(np.uint8)


def _cell_lut(l: np.ndarray, clip_limit: float) -> np.ndarray:
    """One cell's contrast-limited tone curve: clip the histogram, spread the excess, integrate."""
    hist = np.bincount(l.ravel(), minlength=256).astype(np.float32)
    total = float(max(1.0, hist.sum()))
    limit = max(1.0, clip_limit * total / 256.0)
    excess = float(np.maximum(hist - limit, 0.0).sum())
    hist = np.minimum(hist, limit) + excess / 256.0
    return np.cumsum(hist) * (255.0 / total)


def _cell_weights(coords: np.ndarray, grid: int, extent: int):
    """Which two cell curves a coordinate falls between, and how far across it sits."""
    g = coords * (grid / float(max(1, extent))) - 0.5
    i0 = np.clip(np.floor(g), 0, grid - 1).astype(np.intp)
    frac = np.clip(g - i0, 0.0, 1.0).astype(np.float32)
    return i0, np.clip(i0 + 1, 0, grid - 1), frac


def mask_field(mask_fn: Callable[[tuple, List], np.ndarray], shape: Sequence[int],
              boxes: List) -> Field:
    """Build a face/hair mask on the proxy grid so every tile reads the same feathering."""
    h, w = int(shape[0]), int(shape[1])
    s = min(1.0, PROXY_SIDE / float(max(h, w)))
    scaled = [type(b)(int(b.x * s), int(b.y * s), max(1, int(b.w * s)), max(1, int(b.h * s)))
              for b in boxes]
    sh, sw = max(1, int(h * s)), max(1, int(w * s))
    return Field(mask_fn((sh, sw), scaled), (h, w))


# ---------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------
def union_rect(shape: Sequence[int], boxes: List, pad_frac: float, min_pad: int = 24,
               reach: Tuple[float, float] = (0.0, 0.0),
               sigma: Tuple[float, float] = (0.0, 0.0)) -> Rect:
    """The padded region a face-local stage can actually reach, clipped to the frame.

    Everything those stages do is held inside a mask that has already faded to zero well before
    this edge, so cropping to it is not an approximation - PROVIDED the pad is actually wide
    enough for the mask, which a fraction of the box size is not required to be. The pad here is
    therefore the LARGER of the caller's fraction and what the mask geometry demands, computed
    PER AXIS.

    Per axis, because every mask in ops.py feathers with a sigma driven by the face WIDTH while
    the shapes themselves reach much further in y than in x: the face ellipse by 0.18h against
    0.12w, the hair band by 0.9h against 0.55w. One scalar cannot serve both. Sized for x it
    leaves the mask alive on the top and bottom crop edges, and a live mask on a crop edge is a
    straight horizontal boundary between processed and untouched pixels - the "square shape"
    around the restored area.

    `reach` is (rx, ry), how far past the box edge the mask's shape extends as a fraction of the
    box width and height; `sigma` is (floor_px, fraction_of_box_width), the feather. Four sigma
    of feather is added on top, because OpenCV sizes a Gaussian kernel at four sigma for a
    float32 image and every mask here is float32 (see ops.GAUSS_SUPPORT). Anything less and the
    mask is non-zero exactly where the crop stops.

    It stays cheap: for an ordinary portrait the geometric floor does not bind. A 400x520 face
    needs 226 px in x and 271 in y, against the 390 the 0.75 fraction already gives, so the
    region and its cost are unchanged. The floor binds only for small faces, where the old
    `min_pad` of 24 was less than a single feather.
    """
    h, w = int(shape[0]), int(shape[1])
    bw = max(b.w for b in boxes)
    bh = max(b.h for b in boxes)
    sig = max(float(sigma[0]), float(sigma[1]) * bw)
    frac = float(pad_frac) * max(bw, bh)
    pad_x = int(np.ceil(max(float(min_pad), frac, float(reach[0]) * bw + 4.0 * sig)))
    pad_y = int(np.ceil(max(float(min_pad), frac, float(reach[1]) * bh + 4.0 * sig)))
    y0 = max(0, min(b.y for b in boxes) - pad_y)
    x0 = max(0, min(b.x for b in boxes) - pad_x)
    y1 = min(h, max(b.y + b.h for b in boxes) + pad_y)
    x1 = min(w, max(b.x + b.w for b in boxes) + pad_x)
    if y1 <= y0 or x1 <= x0:
        return 0, 0, h, w
    return y0, x0, y1, x1


def _ramp(n: int) -> np.ndarray:
    """A smooth 0..1 feather. Cosine, so its slope is zero at both ends and the join has no
    visible ridge where a linear ramp would leave one."""
    if n <= 0:
        return np.ones(0, np.float32)
    t = (np.arange(n, dtype=np.float32) + 0.5) / float(n)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def _ramp0(n: int) -> np.ndarray:
    """The same cosine ramp, sampled so it starts at EXACTLY zero.

    `_ramp` samples cell centres, which is right for cross-fading two tiles that both exist - it
    never needs to reach a true zero because there is a real, equally valid pixel underneath. A
    border taper is fading against pixels that were never processed at all, so it does: leave a
    first weight of 0.0012 in place and the step it exists to hide survives at a thousandth of
    its size, which on flat content is still a line.
    """
    if n <= 0:
        return np.ones(0, np.float32)
    t = np.arange(n, dtype=np.float32) / float(n)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def border_taper(shape: Sequence[int], rect: Rect, full_shape: Sequence[int],
                 width: int) -> np.ndarray:
    """Write-back weight for a result computed on a sub-rectangle of a larger frame.

    1 through the middle, ramping to 0 at every crop edge that is INTERIOR to the full frame. An
    edge that coincides with the image border is left at 1: there is nothing on the far side of
    it for the result to disagree with, and tapering there would only soften the picture's own
    border.

    This is the SECOND, independent guarantee against the reported rectangle. Mask-exact ops
    (ops.composite_masked) stop a lossy colour round trip from leaking past its mask, and correct
    padding keeps the mask itself away from the crop edge. Both can be defeated - by a face close
    enough to the frame edge that the pad gets clipped, by a mask some future stage builds
    differently, by a model that returns a crop a pixel off. This cannot: whatever the crop
    contains, the write fades out before it reaches an edge anybody can see.

    `shape` is the array being written and may be at a different scale from `rect`; `rect` and
    `full_shape` are used only for the four "is this edge the frame border" comparisons, so they
    may be given in whichever coordinate space the crop was actually clipped in.
    """
    h, w = int(shape[0]), int(shape[1])
    y0, x0, y1, x1 = rect
    fh, fw = int(full_shape[0]), int(full_shape[1])
    ay = np.ones(h, np.float32)
    ax = np.ones(w, np.float32)
    ny = int(max(0, min(int(width), h // 2)))
    nx = int(max(0, min(int(width), w // 2)))
    if ny:
        if y0 > 0:
            ay[:ny] = _ramp0(ny)
        if y1 < fh:
            ay[h - ny:] = _ramp0(ny)[::-1]
    if nx:
        if x0 > 0:
            ax[:nx] = _ramp0(nx)
        if x1 < fw:
            ax[w - nx:] = _ramp0(nx)[::-1]
    return np.minimum(ay[:, None], ax[None, :])


# ---------------------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------------------
class Runner:
    """Decides once whether this photo needs chunking, then runs every stage the same way.

    `chunked` is False for an ordinary photo and every call is a direct passthrough, so the
    existing path for a normal upload does exactly what it did before.
    """

    def __init__(self, settings, shape: Sequence[int], progress=None,
                 should_stop: Optional[Callable[[], bool]] = None) -> None:
        h, w = int(shape[0]), int(shape[1])
        self.threshold = int(settings.CHUNK_THRESHOLD_PIXELS)
        self.chunked = (h * w) > self.threshold
        self.tile = max(128, int(settings.CHUNK_TILE_SIZE))
        self.halo = max(8, int(settings.CHUNK_OVERLAP))
        self.model_tile = max(64, int(settings.MODEL_CHUNK_TILE_SIZE))
        self.model_halo = max(8, int(settings.MODEL_CHUNK_OVERLAP))
        self._progress = progress
        self._should_stop = should_stop
        self._lo, self._hi, self._stage, self._msg = 0, 0, "", ""

    # ---- progress + cooperative cancellation ----
    def stage(self, name: str, lo: int, hi: int, message: str) -> None:
        """Claim a slice of the progress bar. A tiled stage then creeps across its slice, so a
        photo that legitimately takes two minutes never looks frozen to the person waiting."""
        self._stage, self._lo, self._hi, self._msg = name, lo, hi, message
        self._report(0.0)

    def _report(self, fraction: float) -> None:
        if self._progress and self._stage:
            pct = self._lo + (self._hi - self._lo) * max(0.0, min(1.0, fraction))
            self._progress(self._stage, int(pct), self._msg)

    def check(self) -> None:
        """Between-tile abort point. Chunking is also what makes a long job interruptible at
        all: a single whole-image call has nowhere to put this."""
        if self._should_stop and self._should_stop():
            from ..errors import ProcessingTimeout
            raise ProcessingTimeout("This photo took longer than this server is allowed to spend.")

    # ---- the two execution shapes ----
    def map(self, img: np.ndarray, fn: TileFn, halo: Optional[int] = None,
            feather: int = 0, scale: float = 1.0, tile: Optional[int] = None) -> np.ndarray:
        """Run `fn(piece, rect)` over the image tile by tile, into one output buffer.

        `halo` must be at least the op's filter support for the result to match the whole-image
        call exactly. `feather` additionally cross-fades the leading rows/columns of each tile
        into the neighbour already written, which is needed only where the support is not a hard
        number - in this pipeline, the super-resolution network.
        """
        h, w = img.shape[:2]
        if not self.chunked:
            return fn(img, (0, 0, h, w))

        halo = self.halo if halo is None else halo
        halo = max(halo, feather)
        step = max(64, tile or self.tile)
        halo = min(halo, step)   # a very wide margin must not swallow the tile it surrounds

        def S(v: int) -> int:                       # source coordinate -> output coordinate
            return int(round(v * scale))

        out = np.empty((S(h), S(w), img.shape[2]), np.uint8)
        rows = list(range(0, h, step))
        cols = list(range(0, w, step))
        total = len(rows) * len(cols)
        done = 0

        for y0 in rows:
            for x0 in cols:
                self.check()
                y1, x1 = min(h, y0 + step), min(w, x0 + step)
                sy0, sx0 = max(0, y0 - halo), max(0, x0 - halo)
                sy1, sx1 = min(h, y1 + halo), min(w, x1 + halo)

                piece = fn(img[sy0:sy1, sx0:sx1], (sy0, sx0, sy1, sx1))
                want = (S(sx1) - S(sx0), S(sy1) - S(sy0))
                if (piece.shape[1], piece.shape[0]) != want:
                    # The model rounds its own way; pin the piece back onto the grid so the
                    # writes below stay exact instead of drifting a pixel per tile.
                    piece = cv2.resize(piece, want, interpolation=cv2.INTER_LANCZOS4)

                fy = feather if y0 > 0 else 0
                fx = feather if x0 > 0 else 0
                ty0, tx0, ty1, tx1 = S(y0 - fy), S(x0 - fx), S(y1), S(x1)
                sub = piece[ty0 - S(sy0):ty1 - S(sy0), tx0 - S(sx0):tx1 - S(sx0)]

                if fy or fx:
                    self._blend(out, sub, (ty0, tx0, ty1, tx1), S(y0) - ty0, S(x0) - tx0)
                else:
                    out[ty0:ty1, tx0:tx1] = sub

                done += 1
                self._report(done / float(max(1, total)))
        return out

    @staticmethod
    def _blend(out: np.ndarray, sub: np.ndarray, rect: Rect, fy: int, fx: int) -> None:
        """Cross-fade `sub` over the overlap it shares with the tiles already written."""
        ty0, tx0, ty1, tx1 = rect
        ay = np.ones(ty1 - ty0, np.float32)
        ax = np.ones(tx1 - tx0, np.float32)
        if fy > 0:
            ay[:fy] = _ramp(fy)
        if fx > 0:
            ax[:fx] = _ramp(fx)
        alpha = np.minimum(ay[:, None], ax[None, :])[..., None]
        target = out[ty0:ty1, tx0:tx1]
        np.copyto(target, np.clip(target.astype(np.float32) * (1.0 - alpha)
                                  + sub.astype(np.float32) * alpha, 0, 255).astype(np.uint8))

    def region(self, img: np.ndarray, boxes: List, fn, pad_frac: float,
               mask_fn=None, halo: int = 64) -> np.ndarray:
        """Run a face-local stage over only the region it can reach.

        `fn(piece, boxes, mask)` - `mask` is None when the piece is the whole region and the
        stage should build its own, and a slice of one pre-sampled global mask when the region
        was itself too large to hold and had to be tiled.

        The write is in place: on a large photo this is the difference between one buffer and
        five, and no caller holds a reference to the array from before the call.
        """
        if not boxes:
            return img
        h, w = img.shape[:2]
        if not self.chunked:
            return fn(img, boxes, None)

        # The mask function publishes the geometry the crop has to clear; see ops.face_region_mask.
        reach = getattr(mask_fn, "reach", (0.0, 0.0))
        sigma = getattr(mask_fn, "sigma", (0.0, 0.0))
        y0, x0, y1, x1 = union_rect((h, w), boxes, pad_frac, reach=reach, sigma=sigma)
        if (y1 - y0) * (x1 - x0) <= self.threshold:
            local = [type(b)(b.x - x0, b.y - y0, b.w, b.h) for b in boxes]
            view = img[y0:y1, x0:x1]
            # `fn` is allowed to write into the piece it is given - filters.apply_face does, via
            # _in_band - and that piece is a view of `img`. The pixels this write has to fade back
            # to therefore have to be kept before it runs. One copy of the REGION, not the frame.
            base = view.copy()
            piece = fn(view, local, None)
            # Second guarantee, independent of the two above it. The stages are mask-exact and the
            # pad keeps the mask off the crop edge, but neither is something a future stage is
            # obliged to honour. A write that fades to nothing at every crop edge interior to the
            # frame cannot leave a straight boundary whatever it contains.
            #
            # Width is 1.5 feather sigma, floored at 32 px so the taper is never a thin hard line
            # of its own, and capped at a quarter of the crop so it can never reach the face box.
            # That is the whole of the width argument: the pad is reach + 4 sigma, so a 1.5-sigma
            # taper only ever touches pixels lying at least 2.5 sigma outside the mask's shape,
            # where a blurred step is below 0.6% of full strength. Wide enough to spread any
            # residual step over dozens of pixels, provably too narrow to erase the effect.
            sig = max(float(sigma[0]), float(sigma[1]) * max(b.w for b in boxes))
            n = int(min(max(32.0, 1.5 * sig), (y1 - y0) // 4, (x1 - x0) // 4))
            t = border_taper(piece.shape, (y0, x0, y1, x1), (h, w), n)[..., None]
            np.copyto(view, np.clip(base.astype(np.float32) * (1.0 - t)
                                    + piece.astype(np.float32) * t,
                                    0, 255).round().astype(np.uint8))
            self._report(1.0)
            return img

        # A face big enough that even its own region will not fit: tile it, and hand every tile
        # a slice of one globally-built mask so the feathering cannot step at a boundary.
        gm = mask_field(mask_fn, (h, w), boxes) if mask_fn else None
        return self.map(img, lambda piece, rect: fn(piece, boxes, gm.crop(rect) if gm else None),
                        halo=halo)
