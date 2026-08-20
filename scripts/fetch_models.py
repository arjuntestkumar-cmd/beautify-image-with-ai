#!/usr/bin/env python3
"""Download the model weights.

The weights are ~528 MB and two of the files exceed GitHub's 100 MB per-file limit, so they
cannot live in the repository. Any machine that runs this project therefore has to fetch them
once — this script is that step, and it is idempotent: a file already present at the right size
is left alone, so re-running it is free.

Run it directly (`python scripts/fetch_models.py`) or let the Dockerfile call it at build time,
which bakes the weights into the image so restarts are instant.

Auxiliary face-detection weights are fetched here too. The face library would otherwise download
them itself on the first request, which turns a user's first enhancement into a silent 190 MB
download — and fails outright on a host with no outbound access at runtime.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (destination, url, expected size in bytes)
FILES: List[Tuple[str, str, int]] = [
    (
        "models/realesr-general-x4v3.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        4885111,
    ),
    (
        "models/realesr-general-wdn-x4v3.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        4885111,
    ),
    (
        "models/GFPGANv1.4.pth",
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        348632874,
    ),
    (
        "gfpgan/weights/detection_Resnet50_Final.pth",
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        109497761,
    ),
    (
        "gfpgan/weights/parsing_parsenet.pth",
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        85331193,
    ),
]

# A size this far from the expected value means a truncated or wrong file.
TOLERANCE = 0.02


def _human(n: float) -> str:
    return f"{n / 1_048_576:.1f} MB"


def _ok(path: str, expected: int) -> bool:
    if not os.path.isfile(path):
        return False
    actual = os.path.getsize(path)
    return abs(actual - expected) <= expected * TOLERANCE


# How often the progress line is printed. Small enough to prove the download is moving on a
# slow link, large enough not to bury a build log.
PROGRESS_EVERY = 32 * 1_048_576
ATTEMPTS = 3


def _stream(url: str, tmp: str, expected: int, name: str) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or expected)
        done = 0
        next_mark = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if done >= next_mark:
                    # A full line, unconditionally - NOT a \r bar behind `if isatty()`.
                    # A Docker build's stdout is not a tty, so the old version printed nothing at
                    # all for the whole 332 MB file, which made a perfectly healthy download on a
                    # slow link look exactly like a hung one for twenty minutes.
                    pct = done * 100 // max(1, total)
                    print(f"  {name:<32} {pct:3d}%  {_human(done)} / {_human(total)}", flush=True)
                    next_mark = done + PROGRESS_EVERY


def _check_space(dest_dir: str, needed: int, name: str) -> None:
    """Refuse to start a download the disk cannot hold, and say so plainly.

    Running out of space mid-download raises OSError, which the retry loop below then treats as
    a network blip: it deletes the part file, freeing exactly the space that let it get that far,
    downloads to the same point, and fails again. That loop looks like a flaky connection and is
    not one - so check first, and name the real problem.

    Twice the file size, because the download and the finished file coexist briefly.
    """
    try:
        free = shutil.disk_usage(dest_dir).free
    except OSError:
        return                      # cannot tell; let the download try
    if free < needed * 2:
        raise RuntimeError(
            f"not enough disk for {name}: {_human(free)} free, about {_human(needed * 2)} "
            f"needed. On a Docker host, old images are the usual culprit - "
            f"`docker system df` shows what is held, `docker system prune -af` reclaims it."
        )


def download(url: str, dest: str, expected: int) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    name = os.path.basename(dest)
    _check_space(os.path.dirname(dest), expected, name)

    # Retry, because one dropped connection 300 MB into a build should not fail the deploy.
    for attempt in range(1, ATTEMPTS + 1):
        try:
            _stream(url, tmp, expected, name)
            break
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if os.path.exists(tmp):
                os.remove(tmp)          # no resume: a partial file must never look complete
            if attempt == ATTEMPTS:
                raise
            # The MESSAGE, not just the class. "OSError" alone hides the difference between a
            # dropped connection and "No space left on device", which are the two things this
            # actually fails on and want completely different fixes.
            wait = 3 * attempt
            print(f"  {name}: attempt {attempt} of {ATTEMPTS} failed "
                  f"({exc.__class__.__name__}: {exc}); retrying in {wait}s", flush=True)
            free = shutil.disk_usage(os.path.dirname(dest) or ".").free
            print(f"  {name}: {_human(free)} free on the target filesystem", flush=True)
            time.sleep(wait)

    size = os.path.getsize(tmp)
    if abs(size - expected) > expected * TOLERANCE:
        os.remove(tmp)
        raise RuntimeError(
            f"{name}: downloaded {_human(size)} but expected about {_human(expected)}"
        )

    # Atomic: a crash mid-download can never leave a half file looking complete.
    os.replace(tmp, dest)


def main() -> int:
    print("Model weights -> %s" % ROOT)
    missing = 0
    for rel, url, expected in FILES:
        dest = os.path.join(ROOT, rel)
        if _ok(dest, expected):
            print(f"  {os.path.basename(rel):<32} present ({_human(os.path.getsize(dest))})")
            continue
        missing += 1
        print(f"  {os.path.basename(rel):<32} downloading {_human(expected)}...")
        try:
            download(url, dest, expected)
        except Exception as exc:  # noqa: BLE001 - report and fail the build
            print(f"\nFAILED to fetch {rel}: {exc}", file=sys.stderr)
            print(f"  source: {url}", file=sys.stderr)
            return 1

    total = sum(os.path.getsize(os.path.join(ROOT, rel)) for rel, _, _ in FILES)
    print(f"All weights ready ({_human(total)} total, {missing} downloaded).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
