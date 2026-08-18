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
import sys
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


def download(url: str, dest: str, expected: int) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    name = os.path.basename(dest)

    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or expected)
        done = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if sys.stdout.isatty():
                    pct = done * 100 // max(1, total)
                    print(f"\r  {name:<32} {pct:3d}%  {_human(done)}", end="", flush=True)
        if sys.stdout.isatty():
            print()

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
