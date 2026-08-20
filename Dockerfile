# ---------------------------------------------------------------------------
# Beautify — container image
#
# Host-agnostic: anything that runs a container and gives the process ~1.5 GB of
# RAM will work. Verified targets: Oracle Cloud Ampere (arm64), any x86_64 VM.
#
# Three things here matter more than they look:
#
#   1. torch is architecture-dependent. PyTorch's CPU wheel index carries x86_64
#      only; on arm64 that index has nothing and the install fails outright. On
#      arm64 the PyPI wheel IS the CPU build — there is no CUDA for ARM — so
#      plain PyPI is both correct and sufficient there.
#   2. The weights are downloaded during BUILD, not at startup. They cannot live
#      in git (528 MB, two files over GitHub's 100 MB limit), and fetching them
#      at boot would re-download on every restart and leave the first visitor
#      staring at a dead engine.
#   3. One worker, on purpose. Each additional worker loads its own ~960 MB copy
#      of the models.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV needs libGL and glib present even in headless use.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a normal user; everything the app writes must be owned by it.
RUN useradd -m -u 1000 app
WORKDIR /home/app
USER app
ENV PATH="/home/app/.local/bin:${PATH}"

# torch first, so the requirements install finds it already satisfied.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        pip install --no-cache-dir --user torch==2.2.2 torchvision==0.17.2 \
            --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir --user torch==2.2.2 torchvision==0.17.2; \
    fi

COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# The weights come BEFORE the application code, and the order is the whole point.
#
# `COPY . .` invalidates every layer after it, so with the download sitting below the code copy,
# ANY change to any source file re-fetched all 528 MB. That was most of the fifteen to twenty-five
# minutes a deploy took - every single time, for a one-line edit.
#
# Above it, the layer's cache key is this RUN plus the fetch script alone, so the weights are
# downloaded once and every later deploy reuses them. The script imports nothing but the standard
# library and resolves its destination relative to its own path, which is what lets it be copied
# on its own like this.
#
# Moving it costs one more full download - this layer has no cache at its new position - and
# saves it on every deploy after that.
COPY --chown=app:app scripts/fetch_models.py scripts/fetch_models.py
RUN python scripts/fetch_models.py

# Now the code. A change here no longer touches the weights layer above.
# COPY merges into the image rather than replacing directories, and `models/*.pth` and
# `gfpgan/weights/*.pth` are in .dockerignore, so nothing here can overwrite what was just
# downloaded.
COPY --chown=app:app . .

# 7860 suits Hugging Face; PORT is honoured everywhere else.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
