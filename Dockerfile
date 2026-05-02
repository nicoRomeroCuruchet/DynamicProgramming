# syntax=docker/dockerfile:1.6
#
# Dockerfile for CUDA Policy Iteration -- Continuous RL via Dynamic Programming.
#
# What this image gives you:
#   - CUDA 12.x runtime + cuDNN (compatible with cupy-cuda12x)
#   - Python 3.10 (Ubuntu 22.04 default)
#   - All Python deps from requirements.txt
#   - SDL2 + ffmpeg for pygame rendering and GIF/MP4 recording
#   - Xvfb for headless rendering (so --record works without an X server)
#
# Build (default CUDA 12.x):
#   docker build -t cuda-pi .
#
# Build for older CUDA stacks:
#   docker build --build-arg CUDA_VERSION=11.8.0 \
#                --build-arg CUPY_PACKAGE=cupy-cuda11x \
#                -t cuda-pi:cu11 .
#
# Run (requires NVIDIA Container Toolkit on the host):
#   docker run --rm --gpus all \
#              -v "$PWD/results:/app/results" \
#              -v "$PWD/gifs:/app/gifs" \
#              cuda-pi \
#              python3 runners/double_pendulum_swingup_cuda.py --bins 25
#
# Headless render (recording without an X server):
#   docker run --rm --gpus all -v "$PWD/gifs:/app/gifs" cuda-pi \
#       xvfb-run -a python3 runners/pendulum_cuda.py \
#                --record gifs/pendulum.gif --episodes 3 --no-plot

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG CUPY_PACKAGE=cupy-cuda12x

# ── System packages ───────────────────────────────────────────────────────────
# - python3 + pip            : the runtime
# - build-essential, gcc     : for any wheels that compile from source
# - libsdl2-*                : pygame backend
# - libfreetype6, libpng     : matplotlib + pygame text rendering
# - ffmpeg                   : MP4 recording via imageio[ffmpeg]
# - xvfb, x11-utils          : headless display for `--record` without an X server
# - git                      : in case the user clones inside the container
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        gcc \
        libsdl2-2.0-0 \
        libsdl2-image-2.0-0 \
        libsdl2-mixer-2.0-0 \
        libsdl2-ttf-2.0-0 \
        libfreetype6 \
        libpng16-16 \
        libjpeg-turbo8 \
        ffmpeg \
        xvfb \
        x11-utils \
        git \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# Upgrade pip so it understands modern wheel formats.
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install requirements WITHOUT cupy first (we install the right cupy variant
# below based on the CUPY_PACKAGE build arg).
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir \
        $(grep -v '^cupy-' /app/requirements.txt | sed 's/#.*$//') \
    && python3 -m pip install --no-cache-dir "${CUPY_PACKAGE}"

# ── Project files ─────────────────────────────────────────────────────────────
COPY src/      /app/src/
COPY runners/  /app/runners/
COPY utils/    /app/utils/
COPY README.md /app/README.md

# results/ and gifs/ are runtime outputs -- mount them as volumes from the host.
RUN mkdir -p /app/results /app/gifs

# ── Runtime config ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# SDL_VIDEODRIVER=dummy lets pygame initialise without an X server when only
# recording GIFs (no live --render). When --render is needed, run via xvfb-run
# or pass DISPLAY through and unset this variable.
ENV SDL_VIDEODRIVER=dummy

WORKDIR /app

# Default command lists the available runners; the user is expected to pass
# an explicit `python3 runners/<runner>.py ...` command at `docker run` time.
CMD ["bash", "-lc", "echo 'Available runners:' && ls runners/*.py | sed 's|runners/||;s|.py$||' && echo && echo 'Run one with: docker run --gpus all <image> python3 runners/<runner>.py --random 5'"]
