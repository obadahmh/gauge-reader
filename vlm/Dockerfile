# Build from the repo root: docker build -f vlm/Dockerfile -t gauge-vlm:$(git rev-parse --short HEAD) .
FROM python:3.12-slim

# CUDA 12.8 build: the newest the host driver (570) supports
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY vlm/requirements.txt vlm/requirements.txt
RUN pip install --no-cache-dir -r vlm/requirements.txt
COPY vlm vlm

# The pod runs as a non-root user, so caches go to writable paths.
# HF_HOME is on the runs/ hostPath, so the model is downloaded once.
ENV HOME=/tmp WANDB_DIR=/tmp HF_HOME=/data/runs/hf-cache PYTHONUNBUFFERED=1
