# Build the image from the repo root: docker build -f vlm/Dockerfile -t gauge-vlm:$(git rev-parse --short HEAD) .
FROM python:3.12-slim

# PyTorch for CUDA 12.8. The driver on this server (570) supports CUDA 12.8 and older.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY vlm/requirements.txt vlm/requirements.txt
RUN pip install --no-cache-dir -r vlm/requirements.txt
COPY vlm vlm

# The pod runs as your user, not as root. Thus the caches must be in folders that your user can write to.
# HF_HOME is in runs/ on the server (see submit.sh). The model download occurs only one time.
ENV HOME=/tmp WANDB_DIR=/tmp HF_HOME=/data/runs/hf-cache PYTHONUNBUFFERED=1
