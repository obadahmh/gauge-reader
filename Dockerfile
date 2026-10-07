FROM python:3.12-slim

# OpenCV (installed with Ultralytics) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# PyTorch built for CUDA 12.6: the newest build this server's driver (CUDA 12.8) supports
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# The pod runs as your user, not root; point caches and settings at a writable folder
ENV HOME=/tmp YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/mpl
ENTRYPOINT ["bash", "k8s/train.sh"]
