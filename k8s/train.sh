#!/usr/bin/env bash
# Runs one training inside a pod. Settings come from environment variables set in the Job.
set -euo pipefail
: "${RUN_NAME:?set RUN_NAME}"
OUT=/data/runs/pose/$RUN_NAME

# A finished run leaves a DONE marker; a restarted pod then exits instead of training again
if [ -f "$OUT/DONE" ]; then echo "$RUN_NAME already finished"; exit 0; fi

# data.yaml on disk has the machine's path; inside the pod the dataset is at /data/datasets/gauges
sed "s#^path: .*#path: /data/datasets/gauges#" data.yaml > /tmp/data.yaml

if [ -f "$OUT/weights/last.pt" ]; then
  echo "Resuming $RUN_NAME from last.pt"
  yolo pose train resume model="$OUT/weights/last.pt"
else
  yolo pose train data=/tmp/data.yaml model="${MODEL:-yolo26n-pose.pt}" \
    imgsz="${IMGSZ:-320}" epochs="${EPOCHS:-100}" batch="${BATCH:-64}" \
    workers="${WORKERS:-3}" device="${DEVICE:-0}" \
    project=/data/runs/pose name="$RUN_NAME" exist_ok=True ${EXTRA_ARGS:-}
fi
touch "$OUT/DONE"
