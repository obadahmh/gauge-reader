#!/usr/bin/env bash
# Run a VLM command as a single-GPU Kubernetes Job.
# Usage: [TAG=..] [MEM=14Gi] vlm/submit.sh NAME python vlm/<script>.py ARGS...
# Example: vlm/submit.sh zeroshot python vlm/predict.py /data/datasets/gauges_vlm test /data/runs/vlm/zeroshot
# Logs: kubectl logs -f job/vlm-NAME
# k3d mounts the repo's datasets/ and runs/ at /data/datasets and /data/runs.
set -euo pipefail
NAME=$1; shift
TAG=${TAG:-$(git rev-parse --short HEAD)}
MEM=${MEM:-14Gi}
ARGS=$(printf '"%s", ' "$@"); ARGS="[${ARGS%, }]"

kubectl apply -f - <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: vlm-$NAME
  labels: {app: gauge-vlm}
spec:
  backoffLimit: 2
  template:
    metadata:
      labels: {app: gauge-vlm, run: "$NAME"}
    spec:
      restartPolicy: Never
      runtimeClassName: nvidia
      securityContext:
        runAsUser: $(id -u)
        runAsGroup: $(id -g)
      containers:
      - name: vlm
        image: gauge-vlm:$TAG
        imagePullPolicy: Never
        command: $ARGS
        env:
        - {name: WANDB_PROJECT, value: gauge-vlm}
        - name: WANDB_API_KEY
          valueFrom: {secretKeyRef: {name: wandb, key: api-key, optional: true}}
        resources:
          requests: {nvidia.com/gpu: 1, memory: $MEM, cpu: "4"}
          limits:   {nvidia.com/gpu: 1, memory: $MEM}
        volumeMounts:
        - {name: datasets, mountPath: /data/datasets, readOnly: true}
        - {name: runs, mountPath: /data/runs}
        - {name: shm, mountPath: /dev/shm}
      volumes:
      - {name: datasets, hostPath: {path: /data/datasets, type: Directory}}
      - {name: runs, hostPath: {path: /data/runs, type: Directory}}
      - {name: shm, emptyDir: {medium: Memory, sizeLimit: 2Gi}}
YAML
