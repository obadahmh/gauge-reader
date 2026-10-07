#!/usr/bin/env bash
# Submit one training run as a Kubernetes Job.
# Usage: [TAG=..] [GPUS=1] [MEM=9Gi] k8s/submit.sh NAME [KEY=VALUE ...]
#   e.g. k8s/submit.sh n320 IMGSZ=320 EPOCHS=100
set -euo pipefail
NAME=$1; shift
TAG=${TAG:-$(git rev-parse --short HEAD)}
GPUS=${GPUS:-1}
MEM=${MEM:-9Gi}

# Turn KEY=VALUE arguments into environment variables for train.sh
ENV_YAML="        - {name: RUN_NAME, value: \"$NAME\"}"
for kv in "$@"; do
  ENV_YAML+=$'\n'"        - {name: ${kv%%=*}, value: \"${kv#*=}\"}"
done

kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: train-$NAME
  labels: {app: gauge-train}
spec:
  backoffLimit: 3
  template:
    metadata:
      labels: {app: gauge-train, run: "$NAME"}
    spec:
      restartPolicy: Never
      runtimeClassName: nvidia
      securityContext:
        runAsUser: $(id -u)
        runAsGroup: $(id -g)
      containers:
      - name: train
        image: gauge-train:$TAG
        imagePullPolicy: Never
        env:
        - {name: YOLO_CONFIG_DIR, value: /tmp}
$ENV_YAML
        resources:
          requests: {nvidia.com/gpu: $GPUS, memory: $MEM, cpu: "4"}
          limits:   {nvidia.com/gpu: $GPUS, memory: $MEM}
        volumeMounts:
        - {name: datasets, mountPath: /data/datasets, readOnly: true}
        - {name: runs, mountPath: /data/runs}
        - {name: shm, mountPath: /dev/shm}
      volumes:
      - {name: datasets, hostPath: {path: /data/datasets, type: Directory}}
      - {name: runs, hostPath: {path: /data/runs, type: Directory}}
      - {name: shm, emptyDir: {medium: Memory, sizeLimit: 2Gi}}
EOF
