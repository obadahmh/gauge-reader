# VLM gauge reading (QLoRA)

This folder tests a different method. A vision-language model (VLM) reads the full gauge. Can a general VLM learn this task? Is it better or worse than the YOLO needle model?

- Model: [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
- Method: QLoRA. The language model is in 4-bit NF4. Only small LoRA adapters are trained.
- Hardware: one RTX A4000 (16 GB) for each Kubernetes Job.

The VLM gets a front-on dial image and this question: "Read this analog gauge. Reply with the value the needle points to, as a number only." The VLM must find the needle, read the scale and interpolate the value. YOLO finds only the needle. For YOLO, the scale comes from a calibration.

## Files

- `make_data.py`: Makes 512 px dial images from SyncG, with the same warp as `syncg_to_yolo.py`. It also writes JSONL files with the true reading and the scale. The test split has the same dials as the YOLO val folder.
- `yolo_errors.py`: Changes the YOLO angle error into a reading error. It uses the true scale of each dial (value for each degree).
- `predict.py`: Gets the VLM readings for a split, with or without an adapter. It calculates the same scores.
- `train.py`: Does the QLoRA fine-tune and sends the metrics to W&B. Settings: rank 8, alpha 16, learning rate 2e-4, cosine schedule. The loss uses only the answer tokens. The training images get small random changes (rotation, color, blur).
- `compare.py`: Shows the scores of all runs in one table.
- `Dockerfile`, `submit.sh`: The image and a Job with one GPU. They use the same pattern as `k8s/`.

## Scores

- MAE: the mean absolute error, in gauge units.
- Error as a percentage of full scale: mean, median and 95th percentile.
- The percentage of readings in a tolerance of 1%, 2% and 5% of full scale. The accuracy of most gauges is 1 to 2% of full scale.

A VLM answer with no number is a failure for all tolerances. A YOLO image with no needle is also a failure.

## Paths

k3d mounts two folders of this repo into the cluster:

| On the server | In a pod |
|---|---|
| `datasets/` | `/data/datasets` (read-only in the Job) |
| `runs/` | `/data/runs` |

The model download goes into `runs/hf-cache`, so the download occurs only one time.
