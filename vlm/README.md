# VLM gauge reading (QLoRA)

End-to-end gauge reading with a vision-language model (VLM), compared with the YOLO needle model.

- Model: [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- Method: QLoRA. The language model is quantized to 4-bit NF4 and frozen. Only LoRA adapters are trained.
- Hardware: one RTX A4000 (16 GB) per Kubernetes Job

Input: a rectified dial image and the prompt "Read this analog gauge. Reply with the value the needle points to, as a number only." The VLM must locate the needle, read the scale and interpolate. YOLO only locates the needle. Its angle-to-value mapping comes from calibration.

## Files

| File | Function |
|---|---|
| `make_data.py` | Builds 512 px rectified images and JSONL labels from SyncG. The test split is the YOLO val set. |
| `yolo_errors.py` | Converts YOLO angle error to reading error with the true scale of each dial. |
| `predict.py` | Runs inference, zero-shot or with an adapter, and calculates the metrics. |
| `train.py` | QLoRA fine-tune: rank 8, alpha 16, LR 2e-4, cosine schedule, answer-only loss, light augmentation. Logs to W&B. |
| `compare.py` | Tabulates the metrics of all runs. |
| `Dockerfile`, `submit.sh` | Training image and single-GPU Job, as in `k8s/`. |

## Metrics

- MAE in gauge units
- Error as % of full scale (FS): mean, median, p95
- Share of readings within 1%, 2% and 5% FS. Typical gauge accuracy is 1 to 2% FS.

Unparsable VLM answers and missed YOLO detections fail all tolerances.

## Paths

| Host | Pod |
|---|---|
| `datasets/` | `/data/datasets` (read-only) |
| `runs/` | `/data/runs` |

The model cache is `runs/hf-cache`.
