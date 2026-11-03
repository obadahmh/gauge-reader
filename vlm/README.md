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
| `sample_pairs.py` | Samples 8 readings per pool image from the SFT model and makes DPO preference pairs. |
| `dpo.py` | DPO of the SFT adapter on the preference pairs. |
| `paired.py` | Paired comparison of two runs on the same images: win counts, McNemar test, bootstrap intervals. |
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

## Results

SyncG test set (4,000 dials). Error is % of full scale (FS).

| Model | Mean error | Median | p95 | Within 1% FS | Within 2% FS | Within 5% FS |
|---|---|---|---|---|---|---|
| YOLO needle model + true scale | 0.40% | 0.30% | 1.07% | 93.9% | 99.5% | 100% |
| Qwen3-VL-8B, zero-shot | 11.83% | 4.64% | 45.79% | 14.3% | 27.2% | 51.8% |
| Qwen3-VL-8B, QLoRA | 1.40% | 0.70% | 3.37% | 64.5% | 87.4% | 97.5% |

Real photos (45, full frame, no crop or calibration):

| Model | Mean error | Median | Within 1% FS | Within 2% FS | Within 5% FS |
|---|---|---|---|---|---|
| Qwen3-VL-8B, zero-shot | 8.12% | 3.64% | 17.8% | 28.9% | 66.7% |
| Qwen3-VL-8B, QLoRA | 5.05% | 1.00% | 48.9% | 82.2% | 95.6% |

Training: 4,000 images, 2 epochs, one RTX A4000 at a 100 W power cap, approximately 2 hours, 8.6 GiB peak GPU memory.
YOLO remains more accurate on SyncG. The VLM reads the value directly, with no per-gauge calibration.
On real photos, two large errors raise the mean above the 95th percentile.

## DPO (preference tuning)

DPO (Direct Preference Optimization) trains the model to give a higher probability to a preferred answer ("chosen") than to a worse answer ("rejected") for the same input. A frozen reference model keeps the trained model near its start point. Here, the reference is the SFT adapter.

Data: the pool split, 3,000 SyncG train images that SFT did not see. No new labels are necessary: the true reading of each image is known.

1. `sample_pairs.py`: the SFT model reads each pool image 8 times with sampling (temperature 1.0). The reading with the smallest error is "chosen". The reading with the largest error, or an unparsable answer, is "rejected". A pair is kept only if the chosen error is at most 1% FS and the rejected error is at least 1% FS larger. The pairs are on-policy: they come from the model that DPO trains.
2. `dpo.py`: continues the SFT LoRA adapter (rank 8) with the DPO loss. beta 0.1, LR 2e-5, cosine schedule, 16 pairs per optimizer step, 1 epoch. 100 pairs are held out for the eval loss and the pair accuracy. The reference log-probabilities are calculated once before training. Loss and format are the same as in SFT: the loss covers only the answer tokens.
3. `predict.py` with `--adapter <dpo run>/adapter`: the same test and real metrics as above.

```
python vlm/make_data.py ~/data/syncG_raw/syncG datasets/gauges_vlm --splits pool       # on the host
vlm/submit.sh pairs python vlm/sample_pairs.py /data/datasets/gauges_vlm /data/runs/vlm/dpo --adapter /data/runs/vlm/r8/adapter
vlm/submit.sh dpo python vlm/dpo.py /data/datasets/gauges_vlm /data/runs/vlm/dpo/pairs.jsonl /data/runs/vlm/dpo/b0.1 --adapter /data/runs/vlm/r8/adapter
vlm/submit.sh dpo-test python vlm/predict.py /data/datasets/gauges_vlm test /data/runs/vlm/dpo/b0.1 --adapter /data/runs/vlm/dpo/b0.1/adapter
```

### DPO results

Pairs: 3,000 pool images, 8 samples each, 2,382 pairs (79%). A single sample has a 2.61% FS mean error. The best of 8 samples has a 0.51% FS mean error. Sampling time: 2.2 s per image.

Training: 2,282 pairs, 143 optimizer steps, 1 hour, 8.0 GiB peak GPU memory. On the 100 held-out pairs, the eval loss decreased from 0.693 to 0.273, and the share of pairs where the model prefers the chosen answer increased from 50% to 88%.

Greedy decoding, the same images as above:

| Split | Model | Mean error | Median | p95 | Within 1% FS | Within 2% FS | Within 5% FS |
|---|---|---|---|---|---|---|---|
| SyncG test (4,000) | SFT | 1.40% | 0.70% | 3.37% | 64.5% | 87.4% | 97.5% |
| SyncG test (4,000) | SFT + DPO | 1.37% | 0.66% | 3.14% | 66.0% | 88.7% | 97.3% |
| Real photos (45) | SFT | 5.05% | 1.00% | 2.44% | 48.9% | 82.2% | 95.6% |
| Real photos (45) | SFT + DPO | 4.78% | 0.64% | 2.87% | 66.7% | 91.1% | 95.6% |

Paired comparison (`paired.py`). McNemar p-values are two-sided. The intervals are 95% paired bootstrap intervals.

| Split | Within 1% FS, DPO minus SFT | McNemar p, 1% FS | McNemar p, 2% FS | Mean error, DPO minus SFT |
|---|---|---|---|---|
| SyncG test | +1.5 points [+0.2, +2.8] | 0.020 | 0.003 | −0.03% FS [−0.16, +0.08] |
| Real photos | +17.8 points [+4.4, +31.1] (22 to 30 photos) | 0.039 | 0.219 | −0.27% FS [−0.46, −0.08] |

DPO increases the share of readings within 1% and 2% FS on SyncG. The increase is small but statistically significant. The change of the mean error on SyncG is not significant. On the real photos, the increase is larger, but the sample is small and the interval is wide. No hyperparameter was tuned on the test or real splits: this is the only DPO run.
