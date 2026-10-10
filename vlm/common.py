"""Prompt, model loading and metrics for train.py, predict.py and yolo_errors.py."""
import json
import re

import numpy as np

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = "Read this analog gauge. Reply with the value the needle points to, as a number only."
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def messages(answer=None):
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
    if answer is None:
        return [user]
    return [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]


def load(model_id=MODEL, adapter=None):
    """Load Qwen3-VL with the language model in 4-bit NF4.
    The vision encoder and lm_head stay in bf16: they are small and sensitive to quantization."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=bnb, dtype=torch.bfloat16,
                                                        device_map={"": 0})
    processor = AutoProcessor.from_pretrained(model_id)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, processor


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse(text):
    m = NUMBER.search(text.replace(",", ""))
    return float(m.group()) if m else None


def summary(rows):
    """Calculate the metrics from rows with value, pred and full_scale.
    pred is None for an unparsable answer. Such rows are excluded from the mean errors and fail all tolerances."""
    ok = [r for r in rows if r["pred"] is not None]
    err = np.array([abs(r["pred"] - r["value"]) for r in ok])
    pct = np.array([e / r["full_scale"] * 100 for e, r in zip(err, ok)])
    n = len(rows)
    out = {"n": n, "unparsed": n - len(ok)}
    if ok:
        out.update({"mae": float(err.mean()), "mean_pct_fs": float(pct.mean()), "median_pct_fs": float(np.median(pct)),
                    "p95_pct_fs": float(np.percentile(pct, 95))})
        for t in (1, 2, 5):
            out[f"within_{t}pct_fs"] = float((pct <= t).sum() / n)
    return out


def show(name, s):
    if "mae" not in s:
        return print(f"{name}: n={s['n']}, no parseable answers")
    print(f"{name}: n={s['n']}  unparsed {s['unparsed']}  MAE {s['mae']:.3f} units  "
          f"mean {s['mean_pct_fs']:.2f}% / median {s['median_pct_fs']:.2f}% / p95 {s['p95_pct_fs']:.2f}% of full scale  "
          f"within 1/2/5% FS: {s['within_1pct_fs']:.1%} / {s['within_2pct_fs']:.1%} / {s['within_5pct_fs']:.1%}")
