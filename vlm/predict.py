"""Run VLM inference on a split and calculate the metrics.

    python vlm/predict.py /data/datasets/gauges_vlm test /data/runs/vlm/zeroshot
    python vlm/predict.py /data/datasets/gauges_vlm test /data/runs/vlm/r8 --adapter /data/runs/vlm/r8/adapter

Run in a pod (see submit.sh). Writes <out>/<split>_pred.csv (per-image predictions and raw answers)
and <out>/<split>_summary.json. Logs to W&B if WANDB_API_KEY is set.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL, load, messages, parse, read_jsonl, show, summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("split", help="test, real or val")
    ap.add_argument("out")
    ap.add_argument("--adapter")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, help="only the first N images, for a quick check")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.data) / f"{args.split}.jsonl")[:args.limit]
    model, processor = load(args.model, args.adapter)
    model.eval()
    processor.tokenizer.padding_side = "left"  # left padding for batched generation
    prompt = processor.apply_chat_template(messages(), add_generation_prompt=True, tokenize=False)

    t0 = time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        images = [Image.open(Path(args.data) / r["image"]).convert("RGB") for r in chunk]
        inputs = processor(text=[prompt] * len(chunk), images=images, padding=True, return_tensors="pt").to(0)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        texts = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(chunk, texts):
            r["raw"], r["pred"] = t.strip(), parse(t)
        done = i + len(chunk)
        if done % (args.batch * 25) == 0 or done == len(rows):
            print(f"{done}/{len(rows)}  {(time.time() - t0) / done:.2f} s per image", flush=True)
    sec_per_image = (time.time() - t0) / max(len(rows), 1)

    with open(out / f"{args.split}_pred.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "gauge_type", "value", "pred", "abs_err", "pct_fs", "raw"])
        for r in rows:
            e = None if r["pred"] is None else abs(r["pred"] - r["value"])
            w.writerow([r["stem"], r["gauge_type"], r["value"], r["pred"], e,
                        None if e is None else e / r["full_scale"] * 100, r["raw"]])
    s = summary(rows) | {"sec_per_image": sec_per_image, "adapter": args.adapter, "model": args.model}
    (out / f"{args.split}_summary.json").write_text(json.dumps(s, indent=1))
    show(f"{args.split} ({'fine-tuned' if args.adapter else 'zero-shot'})", s)

    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "gauge-vlm"), job_type="eval",
                         name=f"{out.name}-{args.split}", config=vars(args))
        run.summary.update({f"{args.split}/{k}": v for k, v in s.items() if isinstance(v, (int, float))})
        run.log({f"{args.split}/predictions": wandb.Table(
            columns=["stem", "value", "pred", "raw"], data=[[r["stem"], r["value"], r["pred"], r["raw"]] for r in rows])})
        run.finish()


if __name__ == "__main__":
    main()
