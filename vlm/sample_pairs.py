"""Sample readings from the SFT model and build DPO preference pairs.

    python vlm/sample_pairs.py /data/datasets/gauges_vlm /data/runs/vlm/dpo --adapter /data/runs/vlm/r8/adapter

Run in a pod (see submit.sh). The model reads each pool image K times with sampling (temperature 1.0).
The sample with the smallest error is "chosen". The sample with the largest error is "rejected".
An unparsable sample has an infinite error. A pair is kept only if the chosen error is at most --keep % FS
and the rejected error is at least --margin % FS larger. Images with K near-equal readings give no pair.

Writes <out>/samples.jsonl (all samples, appended per batch, so a restarted pod continues),
<out>/pairs.jsonl and <out>/pairs_summary.json. --rebuild makes the pairs again from samples.jsonl with no GPU.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL, messages, parse, read_jsonl  # noqa: E402


def pct_err(text, row):
    pred = parse(text)
    return math.inf if pred is None else abs(pred - row["value"]) / row["full_scale"] * 100


def make_pairs(samples, keep, margin):
    pairs = []
    for s in samples:
        errs = [pct_err(t, s) for t in s["samples"]]
        i, j = int(np.argmin(errs)), int(np.argmax(errs))
        if errs[i] <= keep and errs[j] >= errs[i] + margin:
            pairs.append({k: s[k] for k in ("image", "stem", "value", "full_scale")}
                         | {"chosen": s["samples"][i], "rejected": s["samples"][j],
                            "chosen_err": errs[i], "rejected_err": errs[j] if math.isfinite(errs[j]) else None})
    return pairs


def summarize(samples, pairs, keep, margin):
    errs = np.array([[pct_err(t, s) for t in s["samples"]] for s in samples])
    finite = errs[np.isfinite(errs)]
    rej = [p["rejected_err"] for p in pairs if p["rejected_err"] is not None]
    return {"images": len(samples), "pairs": len(pairs), "k": errs.shape[1], "keep_pct_fs": keep, "margin_pct_fs": margin,
            "sample_unparsed": float((~np.isfinite(errs)).mean()),
            "sample_mean_pct_fs": float(finite.mean()), "sample_median_pct_fs": float(np.median(finite)),
            "best_of_k_mean_pct_fs": float(np.mean([e.min() for e in errs if np.isfinite(e.min())])),
            "images_all_samples_equal": int(sum(len(set(s["samples"])) == 1 for s in samples)),
            "images_best_above_keep": int((errs.min(1) > keep).sum()),
            "rejected_unparsed": sum(p["rejected_err"] is None for p in pairs),
            "chosen_mean_pct_fs": float(np.mean([p["chosen_err"] for p in pairs])) if pairs else None,
            "rejected_median_pct_fs": float(np.median(rej)) if rej else None}


def generate(args, rows, out_file):
    import torch
    from PIL import Image

    from common import load
    model, processor = load(args.model, args.adapter)
    model.eval()
    processor.tokenizer.padding_side = "left"  # left padding for batched generation
    prompt = processor.apply_chat_template(messages(), add_generation_prompt=True, tokenize=False)
    torch.manual_seed(args.seed)
    t0, done = time.time(), 0
    with open(out_file, "a") as f:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            images = [Image.open(Path(args.data) / r["image"]).convert("RGB") for r in chunk]
            # Each image is repeated K times in the batch, so all K samples see the same prompt.
            inputs = processor(text=[prompt] * (len(chunk) * args.k), images=[im for im in images for _ in range(args.k)],
                               padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=16, do_sample=True, temperature=args.temperature,
                                     top_p=1.0, top_k=0)
            texts = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for n, r in enumerate(chunk):
                f.write(json.dumps({k: r[k] for k in ("image", "stem", "value", "full_scale", "gauge_type")}
                                   | {"samples": [t.strip() for t in texts[n * args.k:(n + 1) * args.k]]}) + "\n")
            f.flush()
            done += len(chunk)
            if done % (args.batch * 25) == 0 or i + len(chunk) == len(rows):
                print(f"{done}/{len(rows)}  {(time.time() - t0) / done:.2f} s per image", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--split", default="pool")
    ap.add_argument("--adapter", help="the SFT adapter")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--k", type=int, default=8, help="samples per image")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4, help="images per batch (the batch has batch x k sequences)")
    ap.add_argument("--keep", type=float, default=1.0, help="largest chosen error, %% FS")
    ap.add_argument("--margin", type=float, default=1.0, help="smallest rejected-minus-chosen error, %% FS")
    ap.add_argument("--limit", type=int, help="only the first N images, for a quick check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true", help="make pairs from samples.jsonl only, with no generation")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples_file = out / "samples.jsonl"
    if not args.rebuild:
        rows = read_jsonl(Path(args.data) / f"{args.split}.jsonl")[:args.limit]
        have = {s["stem"] for s in read_jsonl(samples_file)} if samples_file.exists() else set()
        todo = [r for r in rows if r["stem"] not in have]
        print(f"{len(rows)} images, {len(have)} already sampled, {len(todo)} to do")
        if todo:
            generate(args, todo, samples_file)

    samples = read_jsonl(samples_file)
    pairs = make_pairs(samples, args.keep, args.margin)
    with open(out / "pairs.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in pairs)
    s = summarize(samples, pairs, args.keep, args.margin)
    (out / "pairs_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
