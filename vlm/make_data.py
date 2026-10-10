"""Make the VLM dataset. Each item is a front-on dial image, a question and the true reading.

    python vlm/make_data.py ~/data/syncG_raw/syncG datasets/gauges_vlm --real test/

Run this script on the server, not in a pod.
The script uses the same warp as syncg_to_yolo.py. Thus the test images show the same dials as the YOLO test images.
The image size is 512 px, not 320 px. At 512 px, the numbers on the scale are easier to read.

For each split, the script writes images/<split>/*.jpg and <split>.jsonl. Each JSONL line has these fields:
- value: the true reading.
- start and full_scale: these give the error as a percentage of full scale.
- units_per_degree: this changes the YOLO angle error into a reading error.

The splits are:
- train: --train-n random images from the SyncG train set.
- val: --val-n other images from the SyncG train set. Use them only for the validation loss.
- test: all SyncG test images that the warp accepts. This is the same set as the YOLO val folder.
- real: your photos. The script makes this split only if you give --real.
"""
import argparse
import csv
import json
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syncg_to_yolo import convert  # noqa: E402

cv2.setNumThreads(1)
SIZE = 512


def answer(value, interval):
    """Write the reading as text. The precision is approximately 1/100 of a major interval."""
    decimals = max(0, 2 - math.floor(math.log10(interval))) if interval > 0 else 2
    return f"{value:.{decimals}f}"


def work(job):
    jp, ip, out = job
    ann = json.loads(jp.read_text())
    crop, _ = convert(ann, cv2.imread(str(ip)), size=SIZE)
    if crop is None:
        return None
    cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    step = float(ann["long_interval_value"])
    return dict(image=f"images/{out.parent.name}/{out.name}", stem=jp.stem,
                value=float(ann["ground_truth"]), answer=answer(float(ann["ground_truth"]), step),
                start=float(ann["start_value"]), full_scale=(int(ann["long_num"]) - 1) * step,
                units_per_degree=step / float(ann["long_interval_degree"]),
                gauge_type=ann.get("gauge_type", ""))


def build(src, dst, split, out_split, stems=None):
    (dst / "images" / out_split).mkdir(parents=True, exist_ok=True)
    images = {p.stem: p for p in (src / "images" / split).iterdir()}
    jsons = sorted((src / "annotations" / split).glob("*.json"))
    jobs = [(jp, images[jp.stem], dst / "images" / out_split / f"{jp.stem}.jpg")
            for jp in jsons if jp.stem in images and (stems is None or jp.stem in stems)]
    with Pool() as pool:
        rows = [r for r in tqdm(pool.imap(work, jobs, chunksize=16), total=len(jobs), desc=out_split) if r]
    with open(dst / f"{out_split}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{out_split}: {len(rows)} of {len(jobs)} converted")


def build_real(folder, dst, max_side=768):
    (dst / "images" / "real").mkdir(parents=True, exist_ok=True)
    rows = []
    for row in csv.DictReader(open(Path(folder) / "truth.csv")):
        stem = Path(row["file"]).stem
        im = cv2.imread(str(Path(folder) / row["file"]))
        if im is None:
            im = cv2.imread(str(Path(folder) / f"{stem}.jpg"))  # truth.csv can have the .HEIC name
        if im is None or not row["value"].strip():
            print(f"{row['file']}: skipped (can't read it, or no value)")
            continue
        s = max_side / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / "images" / "real" / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lo, hi = float(row["min"]), float(row["max"])
        rows.append(dict(image=f"images/real/{stem}.jpg", stem=stem, value=float(row["value"]),
                         answer=row["value"].strip(), start=lo, full_scale=hi - lo, units_per_degree=None,
                         gauge_type="real"))
    with open(dst / "real.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"real: {len(rows)} photos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="unpacked SyncG folder (has images/ and annotations/)")
    ap.add_argument("dst")
    ap.add_argument("--train-n", type=int, default=4000)
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--real", help="folder with your photos and truth.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    stems = sorted(p.stem for p in (src / "annotations" / "train").glob("*.json"))
    random.Random(args.seed).shuffle(stems)
    build(src, dst, "train", "train", set(stems[:args.train_n]))
    build(src, dst, "train", "val", set(stems[args.train_n:args.train_n + args.val_n]))
    build(src, dst, "test", "test")
    if args.real:
        build_real(args.real, dst)


if __name__ == "__main__":
    main()
