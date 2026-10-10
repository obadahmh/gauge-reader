"""Measure how well the model reads gauges.

    python eval.py synth path/to/best.pt            # SyncG val set: needle angle error
    python eval.py real path/to/best.pt photos/     # your photos: value error against photos/truth.csv

synth saves the worst predictions (worst_*.jpg) and 5 random ones (grid.jpg),
red = model, green = label.
real needs photos/truth.csv with columns file,value (and optionally tag, if a
photo shows more than one gauge). Each flattened dial is saved to crops/<photo>/
with the predicted needle drawn in red.
"""
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np
import yaml

from gauge import Needle, needle_angle, read_frames


def angle_err(a, b):
    e = abs(a - b) % 360
    return min(e, 360 - e)


def draw(path, gt, pr, err):
    im = cv2.imread(path)
    s = im.shape[0]
    for kp, color in ((gt, (0, 200, 0)), (pr, (0, 0, 255))):
        if kp is not None:
            b, t = (np.array(kp) * s).astype(int)
            cv2.line(im, tuple(b), tuple(t), color, 2)
    cv2.putText(im, f"err {err:.2f} deg", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return im


def synth(weights, data="datasets/gauges"):
    model = Needle(weights)
    res = []
    for lp in sorted(glob.glob(f"{data}/labels/val/*.txt")):
        v = list(map(float, open(lp).read().split()))
        ip = lp.replace("/labels/", "/images/").replace(".txt", ".jpg")
        im = cv2.imread(ip)
        gt = (v[5:7], v[8:10])
        out = model(im)
        if out is None:
            res.append((180.0, ip, gt, None))
            continue
        pr = (out[0] / im.shape[1], out[1] / im.shape[0])
        res.append((angle_err(needle_angle(*pr), needle_angle(*gt)), ip, gt, pr))
    if not res:
        raise SystemExit(f"No labels in {data}/labels/val. Run from the repo root.")
    errs = np.array([r[0] for r in res])
    print(f"n={len(errs)}  median {np.median(errs):.2f} deg  p95 {np.percentile(errs, 95):.2f} deg  "
          f"over 5 deg {(errs > 5).mean():.1%}  no detection {sum(r[3] is None for r in res)}")
    for i, (e, ip, gt, pr) in enumerate(sorted(res, key=lambda r: -r[0])[:8]):
        cv2.imwrite(f"worst_{i}_{e:.0f}deg.jpg", draw(ip, gt, pr, e))
    cv2.imwrite("grid.jpg", np.hstack([draw(ip, gt, pr, e) for e, ip, gt, pr in random.sample(res, min(5, len(res)))]))
    print("saved worst_*.jpg and grid.jpg")


def real(weights, folder):
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["min_sharpness"] = 0
    model = Needle(weights)
    errs = []
    for row in csv.DictReader(open(os.path.join(folder, "truth.csv"))):
        stem = os.path.splitext(row["file"])[0]
        im = cv2.imread(os.path.join(folder, row["file"]))
        if im is None:
            im = cv2.imread(os.path.join(folder, stem + ".jpg"))  # truth.csv may still list the .HEIC name
        if im is None:
            print(f"{row['file']}: can't read it (HEIC? run heic_to_jpg.py first)")
            continue
        os.makedirs(f"crops/{stem}", exist_ok=True)
        readings = read_frames([im], cfg, model, crop_dir=f"crops/{stem}")
        if row.get("tag"):
            readings = [r for r in readings if r["tag"] == int(row["tag"])]
        if not readings:
            print(f"{row['file']}: no reading (no known tag found, or low confidence)")
            continue
        r = readings[0]
        g = cfg["gauges"][r["tag"]]
        vals = [v for _, v in g["scale"]]
        e = abs(r["value"] - float(row["value"])) / (max(vals) - min(vals)) * 100
        errs.append(e)
        print(f"{row['file']}: true {row['value']}  pred {r['value']:.2f} {g['unit']}  error {e:.2f}% of full scale")
    if errs:
        print(f"median {np.median(errs):.2f}%  max {max(errs):.2f}% of full scale over {len(errs)} photos")


if __name__ == "__main__":
    {"synth": synth, "real": real}[sys.argv[1]](*sys.argv[2:])
