"""Evaluate the YOLO needle model in reading units on the VLM test set.

    python vlm/yolo_errors.py runs/pose/n416/weights/best.pt datasets/gauges_vlm runs/vlm/yolo

Run on the host. Converts needle angle error to reading error with the true SyncG scale (units per degree).
This assumes perfect calibration, so it is a best case for YOLO. Uses the 320 px images in datasets/gauges/images/val.
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl, show, summary  # noqa: E402
from gauge import Needle, needle_angle  # noqa: E402


def angle_err(a, b):
    e = abs(a - b) % 360
    return min(e, 360 - e)


def main(weights, vlm_data, out, yolo_data="datasets/gauges"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = Needle(weights)
    rows, t0 = read_jsonl(Path(vlm_data) / "test.jsonl"), time.time()
    for r in rows:
        lp = Path(yolo_data) / "labels" / "val" / f"{r['stem']}.txt"
        v = list(map(float, lp.read_text().split()))
        im = cv2.imread(str(lp).replace("/labels/", "/images/").replace(".txt", ".jpg"))
        res = model(im)
        if res is None:
            r["angle_err"], r["pred"] = 180.0, None  # no detection: fails all tolerances, like an unparsable VLM answer
            continue
        pr = (res[0] / im.shape[1], res[1] / im.shape[0])
        r["angle_err"] = angle_err(needle_angle(*pr), needle_angle(v[5:7], v[8:10]))
        r["pred"] = r["value"] + r["angle_err"] * r["units_per_degree"]  # only the magnitude matters
    s = summary(rows) | {"sec_per_image": (time.time() - t0) / len(rows), "weights": weights}
    with open(out / "test_pred.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "gauge_type", "value", "angle_err", "abs_err", "pct_fs"])
        for r in rows:
            e = None if r["pred"] is None else abs(r["pred"] - r["value"])
            w.writerow([r["stem"], r["gauge_type"], r["value"], r["angle_err"], e,
                        None if e is None else e / r["full_scale"] * 100])
    (out / "test_summary.json").write_text(json.dumps(s, indent=1))
    show("test (YOLO)", s)


if __name__ == "__main__":
    main(*sys.argv[1:])
