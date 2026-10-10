"""Calculate the YOLO scores in reading units, on the same test images as the VLM.

    python vlm/yolo_errors.py runs/pose/n416/weights/best.pt datasets/gauges_vlm runs/vlm/yolo

Run this script on the server, not in a pod.
YOLO gives a needle angle. The true scale of each SyncG dial gives the value for each degree.
The script uses this scale to change the angle error into a reading error.
This method uses a perfect calibration. Real gauges do not have a perfect calibration.
Thus the YOLO scores are slightly better than in real use.
The script uses the 320 px images in datasets/gauges/images/val. These are the YOLO test images.
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
            r["angle_err"], r["pred"] = 180.0, None  # no needle found: this counts as a failure, as for a VLM answer with no number
            continue
        pr = (res[0] / im.shape[1], res[1] / im.shape[0])
        r["angle_err"] = angle_err(needle_angle(*pr), needle_angle(v[5:7], v[8:10]))
        r["pred"] = r["value"] + r["angle_err"] * r["units_per_degree"]  # only the size of the error is important
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
