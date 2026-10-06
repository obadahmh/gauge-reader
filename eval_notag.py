"""Test the needle model on photos without AprilTags, by clicking each dial.

    python eval_notag.py path/to/best.pt photos/

photos/truth.csv has columns file,value,min,max: the value you read off the
gauge and the values at the two ends of its scale. For each photo, click the
dial center, a point on the rim, the min mark and the max mark (Esc skips).
There is no perspective correction, so photos should be roughly head-on.
The scale is assumed linear between min and max.
"""
import csv
import os
import sys

import cv2
import numpy as np

from gauge import CROP, Needle, angle_to_value, needle_angle

STEPS = ["center", "rim", "min mark", "max mark"]


def click(img, view=1000):
    s = view / max(img.shape[:2])
    disp = cv2.resize(img, None, fx=s, fy=s)
    pts = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(STEPS):
            cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(disp, STEPS[len(pts)], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            pts.append((x / s, y / s))

    cv2.namedWindow("click")
    cv2.setMouseCallback("click", on_click)
    while len(pts) < len(STEPS):
        cv2.imshow("click", disp)
        cv2.setWindowTitle("click", f"click: {STEPS[len(pts)]}  (Esc skips)")
        if cv2.waitKey(20) == 27:
            return None
    return np.array(pts)


def crop(img, center, radius, size=CROP, margin=1.15):
    """Square crop centered on the dial, same framing as the tag-rectified crops."""
    half = radius * margin
    s = size / (2 * half)
    M = np.float32([[s, 0, (half - center[0]) * s], [0, s, (half - center[1]) * s]])
    return cv2.warpAffine(img, M, (size, size))


def main(weights, folder):
    model = Needle(weights)
    os.makedirs("crops_notag", exist_ok=True)
    errs = []
    for row in csv.DictReader(open(os.path.join(folder, "truth.csv"))):
        stem = os.path.splitext(row["file"])[0]
        img = cv2.imread(os.path.join(folder, row["file"]))
        if img is None:
            img = cv2.imread(os.path.join(folder, stem + ".jpg"))  # truth.csv may still list the .HEIC name
        if img is None:
            print(f"{row['file']}: can't read it (HEIC? run heic_to_jpg.py first)")
            continue
        pts = click(img)
        if pts is None:
            print(f"{row['file']}: skipped")
            continue
        c, rim, lo, hi = pts
        dial = crop(img, c, np.linalg.norm(rim - c))
        out = model(dial)
        if out is None:
            print(f"{row['file']}: no needle found")
            continue
        base, tip, _ = out
        vmin, vmax = float(row["min"]), float(row["max"])
        value = angle_to_value(needle_angle(base, tip), [[needle_angle(c, lo), vmin], [needle_angle(c, hi), vmax]])
        e = abs(value - float(row["value"])) / (vmax - vmin) * 100
        errs.append(e)
        cv2.line(dial, tuple(map(int, base)), tuple(map(int, tip)), (0, 0, 255), 2)
        cv2.imwrite(f"crops_notag/{stem}.jpg", dial)
        print(f"{row['file']}: true {row['value']}  pred {value:.2f}  error {e:.2f}% of full scale")
    cv2.destroyAllWindows()
    if errs:
        print(f"median {np.median(errs):.2f}%  max {max(errs):.2f}% of full scale over {len(errs)} photos")


if __name__ == "__main__":
    main(*sys.argv[1:3])