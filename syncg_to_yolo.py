"""Convert SyncG (huggingface.co/datasets/YihengDeng/syncG) to front-on YOLO pose crops.

    python syncg_to_yolo.py path/to/syncG datasets/gauges

Each image is warped to a front-on square with the homography SyncG provides,
so training crops look like the tag-rectified crops seen at runtime.
Keypoints: 0 = needle base, 1 = needle tip.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from gauge import CROP


def convert(ann, img, size=CROP):
    x0, y0, x1, y1 = ann["dial_bbox_annotations"]
    w, h = x1 - x0, y1 - y0
    to_crop = np.array([[1 / w, 0, -x0 / w], [0, 1 / h, -y0 / h], [0, 0, 1]])  # px to bbox-normalized
    M = np.diag([size, size, 1.0]) @ np.array(ann["homo_matrix"]) @ to_crop
    crop = cv2.warpPerspective(img, M, (size, size))
    p = next(k for k in ann["keypoints_annotations"] if k["type"] == "Pointer")
    kp = cv2.perspectiveTransform(np.float32([[p["origin_kp"], p["outside_kp"]]]), M)[0] / size
    if not ((kp > 0) & (kp < 1)).all():
        return None, None
    (bx, by), (tx, ty) = kp
    return crop, f"0 0.5 0.5 1 1 {bx:.5f} {by:.5f} 2 {tx:.5f} {ty:.5f} 2\n"


def main(src, dst):
    src, dst = Path(src), Path(dst)
    for split, out in (("train", "train"), ("test", "val")):
        (dst / "images" / out).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / out).mkdir(parents=True, exist_ok=True)
        n = 0
        for jp in sorted((src / "annotations" / split).glob("*.json")):
            imgs = list((src / "images" / split).glob(jp.stem + ".*"))
            if not imgs:
                continue
            crop, label = convert(json.loads(jp.read_text()), cv2.imread(str(imgs[0])))
            if crop is None:
                continue
            cv2.imwrite(str(dst / "images" / out / f"{jp.stem}.jpg"), crop)
            (dst / "labels" / out / f"{jp.stem}.txt").write_text(label)
            n += 1
        print(f"{split}: {n} crops")


if __name__ == "__main__":
    main(*sys.argv[1:3])
