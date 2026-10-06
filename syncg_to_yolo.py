"""Convert SyncG (huggingface.co/datasets/YihengDeng/syncG) to front-on YOLO pose crops.

    python syncg_to_yolo.py path/to/syncG datasets/gauges

Each image is warped to a front-on square with the homography SyncG provides,
so training crops look like the tag-rectified crops seen at runtime.
Keypoints: 0 = needle base, 1 = needle tip.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from gauge import CROP

cv2.setNumThreads(1)  # one process per core instead


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
        images = {p.stem: p for p in (src / "images" / split).iterdir()}
        jobs = [(jp, images[jp.stem], dst / "images" / out / f"{jp.stem}.jpg", dst / "labels" / out / f"{jp.stem}.txt")
                for jp in sorted((src / "annotations" / split).glob("*.json")) if jp.stem in images]
        with Pool() as pool:
            n = sum(tqdm(pool.imap_unordered(work, jobs, chunksize=16), total=len(jobs), desc=split))
        print(f"{split}: {n} of {len(jobs)} converted")


def work(job):
    jp, ip, img_out, label_out = job
    crop, label = convert(json.loads(jp.read_text()), cv2.imread(str(ip)))
    if crop is None:
        return 0
    cv2.imwrite(str(img_out), crop)
    label_out.write_text(label)
    return 1


if __name__ == "__main__":
    main(*sys.argv[1:3])