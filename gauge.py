"""Read analog gauges marked with AprilTags and publish the values.

    python gauge.py                     # grab frames from the camera, read, publish
    python gauge.py --image a.jpg       # read from image files instead
    python gauge.py --dump crops/       # save rectified crops for labeling (no model needed)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

CROP = 320  # rectified crop size in px, same as training imgsz
TAGS = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))


def detect_tags(gray):
    corners, ids, _ = TAGS.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}


def rectify(img, corners, center, radius, size=CROP, margin=1.15):
    """Warp the dial to a front-on square crop centered on the dial.

    center and radius are in tag side lengths, measured from the tag's top-left corner.
    Assumes the tag is flat and roughly coplanar with the dial face.
    """
    half = radius * margin
    s = size / (2 * half)
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    dst = (unit - np.float32(center) + half) * s
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst.astype(np.float32))
    return cv2.warpPerspective(img, H, (size, size))


def needle_angle(base, tip):
    """Degrees clockwise from 12 o'clock."""
    return float(np.degrees(np.arctan2(tip[0] - base[0], base[1] - tip[1])) % 360)


def angle_to_value(angle, scale):
    """Piecewise linear map. scale is [[angle, value], ...] in clockwise order."""
    a0 = scale[0][0]
    rel = [(a - a0) % 360 for a, _ in scale]
    vals = [v for _, v in scale]
    r = (angle - a0) % 360
    if r > rel[-1]:  # needle in the dead zone: snap to the nearer end
        r = rel[-1] if r - rel[-1] < 360 - r else 0
    return float(np.interp(r, rel, vals))


class Needle:
    """Keypoint model: returns (base, tip, confidence) or None."""

    def __init__(self, path):
        from ultralytics import YOLO
        self.model = YOLO(path, task="pose")

    def __call__(self, crop):
        r = self.model(crop, imgsz=CROP, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        base, tip = r.keypoints.xy[i].cpu().numpy()[:2]
        kc = r.keypoints.conf
        conf = float(kc[i].min()) if kc is not None else float(r.boxes.conf[i])
        return base, tip, conf


def read_frames(frames, cfg, model, crop_dir=None):
    gauges, hits = cfg["gauges"], {}
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < cfg["min_sharpness"]:
            continue
        for tag, corners in detect_tags(gray).items():
            g = gauges.get(tag)
            if g is None:
                continue
            crop = rectify(frame, corners, g["center"], g["radius"])
            out = model(crop)
            if out is None or out[2] < cfg["min_conf"]:
                continue
            base, tip, conf = out
            hits.setdefault(tag, []).append(angle_to_value(needle_angle(base, tip), g["scale"]))
            if crop_dir:
                cv2.line(crop, tuple(map(int, base)), tuple(map(int, tip)), (0, 0, 255), 2)
                cv2.imwrite(str(Path(crop_dir) / f"{tag}_{time.time_ns()}.jpg"), crop)
    return [dict(gauge=gauges[t]["name"], tag=t, value=round(float(np.median(v)), 4),
                 unit=gauges[t]["unit"], n=len(v), spread=round(float(np.std(v)), 4), ts=time.time())
            for t, v in hits.items()]


def publish(readings, cfg):
    with open(cfg["log"], "a") as f:
        for r in readings:
            f.write(json.dumps(r) + "\n")
    m = cfg.get("mqtt")
    if not m or not readings:
        return
    import paho.mqtt.publish as mqtt
    msgs = [(f"{m['topic']}/{r['gauge']}", json.dumps(r), 1, False) for r in readings]
    try:
        mqtt.multiple(msgs, hostname=m["host"], port=m.get("port", 1883))
    except OSError as e:
        print(f"mqtt publish failed ({e}), readings kept in {cfg['log']}")


def grab(cfg):
    cap = cv2.VideoCapture(cfg["camera"])
    frames = [f for ok, f in (cap.read() for _ in range(cfg["frames"])) if ok]
    cap.release()
    return frames


def dump(frames, cfg, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    for frame in frames:
        for tag, corners in detect_tags(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).items():
            g = cfg["gauges"].get(tag)
            if g:
                cv2.imwrite(f"{out}/{tag}_{time.time_ns()}.jpg", rectify(frame, corners, g["center"], g["radius"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--image", nargs="*", help="read these images instead of the camera")
    ap.add_argument("--dump", metavar="DIR", help="only save rectified crops to DIR")
    ap.add_argument("--crops", metavar="DIR", help="also save annotated crops to DIR")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    frames = [cv2.imread(p) for p in args.image] if args.image else grab(cfg)
    if args.dump:
        return dump(frames, cfg, args.dump)
    if args.crops:
        Path(args.crops).mkdir(parents=True, exist_ok=True)
    readings = read_frames(frames, cfg, Needle(cfg["model"]), args.crops)
    for r in readings:
        print(json.dumps(r))
    publish(readings, cfg)


if __name__ == "__main__":
    main()
