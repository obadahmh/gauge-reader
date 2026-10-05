"""Generate a config entry for one gauge by clicking on a photo of it.

    python calibrate.py photo.jpg --name P-101 --unit bar

Clicks, in order: dial center, any point on the dial rim, then two or more
scale marks going clockwise. Press Enter when done, then type the value of
each scale mark in the terminal. Paste the printed YAML under `gauges:`.
"""
import argparse

import cv2
import numpy as np
import yaml

from gauge import detect_tags, needle_angle

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--name", required=True)
ap.add_argument("--unit", required=True)
ap.add_argument("--tag", type=int, help="tag id, defaults to the first one found")
ap.add_argument("--span", type=float, default=6.0, help="view extent around the tag, in tag sides")
ap.add_argument("--view", type=int, default=900, help="window size in px")
args = ap.parse_args()

img = cv2.imread(args.image)
tags = detect_tags(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
if not tags:
    raise SystemExit("no tag found")
tag = args.tag if args.tag is not None else next(iter(tags))

# Front-on view with the tag in the middle, s px per tag side.
s = args.view / (2 * args.span)
off = args.span - 0.5
unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
H = cv2.getPerspectiveTransform(tags[tag].astype(np.float32), (unit + off) * s)
view = cv2.warpPerspective(img, H, (args.view, args.view))

clicks = []
labels = ["center", "rim"]


def on_click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        cv2.circle(view, (x, y), 4, (0, 0, 255), -1)
        name = labels[len(clicks) - 1] if len(clicks) <= 2 else f"mark {len(clicks) - 2}"
        cv2.putText(view, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


cv2.namedWindow("calibrate")
cv2.setMouseCallback("calibrate", on_click)
while cv2.waitKey(20) not in (13, 10, ord("q")):
    cv2.imshow("calibrate", view)
cv2.destroyAllWindows()
if len(clicks) < 4:
    raise SystemExit("need center, rim and at least two scale marks")

pts = np.array(clicks) / s - off  # view px to tag sides
center = pts[0]
scale = [[round(needle_angle(center, p), 1), float(input(f"value at mark {i + 1}: "))]
         for i, p in enumerate(pts[2:])]
entry = {tag: dict(name=args.name, unit=args.unit,
                   center=[round(float(c), 3) for c in center],
                   radius=round(float(np.linalg.norm(pts[1] - center)), 3),
                   scale=scale)}
print(yaml.safe_dump(entry, sort_keys=False, default_flow_style=None))
