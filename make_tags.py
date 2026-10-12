"""Make a printable A4 sheet of AprilTag 36h11 markers, labelled with their IDs.

    python make_tags.py                       # tags 1 to 4, 50 mm, tags_a4.pdf
    python make_tags.py --ids 5 6 --mm 65

Print the PDF at 100% ("actual size"), then check with a ruler that one tag measures --mm.
"""
import argparse

import cv2
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ids", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--mm", type=float, default=50, help="side of the black square in mm")
ap.add_argument("--out", default="tags_a4.pdf")
args = ap.parse_args()

DPI = 300
px = DPI / 25.4                                     # pixels per mm
W, H, MARGIN = 210, 297, 20                         # A4 portrait, in mm
cell_w, cell_h = (W - 2 * MARGIN) / 2, args.mm + 30   # two columns, room for white border and label
rows = -(-len(args.ids) // 2)
if 2 * MARGIN + rows * cell_h > H or args.mm + 20 > cell_w:
    raise SystemExit("Too many or too large tags for one A4 page: use fewer --ids or a smaller --mm")

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
page = np.full((round(H * px), round(W * px)), 255, np.uint8)
side = round(args.mm * px)
for i, tid in enumerate(args.ids):
    x = round((MARGIN + (i % 2) * cell_w + (cell_w - args.mm) / 2) * px)
    y = round((MARGIN + (i // 2) * cell_h + 10) * px)
    page[y:y + side, x:x + side] = cv2.aruco.generateImageMarker(d, tid, side)
    cv2.putText(page, f"id {tid}", (x, y + side + round(12 * px)), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)

# Saving as PDF with the DPI makes the page exactly A4, so printers don't rescale it
Image.fromarray(page).save(args.out, resolution=DPI)
print(f"wrote {args.out}: ids {args.ids}, {args.mm:g} mm")