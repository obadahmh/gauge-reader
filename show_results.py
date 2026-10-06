import glob, random, sys, cv2, numpy as np
from ultralytics import YOLO
from gauge import needle_angle
m = YOLO(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
tiles = []
for lp in random.sample(sorted(glob.glob("datasets/gauges/labels/val/*.txt")), n):
    v = list(map(float, open(lp).read().split()))
    im = cv2.imread(lp.replace("/labels/", "/images/").replace(".txt", ".jpg"))
    s = im.shape[0]
    gt = np.array([v[5:7], v[8:10]])
    r = m(im, imgsz=320, verbose=False)[0]
    pr = r.keypoints.xyn[int(r.boxes.conf.argmax())].cpu().numpy()[:2]
    for (b, t), c in ((gt, (0, 200, 0)), (pr, (0, 0, 255))):
        cv2.line(im, tuple((b * s).astype(int)), tuple((t * s).astype(int)), c, 2)
    e = abs(needle_angle(*pr) - needle_angle(*gt)) % 360
    cv2.putText(im, f"err {min(e, 360 - e):.2f} deg", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    tiles.append(im)
cv2.imwrite("results_grid.jpg", np.hstack(tiles))
print("saved results_grid.jpg")
