import glob, sys, cv2, numpy as np
from ultralytics import YOLO
from gauge import needle_angle
m = YOLO(sys.argv[1])
res = []
for lp in sorted(glob.glob("datasets/gauges/labels/val/*.txt")):
    v = list(map(float, open(lp).read().split()))
    ip = lp.replace("/labels/", "/images/").replace(".txt", ".jpg")
    r = m(ip, imgsz=320, verbose=False)[0]
    if len(r.boxes) == 0:
        res.append((180.0, ip, None)); continue
    kp = r.keypoints.xyn[int(r.boxes.conf.argmax())].cpu().numpy()[:2]
    e = abs(needle_angle(*kp) - needle_angle(v[5:7], v[8:10])) % 360
    res.append((min(e, 360 - e), ip, kp))
errs = np.array([e for e, _, _ in res])
print(f"n={len(errs)}  median {np.median(errs):.2f} deg  p95 {np.percentile(errs, 95):.2f} deg  "
      f"over 5 deg {(errs > 5).mean():.1%}  no detection {sum(k is None for *_, k in res)}")
for i, (e, ip, kp) in enumerate(sorted(res, key=lambda x: -x[0])[:8]):
    im = cv2.imread(ip)
    if kp is not None:
        s = im.shape[0]; b, t = (kp * s).astype(int)
        cv2.line(im, tuple(b), tuple(t), (0, 0, 255), 2)
    cv2.imwrite(f"worst_{i}_{e:.0f}deg.jpg", im)
