"""Paired comparison of two runs on the same images.

    python vlm/paired.py runs/vlm/r8 runs/vlm/dpo/b0.1 test real

Joins <run>/<split>_pred.csv of run A and run B by image. For each split it prints:
- the number of images where B has a smaller, larger or equal error than A
- for the 1% and 2% FS tolerances: images that only A passes, images that only B passes,
  and the exact McNemar p-value (two-sided binomial test on these two counts)
- the 95% paired bootstrap interval of B minus A, for the mean error and for the share within 1% FS
An unparsable answer has an infinite error.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np


def errors(run, split):
    with open(Path(run) / f"{split}_pred.csv") as f:
        return {r["stem"]: float(r["pct_fs"]) if r["pct_fs"] else math.inf for r in csv.DictReader(f)}


def mcnemar(only_a, only_b):
    n, k = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    a_run, b_run, *splits = sys.argv[1:]
    rng = np.random.default_rng(0)
    for split in splits:
        a, b = errors(a_run, split), errors(b_run, split)
        stems = sorted(a.keys() & b.keys())
        ea, eb = np.array([a[s] for s in stems]), np.array([b[s] for s in stems])
        n = len(stems)
        print(f"{split}: {n} images. B smaller error {int((eb < ea).sum())}, larger {int((eb > ea).sum())}, "
              f"equal {int((eb == ea).sum())}")
        for t in (1, 2):
            pa, pb = ea <= t, eb <= t
            only_a, only_b = int((pa & ~pb).sum()), int((pb & ~pa).sum())
            print(f"  within {t}% FS: A {pa.mean():.1%}, B {pb.mean():.1%}. Only A passes {only_a}, only B passes "
                  f"{only_b}, McNemar p = {mcnemar(only_a, only_b):.3g}")
        idx = rng.integers(0, n, size=(10000, n))
        finite = np.isfinite(ea) & np.isfinite(eb)
        d = np.where(finite, eb - ea, 0.0)
        mean_d = d[idx].mean(1)
        w1_d = ((eb <= 1).astype(float) - (ea <= 1))[idx].mean(1)
        print(f"  B - A, mean error: {d.mean():+.3f} % FS, 95% CI [{np.percentile(mean_d, 2.5):+.3f}, "
              f"{np.percentile(mean_d, 97.5):+.3f}]" + ("" if finite.all() else f" ({n - finite.sum()} unparsed left out)"))
        print(f"  B - A, within 1% FS: {w1_d.mean():+.1%}, 95% CI [{np.percentile(w1_d, 2.5):+.1%}, "
              f"{np.percentile(w1_d, 97.5):+.1%}]")


if __name__ == "__main__":
    main()
