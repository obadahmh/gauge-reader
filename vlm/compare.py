"""Tabulate the *_summary.json files of several runs.

    python vlm/compare.py runs/vlm/yolo runs/vlm/zeroshot runs/vlm/r8
"""
import json
import sys
from pathlib import Path

COLS = [("n", "{:.0f}"), ("unparsed", "{:.0f}"), ("mae", "{:.3f}"), ("mean_pct_fs", "{:.2f}"),
        ("median_pct_fs", "{:.2f}"), ("p95_pct_fs", "{:.2f}"), ("within_1pct_fs", "{:.1%}"),
        ("within_2pct_fs", "{:.1%}"), ("within_5pct_fs", "{:.1%}"), ("sec_per_image", "{:.3f}")]

print("| run | split | " + " | ".join(c for c, _ in COLS) + " |")
print("|---" * (len(COLS) + 2) + "|")
for run in map(Path, sys.argv[1:]):
    for f in sorted(run.glob("*_summary.json")):
        s = json.loads(f.read_text())
        cells = [fmt.format(s[c]) if s.get(c) is not None else "" for c, fmt in COLS]
        print(f"| {run.name} | {f.name.removesuffix('_summary.json')} | " + " | ".join(cells) + " |")
