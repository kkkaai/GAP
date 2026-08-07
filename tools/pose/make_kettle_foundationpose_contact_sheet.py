#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


ROOT = Path("outputs/0713test_phase1_4_vlm_qwen37_test")
SAMPLES = [
    "20260713_172042_054_kettle-1",
    "20260713_172116_649_kettle-2",
    "20260713_172149_129_kettle-3",
]


def main() -> None:
    paths = []
    for sample in SAMPLES:
        for ldir in sorted((ROOT / sample).glob("lollipop_*")):
            p = ldir / "pose_foundationpose" / "comparison.png"
            if p.exists():
                paths.append(p)

    imgs = []
    for p in paths:
        img = imageio.imread(p)[..., :3]
        img = cv2.resize(img, (960, 240), interpolation=cv2.INTER_AREA)
        label = f"{p.parents[1].name}/{p.parents[0].name}"
        cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
        imgs.append(img)

    rows = []
    for i in range(0, len(imgs), 2):
        row = imgs[i : i + 2]
        if len(row) < 2:
            row.append(np.zeros_like(row[0]))
        rows.append(np.concatenate(row, axis=1))

    sheet = np.concatenate(rows, axis=0)
    out = ROOT / "kettle_foundationpose_contact_sheet.png"
    imageio.imwrite(out, sheet)
    print(f"wrote {out} from {len(paths)} comparisons shape={sheet.shape}")

    summary = json.loads((ROOT / "kettle_foundationpose_summary.json").read_text())
    for item in summary:
        t = np.asarray(item["pose_object_in_camera"])[:3, 3]
        print(item["sample_id"], item["lollipop"], "t=", np.round(t, 4).tolist())


if __name__ == "__main__":
    main()
