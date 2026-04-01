"""
Ensure every CXR id in GT has every CheXlocalize localization task in the pred JSON.
Missing tasks (e.g. Support Devices when XRV has no logit) get an empty mask matching GT size.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_lib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "externals" / "chexlocalize"))
from eval_constants import LOCALIZATION_TASKS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-path", type=Path, required=True)
    p.add_argument("--pred-path", type=Path, required=True, help="Updated in place unless --out set.")
    p.add_argument("--out", type=Path, help="Optional output path (default: overwrite pred-path).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.gt_path) as f:
        gt = json.load(f)
    with open(args.pred_path) as f:
        pred = json.load(f)

    for cxr_id in gt:
        if cxr_id not in pred:
            pred[cxr_id] = {}
        for task in LOCALIZATION_TASKS:
            if task not in pred[cxr_id]:
                h, w = gt[cxr_id][task]["size"]
                empty = np.zeros((h, w), dtype=np.uint8)
                enc = mask_lib.encode(np.asfortranarray(empty))
                enc["counts"] = enc["counts"].decode()
                pred[cxr_id][task] = enc

    out = args.out or args.pred_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(pred, f)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
