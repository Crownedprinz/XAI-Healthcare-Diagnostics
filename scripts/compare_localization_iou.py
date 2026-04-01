"""
Merge two CheXlocalize eval.py iou_summary_results.csv files for side-by-side comparison.
Usage:
  python scripts/compare_localization_iou.py \\
    --a artifacts/my_gradcam_eval_val/iou_summary_results.csv --label-a Grad-CAM \\
    --b artifacts/ig_eval_val/iou_summary_results.csv --label-b Integrated-Gradients \\
    --out artifacts/gradcam_vs_ig_val_comparison.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--label-a", type=str, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--label-b", type=str, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    da = pd.read_csv(args.a).rename(
        columns={"mean": f"{args.label_a}_mIoU", "lower": f"{args.label_a}_lower", "upper": f"{args.label_a}_upper"}
    )
    db = pd.read_csv(args.b).rename(
        columns={"mean": f"{args.label_b}_mIoU", "lower": f"{args.label_b}_lower", "upper": f"{args.label_b}_upper"}
    )
    merged = da.merge(db, on="name", how="outer")
    diff = f"delta_mIoU_{args.label_b}_minus_{args.label_a}".replace(" ", "_").replace("-", "_")
    merged[diff] = merged[f"{args.label_b}_mIoU"] - merged[f"{args.label_a}_mIoU"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(merged.to_string(index=False))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
