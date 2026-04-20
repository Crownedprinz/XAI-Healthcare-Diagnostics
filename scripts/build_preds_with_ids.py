"""
Build an img_id-keyed prediction table from preds.npz + labels CSV.

This avoids row-order assumptions and enables joins with localization outputs
that are keyed by `img_id` (e.g., iou_results_per_cxr.csv).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds", type=Path, required=True, help="artifacts/chexpert_val_preds.npz")
    p.add_argument("--labels-csv", type=Path, required=True, help="CheXpert val/test labels CSV")
    p.add_argument("--out", type=Path, default=Path("artifacts/chexpert_val_preds_with_ids.csv"))
    return p.parse_args()


def make_img_id(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if len(parts) >= 3:
        return f"{parts[-3]}_{parts[-2]}_{Path(parts[-1]).stem}"
    return Path(rel_path).stem


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.preds, allow_pickle=True)
    y_true = d["y_true"]
    y_prob = d["y_prob"]
    tasks = [str(t) for t in d["tasks"]]

    labels = pd.read_csv(args.labels_csv)
    if len(labels) != y_true.shape[0]:
        raise ValueError(
            f"Row mismatch: labels rows={len(labels)} vs preds rows={y_true.shape[0]}."
            " Ensure labels CSV matches the preds split."
        )
    if "Path" not in labels.columns:
        raise ValueError("labels CSV must contain column 'Path'.")

    out = pd.DataFrame({"img_id": labels["Path"].astype(str).map(make_img_id)})
    for j, task in enumerate(tasks):
        out[f"y_true|{task}"] = y_true[:, j]
        out[f"y_prob|{task}"] = y_prob[:, j]

    out.to_csv(args.out, index=False)
    print(f"Saved {args.out}")
    print(f"Rows: {len(out)}, tasks: {len(tasks)}")


if __name__ == "__main__":
    main()

