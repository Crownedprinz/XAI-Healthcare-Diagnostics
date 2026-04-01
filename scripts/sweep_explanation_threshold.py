"""
Uniform heatmap binarization threshold sweep on validation Grad-CAM/IG pickles.
Writes a CSV: threshold -> mean mIoU (from CheXlocalize eval summary) across tasks.
Requires: externals/chexlocalize scripts, GT json, and a directory of *_map.pkl files.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT = REPO_ROOT / "externals" / "chexlocalize"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-dir", type=Path, required=True, help="Directory with *_map.pkl heatmaps.")
    p.add_argument("--gt-path", type=Path, required=True, help="gt_segmentations_val.json")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/threshold_sweep"))
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.map_dir = args.map_dir.resolve()
    args.gt_path = args.gt_path.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(EXT))
    from eval_constants import LOCALIZATION_TASKS  # noqa: E402

    rows = []
    for t in args.thresholds:
        th_csv = args.out_dir / f"uniform_threshold_{t:.2f}.csv"
        df = pd.DataFrame([{"threshold": t, "task": task} for task in LOCALIZATION_TASKS])
        df.to_csv(th_csv, index=False)

        pred_json = (args.out_dir / f"pred_seg_thresh_{t:.2f}.json").resolve()
        th_csv = th_csv.resolve()
        subprocess.run(
            [
                sys.executable,
                str(EXT / "heatmap_to_segmentation.py"),
                "--map_dir",
                str(args.map_dir),
                "--threshold_path",
                str(th_csv),
                "--output_path",
                str(pred_json),
            ],
            check=True,
            cwd=str(EXT),
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "pad_pred_segmentation_json.py"),
                "--gt-path",
                str(args.gt_path),
                "--pred-path",
                str(pred_json),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )

        save_dir = args.out_dir / f"eval_thresh_{t:.2f}"
        save_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(EXT / "eval.py"),
                "--metric",
                "iou",
                "--gt_path",
                str(args.gt_path),
                "--pred_path",
                str(pred_json),
                "--save_dir",
                str(save_dir),
                "--true_pos_only",
                "True",
                "--if_human_benchmark",
                "False",
            ],
            check=True,
            cwd=str(EXT),
        )

        summary = pd.read_csv(save_dir / "iou_summary_results.csv")
        mean_miou = float(summary["mean"].mean(skipna=True))
        rows.append({"threshold": t, "val_mean_miou_across_tasks": mean_miou})

    out_csv = args.out_dir / "explanation_threshold_sweep.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "val_mean_miou_across_tasks"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
