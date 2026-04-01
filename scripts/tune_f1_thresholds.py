"""
Search per-class probability thresholds that maximize F1 on validation predictions,
then report F1@0.5 vs F1@tuned on test (or another split).
Expects NPZ files from run_pred_baseline.py (y_true, y_prob, tasks).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-preds", type=Path, required=True, help="chexpert_val_preds.npz")
    p.add_argument("--test-preds", type=Path, help="chexpert_test_preds.npz (optional)")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    return p.parse_args()


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    tasks = [t if isinstance(t, str) else t.item() for t in data["tasks"]]
    return y_true, y_prob, tasks


def best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, 101):
        f1 = f1_score(y_true, y_prob >= t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    y_val, p_val, tasks = load_npz(args.val_preds)
    rows = []
    thresholds = []
    for j, task in enumerate(tasks):
        t_star, f1_tuned = best_threshold_f1(y_val[:, j], p_val[:, j])
        f1_05 = f1_score(y_val[:, j], p_val[:, j] >= 0.5, zero_division=0)
        rows.append(
            {
                "task": task,
                "threshold_f1_max": t_star,
                "f1_val@0.5": f1_05,
                "f1_val@tuned": f1_tuned,
            }
        )
        thresholds.append(t_star)

    th_path = args.out_dir / "f1_thresholds_from_val.csv"
    pd.DataFrame(rows).to_csv(th_path, index=False)
    print(f"Saved {th_path}")

    if args.test_preds and args.test_preds.exists():
        y_te, p_te, tasks_te = load_npz(args.test_preds)
        assert tasks_te == tasks
        out_rows = []
        for j, task in enumerate(tasks):
            t_star = thresholds[j]
            out_rows.append(
                {
                    "task": task,
                    "f1_test@0.5": f1_score(y_te[:, j], p_te[:, j] >= 0.5, zero_division=0),
                    "f1_test@tuned": f1_score(y_te[:, j], p_te[:, j] >= t_star, zero_division=0),
                    "threshold_used": t_star,
                }
            )
        macro_05 = float(np.mean([r["f1_test@0.5"] for r in out_rows]))
        macro_tuned = float(np.mean([r["f1_test@tuned"] for r in out_rows]))
        out_rows.append(
            {
                "task": "MACRO (avg classes)",
                "f1_test@0.5": macro_05,
                "f1_test@tuned": macro_tuned,
                "threshold_used": float("nan"),
            }
        )
        test_path = args.out_dir / "f1_test_fixed_vs_tuned.csv"
        pd.DataFrame(out_rows).to_csv(test_path, index=False)
        print(f"Saved {test_path}")


if __name__ == "__main__":
    main()
