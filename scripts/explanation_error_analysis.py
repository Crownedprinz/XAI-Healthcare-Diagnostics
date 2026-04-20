"""
Analyze relationship between explanation quality and prediction error on validation data.

Inputs:
- preds NPZ from scripts/run_pred_baseline.py
- IoU-per-image CSV from CheXlocalize eval (iou_results_per_cxr.csv)

Outputs:
- per-sample CSV with explanation score, confidence, and error
- summary CSV with Pearson/Spearman correlations and binned error stats
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IOU_TO_PRED_TASK = {
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Cardiomegaly": "Cardiomegaly",
    "Airspace Opacity": "Lung Opacity",
    "Lung Lesion": "Lung Lesion",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Pleural Effusion",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds", type=Path, help="artifacts/chexpert_val_preds.npz")
    p.add_argument(
        "--preds-with-ids",
        type=Path,
        help="artifacts/chexpert_val_preds_with_ids.csv (preferred for robust join)",
    )
    p.add_argument(
        "--iou-per-cxr",
        type=Path,
        required=True,
        help="artifacts/my_gradcam_eval_val_v3/iou_results_per_cxr.csv",
    )
    p.add_argument(
        "--thresholds-csv",
        type=Path,
        default=Path("artifacts/f1_thresholds_from_val.csv"),
        help="Optional tuned thresholds CSV with columns: task,threshold_f1_max",
    )
    p.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--n-bins", type=int, default=5)
    return p.parse_args()


def load_preds(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    y_true = d["y_true"].astype(np.float64)
    y_prob = d["y_prob"].astype(np.float64)
    tasks = [str(t) for t in d["tasks"]]
    return y_true, y_prob, tasks


def load_preds_with_ids(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "img_id" not in df.columns:
        raise ValueError("preds-with-ids CSV must contain `img_id`.")
    prob_cols = [c for c in df.columns if c.startswith("y_prob|")]
    true_cols = [c for c in df.columns if c.startswith("y_true|")]
    tasks = sorted([c.split("|", 1)[1] for c in prob_cols])
    if not tasks:
        raise ValueError("No `y_prob|<task>` columns found in preds-with-ids CSV.")
    y_prob = np.column_stack([df[f"y_prob|{t}"].to_numpy(dtype=np.float64) for t in tasks])
    y_true = np.column_stack([df[f"y_true|{t}"].to_numpy(dtype=np.float64) for t in tasks])
    return df, y_true, y_prob, tasks


def load_thresholds(path: Path, tasks: list[str]) -> np.ndarray:
    if not path.exists():
        return np.full(len(tasks), 0.5, dtype=np.float64)
    df = pd.read_csv(path)
    if "task" not in df.columns or "threshold_f1_max" not in df.columns:
        return np.full(len(tasks), 0.5, dtype=np.float64)
    m = dict(zip(df["task"].astype(str), df["threshold_f1_max"].astype(float)))
    return np.array([float(m.get(t, 0.5)) for t in tasks], dtype=np.float64)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    iou = pd.read_csv(args.iou_per_cxr)
    if "img_id" not in iou.columns:
        raise ValueError("IoU CSV must contain `img_id`.")

    if args.preds_with_ids:
        pred_df, _, _, pred_tasks = load_preds_with_ids(args.preds_with_ids)
        merged = iou.merge(pred_df, on="img_id", how="inner")
        if merged.empty:
            raise ValueError("No overlapping img_id values between IoU CSV and preds-with-ids CSV.")
        y_true = np.column_stack([merged[f"y_true|{t}"].to_numpy(dtype=np.float64) for t in pred_tasks])
        y_prob = np.column_stack([merged[f"y_prob|{t}"].to_numpy(dtype=np.float64) for t in pred_tasks])
        iou = merged
    else:
        if not args.preds:
            raise ValueError("Provide either --preds-with-ids or --preds.")
        y_true, y_prob, pred_tasks = load_preds(args.preds)
        if len(iou) != y_true.shape[0]:
            raise ValueError(
                f"Row mismatch: iou rows={len(iou)} vs preds rows={y_true.shape[0]}."
                " Recreate preds with matching split/order or use --preds-with-ids."
            )

    # Build explanation score as mean IoU across mapped tasks (NaN-safe).
    iou_cols = []
    for iou_name, pred_name in IOU_TO_PRED_TASK.items():
        if iou_name in iou.columns and pred_name in pred_tasks:
            iou_cols.append(iou_name)
    if not iou_cols:
        raise ValueError("No overlapping IoU columns found for task mapping.")

    exp_scores = iou[iou_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

    thresholds = load_thresholds(args.thresholds_csv, pred_tasks)
    y_hat = (y_prob >= thresholds[None, :]).astype(np.float64)
    label_error = (y_true != y_hat).mean(axis=1)
    max_conf = y_prob.max(axis=1)

    per_sample = pd.DataFrame(
        {
            "img_id": iou.get("img_id", pd.Series([f"row_{i}" for i in range(len(iou))])),
            "explanation_score_mean_iou": exp_scores,
            "max_confidence": max_conf,
            "label_error_rate": label_error,
            "label_correct_rate": 1.0 - label_error,
        }
    )
    per_sample_path = args.out_dir / "explanation_vs_error_per_sample.csv"
    per_sample.to_csv(per_sample_path, index=False)

    valid = per_sample.dropna(subset=["explanation_score_mean_iou"]).copy()
    pearson = float(valid["explanation_score_mean_iou"].corr(valid["label_error_rate"], method="pearson"))
    spearman = float(valid["explanation_score_mean_iou"].corr(valid["label_error_rate"], method="spearman"))

    binned = valid.copy()
    binned["exp_bin"] = pd.qcut(
        binned["explanation_score_mean_iou"],
        q=min(args.n_bins, max(2, binned["explanation_score_mean_iou"].nunique())),
        duplicates="drop",
    )
    binned_tbl = (
        binned.groupby("exp_bin", observed=True)
        .agg(
            n=("label_error_rate", "size"),
            mean_expl=("explanation_score_mean_iou", "mean"),
            mean_error=("label_error_rate", "mean"),
            mean_conf=("max_confidence", "mean"),
        )
        .reset_index()
    )

    summary_rows = [
        {"metric": "n_total", "value": float(len(per_sample))},
        {"metric": "n_with_expl_score", "value": float(len(valid))},
        {"metric": "pearson_corr_expl_vs_error", "value": pearson},
        {"metric": "spearman_corr_expl_vs_error", "value": spearman},
    ]
    summary = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "explanation_vs_error_summary.csv"
    summary.to_csv(summary_path, index=False)

    bins_path = args.out_dir / "explanation_vs_error_binned.csv"
    binned_tbl.to_csv(bins_path, index=False)

    print(f"Saved {per_sample_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {bins_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
