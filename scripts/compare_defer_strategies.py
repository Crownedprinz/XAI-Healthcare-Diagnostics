"""
Compare deferral strategies:
1) Confidence-only
2) Confidence + explanation quality

Uses per-image IoU summary as explanation-quality signal.
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
    p.add_argument(
        "--defer-pcts",
        type=int,
        nargs="+",
        default=[0, 10, 20, 30, 40, 50],
        help="Deferred percentages for table output.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for confidence in combined strategy: alpha*conf + (1-alpha)*expl_score",
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/defer_strategy_comparison.csv"))
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


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def evaluate_strategy(
    name: str,
    reliability: np.ndarray,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    defer_pcts: list[int],
) -> pd.DataFrame:
    n = len(reliability)
    order = np.argsort(reliability)  # low reliability deferred first
    rows = []
    for pct in defer_pcts:
        n_defer = int(round(n * pct / 100.0))
        keep_idx = order[n_defer:]
        if len(keep_idx) == 0:
            continue
        err = float((y_true[keep_idx] != y_hat[keep_idx]).mean())
        rows.append(
            {
                "strategy": name,
                "defer_pct": pct,
                "deferred_frac": n_defer / n,
                "coverage": len(keep_idx) / n,
                "label_error_rate_on_kept": err,
                "n_kept": int(len(keep_idx)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

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

    iou_cols = []
    for iou_name, pred_name in IOU_TO_PRED_TASK.items():
        if iou_name in iou.columns and pred_name in pred_tasks:
            iou_cols.append(iou_name)
    if not iou_cols:
        raise ValueError("No overlapping IoU columns found for task mapping.")

    exp_score = iou[iou_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True).to_numpy()
    max_conf = y_prob.max(axis=1)

    # Restrict comparison to rows where explanation score exists.
    valid = ~np.isnan(exp_score)
    y_true = y_true[valid]
    y_prob = y_prob[valid]
    exp_score = exp_score[valid]
    max_conf = max_conf[valid]

    thresholds = load_thresholds(args.thresholds_csv, pred_tasks)
    y_hat = (y_prob >= thresholds[None, :]).astype(np.float64)

    conf_norm = normalized(max_conf)
    expl_norm = normalized(exp_score)
    combined = args.alpha * conf_norm + (1.0 - args.alpha) * expl_norm

    conf_df = evaluate_strategy("confidence_only", conf_norm, y_true, y_hat, args.defer_pcts)
    combo_df = evaluate_strategy(
        f"confidence_plus_expl_alpha_{args.alpha:.2f}",
        combined,
        y_true,
        y_hat,
        args.defer_pcts,
    )

    out = pd.concat([conf_df, combo_df], ignore_index=True)
    out.to_csv(args.out, index=False)

    # Small summary
    summary = []
    for strat, sdf in out.groupby("strategy"):
        x = sdf["coverage"].to_numpy()
        y = sdf["label_error_rate_on_kept"].to_numpy()
        order = np.argsort(x)
        aurc_like = float(np.trapz(y[order], x=x[order]))
        summary.append({"strategy": strat, "n_eval_samples": int(valid.sum()), "aurc_like": aurc_like})
    summary_df = pd.DataFrame(summary)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved {args.out}")
    print(f"Saved {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
