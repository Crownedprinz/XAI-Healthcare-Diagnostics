"""
Compute risk-coverage curve and AURC from saved preds.npz.
Usage:
  python scripts/risk_coverage.py --preds artifacts/chexpert_val_preds.npz --out artifacts/risk_coverage_val.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

def load_preds(path: str):
    data = np.load(path, allow_pickle=True)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    tasks = [t if isinstance(t, str) else t.item() for t in data["tasks"]]
    return y_true, y_prob, tasks


def risk_coverage(y_true: np.ndarray, y_prob: np.ndarray, n_points: int = 50):
    # uncertainty = 1 - max prob per sample across tasks (multi-label -> use max)
    max_conf = y_prob.max(axis=1)
    order = np.argsort(max_conf)  # low confidence first for deferral
    coverages = np.linspace(1.0, 0.0, n_points)
    risks = []
    thresholds = []
    for c in coverages:
        k = int(np.ceil(c * len(order)))
        keep_idx = order[-k:] if k > 0 else []
        if k == 0:
            risks.append(0.0)
            thresholds.append(1.0)
            continue
        yk = y_true[keep_idx]
        yp = (y_prob[keep_idx] >= 0.5).astype(float)
        # system error = 1 - macro accuracy over kept samples/labels
        correct = (yk == yp).mean()
        risks.append(1 - correct)
        thresholds.append(max_conf[order[-k]])
    aurc = np.trapz(risks, x=coverages)
    return coverages, risks, thresholds, aurc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--points", type=int, default=50)
    args = parser.parse_args()

    y_true, y_prob, _ = load_preds(args.preds)
    coverages, risks, thresholds, aurc = risk_coverage(y_true, y_prob, args.points)

    df = pd.DataFrame({
        "coverage": coverages,
        "risk": risks,
        "threshold": thresholds,
    })
    df.to_csv(args.out, index=False)
    print(f"AURC: {aurc:.4f}")
    print(df.head())


if __name__ == "__main__":
    main()
