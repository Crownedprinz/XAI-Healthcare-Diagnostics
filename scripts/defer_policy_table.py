"""
Simple predict-or-defer table: defer cases with the lowest max label probability
(multi-label confidence). Report coverage, deferred fraction, and mean per-label error
on kept cases (same notion as risk–coverage; complements scripts/risk_coverage.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_preds(path: str):
    data = np.load(path, allow_pickle=True)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    tasks = [t if isinstance(t, str) else t.item() for t in data["tasks"]]
    return y_true, y_prob, tasks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds", type=Path, required=True, help="chexpert_val_preds.npz or test")
    p.add_argument("--out", type=Path, required=True, help="CSV path for the defer table")
    p.add_argument(
        "--defer-pcts",
        type=int,
        nargs="+",
        default=[0, 10, 20, 30, 40, 50],
        help="Percent of cases to defer (lowest max-probability first).",
    )
    args = p.parse_args()

    y_true, y_prob, _ = load_preds(str(args.preds))
    n = y_true.shape[0]
    max_conf = y_prob.max(axis=1)
    order = np.argsort(max_conf)

    rows = []
    for defer_pct in args.defer_pcts:
        n_defer = int(round(n * defer_pct / 100.0))
        kept = order[n_defer:]
        if len(kept) == 0:
            continue
        yt = y_true[kept]
        yp = (y_prob[kept] >= 0.5).astype(np.float64)
        label_error_rate = float(np.mean(yt != yp))
        coverage = len(kept) / n
        rows.append(
            {
                "defer_pct": defer_pct,
                "deferred_frac": n_defer / n,
                "coverage": coverage,
                "label_error_rate_on_kept": label_error_rate,
            }
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
