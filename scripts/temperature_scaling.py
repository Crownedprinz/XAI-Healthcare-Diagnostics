"""
Fit a single temperature for sigmoid outputs and report calibration metrics (ECE, Brier).
Usage:
  python scripts/temperature_scaling.py --preds artifacts/chexpert_val_preds.npz --out artifacts/chexpert_val_calibration.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import pandas as pd


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return ((p - y) ** 2).mean(axis=0)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> np.ndarray:
    c = y.shape[1]
    ece_vals = np.zeros(c)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for j in range(c):
        yj = y[:, j]
        pj = p[:, j]
        ece_j = 0.0
        for k in range(n_bins):
            mask = (pj >= bins[k]) & (pj < bins[k + 1])
            if not np.any(mask):
                continue
            conf = pj[mask].mean()
            acc = yj[mask].mean()
            ece_j += np.abs(acc - conf) * (mask.sum() / len(pj))
        ece_vals[j] = ece_j
    return ece_vals


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_steps: int = 500) -> float:
    temperature = torch.nn.Parameter(torch.tensor(1.0, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_steps)
    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature.clamp(min=0.05, max=5.0), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.data.clamp(min=0.05, max=5.0).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=str, required=True, help="Path to npz with y_true,y_prob,tasks")
    parser.add_argument("--out", type=str, required=True, help="CSV output path for calibration metrics")
    parser.add_argument("--bins", type=int, default=15, help="Number of bins for ECE")
    args = parser.parse_args()

    data = np.load(args.preds, allow_pickle=True)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    tasks = [t if isinstance(t, str) else t.item() for t in data["tasks"]]

    logits_np = logit(y_prob)
    logits = torch.tensor(logits_np, dtype=torch.float32)
    labels = torch.tensor(y_true, dtype=torch.float32)

    temp = fit_temperature(logits, labels)
    scaled_logits = logits / temp
    scaled_prob = torch.sigmoid(scaled_logits).cpu().numpy()

    brier_before = brier(y_true, y_prob)
    brier_after = brier(y_true, scaled_prob)
    ece_before = ece(y_true, y_prob, args.bins)
    ece_after = ece(y_true, scaled_prob, args.bins)

    df = pd.DataFrame(
        {
            "task": tasks,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
        }
    )
    df["temperature"] = temp
    df.to_csv(args.out, index=False)

    print(f"Fitted temperature: {temp:.3f}")
    print(df)


if __name__ == "__main__":
    main()
