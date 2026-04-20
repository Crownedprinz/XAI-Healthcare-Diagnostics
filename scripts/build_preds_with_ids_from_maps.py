"""
Build img_id-keyed prediction table by matching map-embedded probabilities to preds.npz.

Use when labels CSV is unavailable but Grad-CAM map pickles exist and include `prob`.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


MAP_TO_PRED_TASK = {
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Cardiomegaly": "Cardiomegaly",
    "Airspace Opacity": "Lung Opacity",
    "Lung Lesion": "Lung Lesion",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Pleural Effusion",
    "Support Devices": None,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds", type=Path, required=True, help="artifacts/chexpert_val_preds.npz")
    p.add_argument("--map-dir", type=Path, required=True, help="artifacts/gradcam_maps_val_v3")
    p.add_argument("--out", type=Path, default=Path("artifacts/chexpert_val_preds_with_ids.csv"))
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    return p.parse_args()


def parse_img_id(file_name: str, task: str) -> str:
    suffix = f"_{task}_map.pkl"
    if not file_name.endswith(suffix):
        raise ValueError(f"Unexpected map filename/task mismatch: {file_name} vs task={task}")
    return file_name[: -len(suffix)]


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.preds, allow_pickle=True)
    y_true = d["y_true"].astype(np.float64)
    y_prob = d["y_prob"].astype(np.float64)
    pred_tasks = [str(t) for t in d["tasks"]]
    pred_task_to_idx = {t: i for i, t in enumerate(pred_tasks)}

    # Build probability vectors per img_id from map pickles.
    img_prob: dict[str, np.ndarray] = {}
    for pkl_path in sorted(args.map_dir.glob("*_map.pkl")):
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        task = str(payload.get("task"))
        mapped = MAP_TO_PRED_TASK.get(task)
        if mapped is None:
            continue
        if mapped not in pred_task_to_idx:
            continue
        img_id = parse_img_id(pkl_path.name, task)
        vec = img_prob.setdefault(img_id, np.full(len(pred_tasks), np.nan, dtype=np.float64))
        vec[pred_task_to_idx[mapped]] = float(payload.get("prob"))

    if not img_prob:
        raise ValueError("No map-derived probabilities found. Check --map-dir.")

    # Match each img_id vector to one row in npz by exact/numerical equality across available tasks.
    used_rows: set[int] = set()
    rows = []
    for img_id, vec in img_prob.items():
        known = ~np.isnan(vec)
        if not np.any(known):
            continue
        candidates = np.where(np.all(np.isclose(y_prob[:, known], vec[known], rtol=args.rtol, atol=args.atol), axis=1))[0]
        if len(candidates) == 0:
            # Fallback nearest row by L1 on known dims.
            diffs = np.abs(y_prob[:, known] - vec[known]).mean(axis=1)
            candidates = np.array([int(np.argmin(diffs))], dtype=int)

        chosen = None
        for c in candidates:
            if int(c) not in used_rows:
                chosen = int(c)
                break
        if chosen is None:
            continue
        used_rows.add(chosen)

        row = {"img_id": img_id}
        for j, t in enumerate(pred_tasks):
            row[f"y_true|{t}"] = float(y_true[chosen, j])
            row[f"y_prob|{t}"] = float(y_prob[chosen, j])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("img_id").reset_index(drop=True)
    out.to_csv(args.out, index=False)

    print(f"Saved {args.out}")
    print(f"Matched images: {len(out)} / npz rows: {len(y_prob)}")


if __name__ == "__main__":
    main()

