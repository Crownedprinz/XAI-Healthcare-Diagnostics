"""
Run TorchXRayVision CheXpert-pretrained DenseNet on CheXpert val/test slices
from the CheXlocalize download and emit AUROC/AUPRC/F1 per task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchxrayvision as xrv
from PIL import Image
from rich.console import Console
from rich.table import Table
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

console = Console()

# Tasks aligned to CheXlocalize localization labels (10 tasks).
TASKS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Support Devices",
]

# Map CheXpert/CheXlocalize labels to TorchXRayVision output names.
# XRV DenseNet exposes the following relevant targets (ordering matters):
# ['Atelectasis', 'Consolidation', '', 'Pneumothorax', 'Edema', '', '', 'Effusion', 'Pneumonia', '', 'Cardiomegaly', '', '', '', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum']
TASK_TO_XRV = {
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Cardiomegaly": "Cardiomegaly",
    "Lung Opacity": "Lung Opacity",
    "Lung Lesion": "Lung Lesion",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Effusion",  # XRV target name
    # Support Devices is not available in the XRV target list; it will be skipped.
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("datasets/chexlocalize/CheXpert/val_labels.csv"),
        help="Path to CheXpert label CSV (val or test).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("datasets/chexlocalize/CheXpert"),
        help="Root folder containing `val/` and `test/` image folders.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda, cpu, mps, or auto",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts"),
        help="Where to save metrics CSV/NPZ.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="val",
        help="Used for output filenames (val/test).",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class CheXpertLocalDataset(data.Dataset):
    def __init__(
        self,
        csv_path: Path,
        images_root: Path,
        tasks: List[str],
        resize: Tuple[int, int] = (224, 224),
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.replace(-1, 0).fillna(0)
        self.images_root = images_root
        self.tasks = tasks
        self.resize = resize

        missing = [t for t in tasks if t not in self.df.columns]
        if missing:
            raise ValueError(f"Tasks missing from CSV columns: {missing}")

        self.labels = self.df[tasks].astype(np.float32).values
        self.paths = self.df["Path"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_path(self, rel: str) -> Path:
        # CSV paths use CheXpert-v1.0/valid or /test naming; map to local folders.
        rel = rel.replace("CheXpert-v1.0/valid", "val")
        rel = rel.replace("CheXpert-v1.0/test", "test")
        return (self.images_root / rel).resolve()

    def __getitem__(self, idx: int):
        img_path = self._resolve_path(self.paths[idx])
        img = Image.open(img_path).convert("L")
        if self.resize:
            img = img.resize(self.resize)
        arr = np.asarray(img, dtype=np.float32)
        arr = xrv.datasets.normalize(arr, 255)  # scale to match XRV expectations
        tensor = torch.from_numpy(arr)[None, ...]  # [1, H, W]
        label = torch.from_numpy(self.labels[idx])
        return {"img": tensor, "y": label, "path": str(img_path)}


def build_model(device: torch.device) -> torch.nn.Module:
    model = xrv.models.DenseNet(weights="densenet121-res224-chex", apply_sigmoid=False)
    model = model.to(device)
    model.eval()
    return model


def select_tasks(model: torch.nn.Module) -> Tuple[List[str], List[int]]:
    model_targets = list(model.targets)
    selected_tasks: List[str] = []
    target_indices: List[int] = []
    skipped: List[str] = []
    for t in TASKS:
        mapped = TASK_TO_XRV.get(t, "")
        if not mapped or mapped not in model_targets:
            skipped.append(t)
            continue
        selected_tasks.append(t)
        target_indices.append(model_targets.index(mapped))
    if skipped:
        console.print(f"[yellow]Skipping tasks not present in XRV targets: {skipped}[/yellow]")
    return selected_tasks, target_indices


def evaluate(
    model: torch.nn.Module,
    loader: data.DataLoader,
    device: torch.device,
    tasks: List[str],
    target_indices: List[int],
) -> Dict[str, Dict[str, float]]:
    y_true: List[np.ndarray] = []
    y_logit: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", ncols=100):
            imgs = batch["img"].to(device)
            logits = model(imgs)
            logits = logits[:, target_indices]
            y_true.append(batch["y"].cpu().numpy())
            y_logit.append(logits.cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0)
    y_logit_np = np.concatenate(y_logit, axis=0)
    y_prob_np = 1 / (1 + np.exp(-y_logit_np))

    results: Dict[str, Dict[str, float]] = {}
    for j, task in enumerate(tasks):
        yt = y_true_np[:, j]
        yp = y_prob_np[:, j]
        metrics: Dict[str, float] = {}
        # AUROC/AUPRC only computed when both classes present.
        if len(np.unique(yt)) > 1:
            metrics["auroc"] = float(roc_auc_score(yt, yp))
            metrics["auprc"] = float(average_precision_score(yt, yp))
        else:
            metrics["auroc"] = np.nan
            metrics["auprc"] = np.nan
        metrics["f1@0.5"] = float(f1_score(yt, yp >= 0.5))
        results[task] = metrics

    return results, y_true_np, y_prob_np


def save_outputs(
    out_dir: Path,
    split_name: str,
    results: Dict[str, Dict[str, float]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tasks: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"chexpert_{split_name}_metrics.csv"
    pd.DataFrame(results).T.to_csv(metrics_path, index=True)

    np.savez_compressed(
        out_dir / f"chexpert_{split_name}_preds.npz",
        y_true=y_true,
        y_prob=y_prob,
        tasks=np.array(tasks),
    )

    console.print(f"[green]Saved metrics:[/green] {metrics_path}")


def print_table(results: Dict[str, Dict[str, float]]) -> None:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Task")
    table.add_column("AUROC", justify="right")
    table.add_column("AUPRC", justify="right")
    table.add_column("F1@0.5", justify="right")
    for task, m in results.items():
        table.add_row(
            task,
            f"{m['auroc']:.3f}" if not np.isnan(m["auroc"]) else "n/a",
            f"{m['auprc']:.3f}" if not np.isnan(m["auprc"]) else "n/a",
            f"{m['f1@0.5']:.3f}",
        )
    console.print(table)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    console.print(f"Using device: {device}")

    model = build_model(device)
    tasks_used, target_indices = select_tasks(model)
    dataset = CheXpertLocalDataset(args.csv, args.images_root, tasks_used)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    results, y_true, y_prob = evaluate(model, loader, device, tasks_used, target_indices)
    print_table(results)
    save_outputs(args.out_dir, args.split_name, results, y_true, y_prob, tasks_used)


if __name__ == "__main__":
    main()
