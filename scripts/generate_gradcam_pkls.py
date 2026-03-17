"""
Generate Grad-CAM heatmap pickles in the CheXlocalize format for val/test images.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data as data
import torchxrayvision as xrv
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from rich.console import Console
from tqdm import tqdm

console = Console()

LOCALIZATION_TASKS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Airspace Opacity",  # maps to Lung Opacity in XRV
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Support Devices",
]

TASK_TO_XRV = {t: t for t in LOCALIZATION_TASKS}
TASK_TO_XRV["Airspace Opacity"] = "Lung Opacity"
TASK_TO_XRV["Pleural Effusion"] = "Effusion"
TASK_TO_XRV["Support Devices"] = None  # not available in XRV targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("datasets/chexlocalize/CheXpert/val_labels.csv"),
        help="CheXpert labels CSV (val or test).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("datasets/chexlocalize/CheXpert"),
        help="Root folder containing val/ and test/ images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/gradcam_maps"),
        help="Output directory for *_map.pkl files.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", help="cuda/cpu/mps/auto")
    parser.add_argument(
        "--resize", type=int, default=224, help="Resize shorter side to this size."
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="val",
        help="Used in logging; does not affect file naming.",
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


class CheXpertForCam(data.Dataset):
    def __init__(self, csv_path: Path, images_root: Path, resize: int = 224):
        import pandas as pd

        self.df = pd.read_csv(csv_path).replace(-1, 0).fillna(0)
        self.paths = self.df["Path"].tolist()
        self.resize = resize
        self.images_root = images_root

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_path(self, rel: str) -> Path:
        rel = rel.replace("CheXpert-v1.0/valid", "val")
        rel = rel.replace("CheXpert-v1.0/test", "test")
        return (self.images_root / rel).resolve()

    def __getitem__(self, idx: int):
        rel = self.paths[idx]
        img_path = self._resolve_path(rel)
        img = Image.open(img_path).convert("L")
        orig_w, orig_h = img.size
        img_resized = img.resize((self.resize, self.resize))
        arr = np.asarray(img_resized, dtype=np.float32)
        arr = xrv.datasets.normalize(arr, 255)
        tensor = torch.from_numpy(arr)[None, ...]  # [1, H, W]
        parts = Path(rel).parts
        if len(parts) >= 3:
            sample_id = f"{parts[-3]}_{parts[-2]}_{Path(parts[-1]).stem}"
        else:
            sample_id = Path(rel).stem.replace(".jpg", "")
        return {
            "img": tensor,
            "path": str(img_path),
            "id": sample_id,
            "orig_wh": torch.tensor([orig_w, orig_h]),
        }


def save_pkl(out_dir: Path, img_id: str, task: str, heatmap: np.ndarray, cxr_dims, prob: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"{img_id}_{task}_map.pkl"
    payload = {
        "map": torch.tensor(heatmap, dtype=torch.float32)[None, None, :, :],
        "cxr_dims": cxr_dims,
        "prob": float(prob),
        "task": task,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    console.print(f"Using device: {device}")

    dataset = CheXpertForCam(args.csv, args.images_root, resize=args.resize)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = xrv.models.DenseNet(weights="densenet121-res224-chex", apply_sigmoid=False).to(device)
    model.eval()
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    xrv_targets = list(model.targets)

    for batch in tqdm(loader, desc="Grad-CAM", ncols=100):
        imgs = batch["img"].to(device)
        imgs.requires_grad_(True)
        logits = model(imgs)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        for i in range(imgs.shape[0]):
            img_tensor = imgs[i : i + 1]
            img_id = batch["id"][i]
            cxr_dims = tuple(int(x) for x in batch["orig_wh"][i].tolist())
            for task in LOCALIZATION_TASKS:
                class_name = TASK_TO_XRV.get(task)
                if not class_name or class_name not in xrv_targets:
                    continue
                class_idx = xrv_targets.index(class_name)
                heatmap = cam(
                    input_tensor=img_tensor,
                    targets=[ClassifierOutputTarget(class_idx)],
                )[0]
                prob_scalar = probs[i, class_idx]
                save_pkl(args.out_dir, img_id, task, heatmap, cxr_dims, prob_scalar)

    console.print(f"[green]Saved Grad-CAM pickles to {args.out_dir}[/green]")


if __name__ == "__main__":
    main()
