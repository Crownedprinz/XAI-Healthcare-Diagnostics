"""
Minimal utilities for heatmap loading and RLE encoding (CheXlocalize).
Upstream repo includes regression helpers; this repo only vendors what eval/heatmap need.
"""
import io
import pickle
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_lib


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def parse_pkl_filename(pkl_path: Path | str) -> tuple[str, str]:
    path = str(pkl_path).split("/")
    task = path[-1].split("_")[-2]
    img_id = "_".join(path[-1].split("_")[:-2])
    return task, img_id


def encode_segmentation(segmentation_arr: np.ndarray):
    segmentation = np.asfortranarray(segmentation_arr.astype("uint8"))
    rs = mask_lib.encode(segmentation)
    rs["counts"] = rs["counts"].decode()
    return rs
