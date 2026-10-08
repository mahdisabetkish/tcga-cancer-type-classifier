from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class TCGAExpressionDataset(Dataset):
    """Loads the preprocessed TCGA pan-cancer matrix built by
    scripts/prepare_data.py and applies train-fit standardization."""

    def __init__(self, processed_dir: str, split: str):
        processed_dir = Path(processed_dir)
        X = np.load(processed_dir / "features.npy")
        y = np.load(processed_dir / "labels.npy")
        mean = np.load(processed_dir / "scaler_mean.npy")
        std = np.load(processed_dir / "scaler_std.npy")
        with open(processed_dir / "splits.json") as f:
            splits = json.load(f)
        with open(processed_dir / "classes.json") as f:
            self.classes = json.load(f)

        idx = np.array(splits[split])
        self.X = ((X[idx] - mean) / std).astype(np.float32)
        self.y = y[idx].astype(np.int64)
        self.n_features = self.X.shape[1]
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])
