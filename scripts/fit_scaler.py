"""Fit a StandardScaler on the training split only, save mean/std for reuse
by the dataset loader, the exported model bundle, and the dashboard."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

X = np.load(PROC / "features.npy")
with open(PROC / "splits.json") as f:
    splits = json.load(f)

train_idx = np.array(splits["train"])
mean = X[train_idx].mean(axis=0)
std = X[train_idx].std(axis=0)
std[std < 1e-6] = 1e-6

np.save(PROC / "scaler_mean.npy", mean.astype(np.float32))
np.save(PROC / "scaler_std.npy", std.astype(np.float32))
print("Saved scaler_mean.npy / scaler_std.npy, shape:", mean.shape)
