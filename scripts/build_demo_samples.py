"""Build a small, portable demo-sample bundle for the live dashboard: a
subset of standardized test-split feature vectors + labels + the raw
value ranges needed to drive the interactive sliders. Kept separate from
the full 267MB test matrix so the dashboard/Docker image/HF Space stay
light -- this is the only data file that ships with the app."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
N_DEMO = 300
TOP_K_SLIDERS = 8

X = np.load(PROC / "features.npy")
y = np.load(PROC / "labels.npy")
mean = np.load(PROC / "scaler_mean.npy")
std = np.load(PROC / "scaler_std.npy")
with open(PROC / "splits.json") as f:
    splits = json.load(f)
with open(PROC / "classes.json") as f:
    classes = json.load(f)

test_idx = np.array(splits["test"])
rng = np.random.RandomState(42)
demo_idx = rng.choice(test_idx, size=min(N_DEMO, len(test_idx)), replace=False)

X_demo_raw = X[demo_idx]  # pre-standardization (log2 fpkm) values, for slider ranges
X_demo_std = ((X_demo_raw - mean) / std).astype(np.float32)
y_demo = y[demo_idx].astype(np.int64)

# pick the highest-variance features (within this demo subset) as the
# interactive sliders in the live-prediction tab
variances = X_demo_raw.var(axis=0)
slider_feature_idx = np.argsort(-variances)[:TOP_K_SLIDERS]

out_dir = ROOT / "dashboard_app" / "data"
out_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_dir / "demo_samples.npz",
    X_std=X_demo_std,
    X_raw=X_demo_raw.astype(np.float32),
    y=y_demo,
    slider_feature_idx=slider_feature_idx,
    raw_min=X_demo_raw.min(axis=0).astype(np.float32),
    raw_max=X_demo_raw.max(axis=0).astype(np.float32),
    scaler_mean=mean,
    scaler_std=std,
)
with open(out_dir / "classes.json", "w") as f:
    json.dump(classes, f)

print(f"Wrote demo_samples.npz with {len(demo_idx)} samples, {TOP_K_SLIDERS} slider features")
print("Slider feature indices:", slider_feature_idx.tolist())
