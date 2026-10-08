"""
Standard non-SSL baselines on the same preprocessed features/splits, so the
SSL + equivariant + Bayesian pipeline can be judged against honest
reference points rather than against itself. Mirrors the three-model
comparison in the original, smaller-scale version of this project.

Usage:
    PYTHONPATH=src python3 -m tcga_bench.training.baselines --config configs/default.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

from tcga_bench.data.dataset import TCGAExpressionDataset


def run_baselines(cfg):
    train_ds = TCGAExpressionDataset(cfg.data.processed_dir, "train")
    test_ds = TCGAExpressionDataset(cfg.data.processed_dir, "test")
    X_train, y_train = train_ds.X, train_ds.y
    X_test, y_test = test_ds.X, test_ds.y

    results = {}

    print("Training Logistic Regression (multinomial, L2)...")
    t0 = time.time()
    lr = LogisticRegression(max_iter=3000, C=1.0, random_state=cfg.seed, n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_time = time.time() - t0
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)
    results["logreg"] = {
        "name": "Logistic Regression",
        "accuracy": float(accuracy_score(y_test, lr_pred)),
        "f1_macro": float(f1_score(y_test, lr_pred, average="macro")),
        "log_loss": float(log_loss(y_test, lr_proba, labels=list(range(train_ds.n_classes)))),
        "train_time_sec": round(lr_time, 2),
    }

    print("Training Random Forest (500 trees)...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=500, random_state=cfg.seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    results["random_forest"] = {
        "name": "Random Forest",
        "accuracy": float(accuracy_score(y_test, rf_pred)),
        "f1_macro": float(f1_score(y_test, rf_pred, average="macro")),
        "log_loss": float(log_loss(y_test, rf_proba, labels=list(range(train_ds.n_classes)))),
        "train_time_sec": round(rf_time, 2),
    }

    print("Training plain MLP (no SSL, no equivariance, no Bayesian head)...")
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class PlainMLP(nn.Module):
        def __init__(self, n_in, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(cfg.seed)
    model = PlainMLP(train_ds.n_features, train_ds.n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    Xt_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt_train = torch.tensor(y_train, dtype=torch.long, device=device)
    Xt_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    yt_test = torch.tensor(y_test, dtype=torch.long, device=device)

    t0 = time.time()
    model.train()
    for epoch in range(100):
        opt.zero_grad()
        out = model(Xt_train)
        loss = loss_fn(out, yt_train)
        loss.backward()
        opt.step()
    mlp_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        logits = model(Xt_test)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()

    results["plain_mlp"] = {
        "name": "Plain MLP (baseline, no SSL/equivariance/Bayesian head)",
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "log_loss": float(log_loss(y_test, proba, labels=list(range(train_ds.n_classes)))),
        "train_time_sec": round(mlp_time, 2),
    }

    print("\n=== Baseline results (test set) ===")
    for k, v in results.items():
        print(f"{v['name']:55s} acc={v['accuracy']:.4f}  f1_macro={v['f1_macro']:.4f}  log_loss={v['log_loss']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/baselines.json")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    results = run_baselines(cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
