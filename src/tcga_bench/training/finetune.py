"""
Supervised stage: backbone representation -> BayesianLinear classification
head, trained with cross-entropy + a KL-to-prior term (a variational ELBO),
optionally starting from the SSL-pretrained backbone.

Supports the ablations needed to actually justify each design choice
rather than assert it:
  --no_ssl            random-init backbone instead of the SSL checkpoint
  --no_equivariance   swap TabularSetBackbone for PlainMLPBackbone
  --freeze_backbone   linear probe (backbone frozen) instead of finetuning

Usage:
    PYTHONPATH=src python3 -m tcga_bench.training.finetune --config configs/default.yaml --tag full
    PYTHONPATH=src python3 -m tcga_bench.training.finetune --config configs/default.yaml --tag no_ssl --no_ssl
    PYTHONPATH=src python3 -m tcga_bench.training.finetune --config configs/default.yaml --tag no_equiv --no_equivariance
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

from tcga_bench.data.dataset import TCGAExpressionDataset
from tcga_bench.eval.calibration import expected_calibration_error, predictive_entropy
from tcga_bench.models.backbone import PlainMLPBackbone, TabularSetBackbone
from tcga_bench.models.bayesian_head import BayesianLinear


def build_backbone(cfg, n_features, no_equivariance: bool):
    if no_equivariance:
        return PlainMLPBackbone(n_features, dim=cfg.model.dim, dropout=cfg.model.dropout)
    return TabularSetBackbone(
        n_features=n_features, dim=cfg.model.dim, n_heads=cfg.model.n_heads,
        n_inducing=cfg.model.n_inducing, n_isab_layers=cfg.model.n_isab_layers,
        dropout=cfg.model.dropout,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default="full")
    parser.add_argument("--no_ssl", action="store_true")
    parser.add_argument("--no_equivariance", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TCGAExpressionDataset(cfg.data.processed_dir, "train")
    val_ds = TCGAExpressionDataset(cfg.data.processed_dir, "val")
    test_ds = TCGAExpressionDataset(cfg.data.processed_dir, "test")

    train_loader = DataLoader(train_ds, batch_size=cfg.finetune.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.finetune.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg.finetune.batch_size, shuffle=False, num_workers=2)

    backbone = build_backbone(cfg, train_ds.n_features, args.no_equivariance).to(device)

    if not args.no_ssl and not args.no_equivariance:
        ckpt = torch.load(cfg.ssl.checkpoint, map_location=device)
        backbone.load_state_dict(ckpt["backbone_state"])
        print(f"Loaded SSL-pretrained backbone from {cfg.ssl.checkpoint} (val_loss={ckpt['val_loss']:.4f}, epoch={ckpt['epoch']})")
    else:
        print("Backbone randomly initialized (SSL pretraining skipped for this run).")

    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)

    head = BayesianLinear(backbone.out_dim, train_ds.n_classes).to(device)

    # class weights to counter the 36..1092 imbalance across the 32 classes
    class_counts = np.bincount(train_ds.y, minlength=train_ds.n_classes).astype(np.float32)
    class_weights = torch.tensor(class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1)), device=device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    params = list(head.parameters())
    if not args.freeze_backbone:
        params += list(backbone.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.finetune.lr, weight_decay=cfg.finetune.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.finetune.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ckpt_path = Path(f"checkpoints/finetuned_{args.tag}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    n_train_batches = len(train_loader)
    best_val_acc = -1.0
    history = []
    t_start = time.time()

    for epoch in range(1, cfg.finetune.epochs + 1):
        backbone.train(mode=not args.freeze_backbone)
        head.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                rep = backbone(x)
                logits = head(rep, sample=True)
                kl = head.kl_divergence() / (n_train_batches * x.size(0))
                loss = ce_loss(logits, y) + cfg.finetune.kl_weight * kl
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item()
        sched.step()

        backbone.eval()
        head.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    logits = head(backbone(x), sample=False)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
            val_acc = correct / total

        history.append({"epoch": epoch, "train_loss": round(running_loss / n_train_batches, 4), "val_acc": round(val_acc, 4)})
        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.finetune.epochs:
            print(f"[{args.tag}] epoch {epoch:3d}/{cfg.finetune.epochs}  train_loss={running_loss/n_train_batches:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"backbone_state": backbone.state_dict(), "head_state": head.state_dict(), "epoch": epoch, "val_acc": val_acc}, ckpt_path)

    # ---- final test evaluation with MC-sampled Bayesian head ----
    best = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(best["backbone_state"])
    head.load_state_dict(best["head_state"])
    backbone.eval()
    head.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                probs = head.predict_distribution(backbone(x), n_samples=cfg.finetune.mc_samples_eval)
            all_probs.append(probs.mean(dim=0).float().cpu().numpy())
            all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)

    ece, bins = expected_calibration_error(probs, labels)
    entropy = predictive_entropy(probs)

    metrics = {
        "tag": args.tag,
        "config": {"no_ssl": args.no_ssl, "no_equivariance": args.no_equivariance, "freeze_backbone": args.freeze_backbone},
        "best_val_acc": best_val_acc,
        "test_accuracy": float(accuracy_score(labels, preds)),
        "test_f1_macro": float(f1_score(labels, preds, average="macro")),
        "ece": ece,
        "mean_predictive_entropy": float(entropy.mean()),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "calibration_bins": bins,
        "train_time_sec": round(time.time() - t_start, 1),
        "history": history,
    }

    out_path = Path(f"results/finetune_{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{args.tag}] TEST accuracy={metrics['test_accuracy']:.4f}  f1_macro={metrics['test_f1_macro']:.4f}  ECE={ece:.4f}")
    print(f"Wrote {out_path} and {ckpt_path}")


if __name__ == "__main__":
    main()
