"""Self-supervised pretraining of the TabularSetBackbone with a VICReg
objective over two stochastically-corrupted views of each sample (see
data/augment.py and training/vicreg.py).

Usage:
    PYTHONPATH=src python3 -m tcga_bench.training.pretrain_ssl --config configs/default.yaml
"""
import argparse
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from tcga_bench.data.augment import augment_batch
from tcga_bench.data.dataset import TCGAExpressionDataset
from tcga_bench.models.backbone import SSLProjector, TabularSetBackbone
from tcga_bench.training.vicreg import vicreg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = TCGAExpressionDataset(cfg.data.processed_dir, "train")
    val_ds = TCGAExpressionDataset(cfg.data.processed_dir, "val")
    print(f"Train samples: {len(train_ds)}  | features: {train_ds.n_features}  | classes: {train_ds.n_classes}")

    train_loader = DataLoader(train_ds, batch_size=cfg.ssl.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.ssl.batch_size, shuffle=False, num_workers=2)

    backbone = TabularSetBackbone(
        n_features=train_ds.n_features,
        dim=cfg.model.dim,
        n_heads=cfg.model.n_heads,
        n_inducing=cfg.model.n_inducing,
        n_isab_layers=cfg.model.n_isab_layers,
        dropout=cfg.model.dropout,
    ).to(device)
    projector = SSLProjector(backbone.out_dim, cfg.ssl.projector_hidden, cfg.ssl.projector_out).to(device)

    opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=cfg.ssl.lr, weight_decay=cfg.ssl.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.ssl.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ckpt_path = Path(cfg.ssl.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_val = float("inf")
    t_start = time.time()

    for epoch in range(1, cfg.ssl.epochs + 1):
        backbone.train()
        projector.train()
        running = {"total": 0.0, "invariance": 0.0, "variance": 0.0, "covariance": 0.0}
        n_batches = 0

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            x_a = augment_batch(x, cfg.ssl.mask_prob, cfg.ssl.noise_std, cfg.ssl.scale_jitter)
            x_b = augment_batch(x, cfg.ssl.mask_prob, cfg.ssl.noise_std, cfg.ssl.scale_jitter)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z_a = projector(backbone(x_a))
                z_b = projector(backbone(x_b))
                loss, parts = vicreg_loss(
                    z_a, z_b,
                    lambda_inv=cfg.ssl.lambda_inv, mu_var=cfg.ssl.mu_var, nu_cov=cfg.ssl.nu_cov,
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running["total"] += loss.item()
            for k in ("invariance", "variance", "covariance"):
                running[k] += parts[k]
            n_batches += 1

        sched.step()
        for k in running:
            running[k] /= n_batches

        # quick validation: same loss, no grad
        backbone.eval()
        projector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                x_a = augment_batch(x, cfg.ssl.mask_prob, cfg.ssl.noise_std, cfg.ssl.scale_jitter)
                x_b = augment_batch(x, cfg.ssl.mask_prob, cfg.ssl.noise_std, cfg.ssl.scale_jitter)
                z_a = projector(backbone(x_a))
                z_b = projector(backbone(x_b))
                loss, _ = vicreg_loss(z_a, z_b, cfg.ssl.lambda_inv, cfg.ssl.mu_var, cfg.ssl.nu_cov)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        history.append({"epoch": epoch, "val_loss": round(val_loss, 4), "elapsed_sec": round(elapsed, 1), **{k: round(v, 4) for k, v in running.items()}})

        if epoch % cfg.ssl.log_every == 0 or epoch == 1 or epoch == cfg.ssl.epochs:
            print(f"[SSL] epoch {epoch:4d}/{cfg.ssl.epochs}  train_loss={running['total']:.4f}  "
                  f"(inv={running['invariance']:.4f} var={running['variance']:.4f} cov={running['covariance']:.4f})  "
                  f"val_loss={val_loss:.4f}  elapsed={elapsed/60:.1f}min")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "backbone_state": backbone.state_dict(),
                "config": OmegaConf.to_container(cfg),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)

    print(f"\nBest SSL val_loss={best_val:.4f}  -> saved to {ckpt_path}")

    import json
    with open(Path(cfg.ssl.checkpoint).parent / "ssl_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
