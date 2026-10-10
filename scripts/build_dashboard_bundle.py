"""
Aggregate baseline results, finetune/ablation results, and a 2D projection
of the learned representation into one JSON bundle the dashboard app reads
at startup. Keeps the dashboard itself free of any training-time
dependencies (sklearn PCA/UMAP, raw feature matrix, etc.).
"""
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

from tcga_bench.data.dataset import TCGAExpressionDataset
from tcga_bench.eval.interpretability import attention_attribution
from tcga_bench.models.backbone import TabularSetBackbone
from tcga_bench.models.bayesian_head import BayesianLinear

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = OmegaConf.load(ROOT / "configs" / "default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = TCGAExpressionDataset(cfg.data.processed_dir, "test")

    backbone = TabularSetBackbone(
        n_features=test_ds.n_features, dim=cfg.model.dim, n_heads=cfg.model.n_heads,
        n_inducing=cfg.model.n_inducing, n_isab_layers=cfg.model.n_isab_layers, dropout=cfg.model.dropout,
    ).to(device)
    head = BayesianLinear(backbone.out_dim, test_ds.n_classes).to(device)

    ckpt = torch.load(ROOT / "checkpoints" / "finetuned_full.pt", map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    backbone.eval()
    head.eval()

    # The ISAB attention tensors scale with (batch * n_inducing * n_features),
    # which is tens of GB for the full test set in one shot on this feature
    # count -- batch the forward passes to keep peak memory bounded.
    batch_size = 16
    reps_list, probs_list = [], []
    with torch.no_grad():
        for start in range(0, len(test_ds.y), batch_size):
            xb = torch.from_numpy(test_ds.X[start:start + batch_size]).to(device)
            reps_b = backbone(xb)
            reps_list.append(reps_b.cpu().numpy())
            probs_b = head.predict_distribution(reps_b, n_samples=cfg.finetune.mc_samples_eval)
            probs_list.append(probs_b.mean(dim=0).cpu().numpy())
    reps = np.concatenate(reps_list)
    probs = np.concatenate(probs_list)

    pca = PCA(n_components=2, random_state=cfg.seed)
    coords = pca.fit_transform(reps)

    print("Computing attention-based global feature attribution...")
    n_attn_samples = min(300, len(test_ds.y))
    attn_idx = np.random.RandomState(cfg.seed).choice(len(test_ds.y), n_attn_samples, replace=False)
    attn_list = []
    with torch.no_grad():
        for start in range(0, len(attn_idx), batch_size):
            idx_b = attn_idx[start:start + batch_size]
            attn_list.append(attention_attribution(backbone, torch.from_numpy(test_ds.X[idx_b]).to(device)).cpu())
    attn = torch.cat(attn_list)
    mean_attn = attn.mean(dim=0).numpy()
    top_k = 25
    top_idx = np.argsort(-mean_attn)[:top_k]
    feature_attribution = {
        "note": "Mean PMA-pooling attention weight per feature, averaged over "
                f"{n_attn_samples} test samples. Feature identity shown as an "
                "anonymized index only -- no biological interpretation is made.",
        "top_features": [
            {"feature_idx": int(i), "mean_attention": round(float(mean_attn[i]), 6)}
            for i in top_idx
        ],
    }

    results = {}
    for name in ["baselines", "finetune_full", "finetune_no_ssl", "finetune_no_equivariance"]:
        p = ROOT / "results" / f"{name}.json"
        if p.exists():
            with open(p) as f:
                results[name] = json.load(f)

    sample_bank = []
    preds = probs.argmax(axis=1)
    for i in range(len(test_ds.y)):
        sample_bank.append({
            "idx": i,
            "true_label": int(test_ds.y[i]),
            "pred_label": int(preds[i]),
            "probs": [round(float(v), 4) for v in probs[i]],
            "pca_x": round(float(coords[i, 0]), 3),
            "pca_y": round(float(coords[i, 1]), 3),
        })

    bundle = {
        "classes": test_ds.classes,
        "n_features": test_ds.n_features,
        "n_test_samples": len(test_ds.y),
        "pca_var_explained": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "feature_attribution": feature_attribution,
        "results": results,
        "samples": sample_bank,
    }

    out_dir = ROOT / "dashboard_app" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bundle.json", "w") as f:
        json.dump(bundle, f)
    print(f"Wrote {out_dir / 'bundle.json'}")


if __name__ == "__main__":
    main()
