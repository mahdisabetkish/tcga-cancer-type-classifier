"""Export the trained full model (backbone + Bayesian head, deterministic
mean-weight forward pass) to ONNX for the client-side, in-browser demo
served from GitHub Pages. The live Gradio dashboard already exposes true
MC-sampled Bayesian uncertainty; this export trades that for a model that
runs anywhere with no backend, no server, and no dependency on this
machine being on.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from tcga_bench.data.dataset import TCGAExpressionDataset
from tcga_bench.models.backbone import TabularSetBackbone
from tcga_bench.models.bayesian_head import BayesianLinear

ROOT = Path(__file__).resolve().parents[1]


class InferenceModel(nn.Module):
    """backbone -> Bayesian head (mean weights) -> softmax, as one graph."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        rep = self.backbone(x)
        logits = self.head(rep, sample=False)
        return torch.softmax(logits, dim=-1)


def main():
    cfg = OmegaConf.load(ROOT / "configs" / "default.yaml")
    test_ds = TCGAExpressionDataset(cfg.data.processed_dir, "test")

    backbone = TabularSetBackbone(
        n_features=test_ds.n_features, dim=cfg.model.dim, n_heads=cfg.model.n_heads,
        n_inducing=cfg.model.n_inducing, n_isab_layers=cfg.model.n_isab_layers, dropout=cfg.model.dropout,
    )
    head = BayesianLinear(backbone.out_dim, test_ds.n_classes)
    ckpt = torch.load(ROOT / "checkpoints" / "finetuned_full.pt", map_location="cpu")
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    backbone.eval()
    head.eval()

    model = InferenceModel(backbone, head).eval()

    dummy = torch.from_numpy(test_ds.X[:2])
    out_dir = ROOT / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["features"], output_names=["probs"],
        dynamic_axes={"features": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
    )
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # verify the exported graph matches the PyTorch model numerically
    with torch.no_grad():
        torch_probs = model(dummy).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_probs = sess.run(["probs"], {"features": dummy.numpy().astype(np.float32)})[0]
    max_diff = np.abs(torch_probs - ort_probs).max()
    print(f"Max abs diff (torch vs onnxruntime): {max_diff:.2e}")
    assert max_diff < 1e-4, "ONNX export does not match PyTorch output"
    print("ONNX export verified: matches PyTorch to within 1e-4.")


if __name__ == "__main__":
    main()
