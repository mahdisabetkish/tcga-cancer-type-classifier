"""
Live, interactive Gradio dashboard for the TCGA pan-cancer representation-
learning benchmark. Pure methods demo: a multi-class classifier over a
large, high-dimensional tabular dataset, with live inference, calibrated
uncertainty, an embedding explorer, and the ablation study that justifies
each architectural choice. No biological, diagnostic, or clinical claims
are made anywhere in this app -- labels are treated strictly as categorical
class ids for a tabular ML problem.

Model weights and the demo-sample bundle are pulled from a Hugging Face
Hub model repo at startup (falling back to local files for development).
"""
import json
import os
import sys
from pathlib import Path

# Two layouts need to work: (1) inside the main repo, this file lives at
# dashboard_app/app.py with src/ one level up; (2) on a Hugging Face Space,
# this file is uploaded as app.py at the repo root with src/ alongside it.
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE if (_HERE / "src").exists() else _HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf

from tcga_bench.eval.interpretability import attention_attribution, integrated_gradients
from tcga_bench.models.backbone import TabularSetBackbone
from tcga_bench.models.bayesian_head import BayesianLinear

ROOT = _HERE
HF_REPO_ID = os.environ.get("HF_REPO_ID", "mahdisabetkish/tcga-pancancer-set-transformer")


def _resolve(filename: str, local_path: Path) -> Path:
    if local_path.exists():
        return local_path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=filename))


def load_everything():
    cfg = OmegaConf.load(_resolve("config.yaml", REPO_ROOT / "configs" / "default.yaml"))

    demo_path = _resolve("demo_samples.npz", ROOT / "data" / "demo_samples.npz")
    demo = np.load(demo_path, allow_pickle=True)

    classes_path = _resolve("classes.json", ROOT / "data" / "classes.json")
    with open(classes_path) as f:
        classes = json.load(f)

    bundle_path = ROOT / "data" / "bundle.json"
    bundle = {}
    if bundle_path.exists():
        with open(bundle_path) as f:
            bundle = json.load(f)

    n_features = demo["X_std"].shape[1]
    backbone = TabularSetBackbone(
        n_features=n_features, dim=cfg.model.dim, n_heads=cfg.model.n_heads,
        n_inducing=cfg.model.n_inducing, n_isab_layers=cfg.model.n_isab_layers, dropout=cfg.model.dropout,
    )
    head = BayesianLinear(backbone.out_dim, len(classes))

    ckpt_path = _resolve("model.pt", REPO_ROOT / "checkpoints" / "finetuned_full.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    backbone.eval()
    head.eval()

    return cfg, demo, classes, bundle, backbone, head


CFG, DEMO, CLASSES, BUNDLE, BACKBONE, HEAD = load_everything()
SLIDER_IDX = DEMO["slider_feature_idx"].tolist()
N_SAMPLES = DEMO["X_std"].shape[0]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color="#1e293b"),
    colorway=["#0d9488", "#6366f1", "#f59e0b", "#ef4444", "#0ea5e9", "#84cc16", "#a855f7", "#ec4899"],
    margin=dict(l=10, r=10, t=44, b=10),
)


def style(fig: go.Figure, **overrides) -> go.Figure:
    fig.update_layout(**{**PLOTLY_LAYOUT, **overrides})
    return fig


@torch.no_grad()
def predict(raw_vector: np.ndarray) -> dict:
    std_vec = (raw_vector - DEMO["scaler_mean"]) / DEMO["scaler_std"]
    x = torch.from_numpy(std_vec.astype(np.float32)).unsqueeze(0)
    rep = BACKBONE(x)
    probs = HEAD.predict_distribution(rep, n_samples=30).mean(dim=0).squeeze(0).numpy()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return probs, entropy


def run_live_prediction(sample_idx: int, *slider_values):
    raw = DEMO["X_raw"][sample_idx].copy()
    for pos, val in zip(SLIDER_IDX, slider_values):
        raw[pos] = val

    probs, entropy = predict(raw)
    true_label = CLASSES[int(DEMO["y"][sample_idx])]

    order = np.argsort(-probs)[:5]
    fig = go.Figure(go.Bar(
        x=[probs[i] for i in order][::-1],
        y=[CLASSES[i] for i in order][::-1],
        orientation="h",
        marker_color="#0d9488",
    ))
    style(fig,
          title=f"Top-5 predicted classes (true: {true_label})",
          xaxis_title="Predicted probability (MC mean, 30 samples)",
          height=320)
    pred_label = CLASSES[int(np.argmax(probs))]
    verdict = "correct" if pred_label == true_label else "incorrect"
    summary = f"Predicted: **{pred_label}** ({verdict}) | True: **{true_label}** | predictive entropy: {entropy:.3f} nats"
    return fig, summary


def explain_prediction(sample_idx: int, *slider_values):
    raw = DEMO["X_raw"][sample_idx].copy()
    for pos, val in zip(SLIDER_IDX, slider_values):
        raw[pos] = val
    std_vec = (raw - DEMO["scaler_mean"]) / DEMO["scaler_std"]
    x = torch.from_numpy(std_vec.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        pred_class = int(torch.argmax(HEAD(BACKBONE(x), sample=False), dim=1).item())

    # fewer steps than the module default (50): a live "explain" button on
    # CPU trades some Integrated-Gradients precision for a few seconds'
    # wait instead of ~15s; see benchmark note in interpretability.py
    attributions = integrated_gradients(BACKBONE, HEAD, x, target_class=pred_class, n_steps=20).numpy()

    order = np.argsort(-np.abs(attributions))[:12]
    colors = ["#0d9488" if v >= 0 else "#ef4444" for v in [attributions[i] for i in order][::-1]]
    fig = go.Figure(go.Bar(
        x=[attributions[i] for i in order][::-1],
        y=[f"feature[{i}]" for i in order][::-1],
        orientation="h",
        marker_color=colors,
    ))
    style(fig,
          title=f"Integrated-Gradients attribution for predicted class: {CLASSES[pred_class]}",
          xaxis_title="attribution (+ pushes toward this class, - pushes away)",
          height=380)
    return fig


def slider_defaults(sample_idx: int):
    raw = DEMO["X_raw"][sample_idx]
    return [float(raw[i]) for i in SLIDER_IDX]


def on_sample_change(sample_idx):
    defaults = slider_defaults(sample_idx)
    return defaults + [None, "", None]


def build_performance_tab():
    results = BUNDLE.get("results", {})
    rows = []
    name_map = {
        "logreg": "Logistic Regression (baseline)",
        "random_forest": "Random Forest (baseline)",
        "plain_mlp": "Plain MLP (baseline)",
    }
    if "baselines" in results:
        for key, v in results["baselines"].items():
            rows.append([name_map.get(key, v["name"]), v["accuracy"], v["f1_macro"], v.get("log_loss"), None])
    tag_map = {
        "finetune_full": "Full model (SSL + equivariant + Bayesian)",
        "finetune_no_ssl": "Ablation: no SSL pretraining",
        "finetune_no_equivariance": "Ablation: no equivariant encoder",
    }
    for key, label in tag_map.items():
        if key in results:
            v = results[key]
            rows.append([label, v["test_accuracy"], v["test_f1_macro"], None, v.get("ece")])

    with gr.Column():
        gr.Markdown("### Full model vs. baselines vs. ablations (held-out test set)")
        gr.Dataframe(
            headers=["Model", "Accuracy", "F1 (macro)", "Log loss", "ECE"],
            value=rows, interactive=False,
        )
        if "finetune_full" in results:
            cm = np.array(results["finetune_full"]["confusion_matrix"])
            fig = go.Figure(go.Heatmap(z=cm, x=CLASSES, y=CLASSES, colorscale="Teal"))
            style(fig, title="Confusion matrix -- full model (test set)", height=600)
            gr.Plot(fig)

            bins = results["finetune_full"]["calibration_bins"]
            xs = [b["avg_confidence"] for b in bins if b["avg_confidence"] is not None]
            ys = [b["avg_accuracy"] for b in bins if b["avg_accuracy"] is not None]
            cal_fig = go.Figure()
            cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                                          line=dict(dash="dash", color="#94a3b8")))
            cal_fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+lines", name="full model",
                                          line=dict(color="#0d9488"), marker=dict(color="#0d9488")))
            style(cal_fig, title=f"Reliability diagram (ECE={results['finetune_full']['ece']:.4f})",
                  xaxis_title="confidence", yaxis_title="accuracy", height=400)
            gr.Plot(cal_fig)


def build_embedding_tab():
    samples = BUNDLE.get("samples", [])
    if not samples:
        gr.Markdown("Embedding projection not available yet.")
        return
    xs = [s["pca_x"] for s in samples]
    ys = [s["pca_y"] for s in samples]
    labels = [CLASSES[s["true_label"]] for s in samples]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", text=labels,
                                marker=dict(size=5, color=[s["true_label"] for s in samples], colorscale="Turbo"),
                                hovertemplate="%{text}<extra></extra>"))
    var = BUNDLE.get("pca_var_explained", [0, 0])
    style(fig, title=f"PCA of learned representation (test set) -- {sum(var)*100:.1f}% variance explained",
          height=600)
    gr.Plot(fig)


def build_interpretability_tab():
    attribution = BUNDLE.get("feature_attribution")
    if not attribution:
        gr.Markdown("Global attention attribution not available yet.")
        return
    gr.Markdown(f"_{attribution['note']}_")
    top = attribution["top_features"]
    fig = go.Figure(go.Bar(
        x=[t["mean_attention"] for t in top][::-1],
        y=[f"feature[{t['feature_idx']}]" for t in top][::-1],
        orientation="h",
        marker_color="#6366f1",
    ))
    style(fig,
          title="Global attribution: mean PMA-pooling attention weight (top 25, averaged over 300 test samples)",
          xaxis_title="mean attention weight", height=600)
    gr.Plot(fig)


CUSTOM_CSS = """
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 18px;
}
.hero-title { font-size: 1.5rem; font-weight: 700; line-height: 1.35; margin-bottom: 6px; color: #f1f5f9; }
.hero-sub { font-size: 0.92rem; color: #cbd5e1; margin-bottom: 18px; }
.stat-grid { display: flex; flex-wrap: wrap; gap: 14px; }
.stat-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    padding: 10px 18px;
    min-width: 128px;
    text-align: center;
}
.stat-num { font-size: 1.35rem; font-weight: 700; color: #2dd4bf; }
.stat-label { font-size: 0.72rem; color: #cbd5e1; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.04em; }
"""


def build_hero_html() -> str:
    results = BUNDLE.get("results", {})
    full = results.get("finetune_full")
    acc = f"{full['test_accuracy']*100:.1f}%" if full else "pending"
    f1 = f"{full['test_f1_macro']*100:.1f}%" if full else "pending"
    ece = f"{full['ece']:.3f}" if full else "pending"
    n_test = BUNDLE.get("n_test_samples", N_SAMPLES)
    n_feat = BUNDLE.get("n_features", DEMO["X_std"].shape[1])
    stats = [
        (n_test, "test samples"), (n_feat, "features"), (len(CLASSES), "classes"),
        (acc, "test accuracy, full model"), (f1, "F1 macro"), (ece, "calibration error (ECE)"),
    ]
    cards = "".join(f'<div class="stat-card"><div class="stat-num">{v}</div>'
                     f'<div class="stat-label">{label}</div></div>' for v, label in stats)
    return (
        '<div class="hero">'
        '<div class="hero-title">Self-Supervised, Equivariant, Uncertainty-Aware '
        'Representation Learning on Large-Scale Tabular Data</div>'
        '<div class="hero-sub">A math and machine learning methods study. '
        'No biological or clinical claim is made anywhere in this project.</div>'
        f'<div class="stat-grid">{cards}</div></div>'
    )


METHODOLOGY_MD = """
## What this is

A methods study in self-supervised, permutation-equivariant, and
uncertainty-aware representation learning, evaluated on a large,
high-dimensional, multi-class tabular dataset (TCGA Pan-Cancer RNA-seq,
32 categories, ~48,500 features, ~9,200 samples).

**This is not a diagnostic or clinical tool, and no biological or
clinical interpretation is made anywhere in this project.** Category
labels are used strictly as categorical class ids for a tabular
classification benchmark.

## Architecture

1. **Feature tokenizer** -- each scalar feature becomes a token via a
   shared value-projection plus a learned per-feature identity embedding.
2. **Set-Transformer encoder** (Lee et al., 2019) with Induced Set
   Attention Blocks (O(n·m) instead of O(n²) self-attention) and Pooling
   by Multihead Attention -- scalable, shared-weight processing of tens
   of thousands of feature tokens.
3. **VICReg self-supervised pretraining** (Bardes, Ponce & LeCun, 2022)
   on two stochastically corrupted views of each sample (feature masking,
   Gaussian noise, scale jitter) -- no negative pairs, no large-batch
   requirement.
4. **Bayesian linear head** (Bayes-by-Backprop; Blundell et al., 2015)
   for classification with calibrated predictive uncertainty via
   Monte-Carlo weight sampling.

## Ablations

The *Model Performance* tab compares the full pipeline against: standard
baselines (logistic regression / random forest / plain MLP), a
no-SSL-pretraining variant, and a no-equivariant-encoder variant -- so
every design choice is checked against the data rather than asserted.
"""


THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="indigo", neutral_hue="slate")

with gr.Blocks(title="TCGA Pan-Cancer Representation Learning Benchmark", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.HTML(build_hero_html())

    with gr.Tab("Live Prediction"):
        gr.Markdown("Pick a held-out sample, optionally perturb its top-variance features, and run a real "
                    "forward pass through the trained model (Bayesian head, 30 Monte-Carlo samples).")
        sample_dd = gr.Dropdown(choices=list(range(N_SAMPLES)), value=0, label="Demo sample index")
        sliders = [
            gr.Slider(
                minimum=float(DEMO["raw_min"][i]), maximum=float(DEMO["raw_max"][i]),
                value=float(DEMO["X_raw"][0][i]), label=f"feature[{i}] (log2 fpkm)",
            )
            for i in SLIDER_IDX
        ]
        predict_btn = gr.Button("Run live prediction", variant="primary")
        plot_out = gr.Plot()
        summary_out = gr.Markdown()

        gr.Markdown("**Explain this prediction** (Integrated Gradients, ~5-10s on CPU): "
                    "which features pushed the model toward / away from its predicted class.")
        explain_btn = gr.Button("Explain prediction")
        explain_plot = gr.Plot()

        sample_dd.change(on_sample_change, inputs=sample_dd, outputs=sliders + [plot_out, summary_out, explain_plot])
        predict_btn.click(run_live_prediction, inputs=[sample_dd] + sliders, outputs=[plot_out, summary_out])
        explain_btn.click(explain_prediction, inputs=[sample_dd] + sliders, outputs=[explain_plot])

    with gr.Tab("Model Performance"):
        build_performance_tab()

    with gr.Tab("Embedding Explorer"):
        build_embedding_tab()

    with gr.Tab("Interpretability"):
        build_interpretability_tab()

    with gr.Tab("Methodology"):
        gr.Markdown(METHODOLOGY_MD)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
