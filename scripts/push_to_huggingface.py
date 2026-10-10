"""
Push the trained model checkpoint, config, demo samples, and a model card
to a Hugging Face Hub model repo.

Requires HF authentication: run `huggingface-cli login` once, or set the
HF_TOKEN environment variable, before running this script. Nothing in
this repo's automation logs in on your behalf.

Usage:
    python3 scripts/push_to_huggingface.py --repo-id <your-username>/tcga-pancancer-set-transformer
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]

MODEL_CARD_TEMPLATE = """---
license: mit
tags:
  - tabular-classification
  - self-supervised-learning
  - set-transformer
  - bayesian-deep-learning
  - pytorch
datasets:
  - custom
---

# TCGA Pan-Cancer Set-Transformer

A self-supervised, permutation-equivariant, uncertainty-aware classifier
for large-scale, high-dimensional tabular data, evaluated on the UCSC
Toil TCGA Pan-Cancer RNA-seq recompute (~{n_samples} samples, ~{n_features}
features, {n_classes} categorical classes).

**This is a methods/representation-learning study. No biological,
diagnostic, or clinical claims are made or implied.** Class labels are
used strictly as categorical ids for a tabular multi-class benchmark.

## Architecture

- Feature tokenizer: shared value projection + learned per-feature
  identity embedding (see `tokenizer.py` in the source repo).
- Set-Transformer encoder (Lee et al., 2019) with Induced Set Attention
  Blocks for O(n*m) scalability across tens of thousands of feature
  tokens, and Pooling by Multihead Attention for the final representation.
- VICReg self-supervised pretraining (Bardes, Ponce & LeCun, 2022).
- Mean-field variational ("Bayes-by-Backprop") linear classification head
  for calibrated predictive uncertainty.

## Files

- `model.pt` -- backbone + Bayesian head state dicts (see source repo for
  the exact module definitions needed to load them).
- `config.yaml` -- architecture / training hyperparameters.
- `demo_samples.npz`, `classes.json` -- small bundle used by the live
  dashboard (not the full training dataset).

## Results, reported honestly

The point of this benchmark is to check whether each piece of the
architecture earns its place, not to assume it does. On this dataset
and split, it mostly doesn't, on raw test accuracy:

| Model | Test accuracy | F1 (macro) | ECE |
|---|---|---|---|
| Logistic regression (baseline) | {logreg_acc} | {logreg_f1} | -- |
| Plain MLP (baseline) | {mlp_acc} | {mlp_f1} | -- |
| Random forest (baseline) | {rf_acc} | {rf_f1} | -- |
| Ablation: no equivariant encoder (MLP + Bayesian head) | {no_equiv_acc} | {no_equiv_f1} | {no_equiv_ece} |
| Ablation: no SSL pretraining | {no_ssl_acc} | {no_ssl_f1} | {no_ssl_ece} |
| **Full model** (SSL + equivariant encoder + Bayesian head) | {full_acc} | {full_f1} | {full_ece} |

The simple baselines win here. This dataset is close to linearly
separable in log-expression space, so the architectural machinery built
for harder, less separable tabular problems doesn't pay for itself on
raw accuracy on this one -- and that's a real finding, not a caveat to
bury. Full history, per-class breakdowns, and the code for every row
above are in the source repository.

## Live demo

- In-browser demo (ONNX Runtime Web, no server to keep running):
  https://mahdisabetkish.github.io/tcga-cancer-type-classifier/
- Full interactive dashboard (Docker image, local run):
  `docker run -p 7860:7860 mahdisabetkish/tcga-pancancer-set-transformer`

## Source

Code: https://github.com/mahdisabetkish/tcga-cancer-type-classifier
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/finetuned_full.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", exist_ok=True)

    with open(ROOT / "dashboard_app" / "data" / "bundle.json") as f:
        bundle = json.load(f)
    results = bundle.get("results", {})

    def pct(section, key, metric="accuracy"):
        v = results.get(section, {}).get(key, {}).get(metric)
        return f"{v*100:.1f}%" if v is not None else "n/a"

    def finetune_pct(tag, metric):
        v = results.get(tag, {}).get(metric)
        if metric == "ece":
            return f"{v:.3f}" if v is not None else "n/a"
        return f"{v*100:.1f}%" if v is not None else "n/a"

    card = MODEL_CARD_TEMPLATE.format(
        n_samples=bundle.get("n_test_samples", "?"),
        n_features=bundle.get("n_features", "?"),
        n_classes=len(bundle.get("classes", [])),
        logreg_acc=pct("baselines", "logreg", "accuracy"),
        logreg_f1=pct("baselines", "logreg", "f1_macro"),
        mlp_acc=pct("baselines", "plain_mlp", "accuracy"),
        mlp_f1=pct("baselines", "plain_mlp", "f1_macro"),
        rf_acc=pct("baselines", "random_forest", "accuracy"),
        rf_f1=pct("baselines", "random_forest", "f1_macro"),
        no_equiv_acc=finetune_pct("finetune_no_equivariance", "test_accuracy"),
        no_equiv_f1=finetune_pct("finetune_no_equivariance", "test_f1_macro"),
        no_equiv_ece=finetune_pct("finetune_no_equivariance", "ece"),
        no_ssl_acc=finetune_pct("finetune_no_ssl", "test_accuracy"),
        no_ssl_f1=finetune_pct("finetune_no_ssl", "test_f1_macro"),
        no_ssl_ece=finetune_pct("finetune_no_ssl", "ece"),
        full_acc=finetune_pct("finetune_full", "test_accuracy"),
        full_f1=finetune_pct("finetune_full", "test_f1_macro"),
        full_ece=finetune_pct("finetune_full", "ece"),
    )
    card_path = ROOT / "README_model_card.md"
    card_path.write_text(card)

    files = {
        args.checkpoint: "model.pt",
        args.config: "config.yaml",
        "dashboard_app/data/demo_samples.npz": "demo_samples.npz",
        "dashboard_app/data/classes.json": "classes.json",
        str(card_path): "README.md",
    }
    for local, remote in files.items():
        local_path = ROOT / local if not Path(local).is_absolute() else Path(local)
        print(f"Uploading {local_path} -> {remote}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="model",
        )

    card_path.unlink()
    print(f"\nDone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
