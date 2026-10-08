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

## Results

See the linked source repository and its live dashboard (Hugging Face
Space) for the full evaluation, including comparisons against standard
baselines and an ablation study (no-SSL-pretraining, no-equivariant-encoder
variants) with calibration metrics.

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

    card = MODEL_CARD_TEMPLATE.format(
        n_samples=bundle.get("n_test_samples", "?"),
        n_features=bundle.get("n_features", "?"),
        n_classes=len(bundle.get("classes", [])),
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
