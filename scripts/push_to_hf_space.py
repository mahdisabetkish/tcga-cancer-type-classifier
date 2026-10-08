"""
Publish the live dashboard as a Hugging Face Space (Gradio SDK). The Space
pulls model weights from the model repo (push_to_huggingface.py) at
startup; this script only ships the app code + small static assets.

Requires HF authentication (`huggingface-cli login` or HF_TOKEN env var).

Usage:
    python3 scripts/push_to_hf_space.py --space-id <your-username>/tcga-pancancer-dashboard
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: TCGA Pan-Cancer Representation Learning
emoji: 🧮
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

Live demo for a self-supervised, permutation-equivariant, uncertainty-aware
classifier on large-scale high-dimensional tabular data (TCGA Pan-Cancer
RNA-seq used purely as a large multi-class tabular benchmark -- no
biological or clinical claims). Source: https://github.com/mahdisabetkish/tcga-cancer-type-classifier
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--hf-repo-id", default="mahdisabetkish/tcga-pancancer-set-transformer",
                         help="Model repo the Space downloads weights from at runtime.")
    args = parser.parse_args()

    create_repo(args.space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api = HfApi()

    readme_path = ROOT / "SPACE_README.md"
    readme_path.write_text(SPACE_README)
    api.upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md",
                     repo_id=args.space_id, repo_type="space")
    readme_path.unlink()

    api.upload_file(path_or_fileobj=str(ROOT / "dashboard_app" / "app.py"), path_in_repo="app.py",
                     repo_id=args.space_id, repo_type="space")
    api.upload_file(path_or_fileobj=str(ROOT / "requirements.txt"), path_in_repo="requirements.txt",
                     repo_id=args.space_id, repo_type="space")
    api.upload_folder(folder_path=str(ROOT / "src"), path_in_repo="src",
                       repo_id=args.space_id, repo_type="space")
    api.upload_folder(folder_path=str(ROOT / "configs"), path_in_repo="configs",
                       repo_id=args.space_id, repo_type="space")

    bundle_dir = ROOT / "dashboard_app" / "data"
    for f in ["classes.json", "bundle.json"]:
        p = bundle_dir / f
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"data/{f}",
                             repo_id=args.space_id, repo_type="space")

    api.add_space_variable(args.space_id, "HF_REPO_ID", args.hf_repo_id)

    print(f"\nDone: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
