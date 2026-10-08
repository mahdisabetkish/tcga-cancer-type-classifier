"""
Build a clean, model-ready dataset from the raw UCSC Toil TCGA Pan-Cancer
RNA-seq recompute (gene-level RSEM, log2(fpkm+0.001)) and its phenotype table.

Scope, by design:
  - Restricted to TCGA "Primary Tumor" samples only (no GTEx/TARGET, no
    normal-tissue or metastatic samples) -> a clean, fixed multi-class
    classification problem: which of 33 categories does this expression
    profile belong to. The label is used purely as a categorical class id
    for a tabular ML benchmark; no biological or clinical interpretation
    is made anywhere in this pipeline.
  - Near-constant genes (floor-value in >99% of samples, i.e. effectively
    undetected across the whole cohort) are dropped: they carry no signal
    and just inflate dimensionality for no benefit.

Outputs (written to data/processed/):
  - features.npy   float32 array, shape (n_samples, n_genes)
  - labels.npy     int64 array,  shape (n_samples,)
  - classes.json   list[str], index -> class name
  - feature_ids.json  list[str], index -> original Ensembl gene id
  - splits.json    train/val/test sample indices (stratified)
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

RNG = 42
FLOOR_VALUE = -9.9658  # log2(0.001 + 0) — RSEM FPKM floor used by this release
MAX_FLOOR_FRACTION = 0.99  # drop genes that are at the floor in >99% of samples

print("Loading phenotype table...")
pheno = pd.read_csv(RAW / "TcgaTargetGTEX_phenotype.txt.gz", sep="\t", encoding="latin-1")
pheno = pheno[pheno["_study"] == "TCGA"]
pheno = pheno[pheno["_sample_type"] == "Primary Tumor"]
pheno = pheno.dropna(subset=["primary disease or tissue"])
pheno = pheno.set_index("sample")
print("TCGA primary-tumor samples with a label:", len(pheno))

print("Loading expression matrix (genes x samples)... this is the slow step")
expr = pd.read_csv(RAW / "tcga_RSEM_gene_fpkm.gz", sep="\t", index_col=0)
print("Raw matrix:", expr.shape, "(genes, samples)")

common = expr.columns.intersection(pheno.index)
print("Samples present in both files:", len(common))

expr = expr[common]
labels_raw = pheno.loc[common, "primary disease or tissue"]

print("Dropping near-constant (floor-value) genes...")
floor_frac = (np.isclose(expr.values, FLOOR_VALUE, atol=1e-3)).mean(axis=1)
keep_mask = floor_frac < MAX_FLOOR_FRACTION
expr = expr.loc[keep_mask]
print(f"Kept {keep_mask.sum()} / {len(keep_mask)} genes after floor-value filter")

X = expr.values.T.astype(np.float32)  # (n_samples, n_genes)
gene_ids = expr.index.tolist()

classes = sorted(labels_raw.unique().tolist())
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[c] for c in labels_raw.values], dtype=np.int64)

print("Final matrix:", X.shape, "| classes:", len(classes))
counts = np.bincount(y)
for c, n in zip(classes, counts):
    print(f"  {n:5d}  {c}")

idx = np.arange(len(y))
idx_train, idx_rest, y_train, y_rest = train_test_split(
    idx, y, test_size=0.30, random_state=RNG, stratify=y
)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_rest, y_rest, test_size=0.50, random_state=RNG, stratify=y_rest
)

np.save(OUT / "features.npy", X)
np.save(OUT / "labels.npy", y)
with open(OUT / "classes.json", "w") as f:
    json.dump(classes, f, indent=2)
with open(OUT / "feature_ids.json", "w") as f:
    json.dump(gene_ids, f)
with open(OUT / "splits.json", "w") as f:
    json.dump(
        {
            "train": idx_train.tolist(),
            "val": idx_val.tolist(),
            "test": idx_test.tolist(),
        },
        f,
    )

size_mb = (X.nbytes) / 1e6
print(f"\nWrote features.npy ({size_mb:.1f} MB), labels, classes, splits to {OUT}")
print(f"train/val/test sizes: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")
