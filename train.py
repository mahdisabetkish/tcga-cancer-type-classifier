"""
Train cancer-type classifiers on the TCGA Pan-Cancer RNA-Seq dataset
(801 samples x 20531 genes, 5 classes: BRCA, KIRC, COAD, LUAD, PRAD).

Produces a single JSON file (dashboard/data/model_bundle.json) containing
everything the static, client-side dashboard needs: metrics, plot data,
and portable model weights so predictions can be recomputed live in the
browser with zero backend / zero server.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

import torch
import torch.nn as nn

RNG = 42
np.random.seed(RNG)
torch.manual_seed(RNG)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "TCGA-PANCAN-HiSeq-801x20531"
OUT_DIR = ROOT / "dashboard" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_GENES = 60  # number of top discriminative genes kept for modeling + UI sliders
CLASS_INFO = {
    "BRCA": "Breast invasive carcinoma",
    "KIRC": "Kidney renal clear cell carcinoma",
    "COAD": "Colon adenocarcinoma",
    "LUAD": "Lung adenocarcinoma",
    "PRAD": "Prostate adenocarcinoma",
}

print("Loading data...")
X_full = pd.read_csv(DATA_DIR / "data.csv", index_col=0)
y_raw = pd.read_csv(DATA_DIR / "labels.csv", index_col=0)["Class"]
classes = sorted(y_raw.unique().tolist())
print("Samples:", X_full.shape, "Classes:", classes)

# log2 transform (RNA-seq expression is heavily right-skewed / non-negative)
X_log = np.log2(X_full.values + 1.0)

# ---- feature selection: top-K most discriminative genes (ANOVA F-test) ----
selector = SelectKBest(score_func=f_classif, k=K_GENES)
X_sel = selector.fit_transform(X_log, y_raw.values)
top_idx = selector.get_support(indices=True)
gene_names = [X_full.columns[i] for i in top_idx]
f_scores = selector.scores_[top_idx]

# ---- train / test split ----
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_sel, y_raw.values, np.arange(len(y_raw)),
    test_size=0.2, random_state=RNG, stratify=y_raw.values,
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

y_train_idx = np.array([classes.index(c) for c in y_train])
y_test_idx = np.array([classes.index(c) for c in y_test])

results = {}
weights = {}

# =========================== Logistic Regression ===========================
print("Training Logistic Regression...")
t0 = time.time()
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=RNG)
lr.fit(X_train_s, y_train_idx)
lr_time = time.time() - t0
lr_pred = lr.predict(X_test_s)
lr_proba = lr.predict_proba(X_test_s)

weights["logreg"] = {
    "coef": lr.coef_.tolist(),       # (5, K)
    "intercept": lr.intercept_.tolist(),  # (5,)
}

# ============================= Random Forest ================================
print("Training Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RNG, n_jobs=-1)
rf.fit(X_train_s, y_train_idx)
rf_time = time.time() - t0
rf_pred = rf.predict(X_test_s)
rf_proba = rf.predict_proba(X_test_s)
rf_importances = rf.feature_importances_

# ======================== PyTorch Neural Network =============================
print("Training Neural Network (PyTorch)...")


class MLP(nn.Module):
    def __init__(self, n_in, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


Xt_train = torch.tensor(X_train_s, dtype=torch.float32)
yt_train = torch.tensor(y_train_idx, dtype=torch.long)
Xt_test = torch.tensor(X_test_s, dtype=torch.float32)
yt_test = torch.tensor(y_test_idx, dtype=torch.long)

model = MLP(K_GENES, len(classes))
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 150
history = {"epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    opt.zero_grad()
    out = model(Xt_train)
    loss = loss_fn(out, yt_train)
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        train_pred = out.argmax(1)
        train_acc = (train_pred == yt_train).float().mean().item()
        test_out = model(Xt_test)
        test_loss = loss_fn(test_out, yt_test).item()
        test_pred = test_out.argmax(1)
        test_acc = (test_pred == yt_test).float().mean().item()

    if epoch % 3 == 0 or epoch == 1:
        history["epoch"].append(epoch)
        history["train_loss"].append(round(loss.item(), 4))
        history["train_acc"].append(round(train_acc, 4))
        history["test_loss"].append(round(test_loss, 4))
        history["test_acc"].append(round(test_acc, 4))

nn_time = time.time() - t0

model.eval()
with torch.no_grad():
    nn_logits = model(Xt_test)
    nn_proba = torch.softmax(nn_logits, dim=1).numpy()
    nn_pred = nn_logits.argmax(1).numpy()

# export MLP weights for JS forward pass
sd = model.state_dict()
weights["mlp"] = {
    "W1": sd["net.0.weight"].tolist(), "b1": sd["net.0.bias"].tolist(),
    "W2": sd["net.3.weight"].tolist(), "b2": sd["net.3.bias"].tolist(),
    "W3": sd["net.6.weight"].tolist(), "b3": sd["net.6.bias"].tolist(),
}

# ================================ Metrics =====================================
def eval_block(name, y_true_idx, y_pred_idx, y_proba, train_time):
    acc = accuracy_score(y_true_idx, y_pred_idx)
    f1_macro = f1_score(y_true_idx, y_pred_idx, average="macro")
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=list(range(len(classes)))
    )
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(classes))))

    y_bin = label_binarize(y_true_idx, classes=list(range(len(classes))))
    roc = {}
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc[c] = {
            "fpr": [round(v, 4) for v in fpr.tolist()],
            "tpr": [round(v, 4) for v in tpr.tolist()],
            "auc": round(float(auc(fpr, tpr)), 4),
        }

    return {
        "name": name,
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "train_time_sec": round(train_time, 3),
        "per_class": [
            {
                "class": classes[i],
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(len(classes))
        ],
        "confusion_matrix": cm.tolist(),
        "roc": roc,
    }


results["logreg"] = eval_block("Logistic Regression", y_test_idx, lr_pred, lr_proba, lr_time)
results["rf"] = eval_block("Random Forest", y_test_idx, rf_pred, rf_proba, rf_time)
results["mlp"] = eval_block("Neural Network (PyTorch MLP)", y_test_idx, nn_pred, nn_proba, nn_time)

print("\n=== Test accuracy ===")
for k, v in results.items():
    print(f"{v['name']:35s} acc={v['accuracy']:.4f}  f1_macro={v['f1_macro']:.4f}  train_time={v['train_time_sec']}s")

# ============================ 2D projections for exploration =================
print("Computing PCA / t-SNE projections...")
X_all_s = scaler.transform(X_sel)  # standardize full set (train+test) with train-fit scaler
pca2 = PCA(n_components=2, random_state=RNG)
pca_coords = pca2.fit_transform(X_all_s)
pca_var = pca2.explained_variance_ratio_.tolist()

tsne = TSNE(n_components=2, random_state=RNG, perplexity=30, init="pca")
tsne_coords = tsne.fit_transform(X_all_s)

split_flag = np.array(["test" if i in set(idx_test) else "train" for i in range(len(y_raw))])

projections = {
    "pca": {
        "var_explained": [round(v, 4) for v in pca_var],
        "points": [
            {"x": round(float(pca_coords[i, 0]), 3), "y": round(float(pca_coords[i, 1]), 3),
             "label": y_raw.values[i], "split": split_flag[i], "sample": str(y_raw.index[i])}
            for i in range(len(y_raw))
        ],
    },
    "tsne": {
        "points": [
            {"x": round(float(tsne_coords[i, 0]), 3), "y": round(float(tsne_coords[i, 1]), 3),
             "label": y_raw.values[i], "split": split_flag[i], "sample": str(y_raw.index[i])}
            for i in range(len(y_raw))
        ],
    },
}

# ============================ Feature importance ==============================
lr_importance = np.mean(np.abs(lr.coef_), axis=0)
feature_importance = [
    {
        "gene": gene_names[i],
        "f_score": round(float(f_scores[i]), 2),
        "rf_importance": round(float(rf_importances[i]), 5),
        "lr_importance": round(float(lr_importance[i]), 5),
    }
    for i in range(K_GENES)
]
feature_importance.sort(key=lambda d: -d["rf_importance"])

# ============================ Sample bank for live demo =======================
# raw (pre-standardization, log2) top-K gene values for every sample, so the UI
# can load a real patient-like profile and let the user perturb it with sliders.
gene_min = X_sel.min(axis=0)
gene_max = X_sel.max(axis=0)
gene_mean_by_class = {
    c: X_sel[y_raw.values == c].mean(axis=0).round(3).tolist() for c in classes
}

samples_bank = []
for i in range(len(y_raw)):
    samples_bank.append({
        "id": str(y_raw.index[i]),
        "label": y_raw.values[i],
        "split": split_flag[i],
        "values": [round(float(v), 3) for v in X_sel[i]],
    })

# =============================== Assemble bundle ================================
bundle = {
    "meta": {
        "dataset": "TCGA Pan-Cancer RNA-Seq (UCI ML Repository)",
        "source_url": "https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq",
        "n_samples": int(X_full.shape[0]),
        "n_genes_total": int(X_full.shape[1]),
        "n_genes_selected": K_GENES,
        "classes": classes,
        "class_info": CLASS_INFO,
        "class_counts": {c: int((y_raw.values == c).sum()) for c in classes},
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "random_state": RNG,
        "note": "Feature selection: SelectKBest ANOVA F-test on log2-transformed counts.",
    },
    "gene_names": gene_names,
    "gene_ranges": {
        "min": [round(float(v), 3) for v in gene_min],
        "max": [round(float(v), 3) for v in gene_max],
    },
    "scaler": {
        "mean": scaler.mean_.round(6).tolist(),
        "scale": scaler.scale_.round(6).tolist(),
    },
    "gene_mean_by_class": gene_mean_by_class,
    "results": results,
    "mlp_history": history,
    "weights": weights,
    "projections": projections,
    "feature_importance": feature_importance,
    "samples": samples_bank,
}

out_path = OUT_DIR / "model_bundle.json"
with open(out_path, "w") as f:
    json.dump(bundle, f)

size_mb = out_path.stat().st_size / 1e6
print(f"\nWrote {out_path} ({size_mb:.2f} MB)")
