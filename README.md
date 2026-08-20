# Pan-Cancer RNA-Seq Classification

A small research project on classifying tumor type from gene expression data.

The question: given RNA-Seq expression levels for about 20,000 genes, how
well can a handful of standard models tell apart five tumor types (breast,
kidney, colon, lung, prostate) using only a small, interpretable subset of
genes?

## Live dashboard

Open `dashboard/index.html` in a browser, or view it hosted here:

**https://mahdisabetkish.github.io/pancancer-rnaseq-dashboard/dashboard/**

The dashboard has five tabs:

- **Overview**: dataset summary, class balance
- **Explore Data**: PCA and t-SNE projections of the samples
- **Model Performance**: accuracy, F1, confusion matrices, and ROC curves for
  each model, plus which genes each model leaned on most
- **Live Prediction**: pick a sample (or drag the gene-value sliders
  yourself) and get a live prediction, recomputed in the browser
- **Methodology**: what was done and why, plus the caveats below

It's a single static HTML file. No server, no API calls, nothing to install.
The trained model weights are baked into the page as JSON, and the forward
pass for the logistic regression and neural net is re-implemented in plain
JavaScript so predictions run client-side.

## Dataset

[TCGA Pan-Cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)
(UCI Machine Learning Repository): 801 patient samples, 20,531 genes, 5
tumor-type labels. Public, de-identified data. Gene identifiers in the source
are anonymized (`gene_0` ... `gene_20530`), so this isn't tied to any specific
biological pathway without further lookup.

## Approach

1. Log2-transform the raw counts (RNA-Seq expression is heavily right-skewed).
2. Keep the 60 genes with the highest ANOVA F-score (`SelectKBest`). That's
   enough signal to work with and small enough to ship to a browser.
3. Standardize, then train three models on an 80/20 split:
   - Logistic regression (multinomial, L2)
   - Random forest (300 trees), also used for feature-importance ranking
   - A small PyTorch MLP (60 to 64 to 32 to 5, dropout, Adam, 150 epochs)
4. Compare accuracy, F1, confusion matrices and ROC curves across the three.
5. Export everything (metrics, projections, model weights) to one JSON file
   the dashboard reads.

## Repo layout

```
train.py              trains the three models, writes dashboard/data/model_bundle.json
build.py               inlines dashboard/src/* + model_bundle.json into dashboard/index.html
dashboard/src/         source for the dashboard (HTML template, CSS, JS)
dashboard/index.html   the built, self-contained dashboard (this is what you open/host)
```

## Reproducing it

```bash
pip install -r requirements.txt
```

Download the dataset from the UCI link above and unzip it so you have
`data/raw/TCGA-PANCAN-HiSeq-801x20531/data.csv` and `labels.csv`, then:

```bash
python train.py    # trains the models, writes dashboard/data/model_bundle.json
python build.py     # rebuilds dashboard/index.html from the template + bundle
```

## Caveats

This is exploratory, not a diagnostic tool. The gene set here is small and
selected purely by statistical significance on this one dataset, so treat any
"important gene" callouts as a starting point for further reading, not a
clinical claim.
