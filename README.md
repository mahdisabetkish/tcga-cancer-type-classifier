# TCGA Pan Cancer Study: Self Supervised Learning on Large Tabular Data

This project is a math and machine learning study. It uses a large table
of numbers with many classes. The goal is to test ideas from self
supervised learning, from set based deep learning, and from Bayesian deep
learning.

**This project is about math, machine learning, deep learning, and
software engineering only. It does not make any biology claim. It does
not make any health claim or medical claim.** The class labels in the
data are only used as plain category ids for a classification task.
Nothing in this project should be read as a biology result or a medical
tool.

## Live demo

- In-browser demo, no server to keep running, nothing to install:
  **https://mahdisabetkish.github.io/tcga-cancer-type-classifier/**
  It runs the trained model in your browser with ONNX Runtime Web and
  lets you try it on held-out test samples.
- Full interactive dashboard, with all five tabs described below, as a
  Docker image: `docker run -p 7860:7860 mahdisabetkish/tcga-pancancer-set-transformer`,
  then open `http://localhost:7860`.
  (https://hub.docker.com/r/mahdisabetkish/tcga-pancancer-set-transformer)
- Trained weights and model card on Hugging Face:
  https://huggingface.co/mahdisabetkish/tcga-pancancer-set-transformer

A note on why there is no live Hugging Face Space: Gradio and Docker
Spaces now require a paid plan even on free CPU hardware, and the one
free exception (ZeroGPU) needs either that same paid plan or a
manually requested grant. So the full dashboard ships as a Docker image
instead, and the always-on demo is the static, in-browser ONNX page
above.

## The question this project asks

Each sample has about 48500 numeric features. There are only about 9200
samples. These samples fall into 32 classes. Some classes have as few as
36 samples. Some classes have more than 1000 samples. This is a hard
setup for learning.

The project tests three ideas together:

1. A model that treats every feature the same way, instead of giving
   each feature its own separate weight. This is called a set based or
   permutation equivariant model. It is built from a Set Transformer.
2. Self supervised pretraining with a method called VICReg. This step
   learns from the data without using any labels at all.
3. A Bayesian output layer. Instead of giving one plain probability
   score, it gives a score with an honest measure of uncertainty.

The project also compares this full model against plain baseline models,
such as logistic regression, random forest, and a plain neural network.
It also removes one part at a time (no pretraining, or no set based
design) to check how much each part actually helps. This comparison is
shown in the dashboard, not just claimed in text.

## Dataset

The data comes from the UCSC Toil TCGA Pan Cancer RNA seq project. You
can read about it here:
https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443

All samples in this source were processed with the same pipeline. This
matters because it removes a common problem in this kind of data, where
different processing steps create fake differences that are not real
signal.

Some facts about the data used here:

- Only "Primary Tumor" samples are used. Normal tissue samples and other
  sample types are removed. This gives one clean 32 class problem.
- Features that almost never change across samples are removed, since
  they carry close to no signal.
- The final table has 9185 samples and 48482 features, with 32 classes.
  The data is split into train, validation, and test sets, with 70
  percent, 15 percent, and 15 percent of the samples.
- The data is public and does not include any personal identity
  information. The exact steps to build this table are in
  `scripts/prepare_data.py`, so anyone can redo them.

## Model design

```
raw feature vector (48482 numbers)
   standardize using train set mean and std
FeatureTokenizer
   one shared small layer turns each number into a token
   plus one learned id vector for each feature position
SetEncoder, a Set Transformer (Lee et al 2019)
   3 ISAB blocks
      instead of comparing every token to every other token,
      it compares tokens to a small set of learned helper points
      this keeps the cost low even with about 48000 tokens
   PMA block
      turns the whole set of tokens into one fixed size vector,
      64 numbers long, in a way that does not care about token order
self supervised step, VICReg
   this part trains the model before it ever sees a label
   it makes two noisy versions of each sample
   then it trains the model so both versions end up close in
   representation space, while still keeping the representation
   spread out and not collapsed
Bayesian output layer
   instead of one fixed weight for each class, each weight is a
   small random distribution
   at test time we sample it many times to get a probability and
   a measure of how sure or unsure the model really is
```

Why not just use a plain neural network with one big first layer? With
48482 input numbers and only about 6400 training rows, a plain first
layer would need one separate weight per input number. That is far too
many free numbers for how little data we have, so it would likely just
memorize the training set. The tokenizer and Set Transformer design
instead reuse the same small layer and the same attention step for every
single feature. Only a small per feature id vector tells the model which
feature is which. This is the exact sense in which the model treats
features as a set instead of a fixed list. You can see this written out
in `src/tcga_bench/models/tokenizer.py`, and there is a test that checks
this property directly in `tests/test_models.py`.

## What is inside this repo

```
src/tcga_bench/
  data       loading data, scaling it, and making noisy views for self supervised training
  models     the tokenizer, the Set Transformer, the Bayesian layer, and the backbones
  training   self supervised training, finetuning, and the baseline models
  eval       tools to check calibration and prediction uncertainty
scripts/
  prepare_data.py            turns raw files into clean feature and label arrays
  fit_scaler.py              computes mean and std from the train set only
  build_demo_samples.py      makes a small sample file for the live dashboard
  build_dashboard_bundle.py  collects all results and plots for the dashboard
  run_pipeline.sh            runs the baselines, the full model, and both ablations in one go
  push_to_huggingface.py     uploads the trained model and a model card to Hugging Face
  push_to_hf_space.py        kept for reference only, see the note on Hugging Face Spaces above
  export_onnx.py             exports the full model to ONNX, used by the in-browser demo
dashboard_app/app.py         the live dashboard, runs locally or as the Docker image above
docs/                        the static, in-browser demo page served by GitHub Pages
configs/default.yaml         every setting used for the model and the training
tests/                       checks for the model, the loss, the calibration tool, and the Bayesian layer
Dockerfile                   builds a small CPU only image that runs the dashboard
.github/workflows/ci.yml     runs the tests on every push, and builds the Docker image on main
```

## How to run it yourself

```bash
pip install -r requirements.txt
pip install -U setuptools wheel
pip install --no-deps --no-build-isolation -e .

# step 1, build the data
python3 scripts/prepare_data.py
python3 scripts/fit_scaler.py

# step 2, self supervised pretraining, no labels used here
PYTHONPATH=src python3 -m tcga_bench.training.pretrain_ssl --config configs/default.yaml

# step 3, baselines, full model, and the two ablations
bash scripts/run_pipeline.sh

# step 4, build the dashboard data
python3 scripts/build_demo_samples.py
PYTHONPATH=src python3 scripts/build_dashboard_bundle.py

# step 5, open the dashboard
python3 dashboard_app/app.py

# step 6, optional, export to ONNX for the static in-browser demo
PYTHONPATH=src python3 scripts/export_onnx.py
```

All settings live in one file, `configs/default.yaml`. All random steps
use a fixed seed. So running these steps again should give you the same
split and close to the same results. If you want to try this method on a
different tabular dataset, you only need to change `scripts/prepare_data.py`
and the data path in the config. The rest of the code does not assume
anything specific about this one dataset.

## Hardware used for training

Training ran on one NVIDIA L40S GPU with 48GB of memory, on a machine
with many CPU cores, using PyTorch with mixed precision. The training
code only uses one GPU right now, but it does not assume only one GPU
can ever be used. Moving it to many GPUs, or to a SLURM cluster, mainly
means changing how the training loop is launched, not changing the model
or the data code.

## Results, reported honestly

The whole point of running baselines and ablations is to find out
whether the extra machinery earns its place, not to assume it does. On
this dataset and split, measured on the held-out test set:

| Model | Test accuracy | F1 (macro) | ECE |
|---|---|---|---|
| Logistic regression (baseline) | 96.1% | 94.4% | -- |
| Plain MLP (baseline) | 95.4% | 93.5% | -- |
| Random forest (baseline) | 94.2% | 89.9% | -- |
| Ablation: no equivariant encoder (MLP + Bayesian head) | 94.0% | 91.7% | 0.053 |
| Ablation: no SSL pretraining | 88.7% | 84.3% | 0.069 |
| **Full model** (SSL + equivariant encoder + Bayesian head) | 88.1% | 83.3% | 0.077 |

The simple baselines come out on top here, on every metric. This
dataset looks close to linearly separable in log-expression space, so
the set-equivariant encoder, the self supervised pretraining, and the
Bayesian head do not pay for themselves on raw accuracy on this
particular problem -- they are built for harder, less separable tabular
problems, and this benchmark is honest about that rather than hiding
it. Full per-epoch history and per-class numbers are in `results/*.json`
and in the dashboard's "Model Performance" tab.

This is also why this repo's results are not compared directly against
an earlier, much smaller version of this project (5 tumor types, 60
hand-picked genes, under 1000 samples): the two are different problems
at different scales, so a side by side number would not mean anything.
This version keeps all 48482 features and all 32 classes on purpose, to
make the comparison between methods the real test, instead of picking
an easy problem that flatters the fancier model.

## Honest limits

This project is a method study. It is not a biology result and not a
medical result. The classes are not balanced, some have very few
samples and some have many. This is handled with class weights in the
loss, and it is shown openly in the per class numbers in the dashboard,
not hidden. Please treat this repo as a demo of method and engineering
skill, and nothing beyond that.
