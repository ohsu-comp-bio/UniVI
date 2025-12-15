# UniVI

[![PyPI version](https://img.shields.io/pypi/v/univi)](https://pypi.org/project/univi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/univi.svg?v=0.2.9)](https://pypi.org/project/univi/)

<picture>
  <!-- Dark mode (GitHub supports this; PyPI may ignore <source>) -->
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.2.9/assets/figures/univi_overview_dark.png">
  <!-- Light mode / fallback (works on GitHub + PyPI) -->
  <img src="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.2.9/assets/figures/univi_overview_light.png"
       alt="UniVI overview and evaluation roadmap"
       width="100%">
</picture>

**UniVI overview and evaluation roadmap.**
(a) Generic UniVI architecture schematic. (b) Core training objective (for UniVI v1 - see documentation for UniVI-lite training objective). (c) Example modality combinations beyond bi-modal data (e.g. TEA-seq (tri-modal RNA + ATAC + ADT)). (d) Evaluation roadmap spanning latent alignment (FOSCTTM), modality mixing, label transfer, reconstruction/prediction NLL, and downstream biological consistency.

---

UniVI is a **multi-modal variational autoencoder (VAE)** framework for aligning and integrating single-cell modalities such as RNA, ADT (CITE-seq), and ATAC. It’s built to support experiments like:

* Joint embedding of RNA + ADT (CITE-seq)
* RNA + ATAC (Multiome) integration
* RNA + ADT + ATAC (TEA-seq) tri-modal data integration
* Independent non-paired modalities from the same tissue type
* Cross-modal reconstruction and imputation
* Data denoising
* Structured evaluation of alignment quality (FOSCTTM, modality mixing, label transfer, etc.)
* Exploratory analysis of the relationships between heterogeneous molecular readouts that inform biological functional dimensions

This repository contains the core UniVI code, training scripts, parameter files, and example notebooks.

---

## Preprint

If you use UniVI in your work, please cite:

> Ashford AJ, Enright T, Nikolova O, Demir E.
> **Unifying Multimodal Single-Cell Data Using a Mixture of Experts β-Variational Autoencoder-Based Framework.**
> *bioRxiv* (2025). doi: [10.1101/2025.02.28.640429](https://www.biorxiv.org/content/10.1101/2025.02.28.640429v1.full)

```bibtex
@article{Ashford2025UniVI,
  title   = {Unifying Multimodal Single-Cell Data Using a Mixture of Experts β-Variational Autoencoder-Based Framework},
  author  = {Ashford, Andrew J. and Enright, Trevor and Nikolova, Olga and Demir, Emek},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.02.28.640429},
  url     = {https://www.biorxiv.org/content/10.1101/2025.02.28.640429v1}
}
````

---

## License

MIT License — see `LICENSE`.

---

## Repository structure

```text
UniVI/
├── README.md                              # Project overview, installation, quickstart
├── LICENSE                                # MIT license text file
├── pyproject.toml                         # Python packaging config (pip / PyPI)
├── assets/                                # Static assets used by README/docs
│   └── figures/                           # Schematic figure(s) for repository front page
├── conda.recipe/                          # Conda build recipe (for conda-build)
│   └── meta.yaml
├── envs/                                  # Example conda environments
│   ├── UniVI_working_environment.yml
│   ├── UniVI_working_environment_v2_full.yml
│   ├── UniVI_working_environment_v2_minimal.yml
│   └── univi_env.yml                      # Recommended env (CUDA-friendly)
├── data/                                  # Small example data notes (datasets are typically external)
│   └── README.md                          # Notes on data sources / formats
├── notebooks/                             # Jupyter notebooks (demos & benchmarks)
│   ├── UniVI_CITE-seq_*.ipynb
│   ├── UniVI_10x_Multiome_*.ipynb
│   └── UniVI_TEA-seq_*.ipynb
├── parameter_files/                       # JSON configs for model + training + data selectors
│   ├── defaults_*.json                    # Default configs (per experiment)
│   └── params_*.json                      # Example “named” configs (RNA, ADT, ATAC, etc.)
├── scripts/                               # Reproducible entry points (revision-friendly)
│   ├── train_univi.py                     # Train UniVI from a parameter JSON
│   ├── evaluate_univi.py                  # Evaluate trained models (FOSCTTM, label transfer, etc.)
│   ├── benchmark_univi_citeseq.py         # CITE-seq-specific benchmarking script
│   ├── run_multiome_hparam_search.py
│   ├── run_frequency_robustness.py        # Composition/frequency mismatch robustness
│   ├── run_do_not_integrate_detection.py  # “Do-not-integrate” unmatched population demo
│   ├── run_benchmarks.py                  # Unified wrapper (includes optional Harmony baseline)
│   └── revision_reproduce_all.sh          # One-click: reproduces figures + supplemental tables
└── univi/                                 # UniVI Python package (importable as `import univi`)
    ├── __init__.py                        # Package exports and __version__
    ├── __main__.py                        # Enables: `python -m univi ...`
    ├── cli.py                             # Minimal CLI (e.g., export-s1, encode)
    ├── pipeline.py                        # Config-driven model+data loading; latent encoding helpers
    ├── diagnostics.py                     # Exports Supplemental_Table_S1.xlsx (env + hparams + dataset stats)
    ├── config.py                          # Config dataclasses (UniVIConfig, ModalityConfig, TrainingConfig)
    ├── data.py                            # Dataset wrappers + matrix selectors (layer/X_key, obsm support)
    ├── evaluation.py                      # Metrics (FOSCTTM, mixing, label transfer, feature recovery)
    ├── matching.py                        # Modality matching / alignment helpers
    ├── objectives.py                      # Losses (ELBO variants, KL/alignment annealing, etc.)
    ├── plotting.py                        # Plotting helpers + consistent style defaults
    ├── trainer.py                         # UniVITrainer: training loop, logging, checkpointing
    ├── figures/                           # Package-internal figure assets (placeholder)
    │   └── .gitkeep
    ├── models/                            # VAE architectures + building blocks
    │   ├── __init__.py
    │   ├── mlp.py                         # Shared MLP building blocks
    │   ├── encoders.py                    # Modality encoders
    │   ├── decoders.py                    # Likelihood-specific decoders (NB, ZINB, Gaussian, etc.)
    │   └── univi.py                       # Core UniVI multi-modal VAE
    ├── hyperparam_optimization/           # Hyperparameter search scripts
    │   ├── __init__.py
    │   ├── common.py
    │   ├── run_adt_hparam_search.py
    │   ├── run_atac_hparam_search.py
    │   ├── run_citeseq_hparam_search.py
    │   ├── run_multiome_hparam_search.py
    │   ├── run_rna_hparam_search.py
    │   └── run_teaseq_hparam_search.py
    └── utils/                             # General utilities
        ├── __init__.py
        ├── io.py                          # I/O helpers (AnnData, configs, checkpoints)
        ├── logging.py                     # Logging configuration / progress reporting
        ├── seed.py                        # Reproducibility helpers (seeding RNGs)
        ├── stats.py                       # Small statistical helpers / transforms
        └── torch_utils.py                 # PyTorch utilities (device, tensor helpers)

```

---

## Generated outputs

Most entry-point scripts write results into a user-specified output directory (commonly `runs/`), which is **not** tracked in git.

A typical `runs/` folder produced by `scripts/revision_reproduce_all.sh` looks like:

```text
runs/
└── <run_name>/
    ├── checkpoints/
    │   └── univi_checkpoint.pt
    ├── eval/
    │   ├── metrics.json                   # Summary metrics (FOSCTTM, label transfer, etc.)
    │   └── metrics.csv                    # Same metrics in tabular form
    ├── robustness/
    │   ├── frequency_perturbation_results.csv
    │   ├── frequency_perturbation_plot.png
    │   ├── frequency_perturbation_plot.pdf
    │   ├── do_not_integrate_summary.csv
    │   ├── do_not_integrate_plot.png
    │   └── do_not_integrate_plot.pdf
    ├── benchmarks/
    │   ├── results.csv                    # Multi-method benchmark table (if enabled)
    │   ├── results.png
    │   └── results.pdf
    └── tables/
        └── Supplemental_Table_S1.xlsx     # Environment + hyperparameters + dataset stats (+ optional metrics)

```

---

## Installation & quickstart

### 1. Install UniVI via PyPI

If you just want to use UniVI:

```bash
pip install univi
```

This installs the `univi` package and all core dependencies.

You can then import it in Python:

```python
import univi
from univi import UniVIMultiModalVAE, ModalityConfig, UniVIConfig
```

> **Note:** UniVI requires `torch`. If `import torch` fails, install PyTorch for your platform/CUDA from:
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### 2. Development install (from source) — recommended for active development

If you want to modify UniVI or run the notebooks exactly as in this repo:

```bash
# Clone the repository
git clone https://github.com/Ashford-A/UniVI.git
cd UniVI

# (Recommended) create a conda env from one of the provided files
conda env create -f envs/univi_env.yml
conda activate univi_env

# Editable install
pip install -e .
```

This makes the `univi` package importable in your scripts and notebooks while keeping it linked to the source tree.

### 3. (Optional) Install via conda / mamba

If UniVI is available on a conda channel (e.g. `conda-forge`), you can install with:

```bash
# Using conda
conda install -c conda-forge univi

# Using mamba
mamba install -c conda-forge univi
```

UniVI is currently available on my custom conda channel and is installable with:

```bash
# Using conda
conda install ashford-a::univi

# Using mamba
mamba install ashford-a::univi
```

To create a fresh environment:

```bash
conda create -n univi_env python=3.10 univi -c conda-forge
conda activate univi_env
```

---

## Preparing input data

UniVI expects per-modality AnnData objects with matching cells (either truly paired data or consistently paired across modalities; `univi/matching.py` contains helper functions for more complex non-joint pairing).

High-level expectations:

* Each modality (e.g. RNA / ADT / ATAC) is an `AnnData` with the **same** `obs_names` (same cells, same order).
* Raw counts are usually stored in `.layers["counts"]`, with a processed view in `.X` used for training.
* Decoder likelihoods should roughly match the distribution of the inputs per modality.

### RNA

* `.layers["counts"]` → raw counts
* `.X` → training representation, e.g.:

  * log1p-normalized HVGs
  * raw counts
  * normalized / scaled counts

Typical decoders:

* `"nb"` or `"zinb"` for raw / count-like data
* `"gaussian"` for log-normalized / scaled data (treated as continuous)

### ADT (CITE-seq)

* `.layers["counts"]` → raw ADT counts
* `.X` → e.g.:

  * CLR-normalized ADT
  * CLR-normalized + scaled ADT
  * raw ADT counts (depending on the experiment)

Typical decoders:

* `"nb"` or `"zinb"` for raw / count-like ADT
* `"gaussian"` for normalized / scaled ADT

### ATAC

* `.layers["counts"]` → raw peak counts
* `.obsm["X_lsi"]` → LSI / TF–IDF components
* `.X` → either:

  * `obsm["X_lsi"]` (continuous LSI space), or
  * `layers["counts"]` (possibly subsetted peaks)

Typical decoders:

* `"gaussian"` / `"mse"` if using continuous LSI
* `"nb"` or `"poisson"` if using (subsetted) raw peak counts

See the notebooks under `notebooks/` for end-to-end preprocessing examples for CITE-seq, Multiome, and TEA-seq.

---

## Training modes & example recipes (v1 vs v2/lite)

UniVI supports two main training regimes:

* **UniVI v1**: per-modality posteriors + reconstruction terms controlled by `v1_recon` (cross/self/avg/etc.) + posterior alignment across modality posteriors. UniVI v1 is the recommended default for paired multimodal data of most configurations and is the main method used in the manuscript.
* **UniVI-lite / v2**: fused latent posterior (precision-weighted MoE/PoE style) + per-modality reconstruction + β·KL(`q_fused||p`) + γ·pairwise alignment between modality posteriors. Scales cleanly to 3+ modalities. This is the recommended default for more loosely-paired or artificially paired data.

---

## Running a minimal training script (UniVI v1 vs UniVI-lite)

### 0) Choose the training objective (`loss_mode`) in your config JSON

In `parameter_files/*.json`, set a single switch that controls the objective.

**Paper objective (v1; `"avg"` trains with 50% weight on self-reconstruction and 50% weight on cross-reconstruction, with weights automatically adjusted so this stays true for any number of modalities):**

```json5
{
  "model": {
    "loss_mode": "v1",
    "v1_recon": "avg",
    "normalize_v1_terms": true
  }
}
```

**UniVI-lite objective (v2; lightweight / fusion-based):**

```json5
{
  "model": {
    "loss_mode": "lite"
  }
}
```

> **Note**
> `loss_mode: "lite"` is an alias for `loss_mode: "v2"` (they run the same objective in the current code).

### 0b) (Optional) Enable supervised labels from config JSON

**Classification head (decoder-only):**

```json5
{
  "model": {
    "loss_mode": "lite",
    "n_label_classes": 30,
    "label_loss_weight": 1.0,
    "label_ignore_index": -1,
    "classify_from_mu": true
  }
}
```

**Lite + label expert injected into fusion (encoder-side):**

```json5
{
  "model": {
    "loss_mode": "lite",
    "n_label_classes": 30,
    "label_loss_weight": 1.0,

    "use_label_encoder": true,
    "label_moe_weight": 1.0,
    "unlabeled_logvar": 20.0,
    "label_encoder_warmup": 5,
    "label_ignore_index": -1
  }
}
```

**Labels as categorical modalities (single or multiple heads):** add one or more categorical modalities in `"data.modalities"` and provide matching AnnData on disk (or build them in Python).

```json5
{
  "model": { "loss_mode": "lite" },
  "data": {
    "modalities": [
      { "name": "rna",      "likelihood": "nb",          "X_key": "X", "layer": "counts" },
      { "name": "adt",      "likelihood": "nb",          "X_key": "X", "layer": "counts" },
      { "name": "celltype", "likelihood": "categorical", "X_key": "X", "layer": null },
      { "name": "patient",  "likelihood": "categorical", "X_key": "X", "layer": null }
    ]
  }
}
```

### 1) Normalization / representation switch (counts vs continuous)

**Important note on selectors:**

* `layer` selects `.layers[layer]` (if `X_key == "X"`).
* `X_key == "X"` selects `.X`/`.layers[layer]`; otherwise `X_key` selects `.obsm[X_key]`.

Correct pattern:

```json5
{
  "data": {
    "modalities": [
      {
        "name": "rna",
        "layer": "log1p",        // uses adata.layers["log1p"] (since X_key=="X")
        "X_key": "X",
        "assume_log1p": true,
        "likelihood": "gaussian"
      },
      {
        "name": "adt",
        "layer": "counts",       // uses adata.layers["counts"] (since X_key=="X")
        "X_key": "X",
        "assume_log1p": false,
        "likelihood": "zinb"
      },
      {
        "name": "atac",
        "layer": null,           // ignored because X_key != "X"
        "X_key": "X_lsi",        // uses adata.obsm["X_lsi"]
        "assume_log1p": false,
        "likelihood": "gaussian"
      }
    ]
  }
}
```

* Use `.layers["counts"]` when you want NB/ZINB/Poisson decoders.
* Use continuous `.X` or `.obsm["X_lsi"]` when you want Gaussian/MSE decoders.

> Jupyter notebooks in this repository (UniVI/notebooks/) show recommended preprocessing per dataset for different data types and analyses. Depending on your research goals, you can use several different methods of preprocessing. The model is robust when it comes to learning underlying biology regardless of preprocessing; the key is that the decoder likelihood should roughly match the input distribution per-modality.

### 2) Train (CLI)

Example: **CITE-seq (RNA + ADT)**

**UniVI v1**

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_cite_seq_scaled_gaussian_v1.json \
  --outdir saved_models/citeseq_v1_run1 \
  --data-root /path/to/your/data
```

**UniVI-lite**

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_cite_seq_scaled_gaussian_lite.json \
  --outdir saved_models/citeseq_lite_run1 \
  --data-root /path/to/your/data
```

Example: **Multiome (RNA + ATAC)**

**UniVI v1**

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_multiome_v1.json \
  --outdir saved_models/multiome_v1_run1 \
  --data-root /path/to/your/data
```

**UniVI-lite**

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_multiome_lite.json \
  --outdir saved_models/multiome_lite_run1 \
  --data-root /path/to/your/data
```

Example: **TEA-seq (RNA + ADT + ATAC)**

**UniVI v1**

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_tea_seq_v1.json \
  --outdir saved_models/teaseq_v1_run1 \
  --data-root /path/to/your/data
```

**UniVI-lite**

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_tea_seq_lite.json \
  --outdir saved_models/teaseq_lite_run1 \
  --data-root /path/to/your/data
```

---

## Quickstart: run UniVI from Python / Jupyter + supervised classification options (if desired)

If you prefer to stay inside a notebook or a Python script instead of calling the CLI, you can build the configs, model, and trainer directly.

Below is a minimal **CITE-seq (RNA + ADT)** example using paired AnnData objects.

```python
import numpy as np
import scanpy as sc
import torch

from torch.utils.data import DataLoader, Subset

from univi import (
    UniVIMultiModalVAE,
    ModalityConfig,
    UniVIConfig,
    TrainingConfig,
)
from univi.data import (
    MultiModalDataset,
    align_paired_obs_names,
    collate_multimodal_xy,   # optional convenience collate for supervised batches
)
from univi.trainer import UniVITrainer
from univi.utils.io import save_anndata_splits
```

### 1) Load preprocessed AnnData (paired cells)

```python
rna = sc.read_h5ad("path/to/rna_citeseq.h5ad")
adt = sc.read_h5ad("path/to/adt_citeseq.h5ad")

adata_dict = {"rna": rna, "adt": adt}
adata_dict = align_paired_obs_names(adata_dict)  # ensures same obs_names/order
```

### 2) Build `MultiModalDataset` and DataLoaders (unsupervised)

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MultiModalDataset(
    adata_dict=adata_dict,
    X_key="X",
    device=None,  # "cpu" or "cuda"
)

n_cells = rna.n_obs
idx = np.arange(n_cells)
rng = np.random.default_rng(0)
rng.shuffle(idx)

split = int(0.8 * n_cells)
train_idx, val_idx = idx[:split], idx[split:]

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

batch_size = 256

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
```

### 2b) (Optional) Supervised batches for Pattern A/B (`(x_dict, y)`)

If you use the classification head and/or label expert injection, supply `y` as integer class indices and mask unlabeled with `-1`.

```python
y_codes = rna.obs["celltype"].astype("category").cat.codes.to_numpy()

dataset_sup = MultiModalDataset(adata_dict=adata_dict, X_key="X", labels=y_codes)

def collate_xy(batch):
    xs, ys = zip(*batch)
    x = {k: torch.stack([d[k] for d in xs], 0) for k in xs[0].keys()}
    y = torch.as_tensor(ys, dtype=torch.long)
    return x, y

train_loader = DataLoader(dataset_sup, batch_size=batch_size, shuffle=True, collate_fn=collate_xy)

# If you prefer using the package helper:
# train_loader = DataLoader(dataset_sup, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_xy)
```

### 2c) Persisting and reusing train/val/test splits (split_map.json)

For reproducibility (and to ensure the exact same cell membership across modalities and reruns), UniVI can write:

* `{prefix}_train.h5ad`, `{prefix}_val.h5ad`, `{prefix}_test.h5ad`
* `{prefix}_split_map.json` containing split membership (stored using `obs_names`)

**Save splits once (from any one modality, typically RNA):**

```python
test_idx = np.array([], dtype=int)  # optional; include if you have a test set

save_anndata_splits(
    rna,
    outdir="splits/citeseq_demo",
    prefix="citeseq_rna",
    split_map={
        "train": train_idx.tolist(),
        "val":   val_idx.tolist(),
        "test":  test_idx.tolist(),
    },
    save_split_map=True,
)
```

**Later: load the split_map and apply to any paired modality using `obs_names`:**

```python
import json

with open("splits/citeseq_demo/citeseq_rna_split_map.json") as f:
    sm = json.load(f)

train_names = sm["train"]
val_names   = sm["val"]
test_names  = sm.get("test", [])

rna_train = rna[train_names].copy()
adt_train = adt[train_names].copy()

rna_val = rna[val_names].copy()
adt_val = adt[val_names].copy()
```

**If you need integer indices for `torch.utils.data.Subset`:**

```python
obs = dataset.obs_names  # canonical ordering used by the dataset

train_idx2 = obs.get_indexer(train_names)
val_idx2   = obs.get_indexer(val_names)

if (train_idx2 < 0).any() or (val_idx2 < 0).any():
    raise ValueError("Some split_map obs_names are missing from the dataset.")

train_ds = Subset(dataset, train_idx2)
val_ds   = Subset(dataset, val_idx2)
```

This pattern lets you:

* reproduce the same split exactly across runs,
* apply the same split membership to other modalities (ADT/ATAC) or derived datasets,
* and recover `Subset` indices deterministically.

### 3) Define UniVI configs (v1 vs UniVI-lite)

```python
univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=5.0,
    gamma=40.0,
    encoder_dropout=0.1,
    decoder_dropout=0.0,
    encoder_batchnorm=True,
    decoder_batchnorm=False,
    kl_anneal_start=0,
    kl_anneal_end=25,
    align_anneal_start=0,
    align_anneal_end=25,
    modalities=[
        ModalityConfig("rna", rna.n_vars, [1024, 512], [512, 1024], likelihood="nb"),
        ModalityConfig("adt", adt.n_vars, [256, 128],  [128, 256],  likelihood="nb"),
    ],
)

train_cfg = TrainingConfig(
    n_epochs=200,
    batch_size=batch_size,
    lr=1e-3,
    weight_decay=1e-4,
    device=device,
    log_every=10,
    grad_clip=5.0,
    num_workers=0,
    seed=42,
    early_stopping=True,
    patience=25,
    min_delta=0.0,
)
```

 + This step is altered if you want to use supervised options - see below for those configurations and considerations for your use-case.

### Should I use prior label-informed supervision/which supervised option should I use?

* Prior label-informed supervised modelling is not necessarily suitable for all research tasks - especially since the latent space can learn a robust biologically-relevant latent space in a label-agnostic training objective.
* If your research goals include either shaping the latent space given labels to inform specific goals or using mapping latent samples back to cell types, the supervised classification methods may be useful.

If desired, you can use labels to “shape” the latent in one of three ways:

1. **Classification head (decoder-only)** — `p(y|z)` (**recommended default**)
   *Works for either `loss_mode="v1"` and `loss_mode="lite"`.*
   Best if you want the latent to be predictive/separable without changing how modalities reconstruct.

2. **Label expert injected into fusion (encoder-side)** — `q(z|y)` (**lite/v2 only**)
   *Works only for `loss_mode="lite"` / `v2`.*
   Best for semi-supervised settings where labels should directly influence the **fused posterior**.

3. **Labels as a full categorical “modality”** — `"celltype"` modality with likelihood `"categorical"`
   *Works for either `loss_mode="v1"` and `loss_mode="lite"`.*
   Useful when you want cell types to behave like a first-class modality (encode/decode/reconstruct), but avoid `v1` cross-reconstruction unless you really know you want it.

---

## Supervised labels (three supported patterns)

### A) Latent classification head (decoder-only): `p(y|z)` (works in **lite/v2** and **v1**)

This is the simplest way to shape the latent. UniVI attaches a categorical head to the latent `z` and adds:

```math
\mathcal{L} \;+=\; \lambda \cdot \mathrm{CE}(\mathrm{logits}(z), y)
```

**How to enable:** initialize the model with:

* `n_label_classes > 0`
* `label_loss_weight` (default `1.0`)
* `label_ignore_index` (default `-1`, used to mask unlabeled rows)

```python
import numpy as np
import torch

from univi import UniVIMultiModalVAE, UniVIConfig, ModalityConfig

# Example labels (0..C-1) from AnnData
y_codes = rna.obs["celltype"].astype("category").cat.codes.to_numpy()
n_classes = int(y_codes.max() + 1)

univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=5.0,
    gamma=40.0,
    modalities=[
        ModalityConfig("rna", rna.n_vars, [1024, 512], [512, 1024], likelihood="nb"),
        ModalityConfig("adt", adt.n_vars, [256, 128],  [128, 256],  likelihood="nb"),
    ],
)

model = UniVIMultiModalVAE(
    univi_cfg,
    loss_mode="lite",            # OR "v1"
    n_label_classes=n_classes,
    label_loss_weight=1.0,
    label_ignore_index=-1,
    classify_from_mu=True,
).to("cuda")
```

During training your batch should provide `y`, and your loop should call:

```python
out = model(x_dict, y=y, epoch=epoch)
loss = out["loss"]
```

Unlabeled cells are supported: set `y=-1` and CE is automatically masked.

---

### B) Label expert injected into fusion: `q(z|y)` (**lite/v2 only**)

In **lite/v2**, UniVI can optionally add a **label encoder** as an additional expert into MoE fusion. Labeled cells get an extra “expert vote” in the fused posterior; unlabeled cells ignore it automatically.

```python
model = UniVIMultiModalVAE(
    univi_cfg,
    loss_mode="lite",

    # Optional: keep the decoder-side classification head too
    n_label_classes=n_classes,
    label_loss_weight=1.0,

    # Encoder-side label expert injected into fusion
    use_label_encoder=True,
    label_moe_weight=1.0,      # >1 => labels influence fusion more
    unlabeled_logvar=20.0,     # very high => tiny precision => ignored in fusion
    label_encoder_warmup=5,    # wait N epochs before injecting labels into fusion
    label_ignore_index=-1,
).to("cuda")
```

**Notes**

* This pathway is **only used in `loss_mode="lite"` / `v2`**, because it is implemented as an extra expert inside fusion.
* Unlabeled cells (`y=-1`) are automatically ignored in fusion via a huge log-variance.

---

### C) Treat labels as a categorical “modality” (best with **lite/v2**)

Instead of providing `y` separately, you can represent labels as another modality (e.g. `"celltype"`) with likelihood `"categorical"`. This makes labels a first-class modality with its own encoder/decoder.

**Recommended representation:** one-hot matrix `(B, C)` stored in `.X`.

```python
import numpy as np
from anndata import AnnData

# y codes (0..C-1)
y_codes = rna.obs["celltype"].astype("category").cat.codes.to_numpy()
C = int(y_codes.max() + 1)

Y = np.eye(C, dtype=np.float32)[y_codes]  # (B, C) one-hot

celltype = AnnData(X=Y)
celltype.obs_names = rna.obs_names.copy()  # MUST match paired modalities
celltype.var_names = [f"class_{i}" for i in range(C)]

adata_dict = {"rna": rna, "adt": adt, "celltype": celltype}

univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=5.0,
    gamma=40.0,
    modalities=[
        ModalityConfig("rna",      rna.n_vars, [1024, 512], [512, 1024], likelihood="nb"),
        ModalityConfig("adt",      adt.n_vars, [256, 128],  [128, 256],  likelihood="nb"),
        ModalityConfig("celltype", C,          [128],       [128],       likelihood="categorical"),
    ],
)

model = UniVIMultiModalVAE(univi_cfg, loss_mode="lite").to("cuda")
```

**Important caveat for `loss_mode="v1"`**
`v1` can perform cross-reconstruction across all modalities. If you include `"celltype"` as a modality, you typically **do not** want cross-recon terms like `celltype → RNA`. If you must run `v1` with label-as-modality, prefer:

```python
model = UniVIMultiModalVAE(univi_cfg, loss_mode="v1", v1_recon="self").to("cuda")
```

If you want full `v1` cross-reconstruction and label shaping, prefer **Pattern A (classification head)** instead.

---

## Multiple classification decoder heads (multi-target supervision)

If you want to learn **multiple categorical predictors at once** (e.g. `celltype`, `patient`, `mutation_status`), the recommended pattern is to add **multiple categorical label modalities**, one per target. Each label modality corresponds to its own categorical decoder head (and optionally its own encoder), and UniVI learns them jointly with the molecular modalities.

This works cleanly for **lite/v2**. For **v1**, prefer `v1_recon="self"` when label modalities are present.

### Example: cell type + patient + TP53 status

```python
import numpy as np
from anndata import AnnData

# --- build integer codes per target ---
celltype_code = rna.obs["celltype"].astype("category").cat.codes.to_numpy()
patient_code  = rna.obs["patient_id"].astype("category").cat.codes.to_numpy()

# TP53_status should be {0, 1} for known labels, and -1 for unknown/unlabeled
# Example: rna.obs["TP53_status"] already holds values in {0,1,-1}
tp53_code = rna.obs["TP53_status"].to_numpy()

C_celltype = int(celltype_code.max() + 1)
C_patient  = int(patient_code.max() + 1)
C_tp53     = 2

# --- one-hot for known labels, and all-zeros for unknown labels ---
def one_hot_with_unknown(codes, n_classes, unknown_val=-1):
    Y = np.zeros((len(codes), n_classes), dtype=np.float32)
    mask = (codes != unknown_val)
    Y[mask] = np.eye(n_classes, dtype=np.float32)[codes[mask]]
    return Y

Y_celltype = one_hot_with_unknown(celltype_code, C_celltype, unknown_val=-1)
Y_patient  = one_hot_with_unknown(patient_code,  C_patient,  unknown_val=-1)
Y_tp53     = one_hot_with_unknown(tp53_code,     C_tp53,     unknown_val=-1)

celltype = AnnData(X=Y_celltype)
patient  = AnnData(X=Y_patient)
tp53     = AnnData(X=Y_tp53)

for a in (celltype, patient, tp53):
    a.obs_names = rna.obs_names.copy()

celltype.var_names = [f"class_{i}" for i in range(C_celltype)]
patient.var_names  = [f"class_{i}" for i in range(C_patient)]
tp53.var_names     = [f"class_{i}" for i in range(C_tp53)]

adata_dict = {
    "rna": rna,
    "adt": adt,                 # optional
    "celltype": celltype,
    "patient": patient,
    "TP53": tp53,
}

univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=5.0,
    gamma=40.0,
    modalities=[
        ModalityConfig("rna",      rna.n_vars, [1024, 512], [512, 1024], likelihood="nb"),
        ModalityConfig("adt",      adt.n_vars, [256, 128],  [128, 256],  likelihood="nb"),
        ModalityConfig("celltype", C_celltype, [128],       [128],       likelihood="categorical"),
        ModalityConfig("patient",  C_patient,  [128],       [128],       likelihood="categorical"),
        ModalityConfig("TP53",     C_tp53,     [64],        [64],        likelihood="categorical"),
    ],
)

# For lite/v2 this is the simplest choice for multi-target labels
model = UniVIMultiModalVAE(univi_cfg, loss_mode="lite").to("cuda")

# For v1, use self-recon if label modalities are present
# model = UniVIMultiModalVAE(univi_cfg, loss_mode="v1", v1_recon="self").to("cuda")
```

**Notes**

* This is the most direct way to get multiple “classification heads” without packing everything into a single label space.
* Unknown/unlabeled entries can be represented as all-zeros one-hot rows (as above). This avoids forcing a dummy class into the label space.

### 4) Choose the objective + supervised option

```python
# Option A: UniVI v1 (unsupervised)
model = UniVIMultiModalVAE(
    univi_cfg,
    loss_mode="v1",
    v1_recon="avg",
    v1_recon_mix=0.0,
    normalize_v1_terms=True,
).to(device)

# Option B: UniVI-lite / v2 (unsupervised)
# model = UniVIMultiModalVAE(univi_cfg, loss_mode="lite").to(device)

# Option C: Add classification head (Pattern A; works in lite/v2 AND v1)
# n_classes = int(y_codes.max() + 1)
# model = UniVIMultiModalVAE(
#     univi_cfg,
#     loss_mode="lite",
#     n_label_classes=n_classes,
#     label_loss_weight=1.0,
#     label_ignore_index=-1,
#     classify_from_mu=True,
# ).to(device)

# Option D: Add label expert injection into fusion (Pattern B; lite/v2 ONLY)
# model = UniVIMultiModalVAE(
#     univi_cfg,
#     loss_mode="lite",
#     n_label_classes=n_classes,
#     label_loss_weight=1.0,
#     use_label_encoder=True,
#     label_moe_weight=1.0,
#     unlabeled_logvar=20.0,
#     label_encoder_warmup=5,
#     label_ignore_index=-1,
# ).to(device)
```

### 5) Train inside Python / Jupyter

```python
trainer = UniVITrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    train_cfg=train_cfg,
    device=device,
)

history = trainer.fit()
```

### Checkpoint format (trainer.save / restore)

`UniVITrainer.save(...)` writes a single `.pt` checkpoint containing:

* model weights
* optimizer state
* AMP scaler state (if enabled)
* trainer state (history, best epoch, etc.)
* optional config / extra metadata

When restoring, the loader checks key compatibility (e.g., label dimensions / head counts) to avoid silent mismatches.

### 6) Write latent `z` into AnnData `.obsm["X_univi"]`

```python
from univi import write_univi_latent

Z = write_univi_latent(model, adata_dict, obsm_key="X_univi", device=device, use_mean=True)
print("Embedding shape:", Z.shape)
```

> **Tip**
> Use `use_mean=True` for deterministic plotting/UMAP. Sampling (`use_mean=False`) is stochastic and useful for generative behavior.

---

## Evaluating / encoding: choosing the latent representation (and what you can do with it)

UniVI exposes a few “one-liner” helpers that make it easy to (i) pick *which* latent you want (per-modality vs fused), (ii) write embeddings back into AnnData, and (iii) run common evaluation metrics without re-plumbing your training code.

### 1) Picking the latent: modality-specific vs fused (MoE/PoE)

Most workflows boil down to one choice:

* **Per-modality latent** (`q(z | x_m)`): best when you want to inspect each modality’s view of the same cells, or compute cross-modality pairing/alignment metrics.
* **Fused latent** (`q(z | x_1, x_2, …)`): best as your “final” integrated embedding for downstream UMAP, neighbors, clustering, label transfer, etc.

`encode_adata` supports both, plus mean vs sampling:

* `"modality_mean"` / `"modality_sample"`: per-modality posterior mean or samples
* `"moe_mean"` / `"moe_sample"`: fused posterior mean or samples (MoE/PoE fusion in lite/v2; consistent fused behavior in current API)

```python
from univi.evaluation import encode_adata

# Per-modality latent (RNA-only view)
Z_rna = encode_adata(
    model, rna,
    modality="rna",
    device=device,
    layer="counts",
    latent="modality_mean",
)

# Fused latent (integrated embedding)
Z_fused = encode_adata(
    model, rna,               # any paired modality AnnData works here (obs_names define rows)
    modality="rna",
    device=device,
    layer="counts",
    latent="moe_mean",
)
```

Tip: prefer `*_mean` for plotting/UMAP (deterministic). Use `*_sample` when you explicitly want stochasticity (e.g., generative behavior, uncertainty-aware downstream work).

---

### 2) Writing embeddings back into AnnData (for Scanpy workflows)

If you want clean Scanpy interoperability, write the embedding into `.obsm[...]` and proceed with neighbors/UMAP/leiden as usual.

```python
from univi import write_univi_latent

# Writes fused mean embedding to each AnnData in the dict by default
Z = write_univi_latent(
    model,
    adata_dict={"rna": rna, "adt": adt},
    obsm_key="X_univi",
    device=device,
    use_mean=True,
)
print(Z.shape)  # (n_cells, latent_dim)
```

Then:

```python
import scanpy as sc

sc.pp.neighbors(rna, use_rep="X_univi", n_neighbors=15)
sc.tl.umap(rna)
sc.tl.leiden(rna, resolution=0.6)
```

---

### 3) Cross-modality alignment metrics (quick sanity checks)

Once you have latents, the standard “is this actually aligned?” checks are:

* **FOSCTTM**: how often non-matching cells are closer than the true match (lower is better).
* **Modality mixing**: kNN neighborhood mixing across modalities (higher usually better).
* **Label transfer**: kNN transfer accuracy across modalities (higher is better).

A common pattern is: compute per-modality latents, then evaluate.

```python
import numpy as np
from univi.evaluation import encode_adata, compute_foscttm, modality_mixing_score, label_transfer_knn

Z_rna = encode_adata(model, rna, modality="rna", device=device, layer="counts", latent="modality_mean")
Z_adt = encode_adata(model, adt, modality="adt", device=device, layer="counts", latent="modality_mean")

foscttm = compute_foscttm(Z_rna, Z_adt, metric="euclidean")
print("FOSCTTM (RNA vs ADT):", float(foscttm))

mix_fused = modality_mixing_score(
    Z=np.vstack([Z_rna, Z_adt]),
    batch=np.array(["rna"] * Z_rna.shape[0] + ["adt"] * Z_adt.shape[0]),
    k=20,
)
print("Mixing (k=20):", float(mix_fused))

labels_rna = rna.obs["cell_type"].astype(str).to_numpy()
labels_adt = adt.obs["cell_type"].astype(str).to_numpy()

pred, acc, cm = label_transfer_knn(
    Z_source=Z_rna, labels_source=labels_rna,
    Z_target=Z_adt, labels_target=labels_adt,
    k=15,
)
print("Label transfer acc (RNA→ADT):", float(acc))
```

(For multi-modal setups you can compute these pairwise across any modality pair.)

---

### 4) “Cool stuff”: multiple classification decoder heads (multi-task labels)

If you enabled the newer multi-head classification support (e.g., `cell_type`, `donor`, `batch`, `mutation_status`, etc.), you can train and later query multiple label predictions from the same latent.

Conceptually: UniVI learns `z`, then each head models `p(y_h | z)` with its own loss weight and masking.

Example (illustrative) inference-time usage pattern:

```python
# Forward pass can return logits for each head when configured
out = model(x_dict, y=y_dict, epoch=epoch)

# Typical keys you may expose in your implementation:
# out["label_logits"] = {"cell_type": ..., "donor": ..., ...}
logits = out.get("label_logits", {})
for head, L in logits.items():
    yhat = L.argmax(-1).detach().cpu().numpy()
    print(head, yhat[:10])
```

Practical uses:

* Train a single latent that’s simultaneously predictive of **cell type** and robust to **batch/donor**
* Add a head for a biological attribute (e.g., **mutation**, **tumor/normal**) while keeping integration objective intact
* Compare separability in `z` with vs without each head (ablation-friendly)

---

### 5) Cross-modal prediction / reconstruction (imputation-style use)

For paired multimodal data, UniVI can be used as a cross-modal predictor: encode with one modality, decode into another.

At a high level:

* Encode: `x_rna → z`
* Decode: `z → \hat{x}_adt` (or `\hat{x}_atac`)

Depending on your codepath, this may be exposed via model forward options or helper utilities. The nice workflow is:

* generate predictions
* aggregate by cell type
* score feature-wise correlation / NLL

This is how you get “RNA→ADT” or “ADT→RNA” performance summaries for marker recovery plots.

---

### 6) Evaluating a trained model via CLI scripts

After training, you can run evaluation to compute alignment metrics and generate UMAPs:

```bash
python scripts/evaluate_univi.py \
  --config parameter_files/defaults_cite_seq.json \
  --model-checkpoint saved_models/citeseq_run1/best_model.pt \
  --outdir figures/citeseq_run1
```

Typical evaluation outputs include:

* FOSCTTM (alignment quality)
* Modality mixing scores
* kNN label transfer accuracy
* UMAPs colored by cell type and modality
* Cross-modal reconstruction summaries

---

### 7) Recommended workflow cheat-sheet

* Use **`moe_mean`** for your “final embedding” (`.obsm["X_univi"]`) and downstream clustering/UMAP.
* Use **`modality_mean`** when computing *pairwise* alignment metrics or debugging a specific modality’s latent.
* Use **sampling** (`*_sample`) only when you explicitly want stochastic behavior.

---

## Hyperparameter tuning (optional)

UniVI includes a hyperparameter optimization module with helpers for:

* **Unimodal** RNA, ADT, ATAC
* **Bi-modal** CITE-seq (RNA+ADT) and Multiome (RNA+ATAC)
* **Tri-modal** TEA-seq (RNA+ADT+ATAC)

Each `run_*_hparam_search` function:

* Randomly samples hyperparameter configurations from a predefined search space
* Trains a UniVI model for each configuration
* Computes validation loss and (for multi-modal setups) alignment metrics (FOSCTTM, label transfer, modality mixing)
* Returns a `pandas.DataFrame` with one row per config, plus the best configuration and its summary metrics

All helpers are available via:

```python
from univi.hyperparam_optimization import (
    run_multiome_hparam_search,
    run_citeseq_hparam_search,
    run_teaseq_hparam_search,
    run_rna_hparam_search,
    run_atac_hparam_search,
    run_adt_hparam_search,
)
```

### CITE-seq (RNA + ADT) hyperparameter search

```python
from univi.hyperparam_optimization import run_citeseq_hparam_search

df, best_result, best_cfg = run_citeseq_hparam_search(
    rna_train=rna_train,
    adt_train=adt_train,
    rna_val=rna_val,
    adt_val=adt_val,
    celltype_key="cell_type",   # or None if you don't have labels
    device="cuda",
    layer="counts",             # raw counts for NB/ZINB decoders
    X_key="X",
    max_configs=100,
    seed=0,
)

df.to_csv("citeseq_hparam_results.csv", index=False)
print("Best config:", best_cfg)
```

### 10x Multiome (RNA + ATAC) hyperparameter search

```python
from univi.hyperparam_optimization import run_multiome_hparam_search

df, best_result, best_cfg = run_multiome_hparam_search(
    rna_train=rna_train,
    atac_train=atac_train,
    rna_val=rna_val,
    atac_val=atac_val,
    celltype_key="cell_type",
    device="cuda",
    layer="counts",
    X_key="X",
    max_configs=100,
    seed=0,
)

df.to_csv("multiome_hparam_results.csv", index=False)
```

### TEA-seq (RNA + ADT + ATAC) hyperparameter search

```python
from univi.hyperparam_optimization import run_teaseq_hparam_search

df, best_result, best_cfg = run_teaseq_hparam_search(
    rna_train=rna_train,
    adt_train=adt_train,
    atac_train=atac_train,
    rna_val=rna_val,
    adt_val=adt_val,
    atac_val=atac_val,
    celltype_key="cell_type",
    device="cuda",
    layer="counts",
    X_key="X",
    max_configs=100,
    seed=0,
)

df.to_csv("teaseq_hparam_results.csv", index=False)
```

### Unimodal RNA / ADT / ATAC hyperparameter search

```python
from univi.hyperparam_optimization import (
    run_rna_hparam_search,
    run_adt_hparam_search,
    run_atac_hparam_search,
)

df_rna, best_result_rna, best_cfg_rna = run_rna_hparam_search(
    rna_train=rna_train,
    rna_val=rna_val,
    device="cuda",
    layer="counts",
    X_key="X",
    max_configs=50,
    seed=0,
)

df_adt, best_result_adt, best_cfg_adt = run_adt_hparam_search(
    adt_train=adt_train,
    adt_val=adt_val,
    device="cuda",
    layer="counts",
    X_key="X",
    max_configs=50,
    seed=0,
)

df_atac, best_result_atac, best_cfg_atac = run_atac_hparam_search(
    atac_train=atac_train,
    atac_val=atac_val,
    device="cuda",
    layer="counts",
    X_key="X",
    max_configs=50,
    seed=0,
)
```
---

For richer, exploratory workflows (TEA-seq tri-modal integration, Multiome RNA+ATAC, non-paired matching, etc.), see the notebooks in `notebooks/`.

---
