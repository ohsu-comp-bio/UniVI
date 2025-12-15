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
(a) Generic UniVI architecture schematic. (b) Core training objective (UniVI v1; see documentation below for UniVI-lite/v2). (c) Example modality combinations beyond bi-modal data (e.g. TEA-seq (tri-modal RNA + ATAC + ADT)). (d) Evaluation roadmap spanning latent alignment (FOSCTTM), modality mixing, label transfer, reconstruction/prediction NLL, and downstream biological consistency.

---

UniVI is a **multi-modal variational autoencoder (VAE)** framework for aligning and integrating single-cell modalities such as RNA, ADT (CITE-seq), and ATAC. It’s built to support experiments like:

* Joint embedding of RNA + ADT (CITE-seq)
* RNA + ATAC (10x Multiome) integration
* RNA + ADT + ATAC (TEA-seq) tri-modal integration
* Independent non-paired modalities from the same tissue type (see `univi/matching.py`)
* Cross-modal reconstruction and imputation
* Data denoising
* Structured evaluation of alignment quality (FOSCTTM, modality mixing, label transfer, etc.)
* Exploratory analysis of relationships between heterogeneous molecular readouts

This repository contains the UniVI package, training scripts, parameter files, and example notebooks.

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
```

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
    │   ├── transformer.py                 # Transformer blocks + encoder
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

Most entry-point scripts write results into a user-specified output directory (commonly `runs/`), which is not tracked in git.

A typical `runs/` folder produced by `scripts/revision_reproduce_all.sh` looks like:

```text
runs/
└── <run_name>/
    ├── checkpoints/
    │   └── univi_checkpoint.pt
    ├── eval/
    │   ├── metrics.json
    │   └── metrics.csv
    ├── robustness/
    │   ├── frequency_perturbation_results.csv
    │   ├── frequency_perturbation_plot.png
    │   ├── frequency_perturbation_plot.pdf
    │   ├── do_not_integrate_summary.csv
    │   ├── do_not_integrate_plot.png
    │   └── do_not_integrate_plot.pdf
    ├── benchmarks/
    │   ├── results.csv
    │   ├── results.png
    │   └── results.pdf
    └── tables/
        └── Supplemental_Table_S1.xlsx
```

---

## Installation & quickstart

### 1. Install UniVI via PyPI

```bash
pip install univi
```

You can then import it:

```python
import univi
from univi import UniVIMultiModalVAE, ModalityConfig, UniVIConfig
```

> **Note:** UniVI requires `torch`. If `import torch` fails, install PyTorch for your platform/CUDA from:
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### 2. Development install (from source)

```bash
git clone https://github.com/Ashford-A/UniVI.git
cd UniVI

conda env create -f envs/univi_env.yml
conda activate univi_env

pip install -e .
```

### 3. (Optional) Install via conda / mamba

When UniVI is available on a conda channel (e.g. `conda-forge`), you can install with:

```bash
conda install -c conda-forge univi
# or
mamba install -c conda-forge univi
```

UniVI is currently available on a custom conda channel and is installable with:

```bash
conda install ashford-a::univi
# or
mamba install ashford-a::univi
```

---

## Preparing input data

UniVI expects per-modality AnnData objects with matching cells (either truly paired data or consistently paired across modalities; `univi/matching.py` contains helper functions for more complex pairing).

High-level expectations:

* Each modality (RNA/ADT/ATAC/etc.) is an `AnnData` with the same `obs_names` (same cells, same order).
* Raw counts are typically stored in `.layers["counts"]`, and a processed representation in `.X` used for training.
* Decoder likelihoods should roughly match the distribution of inputs per modality.

### RNA

* `.layers["counts"]` → raw counts
* `.X` (or `.layers["log1p"]`) → training representation (log1p, scaled, HVG subset, etc.)

Typical decoders:

* `"nb"` or `"zinb"` for count-like inputs
* `"gaussian"` / `"mse"` for continuous inputs

### ADT (CITE-seq)

* `.layers["counts"]` → raw ADT counts
* `.X` → CLR-normalized / scaled ADT, or other continuous representations

Typical decoders:

* `"nb"` or `"zinb"` for count-like ADT
* `"gaussian"` / `"mse"` for continuous ADT

### ATAC

Common representations:

* `.obsm["X_lsi"]` → LSI / TF–IDF components (continuous)
* `.layers["counts"]` → peak counts (sparse counts)

Typical decoders:

* `"gaussian"` / `"mse"` for continuous LSI
* `"poisson"` or `"nb"` for (subsetted) peak counts

See `notebooks/` for end-to-end preprocessing examples.

---

## Training modes & example recipes (v1 vs v2/lite)

UniVI supports two main training regimes:

* **UniVI v1**: per-modality posteriors + reconstruction terms controlled by `v1_recon` (cross/self/avg/etc.) + posterior alignment across modality posteriors. This is the recommended default for paired multimodal data and is the main method used in the manuscript.
* **UniVI-lite / v2**: fused latent posterior (precision-weighted MoE/PoE style) + per-modality reconstruction + β·KL(`q_fused||p`) + γ·pairwise alignment between modality posteriors. Scales cleanly to 3+ modalities and is convenient for loosely-paired settings.

---

## Running a minimal training script (UniVI v1 vs UniVI-lite)

### 0) Choose the training objective (`loss_mode`) in your config JSON

In `parameter_files/*.json`, set a single switch that controls the objective.

**Paper objective (v1; `"avg"` trains with 50% weight on self-recon + 50% weight on cross-recon, automatically normalized across any number of modalities):**

```json5
{
  "model": {
    "loss_mode": "v1",
    "v1_recon": "avg",
    "normalize_v1_terms": true
  }
}
```

**UniVI-lite objective (v2):**

```json5
{
  "model": {
    "loss_mode": "lite"
  }
}
```

> `loss_mode: "lite"` is an alias for `loss_mode: "v2"`.

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

**Labels as categorical modalities:** add one or more categorical modalities in `"data.modalities"` and provide matching AnnData on disk (or build them in Python).

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

### 1) Representation selector (counts vs continuous)

Selector rules:

* `X_key == "X"` selects `.X`, or `.layers[layer]` if `layer` is provided
* `X_key != "X"` selects `.obsm[X_key]` and ignores `layer`

Example:

```json5
{
  "data": {
    "modalities": [
      {
        "name": "rna",
        "layer": "log1p",
        "X_key": "X",
        "assume_log1p": true,
        "likelihood": "gaussian"
      },
      {
        "name": "adt",
        "layer": "counts",
        "X_key": "X",
        "assume_log1p": false,
        "likelihood": "zinb"
      },
      {
        "name": "atac",
        "layer": null,
        "X_key": "X_lsi",
        "assume_log1p": false,
        "likelihood": "gaussian"
      }
    ]
  }
}
```

Notes:

* Use `.layers["counts"]` for NB/ZINB/Poisson decoders.
* Use continuous `.X` or `.obsm["X_lsi"]` for Gaussian/MSE decoders.

### 2) Train (CLI)

**CITE-seq (RNA + ADT)**

UniVI v1:

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_cite_seq_scaled_gaussian_v1.json \
  --outdir saved_models/citeseq_v1_run1 \
  --data-root /path/to/your/data
```

UniVI-lite:

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_cite_seq_scaled_gaussian_lite.json \
  --outdir saved_models/citeseq_lite_run1 \
  --data-root /path/to/your/data
```

**Multiome (RNA + ATAC)**

UniVI v1:

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_multiome_v1.json \
  --outdir saved_models/multiome_v1_run1 \
  --data-root /path/to/your/data
```

UniVI-lite:

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_multiome_lite.json \
  --outdir saved_models/multiome_lite_run1 \
  --data-root /path/to/your/data
```

**TEA-seq (RNA + ADT + ATAC)**

UniVI v1:

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_tea_seq_v1.json \
  --outdir saved_models/teaseq_v1_run1 \
  --data-root /path/to/your/data
```

UniVI-lite:

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_tea_seq_lite.json \
  --outdir saved_models/teaseq_lite_run1 \
  --data-root /path/to/your/data
```

---

## Quickstart: run UniVI from Python / Jupyter + supervised classification options (if desired)

If you prefer to stay in a notebook / Python script instead of calling the CLI, you can build configs, model, and trainer directly.

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
    collate_multimodal_xy,   # optional helper for supervised batches
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
    device=None,
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

### 2b) (Optional) Supervised batches (`(x_dict, y)`)

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

# Or:
# train_loader = DataLoader(dataset_sup, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_xy)
```

### 2c) Persisting and reusing train/val/test splits (split_map.json)

For reproducibility, UniVI can write:

* `{prefix}_train.h5ad`, `{prefix}_val.h5ad`, `{prefix}_test.h5ad`
* `{prefix}_split_map.json` containing split membership (stored using `obs_names`)

**Save splits once (from any one modality):**

```python
test_idx = np.array([], dtype=int)  # optional

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

**Later: load split_map and apply to any paired modality using `obs_names`:**

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
obs = dataset.obs_names

train_idx2 = obs.get_indexer(train_names)
val_idx2   = obs.get_indexer(val_names)

if (train_idx2 < 0).any() or (val_idx2 < 0).any():
    raise ValueError("Some split_map obs_names are missing from the dataset.")

train_ds = Subset(dataset, train_idx2)
val_ds   = Subset(dataset, val_idx2)
```

### 3) Define UniVI configs (v1 vs UniVI-lite)

```python
univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.5,
    gamma=2.5,
    encoder_dropout=0.1,
    decoder_dropout=0.0,
    encoder_batchnorm=True,
    decoder_batchnorm=False,
    kl_anneal_start=0,
    kl_anneal_end=50,
    align_anneal_start=25,
    align_anneal_end=75,
    modalities=[
        ModalityConfig("rna", rna.n_vars, [512, 256, 128], [128, 256, 512], likelihood="nb"),
        ModalityConfig("adt", adt.n_vars, [128, 64],  [64, 128],  likelihood="nb"),
    ],
)

train_cfg = TrainingConfig(
    n_epochs=1000,
    batch_size=batch_size,
    lr=1e-3,
    weight_decay=1e-4,
    device=device,
    log_every=20,
    grad_clip=5.0,
    num_workers=0,
    seed=42,
    early_stopping=True,
    patience=50,
    min_delta=0.0,
)
```

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

# Option C: Add classification head (works in lite/v2 AND v1)
# y_codes = rna.obs["celltype"].astype("category").cat.codes.to_numpy()
# n_classes = int(y_codes.max() + 1)
# model = UniVIMultiModalVAE(
#     univi_cfg,
#     loss_mode="lite",  # OR "v1"
#     n_label_classes=n_classes,
#     label_loss_weight=1.0,
#     label_ignore_index=-1,
#     classify_from_mu=True,
# ).to(device)

# Option D: Label expert injection into fusion (lite/v2 ONLY)
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
* train/val/test split indices used
* features used
* optional config / extra metadata

When restoring, the loader checks key compatibility (e.g., label dimensions / head counts) to avoid silent mismatches.

### 6) Write latent `z` into AnnData `.obsm["X_univi"]`

```python
from univi import write_univi_latent

Z = write_univi_latent(model, adata_dict, obsm_key="X_univi", device=device, use_mean=True)
print("Embedding shape:", Z.shape)
```

> **Tip:** `use_mean=True` is deterministic for plotting/UMAP.

---

## Supervised labels (three supported patterns)

### A) Latent classification head (decoder-only): `p(y|z)` (works in **lite/v2** and **v1**)

```math
\mathcal{L} \;+=\; \lambda \cdot \mathrm{CE}(\mathrm{logits}(z), y)
```

Enable via model init:

```python
y_codes = rna.obs["celltype"].astype("category").cat.codes.to_numpy()
n_classes = int(y_codes.max() + 1)

model = UniVIMultiModalVAE(
    univi_cfg,
    loss_mode="lite",            # OR "v1"
    n_label_classes=n_classes,
    label_loss_weight=1.0,
    label_ignore_index=-1,
    classify_from_mu=True,
).to(device)
```

Training:

```python
out = model(x_dict, y=y, epoch=epoch)
loss = out["loss"]
```

Unlabeled cells supported: set `y=-1` and CE is masked.

---

### B) Label expert injected into fusion: `q(z|y)` (**lite/v2 only**)

```python
model = UniVIMultiModalVAE(
    univi_cfg,
    loss_mode="lite",

    n_label_classes=n_classes,
    label_loss_weight=1.0,

    use_label_encoder=True,
    label_moe_weight=1.0,
    unlabeled_logvar=20.0,
    label_encoder_warmup=5,
    label_ignore_index=-1,
).to(device)
```

Notes:

* Only used in lite/v2 (implemented as an extra expert in fusion).
* Unlabeled (`y=-1`) ignored automatically via huge log-variance.

---

### C) Treat labels as a categorical “modality” (best with **lite/v2**)

```python
import numpy as np
from anndata import AnnData

y_codes = rna.obs["celltype"].astype("category").cat.codes.to_numpy()
C = int(y_codes.max() + 1)

Y = np.eye(C, dtype=np.float32)[y_codes]  # (B,C)

celltype = AnnData(X=Y)
celltype.obs_names = rna.obs_names.copy()
celltype.var_names = [f"class_{i}" for i in range(C)]

adata_dict = {"rna": rna, "adt": adt, "celltype": celltype}

univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.5,
    gamma=2.5,
    modalities=[
        ModalityConfig("rna",      rna.n_vars, [512, 256, 128], [128, 256, 512], likelihood="nb"),
        ModalityConfig("adt",      adt.n_vars, [128, 64],  [64, 128],  likelihood="nb"),
        ModalityConfig("celltype", C,          [128],       [128],       likelihood="categorical"),
    ],
)

model = UniVIMultiModalVAE(univi_cfg, loss_mode="lite").to(device)
```

**Important caveat for `loss_mode="v1"`**
If you include `"celltype"` as a modality, you typically do not want cross-reconstruction terms like `celltype → RNA`. Prefer:

```python
model = UniVIMultiModalVAE(univi_cfg, loss_mode="v1", v1_recon="self").to(device)
```

---

## Multiple classification decoder heads (multi-target supervision)

If you want multiple categorical predictors at once (e.g. celltype, patient, mutation), the simplest pattern is multiple categorical modalities.

This works cleanly for **lite/v2**. For **v1**, prefer `v1_recon="self"` when label modalities are present.

### Example: cell type + patient + TP53 status

```python
import numpy as np
from anndata import AnnData

celltype_code = rna.obs["celltype"].astype("category").cat.codes.to_numpy()
patient_code  = rna.obs["patient_id"].astype("category").cat.codes.to_numpy()
tp53_code     = rna.obs["TP53_status"].to_numpy()  # {0,1,-1} recommended

C_celltype = int(celltype_code.max() + 1)
C_patient  = int(patient_code.max() + 1)
C_tp53     = 2

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
    "adt": adt,
    "celltype": celltype,
    "patient": patient,
    "TP53": tp53,
}

univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.5,
    gamma=2.5,
    modalities=[
        ModalityConfig("rna",      rna.n_vars, [512, 256, 128], [128, 256, 512], likelihood="nb"),
        ModalityConfig("adt",      adt.n_vars, [128, 64],  [64, 128],  likelihood="nb"),
        ModalityConfig("celltype", C_celltype, [128],       [128],       likelihood="categorical"),
        ModalityConfig("patient",  C_patient,  [128],       [128],       likelihood="categorical"),
        ModalityConfig("TP53",     C_tp53,     [64],        [64],        likelihood="categorical"),
    ],
)

model = UniVIMultiModalVAE(univi_cfg, loss_mode="lite").to(device)

# For v1 with label modalities:
# model = UniVIMultiModalVAE(univi_cfg, loss_mode="v1", v1_recon="self").to(device)
```

---

## Evaluating / encoding: choosing the latent representation (and what you can do with it)

UniVI exposes helpers to pick which latent to return and to write embeddings back into AnnData.

### 1) Picking the latent: modality-specific vs fused (MoE/PoE)

* `"modality_mean"` / `"modality_sample"`: per-modality latent
* `"moe_mean"` / `"moe_sample"`: fused latent

```python
from univi.evaluation import encode_adata

Z_rna = encode_adata(model, rna, modality="rna", device=device, layer="counts", latent="modality_mean")
Z_fused = encode_adata(model, rna, modality="rna", device=device, layer="counts", latent="moe_mean")
```

### 2) Writing embeddings back into AnnData (Scanpy-friendly)

```python
from univi import write_univi_latent

Z = write_univi_latent(
    model,
    adata_dict={"rna": rna, "adt": adt},
    obsm_key="X_univi",
    device=device,
    use_mean=True,
)
```

Then:

```python
import scanpy as sc
sc.pp.neighbors(rna, use_rep="X_univi", n_neighbors=15)
sc.tl.umap(rna)
sc.tl.leiden(rna, resolution=0.6)
```

### 3) Cross-modality alignment metrics

```python
import numpy as np
from univi.evaluation import encode_adata, compute_foscttm, modality_mixing_score, label_transfer_knn

Z_rna = encode_adata(model, rna, modality="rna", device=device, layer="counts", latent="modality_mean")
Z_adt = encode_adata(model, adt, modality="adt", device=device, layer="counts", latent="modality_mean")

foscttm = compute_foscttm(Z_rna, Z_adt, metric="euclidean")
print("FOSCTTM (RNA vs ADT):", float(foscttm))

mix = modality_mixing_score(
    Z=np.vstack([Z_rna, Z_adt]),
    batch=np.array(["rna"] * Z_rna.shape[0] + ["adt"] * Z_adt.shape[0]),
    k=20,
)
print("Mixing (k=20):", float(mix))

labels_rna = rna.obs["cell_type"].astype(str).to_numpy()
labels_adt = adt.obs["cell_type"].astype(str).to_numpy()

pred, acc, cm = label_transfer_knn(
    Z_source=Z_rna, labels_source=labels_rna,
    Z_target=Z_adt, labels_target=labels_adt,
    k=15,
)
print("Label transfer acc (RNA→ADT):", float(acc))
```

### 4) “Cool stuff”: multi-head supervised decoders

If you enabled multi-head classification support, you can query predictions at inference time:

```python
probs = model.predict_heads(
    x_dict={"rna": torch.as_tensor(rna.X.A if hasattr(rna.X, "A") else rna.X, dtype=torch.float32, device=model.device)},
    return_probs=True,
)
# probs is a dict: {head_name: (B, C)}
```

(Exact inputs depend on how you build your `x_dict` / dataset; this is usually easiest via `MultiModalDataset` + a loader.)

### 5) Cross-modal prediction / reconstruction

For paired multimodal data, UniVI can be used as a cross-modal predictor: encode with one modality, decode into another. This is the basis of RNA→ADT / ADT→RNA / RNA→ATAC feature recovery plots.

---

## Evaluating a trained model via CLI scripts

```bash
python scripts/evaluate_univi.py \
  --config parameter_files/defaults_cite_seq.json \
  --model-checkpoint saved_models/citeseq_run1/best_model.pt \
  --outdir figures/citeseq_run1
```

---

## Recommended workflow cheat-sheet

* Use **`moe_mean`** for your “final embedding” (`.obsm["X_univi"]`) and downstream clustering/UMAP.
* Use **`modality_mean`** for pairwise alignment metrics and debugging.
* Use sampling (`*_sample`) only when you explicitly want stochastic behavior.

---

## Hyperparameter tuning (optional)

UniVI includes a hyperparameter optimization module with helpers for unimodal and multi-modal regimes.

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
    celltype_key="cell_type",
    device="cuda",
    layer="counts",
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

## Optional: Using Transformer encoders (advanced)

By default, UniVI uses **MLP encoders** (`encoder_type="mlp"`). That means **all existing workflows continue to work unchanged**.

If you want to experiment with the new Transformer-based modality encoders, you can switch a modality to:

- `encoder_type="transformer"`
- provide a `TokenizerConfig` (how `(B,F)` becomes `(B,T,D_in)`)
- provide a `TransformerConfig` (attention depth/width, pooling, dropout, etc.)

### Why use a transformer encoder?

Transformer encoders can be useful when:
- you want a more expressive encoder than an MLP,
- you suspect feature interactions matter more than simple feed-forward mixing,
- you can afford the compute cost (attention scales with token count).

For **very large feature dimensions** (e.g., RNA genes), prefer tokenizers that reduce tokens (smaller `n_tokens`) rather than treating each feature as a token.

### Example: Transformer encoder for RNA, MLP for ADT (CITE-seq)

```python
from univi import UniVIConfig, ModalityConfig
from univi.config import TransformerConfig, TokenizerConfig

univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.5,
    gamma=2.5,
    modalities=[
        ModalityConfig(
            name="rna",
            input_dim=rna.n_vars,
            encoder_hidden=[512, 256, 128],   # unused by transformer encoders, kept for compatibility
            decoder_hidden=[128, 256, 512],
            likelihood="nb",
            encoder_type="transformer",
            tokenizer=TokenizerConfig(
                mode="topk_scalar",           # cheap attention path (F -> T)
                n_tokens=256,                 # number of tokens T
                add_cls_token=False,
            ),
            transformer=TransformerConfig(
                d_model=256,
                num_heads=8,
                num_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
                attn_dropout=0.1,
                activation="gelu",
                pooling="mean",               # "mean" or "cls"
                max_tokens=None,              # optional; if None, UniVI sets it from the tokenizer
            ),
        ),
        ModalityConfig(
            name="adt",
            input_dim=adt.n_vars,
            encoder_hidden=[256, 128],
            decoder_hidden=[128, 256],
            likelihood="nb",
            encoder_type="mlp",               # default
        ),
    ],
)

# Everything else is the same: dataset -> dataloaders -> trainer -> fit()

---

For richer, exploratory workflows (TEA-seq tri-modal integration, Multiome RNA+ATAC, non-paired matching, etc.), see the notebooks in `notebooks/`.

---
