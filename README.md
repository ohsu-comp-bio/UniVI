# UniVI

[![PyPI version](https://img.shields.io/pypi/v/univi)](https://pypi.org/project/univi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/univi.svg?v=0.1.4)](https://pypi.org/project/univi/)

<picture>
  <!-- Dark mode -->
  <source media="(prefers-color-scheme: dark)" srcset="assets/figures/univi_overview_dark.png">
  <!-- Light mode / fallback -->
  <img src="assets/figures/univi_overview_light.png"
       alt="UniVI overview and evaluation roadmap"
       width="100%">
</picture>

**UniVI overview and evaluation roadmap.**  
(a) Generic UniVI architecture schematic. (b) Core training objective (for UniVI v1 - see documentation for UniVI-lite training objective). (c) Example modality combinations beyond bi-modal data (e.g. TEA-seq (tri-modal RNA + ATAC + ADT)). (d) Evaluation roadmap spanning latent alignment (FOSCTTM ↓), modality mixing, label transfer, reconstruction/prediction NLL, and downstream biological consistency.

---

UniVI is a **multi-modal variational autoencoder (VAE)** framework for aligning and integrating single-cell modalities such as RNA, ADT (CITE-seq), and ATAC. It’s built to support experiments like:

- Joint embedding of RNA + ADT (CITE-seq)
- RNA + ATAC (Multiome) integration
- RNA + ADT + ATAC (TEA-seq) tri-modal data integration
- Independent non-paired modalities from the same tissue type
- Cross-modal reconstruction and imputation
- Data denoising
- Structured evaluation of alignment quality (FOSCTTM, modality mixing, label transfer, etc.)
- Exploratory analysis of the relationships between heterogeneous molecular readouts that inform biological functional dimensions

This repository contains the core UniVI code, training scripts, parameter files, and example notebooks.

---

## Preprint

If you use UniVI in your work, please cite:

> Ashford AJ, Enright T, Nikolova O, Demir E.  
> **Unifying Multimodal Single-Cell Data Using a Mixture of Experts β-Variational Autoencoder-Based Framework.**  
> *bioRxiv* (2025). doi: [10.1101/2025.02.28.640429](https://www.biorxiv.org/content/10.1101/2025.02.28.640429v1.full)

```bibtex
@article{ashford2025univi,
  title   = {Unifying Multimodal Single-Cell Data Using a Mixture of Experts β-Variational Autoencoder-Based Framework},
  author  = {Ashford, Andrew J. and Enright, Trevor and Nikolova, Olga and Demir, Emek},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.02.28.640429},
  url     = {https://www.biorxiv.org/content/10.1101/2025.02.28.640429v1}
}
````

---

## Repository structure

At a high level:

```text
UniVI/
├── README.md                      # Project overview, installation, quickstart
├── pyproject.toml                 # Python packaging config (pip / PyPI)
├── assets/                        # Assets folder, currently just houses the figures subfolder
│   └── figures/                   # Contains schematic figure(s) for repository front page
├── conda.recipe/                  # Conda build recipe (for conda-build)
│   └── meta.yaml
├── envs/                          # Example conda environments
│   ├── UniVI_working_environment.yml
│   ├── UniVI_working_environment_v2_full.yml
│   ├── UniVI_working_environment_v2_minimal.yml
│   └── univi_env.yml              # Recommended env to use - has necessary packages for CUDA GPU usage
├── data/                          # Example datasets (Hao CITE-seq, 10x Multiome, TEA-seq)
│   ├── README.md                  # Notes on data sources / formats
│   ├── Hao_CITE-seq_data/
│   ├── PBMC_10x_Multiome_data/
│   └── TEA-seq_data/
├── figures/                       # Generated figures used in docs / paper
├── notebooks/                     # Jupyter notebooks for demos & benchmarks
│   ├── UniVI_CITE-seq_*.ipynb     # CITE-seq examples and benchmarks
│   ├── UniVI_10x_Multiome_*.ipynb # 10x PBMC Multiome examples
│   └── UniVI_TEA-seq_*.ipynb      # TEA-seq (tri-modal) examples
├── parameter_files/               # JSON configs for model & training hyperparameters
│   ├── defaults_*.json            # Default configs (per modality / experiment)
│   └── params_*.json              # Example “named” configs (RNA, ADT, ATAC, etc.)
├── saved_models/                  # Example trained UniVI checkpoints
│   └── univi_*.pt                 # (Reproducibility / ready-made demos)
├── scripts/                       # CLI entry points for training & evaluation
│   ├── train_univi.py             # Train UniVI from a parameter JSON
│   ├── evaluate_univi.py          # Evaluate trained models (FOSCTTM, label transfer, etc.)
│   ├── benchmark_univi_citeseq.py # CITE-seq-specific benchmarking script
│   └── run_multiome_hparam_search.py
└── univi/                         # UniVI Python package (importable as `import univi`)
    ├── __init__.py                # Package exports and __version__
    ├── config.py                  # Config dataclasses & helpers
    ├── data.py                    # Dataset wrappers, loaders (MultiModalDataset, etc.)
    ├── evaluation.py              # Metrics (FOSCTTM, mixing scores, label transfer, etc.)
    ├── matching.py                # Modality matching / alignment helpers
    ├── models/                    # VAE architectures and modality-specific modules
    │   ├── __init__.py
    │   ├── decoders.py            # Likelihood-specific decoders (NB, ZINB, Gaussian, etc.)
    │   ├── encoders.py            # Modality-specific encoders (RNA, ADT, ATAC)
    │   ├── mlp.py                 # Shared MLP building blocks
    │   └── univi.py               # Core UniVI VAE architectures
    ├── objectives.py              # Losses (ELBO, alignment, KL annealing, etc.)
    ├── plotting.py                # Plotting helpers / evaluation visualizations
    ├── trainer.py                 # UniVITrainer: training loop, logging, checkpointing
    ├── hyperparam_optimization/   # Hyperparameter search scripts
    │   ├── __init__.py            # Re-exports run_*_hparam_search helpers
    │   ├── common.py              # Shared hparam search utilities (sampling, training, metrics)
    │   ├── run_adt_hparam_search.py
    │   ├── run_atac_hparam_search.py
    │   ├── run_citeseq_hparam_search.py
    │   ├── run_multiome_hparam_search.py
    │   ├── run_rna_hparam_search.py
    │   └── run_teaseq_hparam_search.py
    └── utils/                     # General utilities
        ├── __init__.py
        ├── io.py                  # I/O helpers (AnnData, configs, checkpoints)
        ├── logging.py             # Logging configuration / progress reporting
        ├── seed.py                # Reproducibility helpers (seeding RNGs)
        ├── stats.py               # Small statistical helpers / transforms
        └── torch_utils.py         # PyTorch utilities (device, tensor helpers)
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

If UniVI is available on a conda channel (e.g. `conda-forge` or a dedicated channel), you can install with:

```bash
# Using conda
conda install -c conda-forge univi

# Using mamba
mamba install -c conda-forge univi
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

## Running a minimal training script (UniVI v1 vs UniVI-lite)

UniVI supports two training regimes:

* **UniVI v1**: paired/pseudo-paired batches + cross-modal reconstruction (e.g., RNA→ADT and ADT→RNA) + posterior alignment.
* **UniVI-lite**: missing-modality friendly (can train when only a subset of modalities are present in a batch), typically with a lighter latent alignment term.

### 0) Choose the training objective (`loss_mode`) in your config JSON

In `parameter_files/*.json`, set a single switch that controls the objective:

**Paper objective (v1; cross-reconstruction + cross-posterior alignment):**

```json
{
  "loss_mode": "v1"
}
````

**UniVI-lite objective (v2; lightweight / fusion-based):**

```json
{
  "loss_mode": "lite"
}
```

> **Note**
> `loss_mode: "lite"` is an alias for `loss_mode: "v2"` (they run the same objective in the current code).

### 1) Normalization / representation switch (counts vs continuous)

UniVI can be trained on **counts** (NB/ZINB/Poisson likelihoods) or **continuous** representations (Gaussian/MSE likelihoods). In your configs, keep this explicit.

Recommended pattern (example):

```json
{
  "input_representation": {
    "RNA":  { "layer": "counts", "X_key": "X", "assume_log1p": false },
    "ADT":  { "layer": "counts", "X_key": "X", "assume_log1p": false },
    "ATAC": { "layer": "counts", "X_key": "X_lsi", "assume_log1p": false }
  }
}
```

* Use `.layers["counts"]` when you want NB/ZINB/Poisson decoders.
* Use continuous `.X` (log1p/CLR/LSI) when you want Gaussian/MSE decoders.

> Jupyter Notebooks in this repository (UniVI/notebooks/) show recommended preprocessing per dataset for different data types and analyses. Depending on your research goals, you can use several different methods of preprocessing. The model is quite robust when it comes to learning underlying biology regardless of input data processing method used; the main key is that the decoder likelihood should roughly match the input distribution per-modality. 

### 2) Train (CLI)

Example: **CITE-seq (RNA + ADT)**

**UniVI v1**

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_cite_seq_v1.json \
  --outdir saved_models/citeseq_v1_run1 \
  --data-root /path/to/your/data
```

**UniVI-lite**

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_cite_seq_lite.json \
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
  --config parameter_files/defaults_teaseq_v1.json \
  --outdir saved_models/teaseq_v1_run1 \
  --data-root /path/to/your/data
```

**UniVI-lite**

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_teaseq_lite.json \
  --outdir saved_models/teaseq_lite_run1 \
  --data-root /path/to/your/data
```

### 3) Quickstart: run UniVI from Python / Jupyter

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
from univi.data import MultiModalDataset
from univi.trainer import UniVITrainer
````

#### 1) Load preprocessed AnnData (paired cells)

```python
# Example: CITE-seq with RNA + ADT
rna = sc.read_h5ad("path/to/rna_citeseq.h5ad")
adt = sc.read_h5ad("path/to/adt_citeseq.h5ad")

# Assumes rna.obs_names == adt.obs_names (same cells, same order)
adata_dict = {
    "rna": rna,
    "adt": adt,
}
```

#### 2) Build `MultiModalDataset` and DataLoaders

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MultiModalDataset(
    adata_dict=adata_dict,
    X_key="X",          # use .X from each AnnData for training
    device=device,      # tensors moved to this device on-the-fly
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

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)
```

#### 3) Define UniVI configs (v1 vs UniVI-lite)

```python
# UniVI model config (architecture + regularization)
univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=5.0,          # KL weight
    gamma=40.0,        # alignment weight (used differently in v1 vs lite)
    encoder_dropout=0.1,
    decoder_dropout=0.0,
    encoder_batchnorm=True,
    decoder_batchnorm=False,
    kl_anneal_start=0,
    kl_anneal_end=25,
    align_anneal_start=0,
    align_anneal_end=25,
    modalities=[
        ModalityConfig(
            name="rna",
            input_dim=rna.n_vars,
            encoder_hidden=[1024, 512],
            decoder_hidden=[512, 1024],
            likelihood="nb",   # counts-like RNA
        ),
        ModalityConfig(
            name="adt",
            input_dim=adt.n_vars,
            encoder_hidden=[256, 128],
            decoder_hidden=[128, 256],
            likelihood="nb",   # counts-like ADT
        ),
    ],
)

# Training config (epochs, LR, device, etc.)
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

#### 4) Choose the objective: **v1** vs **UniVI-lite**

* **v1** (paper objective): cross-reconstruction + cross-posterior alignment
  Best when batches are paired/pseudo-paired and you want explicit cross-prediction.
* **lite** (aka `"v2"`): missing-modality friendly; trains even if some modalities are absent in a batch.

```python
# Option A: UniVI v1 (paper)
model = UniVIMultiModalVAE(
    univi_cfg,
    loss_mode="v1",
    v1_recon="cross",          # "cross" | "self" | "avg" | "moe" | "src:rna" etc.
    v1_recon_mix=0.0,          # optional extra averaged-z recon weight
    normalize_v1_terms=True,
).to(device)

# Option B: UniVI-lite (v2)
# model = UniVIMultiModalVAE(univi_cfg, loss_mode="lite").to(device)
```

#### 5) Train inside Python / Jupyter

```python
trainer = UniVITrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    train_cfg=train_cfg,
    device=device,
)

history = trainer.fit()  # runs the training loop
```

`history` typically contains per-epoch loss and metric curves. After training, you can reuse `model` directly for:

* Computing latent embeddings (`encode_modalities` / `mixture_of_experts`)
* Cross-modal reconstruction (forward passes with different modality subsets)
* Exporting `z` to AnnData or NumPy for downstream analysis (UMAP, clustering, DE, etc.)

### 6) Write latent `z` into AnnData `.obsm["X_univi"]`

```python
@torch.no_grad()
def write_univi_latent(model, adata_dict, *, obsm_key="X_univi", batch_size=512, device="cpu"):
    model.eval()
    names = list(adata_dict.keys())
    n = adata_dict[names[0]].n_obs

    # require paired order
    for nm in names[1:]:
        if not np.array_equal(adata_dict[nm].obs_names.values, adata_dict[names[0]].obs_names.values):
            raise ValueError(f"obs_names mismatch between {names[0]} and {nm}")

    zs = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x_dict = {}
        for nm, ad in adata_dict.items():
            X = ad.X[start:end]
            X = X.A if hasattr(X, "A") else X
            x_dict[nm] = torch.as_tensor(X, dtype=torch.float32, device=device)

        out = model(x_dict)
        zs.append(out["z"].detach().cpu().numpy())

    Z = np.vstack(zs)

    for nm, ad in adata_dict.items():
        ad.obsm[obsm_key] = Z

    return Z

Z = write_univi_latent(model, adata_dict, obsm_key="X_univi", device=device)
print("Embedding shape:", Z.shape)
```

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

## Evaluating a trained model

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

For richer, exploratory workflows (TEA-seq tri-modal integration, Multiome RNA+ATAC, non-paired matching, etc.), see the notebooks in `notebooks/`.

