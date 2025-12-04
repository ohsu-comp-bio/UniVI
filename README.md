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

### 0) Choose your training objective (v1 vs lite) in the config JSON

In your `parameter_files/*.json`, set a single switch controlling the objective. Recommended pattern:

```json
{
  "objective": "v1"
}
```

or:

```json
{
  "objective": "lite"
}
```

> If your repo uses a different field name than `objective` (e.g., `loss_mode`, `training_objective`, `use_cross_recon`), keep your existing key and use the values `"v1"` vs `"lite"` accordingly.

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

> Your notebooks show recommended preprocessing per dataset; the key is that the decoder likelihood should match the input distribution.

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

> Suggestion: keep parallel config templates per dataset: `*_v1.json` and `*_lite.json` that differ only in `objective` (and any recommended default β/γ).

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

