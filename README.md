# UniVI

UniVI is a **multi-modal variational autoencoder (VAE)** framework for aligning and integrating single-cell modalities such as RNA, ADT (CITE-seq), and ATAC. It’s built to support experiments like:

- Joint embedding of RNA + ADT (CITE-seq)
- RNA + ATAC (Multiome) integration
- RNA + ADT + ATAC (TEA-seq) tri-modal data integration
- Independent non-paired modalities from the same tissue type
- Cross-modal reconstruction and imputation
- Data denoising
- Evaluation of alignment quality (FOSCTTM, modality mixing, label transfer, etc.)

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
    │   ├── __init__.py
    │   ├── common.py              # Shared hparam search utilities
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

> **Note (coming soon)**:
> The commands below assume UniVI is available on PyPI and a conda channel (e.g. `conda-forge` or a dedicated `Ashford-A` channel).
> Until the first public release is pushed, please use the **development install from source** (see below).

### 1. Install UniVI via pip (PyPI)

Once UniVI is published on PyPI:

```bash
pip install univi
```

This installs the `univi` package and all core dependencies.

### 2. Install via conda / mamba

Once UniVI is available on a conda channel:

```bash
# Using conda (example: conda-forge)
conda install -c conda-forge univi

# Using mamba
mamba install -c conda-forge univi
```

To create a fresh environment with UniVI:

```bash
conda create -n univi_env python=3.10 univi -c conda-forge
conda activate univi_env
```

### 3. Development install (from source) — **current recommended method**

Until the PyPI / conda packages are live, install from this repository:

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

---

## Prepare input data

UniVI expects per-modality AnnData objects with matching cells (either truly paired data or a well-defined pairing between modalities (the univi/matching.py script contains supplemental functions for different non-joint data pairing):

* Each modality (e.g. RNA / ADT / ATAC) is an `AnnData` with the same `obs_names` (same cells, same order).
* Raw counts are usually stored in `.layers["counts"]`, with a processed view in `.X` used for training.

Typical conventions:

### RNA

* `.layers["counts"]` → raw counts
* `.X` → data used in model training, e.g.:

  * log1p-normalized HVGs
  * raw counts
  * normalized/scaled counts

Decoder likelihood (per RNA modality) should roughly match the input distribution:

* `"nb"` or `"zinb"` for raw or normalized counts
* `"gaussian"` for log-normalized / scaled data (treated as continuous)

### ADT (CITE-seq)

* `.layers["counts"]` → raw ADT counts
* `.X` → one of:

  * CLR-normalized ADT
  * CLR-normalized + scaled ADT
  * raw ADT counts (depending on the experiment)

Decoder likelihood for ADT:

* `"nb"` or `"zinb"` for raw / count-like data
* `"gaussian"` for normalized / scaled ADT

### ATAC

* `.layers["counts"]` → raw peak counts
* `.obsm["X_lsi"]` → LSI/TF–IDF components
* `.X` → can be:

  * `obsm["X_lsi"]` (continuous LSI space)
  * `layers["counts"]` (or a subset / HV peaks)
  * another derived representation

Decoder likelihood for ATAC:

* `"mse"` or `"gaussian"` if using continuous LSI
* `"nb"` or `"poisson"` if using raw peak counts (likely subsetted, highly-variable peaks, etc.)

See the notebooks under `notebooks/` for end-to-end preprocessing examples for CITE-seq, Multiome, and TEA-seq.

---

## Run a minimal training script

Once your data are preprocessed and saved (or loaded directly in a script), you can launch training using `scripts/train_univi.py` and any of the JSON configs under `parameter_files/`.

Example (adjust paths / filenames to your setup):

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_cite_seq.json \
  --outdir saved_models/citeseq_run1 \
  --data-root /path/to/your/data
```

A typical config file in `parameter_files/` specifies:

* Latent dimensionality, β (beta) and γ (gamma)
* Per-modality input dimensions and likelihoods
* Training hyperparameters (epochs, batch size, learning rate, etc.)

---

## Evaluate a trained model

After training, you can run evaluation to compute alignment metrics and generate UMAPs:

```bash
python scripts/evaluate_univi.py \
  --config parameter_files/defaults_cite_seq.json \
  --model-checkpoint saved_models/citeseq_run1/best_model.pt \
  --outdir figures/citeseq_run1
```

These typically compute:

* FOSCTTM
* Modality mixing scores
* kNN label transfer accuracy
* UMAPs colored by cell type and modality
* Cross-modal reconstruction summaries
* More to come!

For more detailed, notebook-style workflows (e.g. TEA-seq tri-modal integration, Multiome RNA+ATAC, or non-paired matching), see the examples under `notebooks/`.

