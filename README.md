# UniVI

UniVI is a **multi-modal variational autoencoder (VAE)** framework for aligning and integrating single-cell modalities such as RNA, ADT (CITE-seq), and ATAC. It’s built to support experiments like:

- Joint embedding of RNA + ADT (CITE-seq)
- RNA + ATAC (Multiome) integration
- RNA + ADT + ATAC (TEA-seq)
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
├── README.md                  # Project overview and quickstart
├── envs/                      # Conda / environment definitions (.yml files)
├── data/                      # Local data (gitignored, except small metadata)
├── figures/                   # Generated plots and figures (gitignored)
├── notebooks/                 # Jupyter notebooks for experiments / exploration
├── parameter_files/           # JSON/YAML configs for models & training runs
├── saved_models/              # Trained model checkpoints / artifacts
├── scripts/                   # CLI scripts / training & eval entry points
├── tutorials/                 # Example workflows / how-to guides
└── univi/                     # UniVI Python package
    ├── __init__.py            # Package exports / version
    ├── config.py              # Config dataclasses & helpers
    ├── data.py                # Dataset wrappers, loaders (MultiModalDataset, etc.)
    ├── evaluation.py          # Metrics (FOSCTTM, mixing scores, label transfer, etc.)
    ├── matching.py            # Helper functions for matching / alignment
    ├── models/                # VAE architectures and modality-specific modules
    │   ├── __init__.py        # Model registry / convenience imports
    │   ├── decoders.py        # Likelihood-specific decoders (NB, ZINB, Gaussian, etc.)
    │   ├── encoders.py        # Modality-specific encoders
    │   ├── mlp.py             # Shared MLP building blocks
    │   └── univi.py           # UniVI core VAE architectures
    ├── objectives.py          # Losses (ELBO, alignment, KL annealing, etc.)
    ├── plotting.py            # Plotting helpers / evaluation visualizations
    ├── trainer.py             # UniVITrainer: training loop & logging
    └── utils/                 # Utility functions, helpers, misc tools
        ├── __init__.py        # Utils namespace
        ├── io.py              # I/O helpers (AnnData, checkpoints, configs)
        ├── logging.py         # Logging setup / progress reporting
        ├── seed.py            # Seeding / reproducibility utilities
        ├── stats.py           # Small statistical helpers / transforms
        └── torch_utils.py     # PyTorch utilities (device, tensors, etc.)
```

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Ashford-A/UniVI.git
cd UniVI
```

### 2. Create and activate a conda environment

Pick one of the environment files under `envs/` (adjust the name if yours is different):

```bash
conda env create -f envs/univi_env.yml
conda activate univi
```

Alternatively, with mamba:

```bash
mamba env create -f envs/univi_env.yml
mamba activate univi
```

### 3. Install UniVI in editable mode

From the repo root:

```bash
pip install -e .
```

This makes the `univi` package importable in your scripts and notebooks.

---

## Prepare input data

UniVI expects per-modality AnnData objects with matching cells, be that paired data or some other pairing metric:

* Each modality (e.g. RNA / ADT / ATAC) is an `AnnData` with the same `obs_names` (same cells, same order).
* Raw counts are usually stored in `.layers["counts"]`, with a processed view in `.X`.

Typical conventions:

**RNA**

* `.layers["counts"]` → raw counts
* `.X` → what's used in model training; can be normalized / log1p counts (e.g. HVGs only), raw counts, normalized / log1p / scaled counts, etc.
* When setting up a model, decoder likelihood for each modality should roughly match its respective input distribution:

  * `"nb"` or `"zinb"` for raw counts or some instances of normalized counts
  * `"gaussian"` for other instances of normalized and normalized/scaled counts

**ADT (CITE-seq)**

* `.layers["counts"]` → raw ADT counts
* `.X` → can be CLR-normalized ADT, CLR-normalized / scaled ADT, raw ADT counts, or even preprocessed similar to RNA depending on integration goals
* Decoder likelihood for ADT data, depending on how it's preprocessed (should roughly match the input distribution):

  * `"nb"` or `"zinb"` for more raw-like counts
  * `"gaussian"` for normalized or normalized/scaled counts

**ATAC**

* `.layers["counts"]` → raw peak counts
* `.obsm["X_lsi"]` → LSI/TF-IDF components
* `.X` → can be set to `obsm["X_lsi"]`, `layers["counts"]`, or other normalization methods for UniVI depending on research goals
* Decoder likelihood for ATAC data, depending on how it's preprocessed (should roughly match the input distribution):

  * `"mse"` if using the continuous LSI space
  * `"nb"` or `"poisson"` if using a subset of raw peak counts / highly variable peaks
  * `"gaussian"` if you treat it similarly to RNA preprocessing depending on integration goals

See `tutorials/` and `notebooks/` for full preprocessing examples.

---

## Run a minimal training script

Once your data are preprocessed and saved (or loaded directly in a script), you can launch training using one of the scripts under `scripts/` and a config file from `parameter_files/`.

Example (adjust paths / filenames to your setup):

```bash
python scripts/train_univi.py \
  --config parameter_files/citeseq_univi_config.json \
  --outdir saved_models/citeseq_run1 \
  --data-root /path/to/your/data
```

A typical config file (`parameter_files/*.json`) specifies:

* Latent dimensionality, beta and gamma
* Per-modality input dimensions and likelihoods
* Training hyperparameters (epochs, batch size, learning rate, etc.)

---

## Evaluate a trained model

After training, you can run evaluation scripts to compute alignment metrics and generate UMAPs:

```bash
python scripts/eval_univi.py \
  --config parameter_files/citeseq_univi_config.json \
  --model-checkpoint saved_models/citeseq_run1/best_model.pt \
  --outdir figures/citeseq_run1
```

These typically compute:

* FOSCTTM
* Modality mixing scores
* kNN label transfer accuracy
* UMAPs colored by cell type and modality
* Cross-modal reconstruction summaries
* We're continuously implementing new evaluation metrics and tools, so this list is not exhaustive.

For more detailed, notebook-style workflows (e.g. TEA-seq tri-modal integration, Multiome RNA+ATAC, or non-paired matching), see the examples under `notebooks/` and `tutorials/`.


