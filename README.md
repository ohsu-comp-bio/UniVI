# UniVI

[![PyPI version](https://img.shields.io/pypi/v/univi)](https://pypi.org/project/univi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/univi.svg?v=0.2.4)](https://pypi.org/project/univi/)

<picture>
  <!-- Dark mode (GitHub supports this; PyPI may ignore <source>) -->
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.2.4/assets/figures/univi_overview_dark.png">
  <!-- Light mode / fallback (works on GitHub + PyPI) -->
  <img src="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.2.4/assets/figures/univi_overview_light.png"
       alt="UniVI overview and evaluation roadmap"
       width="100%">
</picture>

**UniVI overview and evaluation roadmap.**
(a) Generic UniVI architecture schematic. (b) Core training objective (for UniVI v1 - see documentation for UniVI-lite training objective). (c) Example modality combinations beyond bi-modal data (e.g. TEA-seq (tri-modal RNA + ATAC + ADT)). (d) Evaluation roadmap spanning latent alignment (FOSCTTM ↓), modality mixing, label transfer, reconstruction/prediction NLL, and downstream biological consistency.

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
@article{ashford2025univi,
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

## Training modes & example recipes (v1 vs v2/lite + supervised options)

UniVI supports two training regimes:

* **UniVI v1**: per-modality posteriors + reconstruction terms controlled by `v1_recon` (cross/self/avg/etc.) + posterior alignment across modality posteriors.
* **UniVI-lite / v2**: fused latent posterior (precision-weighted MoE/PoE style) + per-modality reconstruction + β·KL(q_fused||p) + γ·pairwise alignment between modality posteriors. Scales cleanly to 3+ modalities and is the recommended default.

### Which supervised option should I use?

Use labels to “shape” the latent in one of three ways:

1. **Classification head (decoder-only)** — `p(y|z)` (**recommended default**)
   *Works for `loss_mode="lite"` and `loss_mode="v1"`.*
   Best if you want the latent to be predictive/separable without changing how modalities reconstruct.

2. **Label expert injected into fusion (encoder-side)** — `q(z|y)` (**lite/v2 only**)
   *Works only for `loss_mode="lite"` / `v2`.*
   Best for semi-supervised settings where labels should directly influence the **fused posterior**.

3. **Labels as a full categorical “modality”** — `"celltype"` modality with likelihood `"categorical"`
   *Best with `loss_mode="lite"`.*
   Useful when you want cell types to behave like a first-class modality (encode/decode/reconstruct), but avoid `v1` cross-reconstruction unless you really know you want it.

---

## Supervised labels (three supported patterns)

```md
### A) Latent classification head (decoder-only): `p(y|z)` (works in **lite/v2** and **v1**)

This is the simplest way to shape the latent. UniVI attaches a categorical head to the latent `z` and adds:

```math
\mathcal{L} \;+=\; \lambda \cdot \mathrm{CE}(\mathrm{logits}(z), y)

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

**Labels as a categorical modality:** add an additional `"celltype"` modality in `"data.modalities"` and provide a matching AnnData on disk (or build it in Python).

```json5
{
  "model": { "loss_mode": "lite" },
  "data": {
    "modalities": [
      { "name": "rna",      "likelihood": "nb",          "X_key": "X", "layer": "counts" },
      { "name": "adt",      "likelihood": "nb",          "X_key": "X", "layer": "counts" },
      { "name": "celltype", "likelihood": "categorical", "X_key": "X", "layer": null }
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

## Quickstart: run UniVI from Python / Jupyter

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
from univi.data import MultiModalDataset, align_paired_obs_names
from univi.trainer import UniVITrainer
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
    device=None,  # recommended: keep dataset on CPU; trainer moves tensors
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
```

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

### 6) Write latent `z` into AnnData `.obsm["X_univi"]`

```python
from univi import write_univi_latent

Z = write_univi_latent(model, adata_dict, obsm_key="X_univi", device=device, use_mean=True)
print("Embedding shape:", Z.shape)
```

> **Tip**
> Use `use_mean=True` for deterministic plotting/UMAP. Sampling (`use_mean=False`) is stochastic and useful for generative behavior.

---

## Evaluating / encoding: choosing the latent representation

Some utilities (e.g., `encode_adata`) support selecting what embedding to return:

* `"moe_mean"` / `"moe_sample"`: fused latent (MoE/PoE)
* `"modality_mean"` / `"modality_sample"`: per-modality latent

```python
from univi.evaluation import encode_adata

Z_rna = encode_adata(model, rna, modality="rna", device=device, layer="counts", latent="modality_mean")
Z_moe = encode_adata(model, rna, modality="rna", device=device, layer="counts", latent="moe_mean")
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

---

