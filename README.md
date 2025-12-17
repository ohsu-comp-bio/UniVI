# UniVI

[![PyPI version](https://img.shields.io/pypi/v/univi)](https://pypi.org/project/univi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/univi.svg?v=0.3.2)](https://pypi.org/project/univi/)

<picture>
  <!-- Dark mode (GitHub supports this; PyPI may ignore <source>) -->
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.3.2/assets/figures/univi_overview_dark.png">
  <!-- Light mode / fallback (works on GitHub + PyPI) -->
  <img src="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.3.2/assets/figures/univi_overview_light.png"
       alt="UniVI overview and evaluation roadmap"
       width="100%">
</picture>

**UniVI** is a **multi-modal variational autoencoder (VAE)** framework for aligning and integrating single-cell modalities such as RNA, ADT (CITE-seq), and ATAC.

It’s designed for experiments like:

- **Joint embedding** of paired multimodal data (CITE-seq, Multiome, TEA-seq)
- **Zero-shot projection** of external unimodal cohorts into a paired “bridge” latent
- **Cross-modal reconstruction / imputation** (RNA→ADT, ATAC→RNA, etc.)
- **Denoising** via learned generative decoders
- **Evaluation** (FOSCTTM, modality mixing, label transfer, feature recovery)
- **Optional supervised heads** for harmonized annotation / domain confusion

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
    ├── interpretability.py                # Helper scripts for transformer token weight interpretability
    ├── figures/                           # Package-internal figure assets (placeholder)
    │   └── .gitkeep
    ├── models/                            # VAE architectures + building blocks
    │   ├── __init__.py
    │   ├── mlp.py                         # Shared MLP building blocks
    │   ├── encoders.py                    # Modality encoders
    │   ├── decoders.py                    # Likelihood-specific decoders (NB, ZINB, Gaussian, etc.)
    │   ├── transformer.py                 # Transformer blocks + encoder
    │   ├── tokenizer.py                   # Handles token i/o for transformer blocks
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

## Installation

### Install via PyPI

```bash
pip install univi
```

> **Note:** UniVI requires `torch`. If `import torch` fails, install PyTorch for your platform/CUDA from:
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Development install (from source)

```bash
git clone https://github.com/Ashford-A/UniVI.git
cd UniVI

conda env create -f envs/univi_env.yml
conda activate univi_env

pip install -e .
```

### (Optional) Install via conda / mamba

```bash
conda install -c conda-forge univi
# or
mamba install -c conda-forge univi
```

UniVI is also installable from a custom channel:

```bash
conda install ashford-a::univi
# or
mamba install ashford-a::univi
```

---

## Data expectations (high-level)

UniVI expects **per-modality AnnData** objects with matching cells (paired data or consistently paired across modalities).

Typical expectations:

* Each modality is an `AnnData` with the same `obs_names` (same cells, same order)
* Raw counts often live in `.layers["counts"]`
* A processed training representation lives in `.X` (or `.obsm["X_*"]` for ATAC LSI)
* Decoder likelihoods should roughly match the training representation:

  * counts-like → `nb` / `zinb` / `poisson`
  * continuous → `gaussian` / `mse`

See `notebooks/` for end-to-end preprocessing examples.

---

## Training objectives (v1 vs v2/lite)

UniVI supports two main training regimes:

* **UniVI v1 (“paper”)**
  Per-modality posteriors + flexible reconstruction scheme (cross/self/avg) + posterior alignment across modalities.

* **UniVI v2 / lite**
  A fused posterior (precision-weighted MoE/PoE-style) + per-modality recon + β·KL + γ·alignment.
  Convenient for 3+ modalities and “loosely paired” settings.

You choose via `loss_mode` at model construction (Python) or config JSON (CLI scripts).

---

## Quickstart (Python / Jupyter)

Below is a minimal paired **CITE-seq (RNA + ADT)** example using `MultiModalDataset` + `UniVITrainer`.

```python
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Subset

from univi import UniVIMultiModalVAE, ModalityConfig, UniVIConfig, TrainingConfig
from univi.data import MultiModalDataset, align_paired_obs_names
from univi.trainer import UniVITrainer
```

### 1) Load paired AnnData

```python
rna = sc.read_h5ad("path/to/rna_citeseq.h5ad")
adt = sc.read_h5ad("path/to/adt_citeseq.h5ad")

adata_dict = {"rna": rna, "adt": adt}
adata_dict = align_paired_obs_names(adata_dict)  # ensures same obs_names/order
```

### 2) Dataset + dataloaders

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MultiModalDataset(
    adata_dict=adata_dict,
    X_key="X",       # uses .X by default
    device=None,     # dataset returns CPU tensors; model moves to GPU
)

n = rna.n_obs
idx = np.arange(n)
rng = np.random.default_rng(0)
rng.shuffle(idx)
split = int(0.8 * n)
train_idx, val_idx = idx[:split], idx[split:]

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
```

### 3) Config + model

```python
univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.5,
    gamma=2.5,
    encoder_dropout=0.1,
    decoder_dropout=0.0,
    modalities=[
        ModalityConfig("rna", rna.n_vars, [512, 256, 128], [128, 256, 512], likelihood="nb"),
        ModalityConfig("adt", adt.n_vars, [128, 64],       [64, 128],       likelihood="nb"),
    ],
)

train_cfg = TrainingConfig(
    n_epochs=1000,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    device=device,
    log_every=20,
    grad_clip=5.0,
    early_stopping=True,
    patience=50,
)

# v1 (paper)
model = UniVIMultiModalVAE(
    univi_cfg,
    loss_mode="v1",
    v1_recon="avg",
    normalize_v1_terms=True,
).to(device)

# Or: v2/lite
# model = UniVIMultiModalVAE(univi_cfg, loss_mode="v2").to(device)
```

### 4) Train

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

---

## Classification (built-in heads)

UniVI supports **in-model supervised classification heads** (single “legacy” label head and/or multi-head auxiliary decoders). This is useful for:

* harmonized cell-type annotation (e.g., bridge → projected cohorts)
* batch/tech/patient prediction (sanity checks, confounding)
* adversarial domain confusion via gradient reversal (GRL)
* multi-task setups (e.g., celltype + patient + mutation flags)

### How it works

* Heads are configured via `UniVIConfig.class_heads` using `ClassHeadConfig`.
* Training targets are passed as `y`, a **dict mapping head name → integer class indices** with shape `(B,)`.
* Unlabeled entries should use `ignore_index` (default `-1`) and are masked out automatically.
* Each head can be delayed with `warmup` and weighted with `loss_weight`.
* Set `adversarial=True` for GRL heads (domain confusion).

### 1) Add heads in the config

```python
from univi.config import ClassHeadConfig

univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.5,
    gamma=2.5,
    modalities=[
        ModalityConfig("rna", rna.n_vars, [512,256,128], [128,256,512], likelihood="nb"),
        ModalityConfig("adt", adt.n_vars, [128,64],      [64,128],      likelihood="nb"),
    ],
    class_heads=[
        ClassHeadConfig(
            name="celltype",
            n_classes=int(rna.obs["celltype"].astype("category").cat.categories.size),
            loss_weight=1.0,
            ignore_index=-1,
            from_mu=True,     # classify from mu_z (more stable)
            warmup=0,
        ),
        ClassHeadConfig(
            name="batch",
            n_classes=int(rna.obs["batch"].astype("category").cat.categories.size),
            loss_weight=0.2,
            ignore_index=-1,
            from_mu=True,
            warmup=10,
            adversarial=True,  # GRL head (domain confusion)
            adv_lambda=1.0,
        ),
    ],
)
```

Optional: attach readable label names (for your own decoding later):

```python
# for multi-heads
model.set_head_label_names("celltype", list(rna.obs["celltype"].astype("category").cat.categories))
model.set_head_label_names("batch",    list(rna.obs["batch"].astype("category").cat.categories))

# for legacy single head (if you use n_label_classes>0)
# model.set_label_names([...])
```

### 2) Pass `y` to the model during training

Your dataloader typically yields only `x_dict`. You can construct `y` from your dataset indices (or store labels inside your dataset object). Example pattern:

```python
# Suppose you have per-cell label arrays aligned to dataset order:
celltype_codes = rna.obs["celltype"].astype("category").cat.codes.to_numpy()
batch_codes    = rna.obs["batch"].astype("category").cat.codes.to_numpy()

# In your training loop / trainer, for a batch of indices `batch_idx`:
y = {
    "celltype": torch.tensor(celltype_codes[batch_idx], device=device),
    "batch":    torch.tensor(batch_codes[batch_idx], device=device),
}

out = model(x_dict, epoch=epoch, y=y)
loss = out["loss"]
loss.backward()
```

What you get back in `out` (when labels are provided) includes:

* `out["head_logits"]`: dict of logits `(B, n_classes)` per head
* `out["head_losses"]`: mean CE per head (masked by `ignore_index`)

### 3) Predict heads after training

```python
model.eval()
batch = next(iter(val_loader))
x_dict = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    probs = model.predict_heads(x_dict, return_probs=True)

for head_name, P in probs.items():
    print(head_name, P.shape)  # (B, n_classes)
```

To inspect which heads exist + their settings:

```python
meta = model.get_classification_meta()
print(meta)
```

---

## After training: what you can do with a trained UniVI model

UniVI isn’t just “map to latent”. With a trained model you can typically:

* **Encode modality-specific posteriors** `q(z|x_rna)`, `q(z|x_adt)`, …
* **Encode a fused posterior** (MoE/PoE; or an optional fused transformer encoder)
* **Reconstruct** inputs via decoders (denoising)
* **Cross-reconstruct / impute** across modalities (RNA→ADT, ADT→RNA, RNA→ATAC, …)
* **Predict supervised targets** via built-in classification heads
* **Inspect per-modality posterior means/variances** for debugging and uncertainty
* (Optional) **Inspect transformer token selection meta** (top-k indices) and (with a small patch) attention weights

### 1) Encode fused latent (deterministic) for plotting / neighbors

```python
model.eval()
batch = next(iter(val_loader))  # dict: {"rna": (B,F), "adt": (B,F)}
x_dict = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    mu_z, logvar_z, z = model.encode_fused(x_dict, use_mean=True)

Z = mu_z.detach().cpu().numpy()
print(Z.shape)
```

### 2) Encode per-modality latents (useful for projection + diagnostics)

```python
with torch.no_grad():
    mu_dict, logvar_dict = model.encode_modalities(x_dict)

Z_rna = mu_dict["rna"].detach().cpu().numpy()
Z_adt = mu_dict["adt"].detach().cpu().numpy()
```

### 3) Reconstruct (denoise) all modalities from the fused latent

```python
with torch.no_grad():
    out = model(x_dict, epoch=0)
    xhat = out["xhat"]  # dict(modality -> decoder output)
```

### 4) Cross-modal reconstruction / imputation (encode one modality, decode another)

Example: **RNA → predicted ADT**

```python
x_rna_only = {"rna": x_dict["rna"]}

with torch.no_grad():
    mu_dict, lv_dict = model.encode_modalities(x_rna_only)
    z_rna = mu_dict["rna"]  # deterministic
    adt_pred = model.decoders["adt"](z_rna)
```

---

## CLI training (from JSON configs)

Most `scripts/*.py` entry points accept a parameter JSON.

**Train:**

```bash
python scripts/train_univi.py \
  --config parameter_files/defaults_cite_seq_scaled_gaussian_v1.json \
  --outdir saved_models/citeseq_v1_run1 \
  --data-root /path/to/your/data
```

**Evaluate:**

```bash
python scripts/evaluate_univi.py \
  --config parameter_files/defaults_cite_seq_scaled_gaussian_v1.json \
  --model-checkpoint saved_models/citeseq_v1_run1/checkpoints/univi_checkpoint.pt \
  --outdir saved_models/citeseq_v1_run1/eval
```

---

## Optional: Transformer encoders (per-modality)

By default, UniVI uses **MLP encoders** (`encoder_type="mlp"`), and all classic workflows work unchanged.

If you want a transformer encoder for a modality, set:

* `encoder_type="transformer"`
* a `TokenizerConfig` (how `(B,F)` becomes `(B,T,D_in)`)
* a `TransformerConfig` (depth/width/pooling)

Example:

```python
from univi.config import TransformerConfig, TokenizerConfig

univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.0,
    gamma=1.25,
    modalities=[
        ModalityConfig(
            name="rna",
            input_dim=rna.n_vars,
            encoder_hidden=[512, 256, 128],   # ignored by transformer encoder; kept for compatibility
            decoder_hidden=[128, 256, 512],
            likelihood="gaussian",
            encoder_type="transformer",
            tokenizer=TokenizerConfig(mode="topk_channels", n_tokens=512, channels=("value","rank","dropout")),
            transformer=TransformerConfig(
                d_model=256, num_heads=8, num_layers=4,
                dim_feedforward=1024, dropout=0.1, attn_dropout=0.1,
                activation="gelu", pooling="mean",
            ),
        ),
        ModalityConfig(
            name="adt",
            input_dim=adt.n_vars,
            encoder_hidden=[128, 64],
            decoder_hidden=[64, 128],
            likelihood="gaussian",
            encoder_type="mlp",
            tokenizer=TokenizerConfig(mode="topk_scalar", n_tokens=min(32, adt.n_vars)),  # useful for fused encoder
        ),
    ],
)
```

Why bother?

* Better inductive bias for **feature interaction modeling**
* Tokenizers focus attention on the most informative features per cell (top-k)
* Interpretability hooks: token indices (and optional attention maps)

---

## Optional: Fused multimodal transformer encoder (advanced)

A single transformer sees **concatenated tokens from multiple modalities** and returns a **single fused posterior** `q(z|all modalities)` using global CLS pooling (or mean pooling).

### Minimal config

```python
from univi.config import TransformerConfig

univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.0,
    gamma=1.25,
    modalities=[...],  # your per-modality configs still exist
    fused_encoder_type="multimodal_transformer",
    fused_modalities=("rna", "adt", "atac"),  # default: all modalities
    fused_transformer=TransformerConfig(
        d_model=256, num_heads=8, num_layers=4,
        dim_feedforward=1024, dropout=0.1, attn_dropout=0.1,
        activation="gelu", pooling="cls",
    ),
)
```

Notes:

* Every modality in `fused_modalities` must define a `tokenizer` (even if its per-modality encoder is MLP).
* If `fused_require_all_modalities=True` and any fused modality is missing at inference, UniVI falls back to MoE/PoE fusion.

---

## Hyperparameter optimization (optional)

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

See `univi/hyperparam_optimization/` and `notebooks/` for examples.

---

## Contact, questions, and bug reports

* **Questions / comments:** open a GitHub Issue with the `question` label (or a Discussion if enabled).
* **Bug reports:** open a GitHub Issue and include:

  * your UniVI version: `python -c "import univi; print(univi.__version__)"`
  * minimal code to reproduce (or a short notebook snippet)
  * stack trace + OS/CUDA/PyTorch versions
* **Feature requests:** open an Issue describing the use-case + expected inputs/outputs (a tiny example is ideal).

---
