# UniVI

[![PyPI version](https://img.shields.io/pypi/v/univi)](https://pypi.org/project/univi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/univi.svg?v=0.3.1)](https://pypi.org/project/univi/)

<picture>
  <!-- Dark mode (GitHub supports this; PyPI may ignore <source>) -->
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.3.1/assets/figures/univi_overview_dark.png">
  <!-- Light mode / fallback (works on GitHub + PyPI) -->
  <img src="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.3.1/assets/figures/univi_overview_light.png"
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

## Training objectives (v1 vs lite/v2)

UniVI supports two main training regimes:

* **UniVI v1 (“paper”)**
  Per-modality posteriors + flexible reconstruction scheme (cross/self/avg) + posterior alignment across modalities.

* **UniVI-lite / v2**
  A fused posterior (precision-weighted MoE/PoE style) + per-modality recon + β·KL + γ·alignment.
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

# Paper objective (v1)
model = UniVIMultiModalVAE(
    univi_cfg,
    loss_mode="v1",
    v1_recon="avg",
    normalize_v1_terms=True,
).to(device)

# Or: UniVI-lite / v2
# model = UniVIMultiModalVAE(univi_cfg, loss_mode="lite").to(device)
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

## Optional classification head addition

UniVI can be used purely unsupervised **or** you can attach lightweight supervised heads for tasks like:

* harmonized cell type annotation (bridge → unimodal projection labels)
* patient / batch prediction (sanity checks, domain effects)
* binary mutation flags, response labels, etc.

Two common patterns are below.

### A) Train a simple classifier *on top of frozen UniVI latents* (easy + robust)

This is the “least magic” option: train UniVI unsupervised as usual, then train a classifier on the latent means.

```python
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

# Example: labels live in RNA AnnData
# (since modalities are aligned, RNA labels index the same cells as ADT/ATAC)
y_cat = rna.obs["celltype"].astype("category")
y = y_cat.cat.codes.to_numpy()  # (n,)

# Encode latents for ALL cells (use your favorite split logic)
model.eval()
Z_all = []

with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=512, shuffle=False):
        x_dict = {k: v.to(device) for k, v in batch.items()}
        mu_z, logvar_z, z = model.encode_fused(x_dict, use_mean=True)
        Z_all.append(mu_z.detach().cpu().numpy())

Z_all = np.concatenate(Z_all, axis=0)  # (n, latent_dim)

# Fit classifier on train split (example)
clf = LogisticRegression(max_iter=2000, n_jobs=1)
clf.fit(Z_all[train_idx], y[train_idx])

# Predict on val/test
yhat = clf.predict(Z_all[val_idx])
```

Notes:

* This gives you a clean “latent classifier” that is easy to audit and compare across models.
* For coarse labels, logistic regression is often surprisingly strong; for many classes you can swap in a linear SVM or a small MLP.

### B) Use UniVI’s built-in heads (single head or multi-head)

If you enabled classification heads, inference is typically one call:

```python
model.eval()
batch = next(iter(val_loader))
x_dict = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    probs = model.predict_heads(x_dict, return_probs=True)

# probs is a dict: {head_name: (B, n_classes)}
for head_name, P in probs.items():
    print(head_name, P.shape)
```

Notes:

* For the legacy single-head path, the key is `model.label_head_name` (default `"label"`).
* For multi-head configs, keys are head names (e.g. `"celltype"`, `"patient"`, `"mutation_TP53"`).
* During supervised training you’ll supply integer targets `y` (class indices) alongside `x_dict` and weight the head loss relative to the VAE losses.

---

## After training: what you can do with a trained UniVI model

UniVI isn’t just “map to latent”. With a trained model you can typically:

* **Encode modality-specific posteriors** `q(z|x_rna)`, `q(z|x_adt)`, …
* **Encode a fused posterior** (lite/v2 MoE/PoE; or an optional fused transformer encoder)
* **Reconstruct** inputs via decoders (denoising)
* **Cross-reconstruct / impute** across modalities (RNA→ADT, ADT→RNA, RNA→ATAC, …)
* **Predict supervised targets** via classification heads (single head or multi-head)
* **Inspect per-modality posterior means/variances** for debugging and uncertainty
* (Optional) **Inspect transformer token selection meta** (top-k indices) and (with a small patch) attention weights

### 1) Encode fused latent (deterministic) for plotting / neighbors

```python
model.eval()
batch = next(iter(val_loader))  # a dict: {"rna": (B,F), "adt": (B,F)}
x_dict = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    mu_z, logvar_z, z = model.encode_fused(x_dict, use_mean=True)

# mu_z is a clean embedding for UMAP/clustering
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

# Depending on decoder likelihood, xhat[m] may be a tensor or a dict (e.g. {"mu":..., "log_theta":...})
# For gaussian-ish decoders, the "mean" is typically the denoised prediction.
```

### 4) Cross-modal reconstruction / imputation (encode one modality, decode another)

Example: **RNA → predicted ADT**

```python
x_rna_only = {"rna": x_dict["rna"]}

with torch.no_grad():
    mu_dict, lv_dict = model.encode_modalities(x_rna_only)
    z_rna = mu_dict["rna"]  # deterministic; or sample via reparameterize

    adt_pred = model.decoders["adt"](z_rna)

# If the ADT decoder returns {"mean": ...}, use adt_pred["mean"] as your predicted profile
```

This is the basis of:

* feature recovery curves (observed vs predicted)
* cross-modal marker plots (e.g., predict ADTs for RNA-only cohorts)
* “denoised” reconstructions (use fused z to reconstruct each modality)

### 5) Predict classification heads (single legacy head or multi-head)

If you enabled classification heads, inference is typically one call:

```python
model.eval()
with torch.no_grad():
    probs = model.predict_heads(x_dict, return_probs=True)

# probs is a dict: {head_name: (B, n_classes)}
for head_name, P in probs.items():
    print(head_name, P.shape)
```

Notes:

* For the legacy single-head path, the key is `model.label_head_name` (default `"label"`).
* For multi-head configs, keys are head names (e.g. `"celltype"`, `"patient"`, `"mutation_TP53"`).

### 6) Get everything in one forward call (loss + recon + latents)

```python
out = model(x_dict, epoch=0, y=None)

print(out.keys())
# includes (typical):
# - "loss", "recon_total", "kl", "align"
# - "mu_z", "logvar_z", "z"
# - "xhat"
# - "mu_dict", "logvar_dict"
# - optional: "class_logits", "head_logits", ...
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

Example: transformer RNA encoder, MLP ADT/ATAC encoders:

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
        ),
        ModalityConfig(
            name="atac",
            input_dim=atac.n_vars,
            encoder_hidden=[128, 64],
            decoder_hidden=[64, 128],
            likelihood="gaussian",
            encoder_type="mlp",
        ),
    ],
)
```

Why bother?

* Better inductive bias for **feature interaction modeling** (within-modality)
* Tokenizers let you focus attention on the most informative features per cell (top-k)
* Provides a natural hook for interpretability (token indices + optional attention maps)

---

## Optional: Fused multimodal transformer encoder (advanced)

This is the “extra fun” mode: a **single transformer** sees **concatenated tokens from multiple modalities** and returns a **single fused posterior** `q(z|all modalities)` using a global CLS token (or mean pooling).

This is optional and does **not** replace the original workflow by default.

### Minimal config

```python
univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.0,
    gamma=1.25,
    modalities=[...],  # your usual per-modality configs still exist

    fused_encoder_type="multimodal_transformer",
    fused_modalities=("rna", "adt", "atac"),  # omit to fuse all modalities

    fused_transformer=TransformerConfig(
        d_model=256, num_heads=8, num_layers=4,
        dim_feedforward=1024, dropout=0.1, attn_dropout=0.1,
        activation="gelu", pooling="cls",   # global CLS over concatenated tokens
    ),
)
```

### What do you gain?

* A “third view” of the cell:

  * modality-specific posteriors `q(z|x_m)` (still useful for projection and diagnostics)
  * MoE/PoE fused posterior (lite/v2)
  * **token-concatenated fused transformer posterior** (when multiple modalities exist)
* Better capacity to represent **cross-modal interactions** (e.g., ADT↔RNA, ATAC↔RNA) without relying solely on posterior averaging
* Interpretability hooks:

  * **token selection meta** (e.g., top-k feature indices per cell per modality)
  * with a small optional patch to `TransformerEncoder`, you can also return **attention weights**

### How to use it well (practical tips)

* **Modality dropout during training** (randomly drop one modality sometimes) helps the model:

  * remain strong for unimodal projection
  * avoid overfitting to “always having everything”
* Keep token counts reasonable:

  * RNA/ATAC can be large → top-k tokenizers keep attention feasible
  * ADT is small → fewer tokens is usually enough
* Use fused transformer embeddings when you *have multiple modalities*,
  and per-modality embeddings when you *don’t*.

---

## Hyperparameter optimization (optional)

UniVI includes utilities for hyperparameter search:

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

* **Questions / comments:** open a GitHub **Discussion** (if enabled) or a GitHub **Issue** with the `question` label.
* **Bug reports:** open a GitHub **Issue** and include:

  * your UniVI version (`python -c "import univi; print(univi.__version__)"`)
  * minimal code to reproduce (or a short notebook snippet)
  * stack trace + OS/CUDA/PyTorch versions
* **Feature requests:** open an Issue describing the use-case + expected inputs/outputs (a tiny example is ideal).
* **PRs are welcome:** if you’re adding new modalities/likelihoods/tokenizers, please include:

  * a minimal example / notebook
  * a small test or sanity check script
  * documentation updates (README + any config fields)

---