# UniVI

[![PyPI version](https://img.shields.io/pypi/v/univi)](https://pypi.org/project/univi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/univi.svg?v=0.3.6)](https://pypi.org/project/univi/)

<picture>
  <!-- Dark mode (GitHub supports this; PyPI may ignore <source>) -->
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.3.6/assets/figures/univi_overview_dark.png">
  <!-- Light mode / fallback (works on GitHub + PyPI) -->
  <img src="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.3.6/assets/figures/univi_overview_light.png"
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
- **Optional supervised heads** for harmonized annotation and domain confusion
- **Optional transformer encoders** (per-modality and/or fused multimodal transformer posterior)
- **Token-level hooks** for interpretability (top-k indices; optional attention maps if enabled)

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
    │   ├── encoders.py                    # Modality encoders (MLP + transformer + fused transformer)
    │   ├── decoders.py                    # Likelihood-specific decoders (NB, ZINB, Gaussian, etc.)
    │   ├── transformer.py                 # Transformer blocks + encoder (+ optional attn bias support)
    │   ├── tokenizer.py                   # Tokenization configs/helpers (top-k / patch)
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
└── <run_name>/                             # user-chosen run name (often includes dataset + date)
    ├── checkpoints/                        # model/trainer state for resuming or export
    │   ├── univi_checkpoint.pt             # primary checkpoint (model + optimizer + config, if enabled)
    │   └── best.pt                         # optional: best-val checkpoint (if early stopping enabled)
    ├── eval/                               # evaluation summaries and derived plots
    │   ├── metrics.json                    # machine-readable metrics summary
    │   ├── metrics.csv                     # flat table for quick comparisons
    │   └── plots/                          # optional: UMAPs, heatmaps, and benchmark figures
    ├── embeddings/                         # optional: exported latents for downstream analysis
    │   ├── mu_z.npy                        # fused mean embedding (cells x latent_dim)
    │   ├── modality_mu/                    # per-modality embeddings q(z|x_m)
    │   │   ├── rna.npy
    │   │   ├── adt.npy
    │   │   └── atac.npy
    │   └── obs_names.txt                   # row order for embeddings (safe joins)
    ├── reconstructions/                    # optional: recon and cross-recon exports
    │   ├── rna_from_rna.npy                # denoised reconstruction
    │   ├── adt_from_adt.npy
    │   ├── adt_from_rna.npy                # cross-modal imputation example
    │   └── rna_from_atac.npy
    ├── robustness/                         # robustness experiments (frequency mismatch, DnI, etc.)
    │   ├── frequency_perturbation_results.csv
    │   ├── frequency_perturbation_plot.png
    │   ├── frequency_perturbation_plot.pdf
    │   ├── do_not_integrate_summary.csv
    │   ├── do_not_integrate_plot.png
    │   └── do_not_integrate_plot.pdf
    ├── benchmarks/                         # baseline comparisons (optionally includes Harmony, etc.)
    │   ├── results.csv
    │   ├── results.png
    │   └── results.pdf
    ├── tables/
    │   └── Supplemental_Table_S1.xlsx       # environment + hparams + dataset statistics snapshot
    └── logs/
        ├── train.log                        # training log (stdout/stderr capture)
        └── history.csv                      # per-epoch train/val traces (if enabled)
```

(Exact subfolders vary by script and flags; the layout above shows the common outputs across the pipeline.)

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
  A fused posterior (precision-weighted MoE/PoE-style by default; optional fused transformer) + per-modality recon + β·KL + γ·alignment.
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

### 3a) Reconstruction loss balancing across modalities (recommended)

In multimodal training, reconstruction losses are often **summed over features** (e.g., RNA has many more features than ADT), which can cause high-dimensional modalities to dominate gradients.

To keep modalities more balanced, `UniVIMultiModalVAE` supports **feature-dimension normalization** of reconstruction loss terms:

- For most likelihoods (`nb`, `zinb`, `poisson`, `bernoulli`, `gaussian`, `categorical`): recon loss is scaled by  
  **`1 / D**recon_dim_power`**, where `D` is the modality feature dimension.
- For `likelihood="mse"`: recon already uses `mean(dim=-1)`, so dimension normalization is **not applied again**.

Defaults:

- `recon_normalize_by_dim=True`
- `recon_dim_power=1.0` (divide by `D`)

```python
# v2/lite with recon balancing (defaults shown explicitly)
model = UniVIMultiModalVAE(
    univi_cfg,
    loss_mode="v2",
    # recon balancing
    recon_normalize_by_dim=True,
    recon_dim_power=1.0,   # try 0.5 to divide by sqrt(D)
).to(device)
````

### 3b) Per-modality reconstruction weights (optional)

You can also apply a manual weight per modality (useful if you want ADT/ATAC to “matter more” than raw dimension normalization would imply).

Set `recon_weight` on each `ModalityConfig` (defaults to `1.0` if omitted):

```python
univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.5,
    gamma=2.5,
    modalities=[
        ModalityConfig(
            "rna", rna.n_vars,
            [512, 256, 128], [128, 256, 512],
            likelihood="nb",
            recon_weight=1.0,
        ),
        ModalityConfig(
            "adt", adt.n_vars,
            [128, 64], [64, 128],
            likelihood="nb",
            recon_weight=2.0,  # emphasize ADT recon
        ),
    ],
)

model = UniVIMultiModalVAE(univi_cfg, loss_mode="v2").to(device)
```

Tip: You can combine both approaches: **dimension normalization (global)** + **per-modality weights (local tuning)**.


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

## Mixed precision (AMP)

AMP (automatic mixed precision) can reduce VRAM usage and speed up training on GPUs by running selected ops in lower precision (fp16 or bf16) while keeping numerically sensitive parts in fp32.

If your trainer supports AMP flags, prefer bf16 where available. If using fp16, gradient scaling is typically used internally to avoid underflow.

---

## Checkpointing and resuming

Training is often run on clusters, so checkpoints are treated as first-class outputs.

Typical checkpoints contain:

* model weights
* optimizer state (for faithful resumption)
* training config/model config (for reproducibility)
* optional AMP scaler state (when using fp16 AMP)

See `univi/utils/io.py` for the exact checkpoint read/write helpers used by the trainer.

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
model.set_head_label_names("celltype", list(rna.obs["celltype"].astype("category").cat.categories))
model.set_head_label_names("batch",    list(rna.obs["batch"].astype("category").cat.categories))
```

### 2) Pass `y` to the model during training

Example pattern (construct labels from arrays aligned to dataset order):

```python
celltype_codes = rna.obs["celltype"].astype("category").cat.codes.to_numpy()
batch_codes    = rna.obs["batch"].astype("category").cat.codes.to_numpy()

y = {
    "celltype": torch.tensor(celltype_codes[batch_idx], device=device),
    "batch":    torch.tensor(batch_codes[batch_idx], device=device),
}

out = model(x_dict, epoch=epoch, y=y)
loss = out["loss"]
loss.backward()
```

When labels are provided, the forward output can include:

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
* **Encode a fused posterior** (MoE/PoE by default; optional fused multimodal transformer posterior)
* **Denoise / reconstruct** inputs via the learned decoders
* **Cross-reconstruct / impute** across modalities (RNA→ADT, ATAC→RNA, etc.)
* **Evaluate alignment** (FOSCTTM, Recall@k, modality mixing, label transfer)
* **Predict supervised targets** via built-in classification heads (if enabled)
* **Inspect uncertainty** via per-modality posterior means/variances
* (Optional) **Inspect transformer token metadata** (top-k indices; attention maps when enabled)

### Fused posterior options

UniVI can produce a fused latent in two ways:

* Default: **precision-weighted MoE/PoE fusion** over per-modality posteriors
* Optional: **fused multimodal transformer posterior** (`fused_encoder_type="multimodal_transformer"`)

In both cases, the standard embedding used for plotting/neighbors is the fused mean:

```python
mu_z, logvar_z, z = model.encode_fused(x_dict, use_mean=True)
````

### 1) Encode embeddings for plotting / neighbors (built-in)

Use `encode_adata` to get either fused (MoE/PoE) or modality-specific latents directly from an AnnData.

```python
import scanpy as sc
import torch
from univi.evaluation import encode_adata

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()

# Fused latent (MoE/PoE) from a single observed modality
Z_fused = encode_adata(
    model,
    rna,
    modality="rna",
    device=device,
    layer="counts",       # or None to use .X
    latent="moe_mean",    # {"moe_mean","moe_sample","modality_mean","modality_sample"}
)

# Modality-specific latent (projection / diagnostics)
Z_rna = encode_adata(
    model,
    rna,
    modality="rna",
    device=device,
    layer="counts",
    latent="modality_mean",
)

rna.obsm["X_univi_fused"] = Z_fused
rna.obsm["X_univi_rna"] = Z_rna

sc.pp.neighbors(rna, use_rep="X_univi_fused")
sc.tl.umap(rna)
sc.pl.umap(rna, color=["celltype"], frameon=False)
```

### 2) Evaluate paired alignment (FOSCTTM, Recall@k, mixing, label transfer)

`evaluate_alignment` is a figure-ready wrapper. It can take precomputed `Z1/Z2`, or compute embeddings from AnnData via `encode_adata`.

```python
from univi.evaluation import evaluate_alignment

# For paired data, you typically pass modality-specific latents for the two modalities
res = evaluate_alignment(
    model=model,
    adata1=rna,
    adata2=adt,
    mod1="rna",
    mod2="adt",
    device=device,
    layer1="counts",
    layer2="counts",
    latent="modality_mean",
    metric="euclidean",
    recall_ks=(1, 5, 10),
    k_mixing=20,
    k_transfer=15,
    # optional label transfer inputs:
    # labels_source=rna.obs["celltype"].to_numpy(),
    # labels_target=adt.obs["celltype"].to_numpy(),
)

print(res)  # dict includes foscttm(+sem), recall@k(+sem), modality_mixing(+sem), label transfer (optional)
```

### 3) Denoise / reconstruct a modality (built-in)

`denoise_adata` runs “encode modality → decode same modality” and can write to a layer.

```python
from univi.evaluation import denoise_adata

Xhat_rna = denoise_adata(
    model,
    rna,
    modality="rna",
    device=device,
    layer="counts",
    out_layer="univi_denoised",  # writes rna.layers["univi_denoised"]
)

# Quick marker plots from denoised values:
import scanpy as sc
markers = ["TRAC", "NKG7", "LYZ", "MS4A1", "CD79A"]

rna_d = rna.copy()
rna_d.X = rna_d.layers["univi_denoised"]
sc.pl.umap(rna_d, color=markers, frameon=False, title=[f"{g} (denoised)" for g in markers])
```

### 4) Cross-modal reconstruction / imputation (built-in)

`cross_modal_predict` runs “encode src modality → decode target modality” and returns a dense numpy array.

```python
from univi.evaluation import cross_modal_predict

# Example: RNA -> predicted ADT
adt_from_rna = cross_modal_predict(
    model,
    adata_src=rna,
    src_mod="rna",
    tgt_mod="adt",
    device=device,
    layer="counts",
    batch_size=512,
    use_moe=True,   # for src-only input, MoE reduces to the src posterior
)
print(adt_from_rna.shape)  # (cells, adt_features)
```

### 5) Direct model calls (advanced / debugging)

If you want full control (or want posterior means/variances explicitly), call the model methods directly.

```python
import torch

model.eval()
batch = next(iter(val_loader))
x_dict = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    # Per-modality posteriors
    mu_dict, logvar_dict = model.encode_modalities(x_dict)

    # Fused posterior (MoE/PoE or fused transformer, depending on config)
    mu_z, logvar_z, z = model.encode_fused(x_dict, use_mean=True)

    # Decode all modalities from a chosen latent (implementation-dependent keys)
    xhat_dict = model.decode_modalities(mu_z)
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

By default, UniVI uses **MLP encoders** (`encoder_type="mlp"`), and classic workflows work unchanged.

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

Notes:

* Tokenizers focus attention on the most informative features per cell (top-k) or local structure (patching).
* Transformer encoders expose optional interpretability hooks (token indices and, when enabled, attention maps).

---

## Optional: ATAC coordinate embeddings and distance attention bias (advanced)

For top-k tokenizers, UniVI can optionally incorporate genomic context:

* **Coordinate embeddings**: chromosome embedding + coordinate MLP per selected feature
* **Distance-based attention bias**: encourages attention between nearby peaks (same chromosome)

### Enable in the tokenizer config (ATAC example)

```python
TokenizerConfig(
    mode="topk_channels",
    n_tokens=512,
    channels=("value","rank","dropout"),
    use_coord_embedding=True,
    n_chroms=<num_chromosomes>,
    coord_scale=1e-6,
)
```

### Attach coordinates and configure distance bias via `UniVITrainer`

If your `UniVITrainer` supports `feature_coords` and `attn_bias_cfg`, you can attach genomic coordinates once and let the trainer build the bias for you:

```python
feature_coords = {
    "atac": {
        "chrom_ids": chrom_ids_long,   # (F,)
        "start": start_bp,             # (F,)
        "end": end_bp,                 # (F,)
    }
}

attn_bias_cfg = {
    "atac": {
        "type": "distance",
        "lengthscale_bp": 50_000.0,
        "same_chrom_only": True,
    }
}

trainer = UniVITrainer(
    model,
    train_loader,
    val_loader=val_loader,
    train_cfg=TrainingConfig(...),
    device="cuda",
    feature_coords=feature_coords,
    attn_bias_cfg=attn_bias_cfg,
)
trainer.fit()
```

This path keeps the model code clean and makes the feature-coordinate plumbing consistent across runs.

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
* If `fused_require_all_modalities=True` and a fused modality is missing at inference, UniVI falls back to MoE/PoE fusion.

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

