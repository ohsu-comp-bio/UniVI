# UniVI

[![PyPI version](https://img.shields.io/pypi/v/univi?v=0.3.9)](https://pypi.org/project/univi/)
[![pypi downloads](https://img.shields.io/pepy/dt/univi?label=pypi%20downloads)](https://pepy.tech/project/univi)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/univi?cacheSeconds=300)](https://anaconda.org/conda-forge/univi)
[![conda-forge downloads](https://img.shields.io/conda/dn/conda-forge/univi?label=conda-forge%20downloads&cacheSeconds=300)](https://anaconda.org/conda-forge/univi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/univi.svg?v=0.3.9)](https://pypi.org/project/univi/)

<picture>
  <!-- Dark mode (GitHub supports this; PyPI may ignore <source>) -->
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.3.9/assets/figures/univi_overview_dark.png">
  <!-- Light mode / fallback (works on GitHub + PyPI) -->
  <img src="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.3.9/assets/figures/univi_overview_light.png"
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

> Ashford AJ, Enright T, Somers J, Nikolova O, Demir E.
> **Unifying multimodal single-cell data with a mixture-of-experts β-variational autoencoder framework.**
> *bioRxiv* (2025; updated 2026). doi: [10.1101/2025.02.28.640429](https://doi.org/10.1101/2025.02.28.640429)

```bibtex
@article{Ashford2025UniVI,
  title   = {Unifying multimodal single-cell data with a mixture-of-experts β-variational autoencoder framework},
  author  = {Ashford, Andrew J. and Enright, Trevor and Somers, Julia and Nikolova, Olga and Demir, Emek},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.02.28.640429},
  url     = {https://www.biorxiv.org/content/10.1101/2025.02.28.640429},
  note    = {Preprint (updated 2026)}
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
├── notebooks/                             # Jupyter notebook analyses to reproduce every figure from our revised manuscript (in progress for Genome Research)
│   ├── UniVI_manuscript_GR-Figure__2__CITE_paired.ipynb
│   ├── UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb
│   ├── UniVI_manuscript_GR-Figure__4__Multiome_paired.ipynb
│   ├── UniVI_manuscript_GR-Figure__5__Multiome_bridge_mapping_and_fine-tuning.ipynb
│   ├── UniVI_manuscript_GR-Figure__6__TEA-seq_tri-modal.ipynb
│   ├── UniVI_manuscript_GR-Figure__7__AML_bridge_mapping_and_fine-tuning.ipynb
│   ├── UniVI_manuscript_GR-Figure__8__benchmarking_against_pytorch_tools.ipynb
│   ├── UniVI_manuscript_GR-Figure__8__benchmarking_against_R_tools.ipynb
│   ├── UniVI_manuscript_GR-Figure__8__benchmarking_merging_and_plotting_runs.ipynb
│   ├── UniVI_manuscript_GR-Figure__9__paired_data_ablation_and_computational_scaling_performance.ipynb
│   ├── UniVI_manuscript_GR-Figure__9__paired_data_ablation_and_computational_scaling_performance_compile_plots_from_results_df.ipynb
│   ├── UniVI_manuscript_GR-Figure_10__cell_population_ablation_MoE.ipynb
│   ├── UniVI_manuscript_GR-Figure_10__cell_population_ablation_MoE_compile_plots_from_results_df.ipynb
│   ├── UniVI_manuscript_GR-Supple_____grid-sweep.ipynb
│   └── UniVI_manuscript_GR-Supple_____grid-sweep_compile_plots_from_results_df.ipynb
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
    │   ├── run_atac_hparam_search.py
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

## Installation

### Install via PyPI

```bash
pip install univi
```

> **Note:** UniVI requires `torch`. If `import torch` fails, install PyTorch for your platform/CUDA from:
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Install via conda / mamba

```bash
conda install -c conda-forge univi
# or
mamba install -c conda-forge univi
```

### Development install (from source)

```bash
git clone https://github.com/Ashford-A/UniVI.git
cd UniVI

conda env create -f envs/univi_env.yml
conda activate univi_env

pip install -e .
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
  A fused posterior (precision-weighted MoE/PoE-style by default; optional fused transformer) + per-modality recon + β·KL + γ·alignment (where alignment is the L2-normed latent means instead of the cross-modal KL term seen in v1).
  Convenient for 3+ modalities and “loosely paired” settings, but with the tradeoff of a weaker one-to-one latent correspondence versus v1.

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

### 3a) Reconstruction loss balancing across modalities (optional)

In multimodal training, reconstruction losses are often **summed over features** (e.g., RNA has many more features than ADT), which can cause high-dimensional modalities to dominate gradients.

To keep modalities more balanced, `UniVIMultiModalVAE` supports **feature-dimension normalization** of reconstruction loss terms:

* For most likelihoods (`nb`, `zinb`, `poisson`, `bernoulli`, `gaussian`, `categorical`): recon loss is scaled by
  **`1 / D**recon_dim_power`**, where `D` is the modality feature dimension.
* For `likelihood="mse"`: recon already uses `mean(dim=-1)`, so dimension normalization is **not applied again**.

Defaults:

* v1: `recon_normalize_by_dim=False`
* v2/lite: `recon_normalize_by_dim=True`, `recon_dim_power=1.0`

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

## Classification (customizable heads)

UniVI supports **in-model supervised heads** to predict labels from the latent space. This is useful for:

* harmonized cell-type annotation (bridge → projected cohorts)
* batch/tech/patient prediction (sanity checks, confounding)
* adversarial domain confusion via gradient reversal (GRL)
* multi-task setups (e.g., celltype + patient + mutation flags)
* lightweight supervision during training without a separate downstream classifier

### Key ideas

* Heads are configured via `UniVIConfig.class_heads` using `ClassHeadConfig`.
* You can define **any number of heads**.
* Each head can be:

  * **categorical** (multi-class softmax / cross-entropy), or
  * **binary** (sigmoid / BCE) if your implementation supports `head_type="binary"` (or just use `n_classes=2` categorical).
* Each head can use a **custom MLP** (depth/width) and its own hyperparameters.
* Training targets are passed as `y`, a dict: **`{head_name: labels}`**.
* Missing labels are masked using `ignore_index` (default `-1`).
* Heads can be delayed via `warmup` and weighted via `loss_weight`.
* Set `adversarial=True` to apply GRL to that head (domain confusion).

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
        # Categorical head (multi-class cell type)
        ClassHeadConfig(
            name="celltype",
            n_classes=int(rna.obs["celltype"].astype("category").cat.categories.size),
            loss_weight=1.0,
            ignore_index=-1,
            from_mu=True,
            warmup=0,
            # Optional (supported in v0.3.9+):
            # head_type="categorical",
            # hidden_dims=(256, 128),
            # dropout=0.1,
            # batchnorm=False,
            # activation="relu",
        ),

        # Adversarial categorical head (domain confusion)
        ClassHeadConfig(
            name="batch",
            n_classes=int(rna.obs["batch"].astype("category").cat.categories.size),
            loss_weight=0.2,
            ignore_index=-1,
            from_mu=True,
            warmup=10,
            adversarial=True,
            adv_lambda=1.0,
            # Optional customization (supported in v0.3.9+):
            # hidden_dims=(128,),
        ),

        # Binary head (e.g. mutation flag) — either true binary (BCE) or categorical with n_classes=2
        ClassHeadConfig(
            name="TP53_mut",
            n_classes=2,           # or use head_type="binary" in configs if supported
            loss_weight=0.5,
            ignore_index=-1,
            from_mu=True,
            warmup=0,
            # Optional customization (supported in v0.3.9+):
            # head_type="binary",
            # hidden_dims=(128, 64),
        ),
    ],
)
```

Optional: attach readable label names for decoding later:

```python
model.set_head_label_names("celltype", list(rna.obs["celltype"].astype("category").cat.categories))
model.set_head_label_names("batch",    list(rna.obs["batch"].astype("category").cat.categories))
```

### 2) Pass `y` to the model during training

```python
celltype_codes = rna.obs["celltype"].astype("category").cat.codes.to_numpy()
batch_codes    = rna.obs["batch"].astype("category").cat.codes.to_numpy()

# Example: binary flag (0/1), with -1 for unknown
tp53 = rna.obs["TP53_mut"].to_numpy().astype(int)     # assume 0/1
tp53 = np.where(np.isnan(tp53), -1, tp53)

y = {
    "celltype": torch.tensor(celltype_codes[batch_idx], device=device),
    "batch":    torch.tensor(batch_codes[batch_idx], device=device),
    "TP53_mut": torch.tensor(tp53[batch_idx], device=device),
}

out = model(x_dict, epoch=epoch, y=y)
loss = out["loss"]
loss.backward()
```

When labels are provided, the forward output can include:

* `out["head_logits"]`: dict of logits per head

  * categorical: `(B, n_classes)`
  * binary (BCE): `(B,)` or `(B, 1)` depending on your implementation
* `out["head_losses"]`: mean loss per head (masked by `ignore_index`)

### 3) Predict heads after training

```python
model.eval()
batch = next(iter(val_loader))
x_dict = {k: v.to(device) for k, v in batch.items()}

with torch.no_grad():
    probs = model.predict_heads(x_dict, return_probs=True)

for head_name, P in probs.items():
    print(head_name, P.shape)
```

### 4) Inspect head metadata

```python
meta = model.get_classification_meta()
print(meta)
```

---

## Categorical variables as a modality (discrete encoder/decoder)

In addition to “heads”, UniVI can treat certain **categorical variables as a full modality**, meaning:

* it has its **own encoder** `q(z|y)` (from discrete inputs),
* it has its **own decoder** `p(y|z)` (predicting the categories),
* and it can participate in fusion/alignment like any other modality.

This can be useful when:

* you want labels to behave like an “anchor modality”
* you want to **inject** label information into the fused posterior
* you want to do **semi-supervised** training where some cells have labels and others do not

### How it works (high-level)

If a modality has `likelihood="categorical"` (or equivalent), UniVI interprets its input as either:

* **integer class indices** of shape `(B,)` or `(B, 1)`, with unlabeled using `ignore_index` (default `-1`), or
* **one-hot** vectors `(B, C)` (optionally sparse/soft), where all-zero rows are treated as unlabeled

Internally, categorical inputs may be converted into one-hot for the encoder, and decoded using cross-entropy.

### Example: add a categorical modality

```python
univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.5,
    gamma=2.5,
    modalities=[
        ModalityConfig("rna", rna.n_vars, [512,256,128], [128,256,512], likelihood="nb"),
        ModalityConfig("adt", adt.n_vars, [128,64],      [64,128],      likelihood="nb"),

        # categorical modality (e.g. coarse cell type)
        ModalityConfig(
            name="celltype_mod",
            input_dim=int(rna.obs["celltype"].astype("category").cat.categories.size),
            encoder_hidden=[128, 64],
            decoder_hidden=[64, 128],
            likelihood="categorical",
            # Optional: ignore_index=-1
        ),
    ],
)
```

### Provide categorical modality data via `x_dict`

If your training loop/dataset returns `x_dict`, include:

* `x_dict["celltype_mod"]` as:

  * `(B,)` integer labels, or
  * `(B, 1)` integer labels, or
  * `(B, C)` one-hot

Unlabeled should be `-1` (or the configured `ignore_index`).

This is separate from `y` (classification heads). Think of categorical modalities as part of the generative model, not auxiliary prediction heads.

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
            tokenizer=TokenizerConfig(mode="topk_scalar", n_tokens=min(32, adt.n_vars)),
        ),
    ],
)
```

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
    modalities=[...],
    fused_encoder_type="multimodal_transformer",
    fused_modalities=("rna", "adt", "atac"),
    fused_transformer=TransformerConfig(
        d_model=256, num_heads=8, num_layers=4,
        dim_feedforward=1024, dropout=0.1, attn_dropout=0.1,
        activation="gelu", pooling="cls",
    ),
)
```

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

* **Questions / comments:** open a GitHub Issue with the `question` label (or make a post in the Discussion thread).
* **Bug reports:** open a GitHub Issue and include:

  * your UniVI version: `python -c "import univi; print(univi.__version__)"`
  * minimal code to reproduce (or a short notebook snippet)
  * stack trace + OS/CUDA/PyTorch versions
* **Feature requests:** open an Issue describing the use-case + expected inputs/outputs (a tiny example is ideal).

