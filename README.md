# UniVI

[![PyPI version](https://img.shields.io/pypi/v/univi?v=0.4.1)](https://pypi.org/project/univi/)
[![pypi downloads](https://img.shields.io/pepy/dt/univi?label=pypi%20downloads)](https://pepy.tech/project/univi)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/univi?cacheSeconds=300)](https://anaconda.org/conda-forge/univi)
[![conda-forge downloads](https://img.shields.io/conda/dn/conda-forge/univi?label=conda-forge%20downloads\&cacheSeconds=300)](https://anaconda.org/conda-forge/univi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/univi.svg?v=0.4.1)](https://pypi.org/project/univi/)

<picture>
  <!-- Dark mode (GitHub supports this; PyPI may ignore <source>) -->
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.4.1/assets/figures/univi_overview_dark.png">
  <!-- Light mode / fallback (works on GitHub + PyPI) -->
  <img src="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.4.1/assets/figures/univi_overview_light.png"
       alt="UniVI overview and evaluation roadmap"
       width="100%">
</picture>

**UniVI** is a **multi-modal variational autoencoder (VAE)** framework for aligning and integrating single-cell modalities such as **RNA**, **ADT (CITE-seq)**, and **ATAC**.

It’s designed for experiments like:

* **Joint embedding** of paired multimodal data (CITE-seq, Multiome, TEA-seq)
* **Zero-shot projection** of external unimodal cohorts into a paired “bridge” latent
* **Cross-modal reconstruction / imputation** (RNA→ADT, ATAC→RNA, etc.)
* **Denoising** via learned generative decoders
* **Evaluation** (FOSCTTM, Recall@k, modality mixing/entropy, label transfer, fused-space clustering)
* **Optional supervised heads** for harmonized annotation and domain confusion
* **Optional transformer encoders** (per-modality and/or fused multimodal transformer posterior)
* **Token-level hooks** for interpretability (top-k indices; optional attention maps if enabled)

---

## Preprint

If you use UniVI in your work, please cite:

> Ashford AJ, Enright T, Somers J, Nikolova O, Demir E.  
> **Unifying multimodal single-cell data with a mixture-of-experts β-variational autoencoder framework.**  
> *bioRxiv* (2025; updated 2026). doi: [10.1101/2025.02.28.640429](https://doi.org/10.1101/2025.02.28.640429)

```bibtex
@article{Ashford2025UniVI,
  title   = {Unifying multimodal single-cell data with a mixture-of-experts β-variational autoencoder framework},
  author  = {Ashford, A. J. and Enright, T. and Somers, J. and Nikolova, O. and Demir, E.},
  journal = {bioRxiv},
  date    = {2025},
  doi     = {10.1101/2025.02.28.640429},
  url     = {https://www.biorxiv.org/content/10.1101/2025.02.28.640429},
  note    = {Preprint (updated 2026)}
}
````

---

## Installation

### PyPI

```bash
pip install univi
```

> **Note:** UniVI requires PyTorch. If `import torch` fails, install PyTorch for your platform/CUDA from PyTorch’s official install instructions.

### Conda / mamba

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

## Data expectations

UniVI expects **per-modality AnnData** objects. For paired settings, modalities should share the same cells:

* Each modality is an `AnnData`
* Paired modalities have the same `obs_names` (same cells, same order)
* Raw counts often live in `.layers["counts"]`
* A model-ready representation lives in `.X` (or `.obsm["X_*"]` for ATAC LSI)

You can keep multiple representations around:

* `.layers["counts"]` = raw
* `.X` = model input (e.g., log1p normalized RNA, CLR ADT, LSI ATAC, etc.)
* `.layers["denoised_*"]` / `.layers["imputed_*"]` = UniVI outputs

---

## Quickstart (Python / Jupyter)

This is the “notebook path”: load paired AnnData → train → encode → evaluate/plot.

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

adata_dict = align_paired_obs_names({"rna": rna, "adt": adt})
```

### 2) Dataset + dataloaders

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MultiModalDataset(
    adata_dict=adata_dict,
    X_key="X",     # uses .X as model input
    device=None,   # dataset yields CPU tensors; model moves to GPU
)

n = rna.n_obs
idx = np.arange(n)
rng = np.random.default_rng(0)
rng.shuffle(idx)
split = int(0.8 * n)
train_idx, val_idx = idx[:split], idx[split:]

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=256, shuffle=True,  num_workers=0)
val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=256, shuffle=False, num_workers=0)
```

### 3) Model config + train

```python
univi_cfg = UniVIConfig(
    latent_dim=40,
    beta=1.5,
    gamma=2.5,
    encoder_dropout=0.1,
    decoder_dropout=0.0,
    modalities=[
        # likelihood could also be: "nb", "zinb", "poisson", "mse", etc.
        # depending on closest modality input distribution
        ModalityConfig(
            "rna",
            rna.n_vars,
            [512, 256, 128],
            [128, 256, 512],
            likelihood="gaussian",
        ),
        ModalityConfig(
            "adt",
            adt.n_vars,
            [128, 64],
            [64, 128],
            likelihood="gaussian",
        ),
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
    best_epoch_warmup=50,  # in UniVI v0.4.1+
    patience=50,
)

model = UniVIMultiModalVAE(
    univi_cfg,
    loss_mode="v1",                # or "v2"
    v1_recon="avg",
    normalize_v1_terms=True,
).to(device)

trainer = UniVITrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    train_cfg=train_cfg,
    device=device,
)

trainer.fit()
```

---

## After training: evaluation + plotting workflows

UniVI ships two post-training workhorse modules:

* `univi.evaluation`: encoding, denoising, cross-modal prediction, alignment metrics, optional MoE gate extraction
* `univi.plotting`: Scanpy/Matplotlib utilities (UMAP panels with compact legends, modality-stacked UMAPs, raw-vs-denoised overlays, confusion matrices, MoE gate plots)

### 0) Imports + plotting defaults

```python
import numpy as np
import scipy.sparse as sp

from univi.evaluation import (
    encode_adata,
    cross_modal_predict,
    denoise_from_multimodal,
    denoise_adata,
    evaluate_alignment,
    encode_moe_gates_from_tensors,
)
from univi.plotting import (
    set_style,
    umap,
    umap_by_modality,
    compare_raw_vs_denoised_umap_features,
    plot_confusion_matrix,
    write_gates_to_obs,
    plot_moe_gate_summary,
)

set_style(font_scale=1.2, dpi=150)
device = "cuda"  # or "cpu"
```

---

### 1) Encode latents and store them in `.obsm["X_univi"]`

`encode_adata(...)` is designed for “one observed modality at a time” (RNA-only, ADT-only, etc.):

```python
Z_rna = encode_adata(
    model,
    adata=rna,
    modality="rna",
    device=device,
    layer=None,         # reads adata.X by default; set layer="counts" if your encoder expects counts
    X_key="X",
    batch_size=1024,
    latent="moe_mean",  # {"moe_mean","moe_sample","modality_mean","modality_sample"}
    random_state=0,
)
rna.obsm["X_univi"] = Z_rna

Z_adt = encode_adata(
    model,
    adata=adt,
    modality="adt",
    device=device,
    layer=None,
    X_key="X",
    batch_size=1024,
    latent="moe_mean",
    random_state=0,
)
adt.obsm["X_univi"] = Z_adt
```

Plot UMAPs from the stored embedding (new API; compact legends by default):

```python
umap(
    rna,
    obsm_key="X_univi",
    color=["celltype.l2", "batch"],
    legend="outside",           # "outside" (recommended), "right_margin", "on_data", "none"
    legend_subset_topk=25,      # optional: show top-k categories by frequency in legend
    savepath="umap_rna_univi.png",
    show=False,
)

umap(
    adt,
    obsm_key="X_univi",
    color=["celltype.l2", "batch"],
    legend="outside",
    legend_subset_topk=25,
    savepath="umap_adt_univi.png",
    show=False,
)
```

---

### 2) Plot modality mixing / co-embedding across modalities

If each modality AnnData has `.obsm["X_univi"]`, you can concatenate and color by modality via `univi_modality`:

```python
umap_by_modality(
    {"rna": rna, "adt": adt},
    obsm_key="X_univi",
    color=["celltype.l2", "univi_modality"],
    legend="outside",
    savepath="umap_rna_adt_by_modality.png",
    show=False,
)
```

---

### 3) Cross-modal prediction (imputation): encode source → decode target

Example: **RNA → ADT** imputation, stored in an ADT layer:

```python
adt_hat_from_rna = cross_modal_predict(
    model,
    adata_src=rna,
    src_mod="rna",
    tgt_mod="adt",
    device=device,
    layer=None,      # uses rna.X by default
    X_key="X",
    batch_size=512,
    use_moe=True,
)

adt.layers["imputed_from_rna"] = adt_hat_from_rna
```

---

### 4) True multimodal denoising (fused latent) and write back to AnnData

If you have multiple observed modalities for the same cells, you can denoise through the fused latent.

```python
def to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)
```

**Option A — provide tensors directly (`denoise_from_multimodal`)**:

```python
rna_denoised = denoise_from_multimodal(
    model,
    x_dict={"rna": to_dense(rna.X), "adt": to_dense(adt.X)},
    target_mod="rna",
    device=device,
    batch_size=512,
    use_mean=True,
)

rna.layers["denoised_fused"] = rna_denoised
```

**Option B — let UniVI pull matrices and write outputs (`denoise_adata` with `adata_by_mod=...`)**:

```python
denoise_adata(
    model,
    adata=rna,                  # output written here
    modality="rna",
    device=device,
    out_layer="denoised_fused",
    overwrite_X=False,
    batch_size=512,
    adata_by_mod={"rna": rna, "adt": adt},
    layer_by_mod={"rna": None, "adt": None},   # None -> use .X
    X_key_by_mod={"rna": "X", "adt": "X"},
    use_mean=True,
)
```

---

### 5) Raw vs denoised feature overlays on the same UMAP

Once you have a denoised layer, you can compare UMAP marker overlays in a 2-row grid:

```python
compare_raw_vs_denoised_umap_features(
    rna,
    obsm_key="X_univi",
    features=["MS4A1", "CD3D", "NKG7"],   # must be in rna.var_names
    raw_layer=None,                      # None -> adata.X
    denoised_layer="denoised_fused",     # must exist in adata.layers
    savepath="umap_raw_vs_denoised_markers.png",
    show=False,
)
```

---

### 6) Alignment evaluation (FOSCTTM, Recall@k, mixing/entropy, label transfer)

`evaluate_alignment(...)` returns a figure-ready metrics dict:

```python
metrics = evaluate_alignment(
    Z1=rna.obsm["X_univi"],
    Z2=adt.obsm["X_univi"],
    metric="euclidean",
    recall_ks=(1, 5, 10),
    k_mixing=20,
    k_entropy=30,
    labels_source=rna.obs["celltype.l2"].to_numpy(),
    labels_target=adt.obs["celltype.l2"].to_numpy(),
    compute_bidirectional_transfer=True,
    k_transfer=15,
    json_safe=True,
)

print("FOSCTTM:", metrics["foscttm"], "+/-", metrics["foscttm_sem"])
print("Recall@10:", metrics["recall_at_10"], "+/-", metrics["recall_at_10_sem"])
print("Mixing:", metrics["modality_mixing"], "+/-", metrics["modality_mixing_sem"])
print("Entropy:", metrics["modality_entropy"], "+/-", metrics["modality_entropy_sem"])
print("Worst-direction macro-F1:", metrics["bidirectional_transfer"]["worst_direction_macro_f1"])
```

Plot the label-transfer confusion matrix:

```python
plot_confusion_matrix(
    np.asarray(metrics["label_transfer_cm"]),
    labels=np.asarray(metrics["label_transfer_label_order"]),
    title="Label transfer (RNA → ADT)",
    normalize="true",                     # None / "true" / "pred"
    savepath="label_transfer_confusion.png",
    show=False,
)
```

---

### 7) (Optional) MoE gating weights: extract + plot

If your model supports `model.mixture_of_experts(..., return_weights=True)` (v0.4.1+), you can inspect per-cell modality reliance.

```python
gate = encode_moe_gates_from_tensors(
    model,
    x_dict={"rna": to_dense(rna.X), "adt": to_dense(adt.X)},
    device=device,
    batch_size=1024,
    modality_order=["rna", "adt"],
    kind="effective_precision",  # recommended: contribution to fused posterior
    return_logits=True,
)

W = gate["weights"]             # (n_cells, n_modalities)
mods = gate["modality_order"]   # ["rna","adt"]
print("Gate means:", gate["per_modality_mean"])
```

Write gates to `.obs` and plot summaries:

```python
write_gates_to_obs(rna, gates=W, modality_names=mods, prefix="moe_gate_")

plot_moe_gate_summary(
    rna,
    gate_prefix="moe_gate_",
    groupby=None,
    savepath="moe_gates_all_cells.png",
    show=False,
)

plot_moe_gate_summary(
    rna,
    gate_prefix="moe_gate_",
    groupby="celltype.l2",
    kind="meanbar",
    max_groups=25,
    savepath="moe_gates_by_celltype.png",
    show=False,
)
```

You can also include gating summaries inside `evaluate_alignment(...)`:

```python
metrics_with_gates = evaluate_alignment(
    Z1=rna.obsm["X_univi"],
    Z2=adt.obsm["X_univi"],
    gate_weights=W,
    gate_modality_order=mods,
    gate_kind=gate.get("kind", None),
    json_safe=True,
)
```

---

## Advanced topics

### Training objectives (v1 vs v2/lite)

* **v1 (“paper”)**: per-modality posteriors + reconstruction scheme (cross/self/avg) + posterior alignment across modalities
* **v2/lite**: fused posterior (MoE/PoE-style by default; optional fused transformer) + per-modality recon + β·KL + γ·alignment (L2 on latent means)

Choose via `loss_mode` at construction time (Python) or config JSON (scripts).

### Optional: transformer encoders and fused multimodal transformer posterior

UniVI can swap MLP encoders for transformers, and can optionally build a **fused transformer posterior** that sees tokens across modalities (when enabled).

(Kept out of the main flow on purpose — see `univi/models/` + notebooks for full examples.)

### Supervised heads and categorical “label modalities”

UniVI supports:

* **classification heads** (predict labels from latent; optionally adversarial/GRL)
* **categorical modalities** (labels as a generative modality with encoder+decoder)

These are great for harmonized annotation / confounding checks / semi-supervision, but are intentionally “advanced” relative to the core train→evaluate loop.

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
├── notebooks/                             # Jupyter notebook analyses to reproduce figures from our revised manuscript (in progress for Genome Research)
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

## License

MIT License — see `LICENSE`.

---

## Contact, questions, and bug reports

* **Questions / comments:** open a GitHub Issue with the `question` label (or use Discussions)
* **Bug reports:** include:

  * UniVI version: `python -c "import univi; print(univi.__version__)"`
  * a minimal notebook/code snippet
  * stack trace + OS/CUDA/PyTorch versions

