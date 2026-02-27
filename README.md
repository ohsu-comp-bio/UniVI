# UniVI

[![PyPI version](https://img.shields.io/pypi/v/univi?v=0.4.7)](https://pypi.org/project/univi/)
[![pypi downloads](https://img.shields.io/pepy/dt/univi?label=pypi%20downloads)](https://pepy.tech/project/univi)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/univi?cacheSeconds=300)](https://anaconda.org/conda-forge/univi)
[![conda-forge downloads](https://img.shields.io/conda/dn/conda-forge/univi?label=conda-forge%20downloads&cacheSeconds=300)](https://anaconda.org/conda-forge/univi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/univi.svg?v=0.4.7)](https://pypi.org/project/univi/)

<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.4.7/assets/figures/univi_overview_dark.png">
  <img src="https://raw.githubusercontent.com/Ashford-A/UniVI/v0.4.7/assets/figures/univi_overview_light.png"
       alt="UniVI overview and evaluation roadmap"
       width="100%">
</picture>

**UniVI** is a **multi-modal variational autoencoder (VAE)** toolkit for aligning and integrating single-cell modalities such as **RNA**, **ADT (CITE-seq)**, **ATAC**, and **coverage-aware / proportion-like assays** (e.g., **single-cell methylome** features).

Common use cases:

- **Joint embedding** of paired multimodal data (CITE-seq, Multiome, TEA-seq)
- **Bridge mapping / projection** of unimodal cohorts into a paired latent
- **Cross-modal imputation** (RNA→ADT, ATAC→RNA, RNA→methylome, …)
- **Denoising / reconstruction** with likelihood-aware decoders
- **Evaluation** (FOSCTTM, Recall@k, mixing/entropy, label transfer, clustering)
- Optional **supervised heads**, **MoE gating diagnostics**, and **transformer encoders**

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

> UniVI requires PyTorch. If `import torch` fails, install PyTorch for your platform/CUDA from PyTorch’s official install instructions.

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

UniVI expects **per-modality AnnData** objects.

* Each modality is an `AnnData`
* For paired settings, modalities share the same cells (`obs_names`, same order)
* Raw counts often live in `.layers["counts"]`
* Model inputs typically live in `.X` (or `.obsm["X_*"]` for ATAC LSI)
* Model input is a dictionary of these `AnnData` objects with the dictionary key specifying the modality (e.g. `rna`, `adt`, `atac`). These keys are used later for evaluation functions (cross-reconstruction etc.).

Recommended convention:

* `.layers["counts"]` = raw counts / raw signal
* `.X` / `.obsm["X_*"]` = model input space (log1p RNA, CLR ADT, LSI ATAC, methyl fractions, etc.)
* `.layers["denoised_*"]` / `.layers["imputed_*"]` = UniVI outputs

---

## Quickstart (Python / Jupyter)

Minimal “notebook path”: load paired AnnData → preprocess data → train → encode/evaluate → plot.

```python
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Subset

from univi import UniVIMultiModalVAE, ModalityConfig, UniVIConfig, TrainingConfig
from univi.data import MultiModalDataset, align_paired_obs_names, collate_multimodal_xy_recon
from univi.trainer import UniVITrainer
```

### 1a) Load paired AnnData

For CITE-seq data
```python
rna = sc.read_h5ad("path/to/rna_citeseq.h5ad")
adt = sc.read_h5ad("path/to/adt_citeseq.h5ad")
```
or for Multiome data (etc.)
```python
rna = sc.read_h5ad("path/to/rna_multiome.h5ad")
atac = sc.read_h5ad("path/to/atac_multiome.h5ad")
```

### 1b) Preprocess each data type as desired

> Note: Make sure to use the appropriate modality decoder distribution in step 3 for your specific data preprocessing steps. See Step 3 `ModalityConfig(likeliehood=..)` input notes for more details.

RNA
```python
# Conventions:
# - keep raw counts in .layers["counts"]
# - set .X to the model input space (e.g., RNA log1p; ADT CLR(+scaled))

# RNA example: log-normalized + HVGs + scale
rna.layers["counts"] = rna.X.copy()  # if raw counts stored in .X, otherwise can try rna.raw.X or similar

# (optional QC metrics)
rna.var["mt"] = rna.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
rna.raw = rna  # snapshot log-space for plotting/DE

sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=2000, subset=True)
sc.pp.scale(rna, max_value=10)
```
ADT
```python
# ADT example: CLR (per-cell) + scale (per-protein)
adt.layers["counts"] = adt.X.copy()  # if raw counts stored in .X, otherwise can try adt.raw.X or similar

def clr_per_cell(X):
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    logX = np.log1p(X)
    return logX - logX.mean(axis=1, keepdims=True)

adt.X = clr_per_cell(adt.layers["counts"])
sc.pp.scale(adt, zero_center=True, max_value=10)
```
ATAC
```python
# ATAC example: TF-IDF -> LSI (store counts in .layers["counts"], put LSI in .obsm["X_lsi"])
atac.layers["counts"] = atac.X.copy()  # if raw counts stored in .X, otherwise can try atac.raw.X or similar

def tfidf(X):
    # X: cells x peaks (counts; sparse ok)
    X = X.tocsr() if hasattr(X, "tocsr") else X

    # TF: normalize each cell by its total counts
    cell_sum = np.asarray(X.sum(axis=1)).ravel()
    cell_sum[cell_sum == 0] = 1.0
    tf = X.multiply(1.0 / cell_sum[:, None])

    # IDF: log(1 + n_cells / (1 + df))
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log1p(X.shape[0] / (1.0 + df))

    return tf.multiply(idf)

X_tfidf = tfidf(atac.layers["counts"])

# LSI via truncated SVD
from sklearn.decomposition import TruncatedSVD  # (keep import local if you want)
svd = TruncatedSVD(n_components=50, random_state=0)
X_lsi = svd.fit_transform(X_tfidf)

# Common convention: drop the first LSI component (often correlates with depth)
atac.obsm["X_lsi"] = X_lsi[:, 1:]

# (optional) you can use this as the model input via X_key="obsm:X_lsi" (depending on your dataset wrapper)
# or keep .X as counts and point UniVI to the obsm key for ATAC inputs.

# (optional alignment sanity check)
assert rna.n_obs == adt.n_obs and np.all(rna.obs_names == adt.obs_names)
```
Put preprocessed per-modality AnnData(s) into dictionary for model (CITE-seq data):
```python
# Put data into `adata_dict` for downstream workflow
adata_dict = align_paired_obs_names({"rna": rna, "adt": adt})
```
or for Multiome data:
```python
# Put data into `adata_dict` for downstream workflow
adata_dict = align_paired_obs_names({"rna": rna, "atac": atac})
```
or for tri-modal data covering RNA+ADT+ATAC(e.g. TEA-seq, DOGMA-seq, ASAP-seq):
```python
# Put data into `adata_dict` for downstream workflow
adata_dict = align_paired_obs_names({"rna": rna, "adt": adt, "atac": atac})
```
or if unimodal VAE use-case (etc.):
```python
# Put data into `adata_dict` for downstream workflow
adata_dict = align_paired_obs_names({"atac": atac})
```

> Note: If you want to use UniVI inductively and avoid data leakage, apply feature selection, scaling,
> and any "learned" transforms (e.g. PCA, LSI) to the training set only, then apply the results elucidated 
> from the training set to the validation and test sets.

### 2) Dataset + dataloaders (MultiModalDataset option)

```python
device = (
    "cuda" if torch.cuda.is_available() else
    ("mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else
     ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else
      "cpu"))
)

dataset = MultiModalDataset(
    adata_dict=adata_dict,
    device=None,              # dataset yields CPU tensors; model moves to GPU
    X_key_by_mod={
        "rna" : "X",          # uses rna.X as model input
        "adt" : "X",          # uses adt.X as model input
        "atac": "obsm:X_lsi"  # uses atac.obsm["X_lsi"] as model input, can replace with "X" if desired features stored there
    },
)

n = rna.n_obs
idx = np.arange(n)
rng = np.random.default_rng(0)
rng.shuffle(idx)

n_train = int(0.8 * n)
n_val   = int(0.1 * n)

train_idx = idx[:n_train]
val_idx   = idx[n_train:n_train + n_val]
test_idx  = idx[n_train + n_val:]

# For reproducibility later, can save indices used for splits
np.savez(
    "splits_rna_adt_seed0.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
)

# Build train/val/test loaders
train_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=256,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_multimodal_xy_recon,
)

val_loader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=256,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_multimodal_xy_recon,
)

test_loader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=256,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_multimodal_xy_recon,
)
```

### 3) Model config + train

```python
univi_cfg = UniVIConfig(
    latent_dim=30,
    beta=1.15,
    gamma=3.25,
    encoder_dropout=0.10,
    decoder_dropout=0.05,
    encoder_batchnorm=True,
    decoder_batchnorm=False,
    kl_anneal_start=50,
    kl_anneal_end=100,
    align_anneal_start=75,
    align_anneal_end=125,
    modalities=[
        #
        # NOTE:
        # Manuscript-style "gaussian" decoders on normalized feature spaces often produce the most
        # cell-to-cell aligned latent spaces for integration-focused use cases. For some assay types
        # (including methylome), a more distribution-matched likelihood may be preferable depending
        # on whether your goal is alignment vs calibrated reconstruction/generation.
        #
        ModalityConfig(
            name="rna",
            input_dim=rna.n_vars,
            encoder_hidden=[1024, 512, 256, 128],
            decoder_hidden=[128, 256, 512, 1024],
            likelihood="gaussian",
        ),
        ModalityConfig(
            name="adt",
            input_dim=adt.n_vars,
            encoder_hidden=[256, 128, 64],
            decoder_hidden=[64, 128, 256],
            likelihood="gaussian",
        ),
        # or
        #ModalityConfig(
        #    name="atac",
        #    input_dim=atac.obsm["X_lsi"].shape[1],
        #    encoder_hidden=[256, 128, 64],
        #    decoder_hidden=[64, 128, 256],
        #    likelihood="gaussian",
        #),
    ],
)
```

### `ModalityConfig` per-modality `likelihood` guidance (pick to match your `.X`/model input preprocessing)

* **RNA**

  * `.X = raw counts` → `likelihood="nb"` (or `"zinb"` if lots of zeros beyond NB)
  * `.X = normalize_total + log1p (+ scale)` → `likelihood="gaussian"`
* **ADT**

  * `.X = CLR (+ scale)` → `likelihood="gaussian"`
  * `.X = raw counts` → usually still better behaved as `"gaussian"` *after* CLR; if you insist on counts-space, consider `"nb"` but it’s often fussier
* **ATAC (common patterns)**

  * `.X = binarized peaks` → `likelihood="bernoulli"`
  * `.X = raw peak counts` → `likelihood="poisson"`
  * `.X = LSI / reduced features` → `likelihood="gaussian"`
* **Methylation**

  * `.X = fractions / beta values (0–1)` → `likelihood="beta"` (common for fraction-like inputs)
  * **Coverage-aware (recommended when you have trials):**

    * `successes + total_count` via `recon_targets` → `likelihood="beta_binomial"` (or `"binomial"` if you don’t need overdispersion)
    * See **Quickstart: RNA + methylome (beta-binomial with recon_targets)** below for the exact setup.
  * `.X = reduced/embedded features` (e.g., PCA/LSI/other continuous reps) → `likelihood="gaussian"` (often best for alignment-focused workflows)
  * `.X = methylated counts only` (no coverage) → `likelihood="nb"` / `"zinb"` (supported, but usually not preferred vs coverage-aware modeling)

Model training configuration and loss mode setup:
```python
train_cfg = TrainingConfig(
    n_epochs=3000,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    device=device,
    early_stopping=True,
    best_epoch_warmup=50,
    patience=50,
    log_every=25,
)

model = UniVIMultiModalVAE(
    univi_cfg,
    #
    # Note: 
    # loss_mode="v1" is recommended (used in the manuscript), can also try "v2" (aka "lite"), although 
    # they are less fleshed-out/robust. "v2" is good if you want a fused latent space (required more 
    # experimental advanced workflows like the transformer fused latent architecture, which is discussed 
    # further in the advanced section below) and no cross-decoder reconstruction term in the loss function
    # (focuses less on paired cross-reconstruction for more experimental unpaired regimes).
    #
    loss_mode="v1",
    v1_recon="avg",
    normalize_v1_terms=True,
).to(device)
```
Finally, train the model:
```python
trainer = UniVITrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    train_cfg=train_cfg,
    device=device,
)

history = trainer.fit()
print("Best epoch:", getattr(trainer, "best_epoch", None))
```

---

## Quickstart: RNA + methylome (beta-binomial with recon_targets)

Assumes:

* `meth.X` is a convenient model input (fractions or embeddings)
* `meth.layers["meth_successes"]` stores methylated counts
* `meth.layers["meth_total_count"]` stores coverage / trials

```python
rna  = sc.read_h5ad("path/to/rna.h5ad")
meth = sc.read_h5ad("path/to/meth.h5ad")

adata_dict = align_paired_obs_names({"rna": rna, "meth": meth})

recon_targets_spec = {
    "meth": {
        "successes_layer": "meth_successes",
        "total_count_layer": "meth_total_count",
    }
}

dataset = MultiModalDataset(
    adata_dict=adata_dict,
    X_key="X",
    device=None,
    recon_targets_spec=recon_targets_spec,
)

train_loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_multimodal_xy_recon,
)

univi_cfg = UniVIConfig(
    latent_dim=30,
    beta=1.15,
    gamma=3.25,
    modalities=[
        ModalityConfig("rna",  rna.n_vars,  [1024, 512, 256, 128], [128, 256, 512, 1024], likelihood="gaussian"),
        ModalityConfig("meth", meth.n_vars, [512, 256, 128],       [128, 256, 512],       likelihood="beta_binomial"),
    ],
)

train_cfg = TrainingConfig(n_epochs=2000, batch_size=256, lr=1e-3, weight_decay=1e-4, device=device, log_every=25,)

model = UniVIMultiModalVAE(univi_cfg, loss_mode="v1", v1_recon="avg", normalize_v1_terms=True).to(device)
trainer = UniVITrainer(model=model, train_loader=train_loader, val_loader=None, train_cfg=train_cfg, device=device)
history = trainer.fit()
print("Best epoch:", getattr(trainer, "best_epoch", None))
```

> When `recon_targets` are present in the batch, `UniVITrainer` forwards them into `model(..., recon_targets=...)` automatically.

---

## Saving + loading

```python
import torch
from univi import UniVIMultiModalVAE

ckpt = {
    "model_state_dict": model.state_dict(),
    "model_config": univi_cfg,
    "train_cfg": train_cfg,
    "history": getattr(trainer, "history", None),
    "best_epoch": getattr(trainer, "best_epoch", None),
}
torch.save(ckpt, "./saved_models/univi_model_state.pt")

ckpt = torch.load("./saved_models/univi_model_state.pt", map_location=device)
model = UniVIMultiModalVAE(ckpt["model_config"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Best epoch:", ckpt.get("best_epoch"))
```

> For additional UniVI examples and preprocessing steps, refer to UniVI/notebooks/ for end-to-end experiments across different data types. Specifically, notebooks/GR_manuscript_reproducibility/ contains code to reproduce all the figures in our revised manuscript, while notebooks/UniVI_additional_examples/ contains examples of training and evaluating UniVI models with less standard data types (e.g. scNMT-seq tri-modal RNA/CpG/GpC data) + additional cool things you can do using our method. The latter folder will be updated with new use-cases as they come up.

---

## After training: what you can do with a UniVI model

UniVI models are **generative** (decoders + likelihoods) and **alignment-oriented** (shared latent space). After training, you typically use two modules:

* `univi.evaluation`: encoding, denoising, cross-modal prediction (imputation), generation, and metrics
* `univi.plotting`: Scanpy/Matplotlib helpers for UMAPs, legends, confusion matrices, MoE gate plots, and reconstruction-error plots

### 0a) Imports + plotting defaults

```python
import numpy as np
import scipy.sparse as sp
import torch

from univi.evaluation import (
    encode_adata,
    encode_fused_adata_pair,
    cross_modal_predict,
    denoise_adata,
    denoise_from_multimodal,
    evaluate_alignment,
    reconstruction_metrics,
    # NEW (generation + recon error workflows)
    generate_from_latent,
    fit_label_latent_gaussians,
    sample_latent_by_label,
    evaluate_cross_reconstruction,
)
from univi.plotting import (
    set_style,
    umap,
    umap_by_modality,
    compare_raw_vs_denoised_umap_features,
    plot_confusion_matrix,
    write_gates_to_obs,
    plot_moe_gate_summary,
    # NEW (reconstruction error plots)
    plot_reconstruction_error_summary,
    plot_featurewise_reconstruction_scatter,
)

set_style(font_scale=1.2, dpi=150)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
```

Helper for sparse matrices:

```python
def to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)
```

### 0b) Explicitly subset the test set indices from the splits prior to training step 

Loading the test set indices for evaluations (if desired, can also just use transductive method (all cells) depending on goals):

```python
# NOTE: Use the same test_idx for both modalities
rna = rna[test_idx].copy()
adt = adt[test_idx].copy()

# Optional sanity checks
assert rna_test.n_obs == adt_test.n_obs
assert np.array_equal(rna_test.obs_names, adt_test.obs_names)
```

---

## 1) Encode a modality into latent space (`.obsm["X_univi"]`)

Use this when you have **one observed modality at a time** (RNA-only, ADT-only, ATAC-only, methylome-only, etc.):

```python
Z_rna = encode_adata(
    model,
    adata=rna,
    modality="rna",
    device=device,
    layer=None,          # uses adata.X by default
    X_key="X",
    batch_size=1024,
    latent="moe_mean",   # {"moe_mean","moe_sample","modality_mean","modality_sample"}
    random_state=0,
)
rna.obsm["X_univi"] = Z_rna
```

Then plot:

```python
umap(
    rna,
    obsm_key="X_univi",
    color=["celltype.l2", "batch"],
    legend="outside",
    legend_subset_topk=25,
    savepath="umap_rna_univi.png",
    show=False,
)
```

---

## 2) Encode a *fused* multimodal latent (true paired/multi-observed cells)

When you have multiple observed modalities for the **same cells**, you can encode the *fused* posterior (and optionally MoE router gates/logits):

```python
fused = encode_fused_adata_pair(
    model,
    adata_by_mod={"rna": rna, "adt": adt},   # same obs_names, same order
    device=device,
    batch_size=1024,
    use_mean=True,
    return_gates=True,
    return_gate_logits=True,
    write_to_adatas=True,                   # writes obsm + gate columns
    fused_obsm_key="X_univi_fused",
    gate_prefix="gate",
)

# fused["Z_fused"] -> (n_cells, latent_dim)
# fused["gates"]  -> (n_cells, n_modalities) or None (if fused transformer posterior is used)
```

Plot fused:

```python
umap(
    rna,
    obsm_key="X_univi_fused",
    color=["celltype.l2", "batch"],
    legend="outside",
    savepath="umap_fused.png",
    show=False,
)
```

Plot fused both modalities by modality and celltype:

```python
umap_by_modality(
    {"rna": rna, "adt": adt},
    obsm_key="X_univi_fused",
    color=["univi_modality", "celltype.l2"],
    legend="outside",
    size=8,
    savepath="umap_fused_both_modalities.png",
    show=False,
)
```

---

## 3) Cross-modal prediction (imputation): encode source → decode target

Example: **RNA → ADT** (same pattern applies to RNA→methylome, methylome→RNA, etc.). UniVI will automatically handle decoder output types internally (e.g. Gaussian returns tensor; NB returns `{"mu","log_theta"}`; ZINB returns `{"mu","log_theta","logit_pi"}`; Poisson returns `{"rate","log_rate"}`; Beta/Beta-Binomial return parameter dicts) and return an appropriate **mean-like** prediction for downstream evaluation/plotting.

```python
adt_hat_from_rna = cross_modal_predict(
    model,
    adata_src=rna,
    src_mod="rna",
    tgt_mod="adt",
    device=device,
    layer=None,
    X_key="X",
    batch_size=512,
    use_moe=True,
)
adt.layers["imputed_from_rna"] = adt_hat_from_rna
```

---

## 4) Denoising (self-reconstruction or true fused denoising)

### Option A — self-denoise a single modality (same as “reconstruct”)

```python
denoise_adata(
    model,
    adata=rna,
    modality="rna",
    device=device,
    out_layer="denoised_self",
    overwrite_X=False,
    batch_size=512,
)
```

### Option B — true multimodal denoising via fused latent

```python
denoise_adata(
    model,
    adata=rna,                         # output written here
    modality="rna",
    device=device,
    out_layer="denoised_fused",
    overwrite_X=False,
    batch_size=512,
    adata_by_mod={"rna": rna, "adt": adt},
    layer_by_mod={"rna": None, "adt": None},  # None -> use .X
    X_key_by_mod={"rna": "X", "adt": "X"},
    use_mean=True,
)
```

Compare raw vs denoised marker overlays:

```python
compare_raw_vs_denoised_umap_features(
    rna,
    obsm_key="X_univi",
    features=["MS4A1", "CD3D", "NKG7"],
    raw_layer=None,
    denoised_layer="denoised_fused",
    savepath="umap_raw_vs_denoised.png",
    show=False,
)
```

---

## 5) Quantify reconstruction / imputation error vs ground truth

You can compute **featurewise + summary** errors between:

* **cross-reconstructed** (RNA→ADT, ATAC→RNA, methylome→RNA, …)
* **denoised** outputs (self or fused)
* and the **true observed** data

### A) Basic metrics on two matrices

```python
true = to_dense(adt.X)
pred = adt.layers["imputed_from_rna"]

m = reconstruction_metrics(true, pred)
print("MSE mean:", m["mse_mean"])
print("Pearson mean:", m["pearson_mean"])
```

### B) One-call evaluation for cross-reconstruction / denoising

This will:

1. generate predictions via UniVI (handling decoder output types correctly),
2. align to the requested truth matrix (layer/X_key), and
3. return metrics + optional per-feature vectors.

```python
rep = evaluate_cross_reconstruction(
    model,
    adata_src=rna,
    adata_tgt=adt,
    src_mod="rna",
    tgt_mod="adt",
    device=device,
    src_layer=None,
    tgt_layer=None,
    batch_size=512,
    # optionally restrict to a feature subset (e.g., top markers)
    feature_names=None,
)
print(rep["summary"])   # mse_mean/median, pearson_mean/median, etc.
```

Plot reconstruction-error summaries:

```python
plot_reconstruction_error_summary(
    rep,
    title="RNA → ADT imputation error",
    savepath="recon_error_summary.png",
    show=False,
)
```

And featurewise scatter (true vs predicted) for selected features:

```python
plot_featurewise_reconstruction_scatter(
    rep,
    features=["CD3", "CD4", "MS4A1"],
    savepath="recon_scatter_selected_features.png",
    show=False,
)
```

---

## 6) Alignment evaluation (FOSCTTM, Recall@k, mixing/entropy, label transfer, gates)

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
```

Confusion matrix:

```python
plot_confusion_matrix(
    np.asarray(metrics["label_transfer_cm"]),
    labels=np.asarray(metrics["label_transfer_label_order"]),
    title="Label transfer (RNA → ADT)",
    normalize="true",
    savepath="label_transfer_confusion.png",
    show=False,
)
```

---

## 7) Generate new data from latent space (sampling / “in silico cells”)

UniVI decoders define a likelihood per modality (Gaussian, NB, ZINB, Poisson, Bernoulli, Beta, Binomial/Beta-Binomial, etc.). Generation is done as:

1. pick latent samples `z ~ p(z)` (or a conditional latent distribution)
2. decode with the modality decoder(s)
3. return **mean-like reconstructions** or (optionally) sample from the likelihood

### A) Unconditional generation (standard normal prior)

```python
Xgen = generate_from_latent(
    model,
    n=5000,
    target_mod="rna",
    device=device,
    z_source="prior",         # "prior" or provide z directly
    return_mean=True,         # mean-like output
    sample_likelihood=False,  # if True: sample from likelihood when supported
)
# Xgen shape: (5000, n_genes)
```

### B) Cell-type–conditioned generation via empirical latent neighborhoods

This is the “no classifier head needed” option:

1. encode a reference cohort
2. pick cells with a given label
3. sample around their latent distribution (Gaussian fit, or jitter)

```python
Z = rna.obsm["X_univi"]
labels = rna.obs["celltype.l2"].to_numpy()

# Fit a per-label Gaussian in latent space
label_gauss = fit_label_latent_gaussians(Z, labels)

# Sample latent points for a chosen label
z_B = sample_latent_by_label(label_gauss, label="B cell", n=2000, random_state=0)

# Decode to RNA space
X_B = generate_from_latent(
    model,
    z=z_B,
    target_mod="rna",
    device=device,
    return_mean=True,
)
```

### C) Cluster-aware generation (no annotations required)

If you don’t have labels, you can cluster `Z` (e.g., k-means), fit cluster Gaussians, then sample by cluster id.

### D) Head-guided generation (optional, when a classifier head exists)

If you trained a classification head, you can optionally *bias* latent selection toward a desired label by filtering or optimizing candidate z’s (implementation depends on your head setup). UniVI supports this workflow when the head is present, but the **label-agnostic Gaussian/cluster methods work everywhere**.

---

## 8) MoE gating diagnostics (precision contributions + optional learnable router)

UniVI can report per-cell modality **contribution weights** for the **analytic fusion** path (MoE/PoE-style).

There are two related notions of “who contributed how much” to the fused latent:

* **Precision-only (always available):** derived from each modality’s posterior uncertainty in latent space.
* **Router × precision (optional):** if your trained model exposes **router logits**, UniVI can combine router probabilities with precision to produce contribution weights.

> Note: This section applies to **analytic fusion** (Gaussian experts in latent space).
> If you use a **fused transformer posterior**, there may be no analytic precision/router attribution
> and gates can be unavailable or not meaningful.

### A) Compute per-cell contribution weights (recommended)

```python
from univi.evaluation import to_dense, encode_moe_gates_from_tensors
from univi.plotting import write_gates_to_obs, plot_moe_gate_summary

gate = encode_moe_gates_from_tensors(
    model,
    x_dict={"rna": to_dense(rna.X), "adt": to_dense(adt.X)},
    device=device,
    batch_size=1024,
    modality_order=["rna", "adt"],
    kind="router_x_precision",  # falls back to "effective_precision" if router logits are unavailable
    return_logits=True,
)

W    = gate["weights"]         # (n_cells, n_modalities), rows sum to 1
mods = gate["modality_order"]  # e.g. ["rna", "adt"]

print("Requested kind:", gate.get("requested_kind"))
print("Effective kind:", gate.get("kind"))
print("Per-modality mean:", gate.get("per_modality_mean"))
print("Has logits:", gate.get("logits") is not None)
```

If you want **precision-only** weights (no router influence), set `kind="effective_precision"`.

### B) Write weights to `.obs` (for plotting / grouping)

```python
write_gates_to_obs(
    rna,
    gates=W,
    modality_names=mods,
    gate_prefix="moe_gate",          # creates obs cols: moe_gate_{mod}
    gate_logits=gate.get("logits"),  # optional; may be None
)
```

### C) Plot contribution usage (overall + grouped)

```python
plot_moe_gate_summary(
    rna,
    gate_prefix="moe_gate",
    groupby="celltype.l3",           # or "celltype.l2", "batch", etc.
    agg="mean",
    savepath="moe_gates_by_celltype.png",
    show=False,
)
```

### D) Optional: log gates alongside alignment metrics

`evaluate_alignment(...)` evaluates geometric alignment (FOSCTTM, Recall@k, mixing/entropy, label transfer).
If you want to save gate summaries alongside those metrics, just merge dictionaries:

```python
from univi.evaluation import evaluate_alignment

metrics = evaluate_alignment(
    Z1=rna.obsm["X_univi"],
    Z2=adt.obsm["X_univi"],
    labels_source=rna.obs["celltype.l3"].to_numpy(),
    labels_target=adt.obs["celltype.l3"].to_numpy(),
    json_safe=True,
)

metrics["moe_gates"] = {
    "kind": gate.get("kind"),
    "requested_kind": gate.get("requested_kind"),
    "modality_order": mods,
    "per_modality_mean": gate.get("per_modality_mean"),
    # (optional) store full matrices; omit if you want small JSON
    # "weights": W,
    # "logits": gate.get("logits"),
}
```

---

## Advanced topics

### Training objectives (v1 vs v2/lite)

* **v1 (“paper”)**: per-modality posteriors + reconstruction scheme (cross/self/avg) + posterior alignment across modalities
* **v2/lite**: fused posterior (MoE/PoE-style by default; optional fused transformer) + per-modality recon + β·KL + γ·alignment (L2 on latent means)

Choose via `loss_mode` at construction time (Python) or config JSON (scripts).

### Decoder output types (what UniVI handles for you)

Decoders can return either:

* a tensor (e.g. Gaussian)
* or a dict (e.g. NB/ZINB/Poisson/Beta/Beta-Binomial parameter dicts)

UniVI evaluation utilities unwrap these and return mean-like matrices for plotting/evaluation.

## Advanced model features

This section covers the “advanced” knobs in `univi/models/univi.py` and when to use them. Everything below is optional: you can train and evaluate UniVI without touching any of it.

---

### 1) Fused multimodal transformer posterior (optional)

**What it is:**
A *single* fused encoder that tokenizes each observed modality, concatenates tokens, runs a multimodal transformer, and outputs a fused posterior `(mu_fused, logvar_fused)`.

**Why you’d use it:**

* You want the posterior to be learned jointly across modalities (rather than fused analytically via PoE/MoE precision fusion).
* You want token-level interpretability hooks (e.g., ATAC top-k peak indices; optional attention maps if enabled in the encoder stack).
* You want a learnable “cross-modality mixing” mechanism beyond precision fusion.

**How to enable (config):**

* Set `cfg.fused_encoder_type = "multimodal_transformer"`.
* Optionally set:

  * `cfg.fused_modalities = ["rna","adt","atac"]` (defaults to all)
  * `cfg.fused_require_all_modalities = True` (default): only use fused posterior when all required modalities are present; otherwise falls back to `mixture_of_experts()`.

**Key API points:**

* Training: the model will automatically decide whether to use fused encoder or fallback based on presence and `fused_require_all_modalities`.
* Encoding: use `model.encode_fused(...)` to get the fused latent and optionally gates from fallback fusion.

```python
mu, logvar, z = model.encode_fused(
    {"rna": X_rna, "adt": X_adt, "atac": X_atac},
    use_mean=True,
)
```

---

### 2) Attention bias for transformer encoders (distance bias for ATAC, optional)

**What it is:**
A safe, optional attention bias that can encourage local genomic context for tokenized ATAC (or any modality tokenizer that supports it). It’s a **no-op** unless:

* the encoder is transformer-based *and*
* its tokenizer exposes `build_distance_attn_bias()` *and*
* you pass `attn_bias_cfg`.

**Why you’d use it:**

* ATAC token sets are sparse and positional: distance-aware attention can help the transformer focus on local regulatory structure.

**How to use (forward / encode / predict):**
Pass `attn_bias_cfg` into `forward(...)`, `encode_fused(...)`, or `predict_heads(...)`.

```python
attn_bias_cfg = {
  "atac": {"type": "distance", "lengthscale_bp": 50_000, "same_chrom_only": True}
}

out = model(x_dict=x_dict, epoch=ep, attn_bias_cfg=attn_bias_cfg)
mu, logvar, z = model.encode_fused(x_dict, attn_bias_cfg=attn_bias_cfg)
pred = model.predict_heads(x_dict, attn_bias_cfg=attn_bias_cfg)
```

**Notes:**

* For the *fused* multimodal transformer posterior, UniVI applies distance bias *within* the ATAC token block and leaves cross-modality blocks neutral (0), so it won’t artificially “force” cross-modality locality.

---

### 3) Learnable MoE gating for fusion (optional)

**What it is:**
A learnable gate that produces per-cell modality weights and uses them to scale per-modality precisions before PoE-style fusion. This is **off by default**; without it, fusion is pure precision fusion.

**Why you’d use it:**

* Modalities have variable quality per cell (e.g., low ADT counts, sparse ATAC, stressed RNA, low methylome coverage).
* You want a *data-driven* “trust score” per modality per cell.
* You want interpretable per-cell reliance weights (gate weights) to diagnose integration behavior.

**How to enable (config):**

* `cfg.use_moe_gating = True`
* Optional:

  * `cfg.moe_gating_type = "per_modality"` (default) or `"shared"`
  * `cfg.moe_gating_hidden = [..]`, `cfg.moe_gating_dropout`, `cfg.moe_gating_batchnorm`, `cfg.moe_gating_activation`
  * `cfg.moe_gate_eps` to avoid exact zeros in gated precisions

**How to retrieve gates:**
Use `encode_fused(..., return_gates=True)` (works when not using fused transformer posterior; if fused posterior is used, gates are `None`).

```python
mu, logvar, z, gates, gate_logits = model.encode_fused(
    x_dict,
    use_mean=True,
    return_gates=True,
    return_gate_logits=True,
)

# gates: (n_cells, n_modalities) in the model's modality order
```

**Tip:**
Gate weights are useful for plots like “ADT reliance by celltype” or identifying low-quality subsets.

---

### 4) Multi-head supervised decoders (classification + adversarial heads)

UniVI supports two supervised head systems:

#### A) Legacy single label head (kept for backwards compatibility)

**What it is:**
A single categorical head via `label_decoder` controlled by init args:

* `n_label_classes`, `label_loss_weight`, `label_ignore_index`, `classify_from_mu`, `label_head_name`

**When to use it:**
If you already rely on the legacy label head in notebooks/scripts and want a stable API.

**Label names helpers:**

```python
model.set_label_names(["B", "T", "NK", ...])
```

#### B) New `cfg.class_heads` multi-head system (recommended for new work)

**What it is:**
Any number of heads defined via `ClassHeadConfig`. Heads can be:

* **categorical**: softmax + cross-entropy
* **binary**: single logit + BCEWithLogitsLoss (optionally with `pos_weight`)

Heads can also be **adversarial**: they apply a gradient reversal layer (GRL) to encourage invariance (domain confusion).

**Why you’d use it:**

* Predict multiple labels simultaneously (celltype, batch, donor, tissue, QC flags, etc.).
* Add domain-adversarial training (e.g., suppress batch/donor information).
* Semi-supervised setups where only some labels exist per head.

**How labels are passed at training time:**
`y` should be a dict keyed by head name:

```python
y = {
  "celltype": celltype_ids,   # categorical (shape [B] or one-hot [B,C])
  "batch": batch_ids,         # adversarial categorical, for batch-invariant latents
  "is_doublet": doublet_01,   # binary head (0/1, ignore_index supported)
}
out = model(x_dict=x_dict, epoch=ep, y=y)
```

**How to predict heads after training:**
Use `predict_heads(...)` to run encoding + head prediction in one call.

```python
pred = model.predict_heads(x_dict, return_probs=True)
# pred[head] returns probabilities (softmax for categorical, sigmoid for binary)
```

**Head label name helpers (categorical):**

```python
model.set_head_label_names("celltype", ["B", "T", "NK", ...])
```

**Inspect head configuration (useful for logging):**

```python
meta = model.get_classification_meta()
```

---

### 5) Label expert injection into the fused posterior (semi-supervised “label as expert”)

**What it is:**
Optionally treats labels as an additional expert by encoding the label into a Gaussian posterior and fusing it with the base fused posterior. Controlled by:

* `use_label_encoder=True` and `n_label_classes>0`
* `label_encoder_warmup` (epoch threshold before injection starts)
* `label_moe_weight` (how strong labels influence fusion)
* `unlabeled_logvar` (large => labels contribute little when missing)

**Why you’d use it:**

* Semi-supervised alignment: labels can stabilize the latent when paired signals are weak.
* Controlled injection after warmup to avoid early collapse.

**How to use in encoding:**
`encode_fused(..., inject_label_expert=True, y=...)`

```python
mu, logvar, z = model.encode_fused(
    x_dict,
    epoch=ep,
    y={"label": y_ids},          # or just pass y_ids if using legacy path
    inject_label_expert=True,
)
```

---

### 6) Recon scaling across modalities (important when dims differ a lot)

**What it is:**
Per-modality reconstruction losses are typically summed across features; large modalities (RNA) can dominate gradients. UniVI supports:

* `recon_normalize_by_dim` + `recon_dim_power` (divide by `D**power`)
* per-modality `ModalityConfig.recon_weight`

**Defaults:**

* v1-style losses: normalize is off by default, `power=0.5`
* v2/lite: normalize is on by default, `power=1.0`

**Why you’d use it:**

* Stabilize training when RNA has 2k–20k dims but ADT has 30–200 dims and ATAC-LSI has ~50–500 dims (and methylome features may vary widely too).
* Tune modality balance without hand-waving.

**How to tune:**

* For “equal per-cell contribution” across modalities: `recon_normalize_by_dim=True` and `recon_dim_power=1.0`
* If you want a softer correction: `power=0.5`
* Or set `recon_weight` per modality.

---

### 7) Convenience APIs

#### `encode_fused(...)`

**Purpose:** Encode any subset of modalities into a fused posterior, with optional gate outputs.

```python
mu, logvar, z = model.encode_fused(
    x_dict,
    epoch=0,
    use_mean=True,                 # True: return mu; False: sample
    inject_label_expert=True,
    attn_bias_cfg=None,
)

# Optional: get fusion gates (only when fused transformer posterior is NOT used)
mu, logvar, z, gates, gate_logits = model.encode_fused(
    x_dict,
    return_gates=True,
    return_gate_logits=True,
)
```

#### `predict_heads(...)`

**Purpose:** Encode fused latent, then emit probabilities/logits for the legacy head + all multi-head configs.

```python
pred = model.predict_heads(x_dict, return_probs=True)
# pred[head] -> probs (softmax/sigmoid)
```

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
├── notebooks/                             # End-to-end Jupyter Notebook analyses and examples
│   ├── GR_manuscript_reproducibility/     # Reproduce figures from revised manuscript (in progress for Genome Research; bioRxiv manuscript v2)
│   │   ├── UniVI_manuscript_GR-Figure__2__CITE_paired.ipynb
│   │   ├── UniVI_manuscript_GR-Figure__3__CITE_paired_biological_latent.ipynb
│   │   ├── UniVI_manuscript_GR-Figure__4__Multiome_paired.ipynb
│   │   ├── UniVI_manuscript_GR-Figure__5__Multiome_bridge_mapping_and_fine-tuning.ipynb
│   │   ├── UniVI_manuscript_GR-Figure__6__TEA-seq_tri-modal.ipynb
│   │   ├── UniVI_manuscript_GR-Figure__7__AML_bridge_mapping_and_fine-tuning.ipynb
│   │   ├── UniVI_manuscript_GR-Figure__8__benchmarking_against_pytorch_tools.ipynb
│   │   ├── UniVI_manuscript_GR-Figure__8__benchmarking_against_R_tools.ipynb
│   │   ├── UniVI_manuscript_GR-Figure__8__benchmarking_merging_and_plotting_runs.ipynb
│   │   ├── UniVI_manuscript_GR-Figure__9__paired_data_ablation_and_computational_scaling_performance.ipynb
│   │   ├── UniVI_manuscript_GR-Figure__9__paired_data_ablation_and_computational_scaling_performance_compile_plots_from_results_df.ipynb
│   │   ├── UniVI_manuscript_GR-Figure_10__cell_population_ablation_MoE.ipynb
│   │   ├── UniVI_manuscript_GR-Figure_10__cell_population_ablation_MoE_compile_plots_from_results_df.ipynb
│   │   ├── UniVI_manuscript_GR-Supple_____grid-sweep.ipynb
│   │   └── UniVI_manuscript_GR-Supple_____grid-sweep_compile_plots_from_results_df.ipynb
│   └── UniVI_additional_examples/         # Additional examples of UniVI workflow functionality
│       ├── Multiome_NB-RNA-counts_Poisson_or_Bernoulli-ATAC_peak-counts_Peak_perturbation_to_RNA_expression_cross-generation_experiment.ipynb
│       └── scNMT-seq_mouse_gastrulation_feature-level_integration_example.ipynb
├── parameter_files/                       # JSON configs for model + training + data selectors
│   ├── defaults_*.json                    # Default configs (per experiment)
│   └── params_*.json                      # Example “named” configs (RNA, ADT, ATAC, methylome, etc.)
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
    │   ├── decoders.py                    # Likelihood-specific decoders (NB, ZINB, Gaussian, Beta, Binomial/Beta-Binomial, etc.)
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

## License

MIT License — see `LICENSE`.

---

## Contact, questions, and bug reports

* Questions / comments: open a GitHub Issue with the `question` label (or use Discussions)
* Bug reports: include:

  * UniVI version: `python -c "import univi; print(univi.__version__)"`
  * minimal notebook/code snippet
  * stack trace + OS/CUDA/PyTorch versions
