# univi/plotting.py

from __future__ import annotations

from typing import Dict, Optional, Sequence, List, Mapping, Any, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData


# ----------------------------
# Style / defaults
# ----------------------------
def set_style(font_scale: float = 1.25, dpi: int = 150) -> None:
    """Readable, manuscript-friendly plotting defaults."""
    import matplotlib as mpl

    base = 10.0 * float(font_scale)
    mpl.rcParams.update({
        "figure.dpi": int(dpi),
        "savefig.dpi": 300,
        "font.size": base,
        "axes.titlesize": base * 1.2,
        "axes.labelsize": base * 1.1,
        "xtick.labelsize": base * 0.95,
        "ytick.labelsize": base * 0.95,
        "legend.fontsize": base * 0.95,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    sc.settings.set_figure_params(dpi=int(dpi), dpi_save=300, frameon=False)


def set_default_layer(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    prefer_layers: Sequence[str] = ("denoised", "denoise", "X_denoised", "univi_denoised"),
) -> Optional[str]:
    """
    Decide which layer to use for expression overlays.

    - If `layer` is provided: use it (and validate it exists).
    - Else, pick the first matching layer in `prefer_layers` if present.
    - Else, return None (meaning use adata.X).
    """
    if layer is not None:
        if layer != "X" and layer not in adata.layers:
            raise KeyError(f"Requested layer={layer!r} not found. Available layers={list(adata.layers.keys())}")
        return None if layer == "X" else layer

    for cand in prefer_layers:
        if cand in adata.layers:
            return cand
    return None


def get_matrix_for_plot(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    X_key: str = "X",
) -> np.ndarray:
    """
    Return a dense matrix for non-scanpy plotting (e.g., explicit heatmaps/scatters).
    For scanpy plotting (UMAP feature overlays), you usually just pass `layer=...`
    directly to scanpy and avoid materializing dense matrices.
    """
    if layer is None or layer == "X":
        X = adata.X
    else:
        X = adata.layers[layer]

    # AnnData matrices can be sparse; convert to dense for plotting if needed
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            return X.toarray()
    except Exception:
        pass

    return np.asarray(X)


# ----------------------------
# Embedding / UMAP helpers
# ----------------------------
def ensure_neighbors_umap(
    adata: AnnData,
    *,
    rep_key: str = "X_univi",
    n_neighbors: int = 30,
    random_state: int = 0,
    neighbors_key: Optional[str] = None,
) -> None:
    """
    Ensure neighbors + UMAP are computed using adata.obsm[rep_key].
    Uses default scanpy slots unless you supply neighbors_key.
    """
    if rep_key not in adata.obsm:
        raise KeyError(f"Missing obsm[{rep_key!r}]. Available: {list(adata.obsm.keys())}")

    # neighbors
    if neighbors_key is None:
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=int(n_neighbors))
    else:
        if neighbors_key not in adata.uns:
            sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=int(n_neighbors), key_added=str(neighbors_key))

    # umap
    if "X_umap" not in adata.obsm:
        if neighbors_key is None:
            sc.tl.umap(adata, random_state=int(random_state))
        else:
            sc.tl.umap(adata, random_state=int(random_state), neighbors_key=str(neighbors_key))


def umap_single_adata(
    adata_obj: AnnData,
    *,
    obsm_key: str = "X_univi",
    color: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
    savepath: Optional[str] = None,
    n_neighbors: int = 30,
    random_state: int = 0,
    title: Optional[str] = None,
    size: Optional[float] = None,
) -> None:
    """
    Plot a UMAP for a single AnnData.

    - `color` can include obs keys and/or var_names features.
    - `layer` controls expression overlay source when `color` includes var_names.
      (If layer is None, scanpy uses adata.X.)
    """
    ensure_neighbors_umap(
        adata_obj,
        rep_key=obsm_key,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    color_list = list(color) if color is not None else []
    sc.pl.umap(
        adata_obj,
        color=color_list,
        layer=None if (layer is None or layer == "X") else layer,
        show=False,
        title=title,
        size=size,
    )

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()


def umap_by_modality(
    adata_dict: Dict[str, AnnData],
    *,
    obsm_key: str = "X_univi",
    color: Union[str, Sequence[str]] = "celltype",
    layer: Optional[str] = None,
    savepath: Optional[str] = None,
    n_neighbors: int = 30,
    random_state: int = 0,
    title: Optional[str] = None,
    size: Optional[float] = None,
) -> None:
    """
    Concatenate adatas; expects each input adata already has the same obsm_key embedding
    (or you should add it before calling this).

    Adds:
      - combined.obs["univi_modality"] = which modality/source
    """
    annotated: List[AnnData] = []
    for mod, a in adata_dict.items():
        aa = a.copy()
        aa.obs["univi_modality"] = str(mod)
        annotated.append(aa)

    combined = annotated[0].concatenate(
        *annotated[1:],
        batch_key="univi_source",
        batch_categories=list(adata_dict.keys()),
        index_unique="-",
        join="outer",
    )

    # concatenate may drop obsm; rebuild if needed
    if obsm_key not in combined.obsm:
        try:
            Zs = [adata_dict[m].obsm[obsm_key] for m in adata_dict.keys()]
            combined.obsm[obsm_key] = np.vstack(Zs)
        except Exception:
            raise KeyError(f"combined missing obsm[{obsm_key!r}] after concatenation; add it manually.")

    if isinstance(color, str):
        color_list = [color, "univi_modality"]
    else:
        color_list = list(color) + ["univi_modality"]

    umap_single_adata(
        combined,
        obsm_key=obsm_key,
        color=color_list,
        layer=layer,
        savepath=savepath,
        n_neighbors=n_neighbors,
        random_state=random_state,
        title=title,
        size=size,
    )


# ----------------------------
# Raw vs Denoised comparisons
# ----------------------------
def compare_raw_vs_denoised_umap_features(
    adata: AnnData,
    *,
    obsm_key: str = "X_univi",
    features: Sequence[str],
    denoised_layer: Optional[str],
    raw_layer: Optional[str] = None,
    n_neighbors: int = 30,
    random_state: int = 0,
    savepath: Optional[str] = None,
    title_prefix: str = "",
    size: Optional[float] = None,
) -> None:
    """
    Convenience for a common manuscript panel:

      UMAP colored by feature overlays from:
        - raw expression (adata.X or raw_layer)
        - denoised expression (adata.layers[denoised_layer])

    Produces a 2-row grid:
      row 1: raw
      row 2: denoised
    """
    ensure_neighbors_umap(adata, rep_key=obsm_key, n_neighbors=n_neighbors, random_state=random_state)

    feats = list(features)
    if len(feats) == 0:
        raise ValueError("features must be non-empty.")

    # Validate layers
    raw_layer_eff = None if (raw_layer is None or raw_layer == "X") else raw_layer
    den_layer_eff = None if (denoised_layer is None or denoised_layer == "X") else denoised_layer
    if den_layer_eff is not None and den_layer_eff not in adata.layers:
        raise KeyError(f"denoised_layer={den_layer_eff!r} not found in adata.layers.")

    # Build figure
    ncol = len(feats)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncol,
        figsize=(3.6 * ncol, 3.2 * 2),
        squeeze=False,
    )

    # scanpy draws into current axes; we must set it
    for j, f in enumerate(feats):
        # RAW
        plt.sca(axes[0, j])
        sc.pl.umap(
            adata,
            color=f,
            layer=raw_layer_eff,
            show=False,
            title=f"{title_prefix}{f} (raw)" if title_prefix else f"{f} (raw)",
            size=size,
        )

        # DENOISED
        plt.sca(axes[1, j])
        sc.pl.umap(
            adata,
            color=f,
            layer=den_layer_eff,
            show=False,
            title=f"{title_prefix}{f} (denoised)" if title_prefix else f"{f} (denoised)",
            size=size,
        )

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Confusion matrix (no seaborn)
# ----------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    labels: np.ndarray,
    *,
    title: str = "Label transfer (source → target)",
    normalize: Optional[str] = None,  # None, "true", "pred"
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (7.0, 6.0),
    savepath: Optional[str] = None,
    rotate_xticks: int = 60,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Plot confusion matrix using matplotlib.

    normalize:
      - None: raw counts
      - "true": divide rows (per-true class)
      - "pred": divide cols (per-pred class)
    """
    cm = np.asarray(cm, dtype=float)
    lab = np.asarray(labels)

    if normalize is not None:
        norm = str(normalize).lower().strip()
        if norm == "true":
            denom = cm.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            cm = cm / denom
        elif norm == "pred":
            denom = cm.sum(axis=0, keepdims=True)
            denom[denom == 0] = 1.0
            cm = cm / denom
        else:
            raise ValueError("normalize must be one of None, 'true', 'pred'")

    plt.figure(figsize=figsize)
    im = plt.imshow(cm, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(np.arange(len(lab)), lab, rotation=rotate_xticks, ha="right")
    plt.yticks(np.arange(len(lab)), lab)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# MoE gating plotting helpers
# ----------------------------
def write_gates_to_obs(
    adata: AnnData,
    gates: np.ndarray,
    modality_names: Sequence[str],
    *,
    prefix: str = "moe_gate_",
    overwrite: bool = True,
) -> None:
    """
    Store MoE gate weights (n_cells, n_modalities) into adata.obs as columns.
    """
    gates = np.asarray(gates)
    if gates.ndim != 2:
        raise ValueError(f"gates must be 2D (n_cells, n_mods). Got {gates.shape}")
    if gates.shape[0] != adata.n_obs:
        raise ValueError(f"gates n_cells={gates.shape[0]} != adata.n_obs={adata.n_obs}")
    if gates.shape[1] != len(modality_names):
        raise ValueError(f"gates n_mods={gates.shape[1]} != len(modality_names)={len(modality_names)}")

    for j, m in enumerate(modality_names):
        col = f"{prefix}{m}"
        if (col in adata.obs) and (not overwrite):
            continue
        adata.obs[col] = gates[:, j].astype(np.float32)


def plot_moe_gate_summary(
    adata: AnnData,
    *,
    gate_prefix: str = "moe_gate_",
    modality_names: Optional[Sequence[str]] = None,
    groupby: Optional[str] = None,   # e.g. "celltype.l2"
    kind: str = "box",               # "box" or "violin" or "meanbar"
    max_groups: int = 30,
    figsize: Tuple[float, float] = (10.0, 4.5),
    savepath: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Summarize MoE gate weights stored in obs columns like moe_gate_rna, moe_gate_adt, ...

    - If groupby is None: plots per-modality distributions across cells.
    - If groupby is set: summarizes by group (mean gate per modality per group).
    """
    if modality_names is None:
        # infer modality names from columns
        modality_names = [c[len(gate_prefix):] for c in adata.obs.columns if str(c).startswith(gate_prefix)]
        modality_names = list(modality_names)

    cols = [f"{gate_prefix}{m}" for m in modality_names]
    for c in cols:
        if c not in adata.obs:
            raise KeyError(f"Missing gate column {c!r} in adata.obs.")

    vals = adata.obs[cols].to_numpy(dtype=float)

    if groupby is None:
        # simple per-modality distributions
        plt.figure(figsize=figsize)
        plt.boxplot([vals[:, i] for i in range(vals.shape[1])], labels=list(modality_names), showfliers=False)
        plt.ylabel("Gate weight")
        plt.title(title or "MoE gate weights (all cells)")
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
        return

    # grouped summary
    groups = np.asarray(adata.obs[groupby]).astype(str)
    uniq = np.unique(groups)
    if uniq.size > int(max_groups):
        # keep top max_groups by count
        counts = {g: int((groups == g).sum()) for g in uniq}
        uniq = np.array(sorted(uniq, key=lambda g: counts[g], reverse=True)[: int(max_groups)])

    group_means = []
    group_labels = []
    for g in uniq:
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        group_means.append(vals[mask].mean(axis=0))
        group_labels.append(g)

    G = np.vstack(group_means) if group_means else np.zeros((0, vals.shape[1]))

    plt.figure(figsize=(max(figsize[0], 0.35 * len(group_labels) + 3.0), figsize[1]))
    # meanbar: one bar cluster per group
    if str(kind).lower().strip() == "meanbar":
        x = np.arange(len(group_labels))
        width = 0.8 / max(1, len(modality_names))
        for i, m in enumerate(modality_names):
            plt.bar(x + i * width, G[:, i], width=width, label=str(m))
        plt.xticks(x + 0.5 * (len(modality_names) - 1) * width, group_labels, rotation=60, ha="right")
        plt.ylabel("Mean gate weight")
        plt.title(title or f"MoE gate weights by {groupby}")
        plt.legend(frameon=False, ncol=min(4, len(modality_names)))
        plt.tight_layout()
    else:
        # fallback: boxplot per modality, grouped (flattened)
        # This is a simpler view: each modality shows distribution of group means
        plt.boxplot([G[:, i] for i in range(G.shape[1])], labels=list(modality_names), showfliers=False)
        plt.ylabel("Mean gate weight (per group)")
        plt.title(title or f"MoE gate weights by {groupby}")
        plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

