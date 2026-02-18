# univi/plotting.py

from __future__ import annotations

from typing import Dict, Optional, Sequence, List, Union, Tuple, Any, Mapping

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData


# =============================================================================
# Style / defaults
# =============================================================================
def set_style(
    font_scale: float = 1.1,
    dpi: int = 150,
    *,
    rc: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Reasonable plotting defaults. You can override any rcParams via `rc`.
    """
    import matplotlib as mpl

    base = 10.0 * float(font_scale)
    defaults = {
        "figure.dpi": int(dpi),
        "savefig.dpi": 300,
        "font.size": base,
        "axes.titlesize": base * 1.15,
        "axes.labelsize": base * 1.05,
        "xtick.labelsize": base * 0.90,
        "ytick.labelsize": base * 0.90,
        "legend.fontsize": base * 0.90,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if rc:
        defaults.update(rc)

    mpl.rcParams.update(defaults)
    sc.settings.set_figure_params(dpi=int(dpi), dpi_save=300, frameon=False)


def _is_categorical_obs(adata: AnnData, key: str) -> bool:
    if key not in adata.obs:
        return False
    s = adata.obs[key]
    # pandas category OR object OR string dtype
    return (str(s.dtype) == "category") or (s.dtype == object) or (str(s.dtype).startswith("string"))


def _ensure_scanpy_colors(adata: AnnData, key: str) -> None:
    """
    Ensure adata.uns[f"{key}_colors"] exists for categorical obs keys.
    Scanpy usually creates this lazily the first time you plot.
    """
    if key not in adata.obs:
        return
    if not _is_categorical_obs(adata, key):
        return
    color_key = f"{key}_colors"
    if color_key in adata.uns and adata.uns[color_key] is not None:
        return
    # Trigger scanpy to assign colors without showing
    try:
        sc.pl.umap(adata, color=key, show=False)
    except Exception:
        # If UMAP isn't computed yet, scanpy might choke; ignore and fall back later.
        pass


def _get_categories_and_colors(
    adata: AnnData,
    key: str,
    *,
    subset_topk: Optional[int] = None,
) -> Tuple[List[str], List[Any]]:
    """
    Recover categories and scanpy palette (if present) for a categorical obs key.

    subset_topk:
      If provided, restrict legend items to top-k categories by frequency
      (helps reduce crowding).
    """
    s = adata.obs[key].astype("category")
    cats_all = [str(c) for c in s.cat.categories]

    # Optionally pick top-k by frequency
    if subset_topk is not None and subset_topk > 0 and len(cats_all) > subset_topk:
        vc = s.value_counts()
        keep = [str(x) for x in vc.index[: int(subset_topk)]]
        cats = keep
    else:
        cats = cats_all

    _ensure_scanpy_colors(adata, key)
    color_key = f"{key}_colors"
    colors_all = adata.uns.get(color_key, None)

    if colors_all is None:
        # fallback to matplotlib default cycle
        colors = [None] * len(cats_all)
    else:
        colors = list(colors_all)

    # align color list length
    if len(colors) < len(cats_all):
        colors = (colors * (len(cats_all) // max(1, len(colors)) + 1))[: len(cats_all)]
    else:
        colors = colors[: len(cats_all)]

    # subset colors to chosen categories
    if cats != cats_all:
        idx = {lab: i for i, lab in enumerate(cats_all)}
        colors = [colors[idx[lab]] for lab in cats]
    else:
        colors = colors[: len(cats)]

    return cats, colors


def _add_outside_legend(
    ax: plt.Axes,
    adata: AnnData,
    key: str,
    *,
    title: Optional[str] = None,
    ncols: int = 2,
    fontsize: Optional[float] = None,
    markerscale: float = 1.2,
    max_items: int = 40,
    subset_topk: Optional[int] = None,
    loc: str = "center left",
    bbox_to_anchor: Tuple[float, float] = (1.02, 0.5),
) -> None:
    """
    Add a compact categorical legend outside the axis.

    Skips legend if too many categories (max_items) unless subset_topk is set.
    """
    if key not in adata.obs or not _is_categorical_obs(adata, key):
        return

    s = adata.obs[key].astype("category")
    n_cats = len(s.cat.categories)

    # If too many items, either subset (if asked) or skip
    if n_cats > int(max_items) and (subset_topk is None or subset_topk <= 0):
        return

    cats_str, colors = _get_categories_and_colors(adata, key, subset_topk=subset_topk)

    # If still too many after subsetting, bail
    if len(cats_str) > int(max_items):
        return

    handles = []
    for lab, col in zip(cats_str, colors):
        h = plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=col,
            markeredgecolor=col,
            label=lab,
        )
        handles.append(h)

    ax.legend(
        handles=handles,
        title=title or key,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=False,
        ncol=int(ncols),
        fontsize=fontsize,
        markerscale=float(markerscale),
        borderaxespad=0.0,
        handletextpad=0.35,
        columnspacing=0.9,
    )


def _finalize_figure(
    fig: plt.Figure,
    *,
    savepath: Optional[str],
    show: bool,
    close: bool,
    tight: bool = True,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
) -> None:
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches=bbox_inches, pad_inches=float(pad_inches))
    if show:
        plt.show()
    if close:
        plt.close(fig)


# =============================================================================
# Embedding / UMAP helpers
# =============================================================================
def ensure_neighbors_umap(
    adata: AnnData,
    *,
    rep_key: str = "X_univi",
    n_neighbors: int = 30,
    random_state: int = 0,
    neighbors_key: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Ensure neighbors + UMAP exist using adata.obsm[rep_key].

    - If `force=True`, recompute neighbors+umap even if they already exist.
    """
    if rep_key not in adata.obsm:
        raise KeyError(f"Missing obsm[{rep_key!r}]. Available: {list(adata.obsm.keys())}")

    # neighbors
    if neighbors_key is None:
        needs_neighbors = force or ("neighbors" not in adata.uns)
        if needs_neighbors:
            sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=int(n_neighbors))
    else:
        needs_neighbors = force or (neighbors_key not in adata.uns)
        if needs_neighbors:
            sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=int(n_neighbors), key_added=str(neighbors_key))

    # umap
    needs_umap = force or ("X_umap" not in adata.obsm)
    if needs_umap:
        if neighbors_key is None:
            sc.tl.umap(adata, random_state=int(random_state))
        else:
            sc.tl.umap(adata, random_state=int(random_state), neighbors_key=str(neighbors_key))


# =============================================================================
# New API (preferred)
# =============================================================================
def umap(
    adata: AnnData,
    *,
    obsm_key: str = "X_univi",
    color: Union[str, Sequence[str]] = (),
    layer: Optional[str] = None,
    title: Optional[Union[str, Sequence[str]]] = None,
    size: Optional[float] = None,
    n_neighbors: int = 30,
    random_state: int = 0,
    force_recompute: bool = False,
    # layout
    ncols: int = 2,
    panel_size: Tuple[float, float] = (4.2, 3.8),
    wspace: float = 0.25,
    hspace: float = 0.25,
    # legend control
    legend: str = "outside",  # "outside", "right_margin", "on_data", "none"
    legend_ncols: int = 2,
    legend_fontsize: Optional[float] = None,
    legend_max_items: int = 40,
    legend_subset_topk: Optional[int] = None,
    legend_bbox_to_anchor: Tuple[float, float] = (1.02, 0.5),
    # output
    savepath: Optional[str] = None,
    show: bool = True,
    close: Optional[bool] = None,
    return_fig: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    # passthrough for scanpy
    **scanpy_kwargs,
) -> Union[None, Tuple[plt.Figure, np.ndarray]]:
    """
    UMAP plotting that avoids huge crowded Scanpy legends by default.

    legend:
      - "outside": compact legend outside categorical panels (recommended)
      - "right_margin": scanpy legend in right margin
      - "on_data": scanpy legend on top of points
      - "none": no legend
    """
    ensure_neighbors_umap(
        adata,
        rep_key=obsm_key,
        n_neighbors=n_neighbors,
        random_state=random_state,
        force=force_recompute,
    )

    colors = [color] if isinstance(color, str) else list(color)
    n_panels = max(1, len(colors)) if len(colors) > 0 else 1

    if close is None:
        # Notebook-friendly: keep open if showing; otherwise close after save
        close = (not show) and (savepath is not None)

    ncols_eff = int(min(max(1, ncols), n_panels))
    nrows_eff = int(np.ceil(n_panels / ncols_eff))

    fig = plt.figure(figsize=(panel_size[0] * ncols_eff, panel_size[1] * nrows_eff))
    gs = fig.add_gridspec(nrows_eff, ncols_eff, wspace=wspace, hspace=hspace)

    axes = np.empty((nrows_eff, ncols_eff), dtype=object)
    for r in range(nrows_eff):
        for c in range(ncols_eff):
            axes[r, c] = fig.add_subplot(gs[r, c])

    # normalize titles
    if title is None:
        titles = [None] * n_panels
    elif isinstance(title, str):
        titles = [title] + [None] * (n_panels - 1)
    else:
        titles = list(title) + [None] * max(0, n_panels - len(title))

    layer_eff = None if (layer is None or layer == "X") else layer

    # If no color requested, plot just the embedding (no coloring)
    if len(colors) == 0:
        ax = axes[0, 0]
        sc.pl.umap(
            adata,
            color=None,
            ax=ax,
            show=False,
            title=titles[0] if titles[0] is not None else "",
            size=size,
            **scanpy_kwargs,
        )
        # hide unused
        for j in range(1, nrows_eff * ncols_eff):
            r, c = divmod(j, ncols_eff)
            axes[r, c].axis("off")

        _finalize_figure(
            fig, savepath=savepath, show=show, close=bool(close),
            tight=False, bbox_inches=bbox_inches, pad_inches=pad_inches
        )
        if return_fig:
            return fig, axes
        return None

    for i, key in enumerate(colors):
        r, c = divmod(i, ncols_eff)
        ax = axes[r, c]
        is_cat = _is_categorical_obs(adata, str(key))

        # Decide scanpy legend behavior
        if legend == "none":
            legend_loc = None
        elif legend == "on_data":
            legend_loc = "on data"
        elif legend == "right_margin":
            legend_loc = "right margin"
        elif legend == "outside":
            legend_loc = None
        else:
            legend_loc = "right margin" if is_cat else None

        sc.pl.umap(
            adata,
            color=str(key),
            layer=layer_eff,
            ax=ax,
            show=False,
            title=titles[i] if titles[i] is not None else str(key),
            size=size,
            legend_loc=legend_loc,
            **scanpy_kwargs,
        )

        if legend == "outside" and is_cat:
            _add_outside_legend(
                ax,
                adata,
                str(key),
                ncols=int(legend_ncols),
                fontsize=legend_fontsize,
                max_items=int(legend_max_items),
                subset_topk=legend_subset_topk,
                bbox_to_anchor=legend_bbox_to_anchor,
            )

    # turn off unused axes
    for j in range(len(colors), nrows_eff * ncols_eff):
        r, c = divmod(j, ncols_eff)
        axes[r, c].axis("off")

    _finalize_figure(
        fig, savepath=savepath, show=show, close=bool(close),
        tight=False, bbox_inches=bbox_inches, pad_inches=pad_inches
    )

    if return_fig:
        return fig, axes
    return None


def umap_by_modality(
    adata_dict: Dict[str, AnnData],
    *,
    obsm_key: str = "X_univi",
    color: Union[str, Sequence[str]] = ("cell_type", "dataset"),
    layer: Optional[str] = None,
    # layout
    ncols: int = 2,
    panel_size: Tuple[float, float] = (4.2, 3.8),
    wspace: float = 0.25,
    hspace: float = 0.25,
    # legend
    legend: str = "outside",
    legend_ncols: int = 2,
    legend_fontsize: Optional[float] = None,
    legend_max_items: int = 40,
    legend_subset_topk: Optional[int] = None,
    legend_bbox_to_anchor: Tuple[float, float] = (1.02, 0.5),
    # neighbors/umap
    n_neighbors: int = 30,
    random_state: int = 0,
    force_recompute: bool = False,
    # output
    savepath: Optional[str] = None,
    show: bool = True,
    close: Optional[bool] = None,
    return_fig: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    **scanpy_kwargs,
) -> Union[None, Tuple[plt.Figure, np.ndarray]]:
    """
    Combine multiple AnnData objects for visualization while preserving obsm stacking.
    Adds combined.obs["univi_modality"].
    """
    mods = list(adata_dict.keys())
    if len(mods) == 0:
        raise ValueError("adata_dict is empty.")

    Zs = []
    obs_frames = []
    for mod in mods:
        a = adata_dict[mod]
        if obsm_key not in a.obsm:
            raise KeyError(f"{mod!r} missing obsm[{obsm_key!r}]")

        Zs.append(np.asarray(a.obsm[obsm_key]))
        obs = a.obs.copy()
        obs["univi_modality"] = str(mod)
        obs_frames.append(obs)

    Z = np.vstack(Zs)
    if len(obs_frames) == 1:
        obs_all = obs_frames[0]
    else:
        # robust concat (pandas)
        import pandas as pd
        obs_all = pd.concat(obs_frames, axis=0)

    combined = AnnData(X=np.zeros((Z.shape[0], 0), dtype=np.float32), obs=obs_all)
    combined.obsm[obsm_key] = Z

    return umap(
        combined,
        obsm_key=obsm_key,
        color=color,
        layer=layer,
        ncols=ncols,
        panel_size=panel_size,
        wspace=wspace,
        hspace=hspace,
        legend=legend,
        legend_ncols=legend_ncols,
        legend_fontsize=legend_fontsize,
        legend_max_items=legend_max_items,
        legend_subset_topk=legend_subset_topk,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        n_neighbors=n_neighbors,
        random_state=random_state,
        force_recompute=force_recompute,
        savepath=savepath,
        show=show,
        close=close,
        return_fig=return_fig,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        **scanpy_kwargs,
    )


# =============================================================================
# Raw vs Denoised comparisons
# =============================================================================
def compare_raw_vs_denoised_umap_features(
    adata: AnnData,
    *,
    obsm_key: str = "X_univi",
    features: Sequence[str],
    denoised_layer: Optional[str],
    raw_layer: Optional[str] = None,
    n_neighbors: int = 30,
    random_state: int = 0,
    force_recompute: bool = False,
    # layout
    panel_size: Tuple[float, float] = (4.0, 3.6),
    wspace: float = 0.15,
    hspace: float = 0.15,
    # output
    savepath: Optional[str] = None,
    show: bool = True,
    close: Optional[bool] = None,
    return_fig: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    **scanpy_kwargs,
) -> Union[None, Tuple[plt.Figure, np.ndarray]]:
    """
    2-row grid:
      row 1: raw overlay
      row 2: denoised overlay
    """
    feats = list(features)
    if len(feats) == 0:
        raise ValueError("features must be non-empty.")

    ensure_neighbors_umap(
        adata,
        rep_key=obsm_key,
        n_neighbors=n_neighbors,
        random_state=random_state,
        force=force_recompute,
    )

    raw_layer_eff = None if (raw_layer is None or raw_layer == "X") else raw_layer
    den_layer_eff = None if (denoised_layer is None or denoised_layer == "X") else denoised_layer
    if den_layer_eff is not None and den_layer_eff not in adata.layers:
        raise KeyError(f"denoised_layer={den_layer_eff!r} not found in adata.layers.")

    if close is None:
        close = (not show) and (savepath is not None)

    ncol = len(feats)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncol,
        figsize=(panel_size[0] * ncol, panel_size[1] * 2),
        squeeze=False,
        gridspec_kw={"wspace": wspace, "hspace": hspace},
    )

    for j, f in enumerate(feats):
        sc.pl.umap(
            adata,
            color=f,
            layer=raw_layer_eff,
            ax=axes[0, j],
            show=False,
            title=f"{f} (raw)",
            **scanpy_kwargs,
        )
        sc.pl.umap(
            adata,
            color=f,
            layer=den_layer_eff,
            ax=axes[1, j],
            show=False,
            title=f"{f} (denoised)",
            **scanpy_kwargs,
        )

    _finalize_figure(
        fig, savepath=savepath, show=show, close=bool(close),
        tight=False, bbox_inches=bbox_inches, pad_inches=pad_inches
    )

    if return_fig:
        return fig, axes
    return None


# =============================================================================
# Confusion matrix (no seaborn)
# =============================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    *,
    title: str = "Label transfer (source → target)",
    normalize: Optional[str] = None,  # None, "true", "pred"
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (7.0, 6.0),
    savepath: Optional[str] = None,
    rotate_xticks: int = 60,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = True,
    close: Optional[bool] = None,
    return_fig: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
) -> Union[None, plt.Figure]:
    cm = np.asarray(cm, dtype=float)
    lab = np.asarray(list(labels))

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

    if close is None:
        close = (not show) and (savepath is not None)

    fig = plt.figure(figsize=figsize)
    im = plt.imshow(cm, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(np.arange(len(lab)), lab, rotation=rotate_xticks, ha="right")
    plt.yticks(np.arange(len(lab)), lab)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    _finalize_figure(
        fig, savepath=savepath, show=show, close=bool(close),
        tight=False, bbox_inches=bbox_inches, pad_inches=pad_inches
    )

    if return_fig:
        return fig
    return None


# =============================================================================
# MoE gating helpers
# =============================================================================
def write_gates_to_obs(
    adata: AnnData,
    gates: np.ndarray,
    modality_names: Sequence[str],
    *,
    prefix: str = "moe_gate_",
    overwrite: bool = True,
) -> None:
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
    groupby: Optional[str] = None,  # e.g. "celltype.l2"
    kind: str = "box",              # "box" or "meanbar"
    max_groups: int = 30,
    figsize: Tuple[float, float] = (10.0, 4.5),
    savepath: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
    close: Optional[bool] = None,
    return_fig: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
) -> Union[None, plt.Figure]:
    if modality_names is None:
        modality_names = [c[len(gate_prefix):] for c in adata.obs.columns if str(c).startswith(gate_prefix)]
        modality_names = list(modality_names)

    cols = [f"{gate_prefix}{m}" for m in modality_names]
    for c in cols:
        if c not in adata.obs:
            raise KeyError(f"Missing gate column {c!r} in adata.obs.")

    vals = adata.obs[cols].to_numpy(dtype=float)

    if close is None:
        close = (not show) and (savepath is not None)

    fig = plt.figure(figsize=figsize)

    if groupby is None:
        plt.boxplot([vals[:, i] for i in range(vals.shape[1])], labels=list(modality_names), showfliers=False)
        plt.ylabel("Gate weight")
        plt.title(title or "MoE gate weights (all cells)")
        plt.tight_layout()
        _finalize_figure(
            fig, savepath=savepath, show=show, close=bool(close),
            tight=False, bbox_inches=bbox_inches, pad_inches=pad_inches
        )
        return fig if return_fig else None

    groups = np.asarray(adata.obs[groupby]).astype(str)
    uniq = np.unique(groups)
    if uniq.size > int(max_groups):
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

    kind_eff = str(kind).lower().strip()
    if kind_eff == "meanbar":
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
        plt.boxplot([G[:, i] for i in range(G.shape[1])], labels=list(modality_names), showfliers=False)
        plt.ylabel("Mean gate weight (per group)")
        plt.title(title or f"MoE gate weights by {groupby}")
        plt.tight_layout()

    _finalize_figure(
        fig, savepath=savepath, show=show, close=bool(close),
        tight=False, bbox_inches=bbox_inches, pad_inches=pad_inches
    )
    return fig if return_fig else None


# =============================================================================
# Backwards-compatible aliases (old API)
# =============================================================================
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
    # new optional controls (won't break old calls)
    show: Optional[bool] = None,
    close: Optional[bool] = None,
    legend: str = "right_margin",
    legend_ncols: int = 2,
    legend_fontsize: Optional[float] = None,
    legend_max_items: int = 40,
    legend_subset_topk: Optional[int] = None,
    legend_bbox_to_anchor: Tuple[float, float] = (1.02, 0.5),
    ncols: int = 2,
    panel_size: Tuple[float, float] = (4.2, 3.8),
    wspace: float = 0.25,
    hspace: float = 0.25,
    force_recompute: bool = False,
    return_fig: bool = False,
    **scanpy_kwargs,
) -> Union[None, Tuple[plt.Figure, np.ndarray]]:
    """
    OLD entrypoint kept alive.

    Old default behavior:
      - show=False
      - if saving, close figure
      - otherwise close figure too (historically)
    """
    if show is None:
        show = False
    if close is None:
        # historical behavior: always close (esp. because old version called plt.close())
        close = True

    color_list = list(color) if color is not None else []
    return umap(
        adata_obj,
        obsm_key=obsm_key,
        color=color_list,
        layer=layer,
        title=title,
        size=size,
        n_neighbors=n_neighbors,
        random_state=random_state,
        force_recompute=force_recompute,
        ncols=ncols,
        panel_size=panel_size,
        wspace=wspace,
        hspace=hspace,
        legend=legend,
        legend_ncols=legend_ncols,
        legend_fontsize=legend_fontsize,
        legend_max_items=legend_max_items,
        legend_subset_topk=legend_subset_topk,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        savepath=savepath,
        show=bool(show),
        close=bool(close),
        return_fig=return_fig,
        **scanpy_kwargs,
    )


def umap_by_modality_old(
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
    show: Optional[bool] = None,
    close: Optional[bool] = None,
    **kwargs,
) -> None:
    """
    Compatibility shim for *very old* umap_by_modality signature.

    Old behavior:
      - always added 'univi_modality' to colors
      - show=False and close=True
    """
    if show is None:
        show = False
    if close is None:
        close = True

    if isinstance(color, str):
        color_list = [color, "univi_modality"]
    else:
        color_list = list(color) + ["univi_modality"]

    # ignore title/size in old implementation unless user passes them
    umap_by_modality(
        adata_dict,
        obsm_key=obsm_key,
        color=color_list,
        layer=layer,
        n_neighbors=n_neighbors,
        random_state=random_state,
        force_recompute=kwargs.pop("force_recompute", False),
        savepath=savepath,
        show=bool(show),
        close=bool(close),
        # allow additional kwargs to influence layout/legend if provided
        **{k: v for k, v in kwargs.items() if k in {
            "ncols", "panel_size", "wspace", "hspace",
            "legend", "legend_ncols", "legend_fontsize", "legend_max_items",
            "legend_subset_topk", "legend_bbox_to_anchor",
            "return_fig", "bbox_inches", "pad_inches",
        }},
    )

