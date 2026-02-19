# univi/plotting.py
from __future__ import annotations

from typing import Dict, Optional, Sequence, List, Union, Tuple, Any, Mapping

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

from .evaluation import to_dense


# =============================================================================
# Style / defaults
# =============================================================================
def set_style(
    font_scale: float = 1.1,
    dpi: int = 150,
    *,
    rc: Optional[Dict[str, Any]] = None,
) -> None:
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
    return (str(s.dtype) == "category") or (s.dtype == object) or (str(s.dtype).startswith("string"))


def _ensure_scanpy_colors(adata: AnnData, key: str) -> None:
    if key not in adata.obs:
        return
    if not _is_categorical_obs(adata, key):
        return
    color_key = f"{key}_colors"
    if color_key in adata.uns and adata.uns[color_key] is not None:
        return
    try:
        sc.pl.umap(adata, color=key, show=False)
    except Exception:
        pass


def _get_categories_and_colors(
    adata: AnnData,
    key: str,
    *,
    subset_topk: Optional[int] = None,
) -> Tuple[List[str], List[Any]]:
    s = adata.obs[key].astype("category")
    cats_all = [str(c) for c in s.cat.categories]

    if subset_topk is not None and subset_topk > 0 and len(cats_all) > subset_topk:
        vc = s.value_counts()
        cats = [str(x) for x in vc.index[: int(subset_topk)]]
    else:
        cats = cats_all

    _ensure_scanpy_colors(adata, key)
    colors_all = adata.uns.get(f"{key}_colors", None)

    if colors_all is None:
        colors_all = [None] * len(cats_all)
    else:
        colors_all = list(colors_all)

    if len(colors_all) < len(cats_all):
        colors_all = (colors_all * (len(cats_all) // max(1, len(colors_all)) + 1))[: len(cats_all)]
    else:
        colors_all = colors_all[: len(cats_all)]

    if cats != cats_all:
        idx = {lab: i for i, lab in enumerate(cats_all)}
        colors = [colors_all[idx[lab]] for lab in cats]
    else:
        colors = colors_all[: len(cats)]

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
    if key not in adata.obs or not _is_categorical_obs(adata, key):
        return

    s = adata.obs[key].astype("category")
    n_cats = len(s.cat.categories)

    if n_cats > int(max_items) and (subset_topk is None or subset_topk <= 0):
        return

    cats_str, colors = _get_categories_and_colors(adata, key, subset_topk=subset_topk)
    if len(cats_str) > int(max_items):
        return

    handles = []
    for lab, col in zip(cats_str, colors):
        h = plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                       markerfacecolor=col, markeredgecolor=col, label=lab)
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
    if rep_key not in adata.obsm:
        raise KeyError(f"Missing obsm[{rep_key!r}]. Available: {list(adata.obsm.keys())}")

    if neighbors_key is None:
        needs_neighbors = force or ("neighbors" not in adata.uns)
        if needs_neighbors:
            sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=int(n_neighbors))
    else:
        needs_neighbors = force or (neighbors_key not in adata.uns)
        if needs_neighbors:
            sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=int(n_neighbors), key_added=str(neighbors_key))

    needs_umap = force or ("X_umap" not in adata.obsm)
    if needs_umap:
        if neighbors_key is None:
            sc.tl.umap(adata, random_state=int(random_state))
        else:
            sc.tl.umap(adata, random_state=int(random_state), neighbors_key=str(neighbors_key))


# =============================================================================
# UMAP plotting
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
    ncols: int = 2,
    panel_size: Tuple[float, float] = (4.2, 3.8),
    wspace: float = 0.25,
    hspace: float = 0.25,
    legend: str = "outside",  # "outside", "right_margin", "on_data", "none"
    legend_ncols: int = 2,
    legend_fontsize: Optional[float] = None,
    legend_max_items: int = 40,
    legend_subset_topk: Optional[int] = None,
    legend_bbox_to_anchor: Tuple[float, float] = (1.02, 0.5),
    savepath: Optional[str] = None,
    show: bool = True,
    close: Optional[bool] = None,
    return_fig: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    **scanpy_kwargs,
) -> Union[None, Tuple[plt.Figure, np.ndarray]]:
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
        close = (not show) and (savepath is not None)

    ncols_eff = int(min(max(1, ncols), n_panels))
    nrows_eff = int(np.ceil(n_panels / ncols_eff))

    fig = plt.figure(figsize=(panel_size[0] * ncols_eff, panel_size[1] * nrows_eff))
    gs = fig.add_gridspec(nrows_eff, ncols_eff, wspace=wspace, hspace=hspace)

    axes = np.empty((nrows_eff, ncols_eff), dtype=object)
    for r in range(nrows_eff):
        for c in range(ncols_eff):
            axes[r, c] = fig.add_subplot(gs[r, c])

    if title is None:
        titles = [None] * n_panels
    elif isinstance(title, str):
        titles = [title] + [None] * (n_panels - 1)
    else:
        titles = list(title) + [None] * max(0, n_panels - len(title))

    layer_eff = None if (layer is None or layer == "X") else layer

    if len(colors) == 0:
        ax = axes[0, 0]
        sc.pl.umap(adata, color=None, ax=ax, show=False,
                   title=titles[0] if titles[0] is not None else "",
                   size=size, **scanpy_kwargs)
        for j in range(1, nrows_eff * ncols_eff):
            r, c = divmod(j, ncols_eff)
            axes[r, c].axis("off")

        _finalize_figure(fig, savepath=savepath, show=show, close=bool(close),
                         tight=False, bbox_inches=bbox_inches, pad_inches=pad_inches)
        return (fig, axes) if return_fig else None

    for i, key in enumerate(colors):
        r, c = divmod(i, ncols_eff)
        ax = axes[r, c]
        is_cat = _is_categorical_obs(adata, str(key))

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
                ax, adata, str(key),
                ncols=int(legend_ncols),
                fontsize=legend_fontsize,
                max_items=int(legend_max_items),
                subset_topk=legend_subset_topk,
                bbox_to_anchor=legend_bbox_to_anchor,
            )

    for j in range(len(colors), nrows_eff * ncols_eff):
        r, c = divmod(j, ncols_eff)
        axes[r, c].axis("off")

    _finalize_figure(fig, savepath=savepath, show=show, close=bool(close),
                     tight=False, bbox_inches=bbox_inches, pad_inches=pad_inches)
    return (fig, axes) if return_fig else None


def umap_by_modality(
    adata_dict: Dict[str, AnnData],
    *,
    obsm_key: str = "X_univi",
    color: Union[str, Sequence[str]] = ("cell_type", "dataset"),
    layer: Optional[str] = None,
    ncols: int = 2,
    panel_size: Tuple[float, float] = (4.2, 3.8),
    wspace: float = 0.25,
    hspace: float = 0.25,
    legend: str = "outside",
    legend_ncols: int = 2,
    legend_fontsize: Optional[float] = None,
    legend_max_items: int = 40,
    legend_subset_topk: Optional[int] = None,
    legend_bbox_to_anchor: Tuple[float, float] = (1.02, 0.5),
    n_neighbors: int = 30,
    random_state: int = 0,
    force_recompute: bool = False,
    savepath: Optional[str] = None,
    show: bool = True,
    close: Optional[bool] = None,
    return_fig: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    **scanpy_kwargs,
) -> Union[None, Tuple[plt.Figure, np.ndarray]]:
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
# NEW: Convenience helpers for logits/probs + error plots
# =============================================================================
def layer_sigmoid_inplace(adata: AnnData, src_layer: str, dst_layer: str) -> None:
    """
    Convert logits layer -> probs layer using sigmoid.
    Works for dense or sparse; for sparse, applies sigmoid to stored data only.
    """
    X = adata.layers[src_layer]
    if sp.issparse(X):
        X2 = X.copy()
        X2.data = 1.0 / (1.0 + np.exp(-X2.data))
        adata.layers[dst_layer] = X2
    else:
        adata.layers[dst_layer] = 1.0 / (1.0 + np.exp(-np.asarray(X)))


def compare_raw_vs_pred_umap_features(
    adata: AnnData,
    *,
    obsm_key: str = "X_univi",
    features: Sequence[str],
    pred_layer: str,
    raw_layer: Optional[str] = None,
    pred_is_logits: bool = False,
    n_neighbors: int = 30,
    random_state: int = 0,
    force_recompute: bool = False,
    panel_size: Tuple[float, float] = (4.0, 3.6),
    wspace: float = 0.15,
    hspace: float = 0.15,
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
      row 1: raw
      row 2: predicted

    pred_is_logits=True will plot sigmoid(pred_layer) by creating a temporary layer.
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
    if pred_layer not in adata.layers:
        raise KeyError(f"pred_layer {pred_layer!r} not found in adata.layers.")

    plot_pred_layer = pred_layer
    tmp_layer = None
    if pred_is_logits:
        tmp_layer = f"__tmp_sigmoid__{pred_layer}"
        layer_sigmoid_inplace(adata, pred_layer, tmp_layer)
        plot_pred_layer = tmp_layer

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
            adata, color=f, layer=raw_layer_eff, ax=axes[0, j], show=False,
            title=f"{f} (raw)", **scanpy_kwargs
        )
        sc.pl.umap(
            adata, color=f, layer=plot_pred_layer, ax=axes[1, j], show=False,
            title=f"{f} (pred)", **scanpy_kwargs
        )

    _finalize_figure(fig, savepath=savepath, show=show, close=bool(close),
                     tight=False, bbox_inches=bbox_inches, pad_inches=pad_inches)

    # cleanup temp layer
    if tmp_layer is not None and tmp_layer in adata.layers:
        try:
            del adata.layers[tmp_layer]
        except Exception:
            pass

    return (fig, axes) if return_fig else None


def plot_feature_scatter_observed_vs_pred(
    adata: AnnData,
    feature: str,
    *,
    layer_obs: Optional[str] = None,
    layer_pred: str,
    pred_is_logits: bool = False,
    max_points: int = 20000,
    random_state: int = 0,
    s: float = 4.0,
    alpha: float = 0.3,
    title: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Scatter: predicted vs observed for one feature.
    """
    j = adata.var_names.get_loc(feature)

    Xo = adata.X if layer_obs is None else adata.layers[layer_obs]
    Xp = adata.layers[layer_pred]

    yo = np.asarray(Xo[:, j].toarray()).ravel() if sp.issparse(Xo) else np.asarray(Xo[:, j]).ravel()
    yp = np.asarray(Xp[:, j].toarray()).ravel() if sp.issparse(Xp) else np.asarray(Xp[:, j]).ravel()

    if pred_is_logits:
        yp = 1.0 / (1.0 + np.exp(-yp))

    n = yo.shape[0]
    if n > int(max_points):
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(n, size=int(max_points), replace=False)
        yo, yp = yo[idx], yp[idx]

    fig = plt.figure(figsize=(4.8, 4.2))
    plt.scatter(yp, yo, s=float(s), alpha=float(alpha))
    plt.xlabel("pred")
    plt.ylabel("obs")
    plt.title(title or feature)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# =============================================================================
# Backwards-compatible aliases
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
    if show is None:
        show = False
    if close is None:
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
    if show is None:
        show = False
    if close is None:
        close = True

    if isinstance(color, str):
        color_list = [color, "univi_modality"]
    else:
        color_list = list(color) + ["univi_modality"]

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
        **{k: v for k, v in kwargs.items() if k in {
            "ncols", "panel_size", "wspace", "hspace",
            "legend", "legend_ncols", "legend_fontsize", "legend_max_items",
            "legend_subset_topk", "legend_bbox_to_anchor",
            "return_fig", "bbox_inches", "pad_inches",
        }},
    )

