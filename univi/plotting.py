# univi/plotting.py

from __future__ import annotations
from typing import Dict, Optional, Sequence, List

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from anndata import AnnData
import anndata as ad


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


def umap_single_adata(
    adata_obj: AnnData,
    obsm_key: str = "X_univi",
    color: Optional[Sequence[str]] = None,
    savepath: Optional[str] = None,
    n_neighbors: int = 30,
    random_state: int = 0,
) -> None:
    if obsm_key not in adata_obj.obsm:
        raise KeyError("Missing obsm[%r]. Available: %s" % (obsm_key, list(adata_obj.obsm.keys())))

    # Compute neighbors/umap if missing
    if "neighbors" not in adata_obj.uns:
        sc.pp.neighbors(adata_obj, use_rep=obsm_key, n_neighbors=int(n_neighbors))
    if "X_umap" not in adata_obj.obsm:
        sc.tl.umap(adata_obj, random_state=int(random_state))

    if color is None:
        color = []
    sc.pl.umap(adata_obj, color=list(color), show=False)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()


def umap_by_modality(
    adata_dict: Dict[str, AnnData],
    obsm_key: str = "X_univi",
    color: str = "celltype",
    savepath: Optional[str] = None,
    n_neighbors: int = 30,
    random_state: int = 0,
) -> None:
    """
    Concatenate adatas; expects each input adata already has the same obsm_key embedding
    (or you should add it before calling this).
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

    # Carry embeddings if needed (common pitfall: concatenate drops obsm keys)
    if obsm_key not in combined.obsm:
        # try to rebuild from parts
        try:
            Zs = [adata_dict[m].obsm[obsm_key] for m in adata_dict.keys()]
            combined.obsm[obsm_key] = np.vstack(Zs)
        except Exception:
            raise KeyError("combined is missing obsm[%r] after concatenation; add it manually." % obsm_key)

    umap_single_adata(
        combined,
        obsm_key=obsm_key,
        color=[color, "univi_modality"],
        savepath=savepath,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: np.ndarray,
    title: str = "Label transfer (source \u2192 target)",
    savepath: Optional[str] = None,
) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=False,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

