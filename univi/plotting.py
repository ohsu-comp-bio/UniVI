# univi/plotting.py

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from anndata import AnnData
import anndata as ad


def umap_by_modality(
    adata_dict: Dict[str, AnnData],
    obsm_key: str = "X_univi",
    color: str = "celltype",
    savepath: Optional[str] = None,
):
    """
    Run UMAP on concatenated AnnData objects (if needed) or on each separately
    and show cells colored by cell type and/or modality.
    """
    # simple version: concatenate
    adatas = []
    for mod, ad in adata_dict.items():
        ad = ad.copy()
        ad.obs["modality"] = mod
        adatas.append(ad)
    concat = adatas[0].concatenate(
        *adatas[1:], join="outer", batch_key="batch", batch_categories=list(adata_dict.keys())
    )

    # ensure UMAP exists
    if obsm_key not in concat.obsm:
        sc.pp.neighbors(concat, use_rep=obsm_key)
        sc.tl.umap(concat)

    sc.pl.umap(
        concat,
        color=[color, "modality"],
        wspace=0.4,
        show=False,
    )
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: np.ndarray,
    title: str = "Label transfer (source → target)",
    savepath: Optional[str] = None,
):
    """
    Simple confusion matrix heatmap.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
    plt.close()

def umap_by_modality(
    adata_dict: Dict[str, "ad.AnnData"],
    obsm_key: str = "X_univi",
    color: str = "celltype",
    savepath: Optional[str] = None,
):
    """
    Convenience wrapper: concatenate multiple adatas (one per modality),
    then call umap_single_adata on the combined object.

    We assume each adata already has obsm[obsm_key] defined.
    """
    # tag each adata with its modality
    annotated = []
    for mod, a in adata_dict.items():
        a = a.copy()
        a.obs["univi_modality"] = mod
        annotated.append(a)

    combined = annotated[0].concatenate(
        *annotated[1:],
        batch_key="univi_source",
        batch_categories=list(adata_dict.keys()),
        index_unique="-",
    )

    umap_single_adata(
        combined,
        obsm_key=obsm_key,
        color=color,
        savepath=savepath,
    )


