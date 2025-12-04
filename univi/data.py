# univi/data.py

from __future__ import annotations

from typing import Dict, Optional, Union, Mapping, Any, Tuple

import os
import numpy as np
import scipy.sparse as sp
import anndata as ad
from anndata import AnnData

import torch
from torch.utils.data import Dataset


LayerSpec = Union[None, str, Mapping[str, Optional[str]]]
XKeySpec = Union[str, Mapping[str, str]]


def _get_matrix(adata: AnnData, *, layer: Optional[str], X_key: str):
    """Return the 2D matrix view selected by (layer, X_key).

    - If X_key == "X": uses adata.layers[layer] if layer is not None, else adata.X
    - Else: uses adata.obsm[X_key]
    """
    if X_key != "X":
        if X_key not in adata.obsm:
            raise KeyError(f"X_key={X_key!r} not found in adata.obsm. Keys={list(adata.obsm.keys())}")
        return adata.obsm[X_key]
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"layer={layer!r} not found in adata.layers. Keys={list(adata.layers.keys())}")
        return adata.layers[layer]
    return adata.X


def infer_input_dim(adata: AnnData, *, layer: Optional[str], X_key: str) -> int:
    X = _get_matrix(adata, layer=layer, X_key=X_key)
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError(f"Selected matrix for (layer={layer!r}, X_key={X_key!r}) is not 2D.")
    return int(X.shape[1])


def align_paired_obs_names(
    adata_dict: Dict[str, AnnData],
    *,
    how: str = "intersection",
    require_nonempty: bool = True,
) -> Dict[str, AnnData]:
    """Align paired modalities by obs_names and order.

    If how == "intersection": take intersection of obs_names across modalities and
    reorder every AnnData to the same shared order.
    """
    if not adata_dict:
        raise ValueError("adata_dict is empty")

    if how != "intersection":
        raise ValueError(f"Unsupported how={how!r}. Only 'intersection' is supported.")

    names = list(adata_dict.keys())
    shared = None
    for nm in names:
        idx = adata_dict[nm].obs_names
        shared = idx if shared is None else shared.intersection(idx)

    if require_nonempty and (shared is None or len(shared) == 0):
        raise ValueError("No shared obs_names across modalities (intersection is empty).")

    shared = shared.sort_values()
    out: Dict[str, AnnData] = {}
    for nm in names:
        out[nm] = adata_dict[nm].loc[shared].copy()
    return out


class MultiModalDataset(Dataset):
    """Multimodal Dataset wrapping multiple AnnData objects.

    Supports either a *single* (layer, X_key) for all modalities **or**
    per-modality {'rna': 'counts', 'atac': None, ...} specifications.

    Parameters
    ----------
    adata_dict
        Dict mapping modality name -> AnnData.
    layer
        None / str / dict[str, Optional[str]].
    X_key
        str / dict[str, str]. Use "X" for adata.X / adata.layers[layer]; otherwise uses adata.obsm[X_key].
    paired
        If True, assumes all modalities have same n_obs and same obs_names order.
    device
        Optional device to move tensors to during __getitem__.
    """

    def __init__(
        self,
        adata_dict: Dict[str, AnnData],
        *,
        layer: LayerSpec = None,
        X_key: XKeySpec = "X",
        paired: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.adata_dict = adata_dict
        self.paired = paired
        self.device = device

        if isinstance(layer, Mapping):
            self.layer_by_modality = dict(layer)
        else:
            self.layer_by_modality = {k: layer for k in adata_dict.keys()}

        if isinstance(X_key, Mapping):
            self.xkey_by_modality = dict(X_key)
        else:
            self.xkey_by_modality = {k: X_key for k in adata_dict.keys()}

        if paired:
            first = next(iter(adata_dict.values()))
            for nm, adata in self.adata_dict.items():
                if adata.n_obs != first.n_obs:
                    raise ValueError(
                        f"Paired dataset requires matching n_obs across modalities. Got {nm}={adata.n_obs}, first={first.n_obs}"
                    )
                if not np.array_equal(adata.obs_names.values, first.obs_names.values):
                    raise ValueError(
                        f"Paired dataset requires identical obs_names order. Modality '{nm}' does not match."
                    )

    @property
    def n_cells(self) -> int:
        return len(self)

    def __len__(self) -> int:
        first = next(iter(self.adata_dict.values()))
        return int(first.n_obs)

    def _get_X_row(self, adata: AnnData, idx: int, *, layer: Optional[str], X_key: str) -> np.ndarray:
        X = _get_matrix(adata, layer=layer, X_key=X_key)
        row = X[idx]
        if sp.issparse(row):
            row = row.toarray()
        row = np.asarray(row).reshape(-1)
        return row.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, adata in self.adata_dict.items():
            layer = self.layer_by_modality.get(name, None)
            xkey = self.xkey_by_modality.get(name, "X")
            row_np = self._get_X_row(adata, idx, layer=layer, X_key=xkey)
            t = torch.from_numpy(row_np)
            if self.device is not None:
                t = t.to(self.device)
            out[name] = t
        return out


def dataset_from_anndata_dict(
    adata_dict: Dict[str, AnnData],
    *,
    layer: LayerSpec = None,
    X_key: XKeySpec = "X",
    paired: bool = True,
    align_obs: bool = True,
) -> Tuple[MultiModalDataset, Dict[str, AnnData]]:
    """Convenience: optionally align obs_names and build MultiModalDataset."""
    if align_obs and paired:
        adata_dict = align_paired_obs_names(adata_dict, how="intersection")
    ds = MultiModalDataset(adata_dict, layer=layer, X_key=X_key, paired=paired)
    return ds, adata_dict


def load_anndata_dict_from_config(
    modality_cfgs: list[dict[str, Any]],
    *,
    data_root: Optional[str] = None,
) -> Dict[str, AnnData]:
    """Load h5ad files described by the JSON config's data.modalities list."""
    out: Dict[str, AnnData] = {}
    for m in modality_cfgs:
        name = m["name"]
        path = m["h5ad_path"]
        if data_root is not None and not os.path.isabs(path):
            path = os.path.join(data_root, path)
        out[name] = ad.read_h5ad(path)
    return out
