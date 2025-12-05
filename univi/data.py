# univi/data.py

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union, List

import os
import numpy as np
import scipy.sparse as sp
import anndata as ad
from anndata import AnnData

import torch
from torch.utils.data import Dataset


LayerSpec = Union[None, str, Mapping[str, Optional[str]]]
XKeySpec = Union[str, Mapping[str, str]]


def _get_matrix(adata_obj: AnnData, *, layer: Optional[str], X_key: str):
    if X_key != "X":
        if X_key not in adata_obj.obsm:
            raise KeyError("X_key=%r not found in adata.obsm. Keys=%s" % (X_key, list(adata_obj.obsm.keys())))
        return adata_obj.obsm[X_key]
    if layer is not None:
        if layer not in adata_obj.layers:
            raise KeyError("layer=%r not found in adata.layers. Keys=%s" % (layer, list(adata_obj.layers.keys())))
        return adata_obj.layers[layer]
    return adata_obj.X


def infer_input_dim(adata_obj: AnnData, *, layer: Optional[str], X_key: str) -> int:
    X = _get_matrix(adata_obj, layer=layer, X_key=X_key)
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("Selected matrix for (layer=%r, X_key=%r) is not 2D." % (layer, X_key))
    return int(X.shape[1])


def align_paired_obs_names(
    adata_dict: Dict[str, AnnData],
    how: str = "intersection",
    require_nonempty: bool = True,
) -> Dict[str, AnnData]:
    if not adata_dict:
        raise ValueError("adata_dict is empty")
    if how != "intersection":
        raise ValueError("Unsupported how=%r. Only 'intersection' is supported." % how)

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
    """
    Returns either:
      - x_dict
      - (x_dict, y) if labels are provided
    """

    def __init__(
        self,
        adata_dict: Dict[str, AnnData],
        layer: LayerSpec = None,
        X_key: XKeySpec = "X",
        paired: bool = True,
        device: Optional[torch.device] = None,
        labels: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,
    ):
        self.adata_dict = adata_dict
        self.paired = bool(paired)
        self.device = device
        self.labels = labels

        if isinstance(layer, Mapping):
            self.layer_by_modality = dict(layer)
        else:
            self.layer_by_modality = {k: layer for k in adata_dict.keys()}

        if isinstance(X_key, Mapping):
            self.xkey_by_modality = dict(X_key)
        else:
            self.xkey_by_modality = {k: X_key for k in adata_dict.keys()}

        if self.paired:
            first = next(iter(adata_dict.values()))
            for nm, adata_obj in self.adata_dict.items():
                if adata_obj.n_obs != first.n_obs:
                    raise ValueError("Paired dataset requires matching n_obs across modalities.")
                if not np.array_equal(adata_obj.obs_names.values, first.obs_names.values):
                    raise ValueError("Paired dataset requires identical obs_names order; %r differs." % nm)

        if self.labels is not None:
            if hasattr(self.labels, "__len__") and len(self.labels) != self.__len__():
                raise ValueError("labels length (%d) must equal n_cells (%d)" % (len(self.labels), self.__len__()))

    def __len__(self) -> int:
        first = next(iter(self.adata_dict.values()))
        return int(first.n_obs)

    def _get_X_row(self, adata_obj: AnnData, idx: int, layer: Optional[str], X_key: str) -> np.ndarray:
        X = _get_matrix(adata_obj, layer=layer, X_key=X_key)
        row = X[idx]
        if sp.issparse(row):
            row = row.toarray()
        row = np.asarray(row).reshape(-1).astype(np.float32)
        return row

    def __getitem__(self, idx: int):
        out: Dict[str, torch.Tensor] = {}
        for name, adata_obj in self.adata_dict.items():
            layer = self.layer_by_modality.get(name, None)
            xkey = self.xkey_by_modality.get(name, "X")
            row_np = self._get_X_row(adata_obj, idx, layer=layer, X_key=xkey)
            t = torch.from_numpy(row_np)
            if self.device is not None:
                t = t.to(self.device)
            out[name] = t

        if self.labels is None:
            return out

        y = self.labels[idx]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.long)
        else:
            y = y.long()
        if self.device is not None:
            y = y.to(self.device)

        return out, y


def dataset_from_anndata_dict(
    adata_dict: Dict[str, AnnData],
    layer: LayerSpec = None,
    X_key: XKeySpec = "X",
    paired: bool = True,
    align_obs: bool = True,
    labels: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,
) -> Tuple[MultiModalDataset, Dict[str, AnnData]]:
    if align_obs and paired:
        adata_dict = align_paired_obs_names(adata_dict, how="intersection")
    ds = MultiModalDataset(adata_dict, layer=layer, X_key=X_key, paired=paired, labels=labels)
    return ds, adata_dict


def load_anndata_dict_from_config(
    modality_cfgs: List[Dict[str, Any]],
    data_root: Optional[str] = None,
) -> Dict[str, AnnData]:
    out: Dict[str, AnnData] = {}
    for m in modality_cfgs:
        name = m["name"]
        path = m["h5ad_path"]
        if data_root is not None and not os.path.isabs(path):
            path = os.path.join(data_root, path)
        out[name] = ad.read_h5ad(path)
    return out

