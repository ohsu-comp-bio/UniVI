# univi/data.py

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union, List, Sequence

import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import anndata as ad
from anndata import AnnData

import torch
from torch.utils.data import Dataset

from .config import ModalityConfig

# -----------------------------
# Types
# -----------------------------
LayerSpec = Union[None, str, Mapping[str, Optional[str]]]
XKeySpec = Union[str, Mapping[str, str]]


# -----------------------------
# Likelihood helpers
# -----------------------------
def _is_categorical_likelihood(lk: Optional[str]) -> bool:
    lk = (lk or "").lower().strip()
    return lk in ("categorical", "cat", "ce", "cross_entropy", "multinomial", "softmax")


# -----------------------------
# Matrix selection helpers
# -----------------------------
def _get_matrix(adata_obj: AnnData, *, layer: Optional[str], X_key: str):
    """
    Resolve a 2D matrix from an AnnData according to:
      - If X_key != "X": use adata.obsm[X_key]
      - Else if layer is not None: use adata.layers[layer]
      - Else: use adata.X
    """
    if X_key != "X":
        if X_key not in adata_obj.obsm:
            raise KeyError(
                "X_key=%r not found in adata.obsm. Keys=%s" % (X_key, list(adata_obj.obsm.keys()))
            )
        return adata_obj.obsm[X_key]

    if layer is not None:
        if layer not in adata_obj.layers:
            raise KeyError(
                "layer=%r not found in adata.layers. Keys=%s" % (layer, list(adata_obj.layers.keys()))
            )
        return adata_obj.layers[layer]

    return adata_obj.X


def infer_input_dim(adata_obj: AnnData, *, layer: Optional[str], X_key: str) -> int:
    X = _get_matrix(adata_obj, layer=layer, X_key=X_key)
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("Selected matrix for (layer=%r, X_key=%r) is not 2D." % (layer, X_key))
    return int(X.shape[1])


# -----------------------------
# Pairing / alignment helpers
# -----------------------------
def align_paired_obs_names(
    adata_dict: Dict[str, AnnData],
    how: str = "intersection",
    require_nonempty: bool = True,
    sort: bool = True,
    copy: bool = True,
) -> Dict[str, AnnData]:
    """
    Align modalities on shared obs_names (by intersection).
    Returns a NEW dict with each AnnData subset/reordered to the shared index.
    """
    if not adata_dict:
        raise ValueError("adata_dict is empty")
    if how != "intersection":
        raise ValueError("Unsupported how=%r. Only 'intersection' is supported." % how)

    names = list(adata_dict.keys())
    shared = None
    for nm in names:
        idx = adata_dict[nm].obs_names
        shared = idx if shared is None else shared.intersection(idx)

    if shared is None:
        shared = pd.Index([])

    if require_nonempty and len(shared) == 0:
        raise ValueError("No shared obs_names across modalities (intersection is empty).")

    if sort:
        shared = shared.sort_values()

    out: Dict[str, AnnData] = {}
    for nm in names:
        slc = adata_dict[nm].loc[shared]
        out[nm] = slc.copy() if copy else slc
    return out


def _as_modality_map(
    spec: Union[str, None, Mapping[str, Any]],
    adata_dict: Dict[str, AnnData],
    kind: str,
) -> Dict[str, Any]:
    """
    Normalize a scalar or mapping spec to a per-modality dict.
    """
    if isinstance(spec, Mapping):
        out = dict(spec)
    else:
        out = {k: spec for k in adata_dict.keys()}

    for k in adata_dict.keys():
        if k not in out:
            out[k] = None if kind == "layer" else "X"
    return out


# -----------------------------
# Dataset
# -----------------------------
class MultiModalDataset(Dataset):
    """
    A simple multi-modal AnnData-backed torch Dataset.

    Returns:
      - x_dict: Dict[modality -> FloatTensor]
      - (x_dict, y) if labels are provided (y is LongTensor)

    Categorical modality support
    ----------------------------
    If you pass modality_cfgs and a modality is categorical with input_kind="obs",
    then x_dict[modality] will be a (D=1,) float tensor holding an integer code.
    Your model code can convert it to one-hot for encoding and to Long targets for CE.
    """

    def __init__(
        self,
        adata_dict: Dict[str, AnnData],
        layer: LayerSpec = None,
        X_key: XKeySpec = "X",
        paired: bool = True,
        device: Optional[torch.device] = None,
        labels: Optional[Union[np.ndarray, torch.Tensor, Sequence[int]]] = None,
        dtype: torch.dtype = torch.float32,
        modality_cfgs: Optional[List[ModalityConfig]] = None,
    ):
        if not adata_dict:
            raise ValueError("adata_dict is empty")

        self.adata_dict: Dict[str, AnnData] = adata_dict
        self.modalities: List[str] = list(adata_dict.keys())
        self.paired = bool(paired)
        self.device = device
        self.dtype = dtype

        self.layer_by_modality: Dict[str, Optional[str]] = _as_modality_map(layer, adata_dict, kind="layer")
        self.xkey_by_modality: Dict[str, str] = _as_modality_map(X_key, adata_dict, kind="xkey")

        self.mod_cfg_by_name: Dict[str, ModalityConfig] = {}
        if modality_cfgs is not None:
            self.mod_cfg_by_name = {m.name: m for m in modality_cfgs}

        # Cache sizes / obs_names from first modality (canonical)
        first = next(iter(adata_dict.values()))
        self._n_cells: int = int(first.n_obs)
        self._obs_names = first.obs_names

        # Validate pairedness
        if self.paired:
            for nm, adata_obj in self.adata_dict.items():
                if int(adata_obj.n_obs) != self._n_cells:
                    raise ValueError(
                        f"Paired dataset requires matching n_obs across modalities. "
                        f"First={self._n_cells}, {nm}={adata_obj.n_obs}"
                    )
                if not np.array_equal(adata_obj.obs_names.values, self._obs_names.values):
                    raise ValueError(
                        "Paired dataset requires identical obs_names order; %r differs. "
                        "Tip: use dataset_from_anndata_dict(..., align_obs=True)." % nm
                    )

        # Labels (optional)
        self.labels: Optional[torch.Tensor] = None
        if labels is not None:
            y = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
            if y.ndim != 1:
                y = y.reshape(-1)
            if int(y.shape[0]) != self._n_cells:
                raise ValueError(f"labels length ({int(y.shape[0])}) must equal n_cells ({self._n_cells})")

            y = y.long()
            if self.device is not None:
                y = y.to(self.device)
            self.labels = y

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def obs_names(self):
        return self._obs_names

    def __len__(self) -> int:
        return self._n_cells

    # ---- row extraction ----
    def _get_X_row(self, adata_obj: AnnData, idx: int, layer: Optional[str], X_key: str) -> np.ndarray:
        X = _get_matrix(adata_obj, layer=layer, X_key=X_key)
        row = X[idx]
        if sp.issparse(row):
            row = row.toarray()
        return np.asarray(row).reshape(-1).astype(np.float32, copy=False)

    def _get_obs_label_row(self, adata_obj: AnnData, idx: int, obs_key: str) -> np.ndarray:
        """
        Read a single label code from adata.obs[obs_key] and return it as (1,) float32.
        We expect integer codes already. If you have strings, encode them first.
        """
        if obs_key not in adata_obj.obs:
            raise KeyError(f"obs_key={obs_key!r} not found in adata.obs columns.")

        col = adata_obj.obs[obs_key]

        if pd.api.types.is_categorical_dtype(col):
            v = int(col.cat.codes.iloc[idx])
            return np.asarray([v], dtype=np.float32)

        v = col.iloc[idx]
        if isinstance(v, (np.integer, int)):
            return np.asarray([int(v)], dtype=np.float32)
        if isinstance(v, (np.floating, float)):
            # allow float that is integer-valued (e.g. 3.0)
            return np.asarray([float(v)], dtype=np.float32)

        raise TypeError(
            f"adata.obs[{obs_key!r}] must be numeric integer codes (or pandas Categorical). "
            f"Got type {type(v)} at row {idx}. Encode categories to int codes first."
        )

    def __getitem__(self, idx: int):
        x_dict: Dict[str, torch.Tensor] = {}

        for name, adata_obj in self.adata_dict.items():
            mcfg = self.mod_cfg_by_name.get(name, None)

            # If this modality is categorical and configured to come from .obs, do that.
            if (
                mcfg is not None
                and _is_categorical_likelihood(mcfg.likelihood)
                and (mcfg.input_kind or "matrix") == "obs"
            ):
                if not mcfg.obs_key:
                    raise ValueError(f"Modality {name!r}: input_kind='obs' requires obs_key.")
                row_np = self._get_obs_label_row(adata_obj, idx, obs_key=mcfg.obs_key)
            else:
                layer = self.layer_by_modality.get(name, None)
                xkey = self.xkey_by_modality.get(name, "X")
                row_np = self._get_X_row(adata_obj, idx, layer=layer, X_key=xkey)

            x = torch.from_numpy(row_np).to(dtype=self.dtype)
            if self.device is not None:
                x = x.to(self.device)
            x_dict[name] = x

        if self.labels is None:
            return x_dict
        return x_dict, self.labels[idx]


# -----------------------------
# Factory helpers
# -----------------------------
def dataset_from_anndata_dict(
    adata_dict: Dict[str, AnnData],
    layer: LayerSpec = None,
    X_key: XKeySpec = "X",
    paired: bool = True,
    align_obs: bool = True,
    labels: Optional[Union[np.ndarray, torch.Tensor, Sequence[int]]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    copy_aligned: bool = True,
    modality_cfgs: Optional[List[ModalityConfig]] = None,
) -> Tuple[MultiModalDataset, Dict[str, AnnData]]:
    """
    Convenience wrapper to optionally align obs_names then build MultiModalDataset.

    Returns:
      (dataset, (possibly aligned) adata_dict)
    """
    if align_obs and paired:
        adata_dict = align_paired_obs_names(adata_dict, how="intersection", copy=copy_aligned)

    ds = MultiModalDataset(
        adata_dict=adata_dict,
        layer=layer,
        X_key=X_key,
        paired=paired,
        device=device,
        labels=labels,
        dtype=dtype,
        modality_cfgs=modality_cfgs,
    )
    return ds, adata_dict


def load_anndata_dict_from_config(
    modality_cfgs: List[Dict[str, Any]],
    data_root: Optional[str] = None,
) -> Dict[str, AnnData]:
    """
    Load a modality->AnnData dict from a list of config dicts.
    Expected keys per modality:
      - "name"
      - "h5ad_path"
    """
    out: Dict[str, AnnData] = {}
    for m in modality_cfgs:
        if "name" not in m or "h5ad_path" not in m:
            raise KeyError("Each modality config must contain keys: 'name' and 'h5ad_path'.")

        name = m["name"]
        path = m["h5ad_path"]

        if data_root is not None and not os.path.isabs(path):
            path = os.path.join(data_root, path)

        out[name] = ad.read_h5ad(path)

    if not out:
        raise ValueError("No modalities loaded (empty modality_cfgs?)")

    return out


def collate_multimodal_xy(batch):
    """
    Explicit collate:
      - works for [x_dict, ...] or [(x_dict, y), ...]
      - stacks per-modality tensors into (B, D)

    For categorical-from-obs modalities, D will be 1 (so batch -> (B,1)).
    """
    if isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2:
        xs, ys = zip(*batch)
        y = torch.as_tensor(ys, dtype=torch.long)
    else:
        xs, y = batch, None

    keys = xs[0].keys()
    x = {k: torch.stack([d[k] for d in xs], dim=0) for k in keys}
    return x if y is None else (x, y)

