# univi/data.py

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from anndata import AnnData


class MultiModalDataset(Dataset):
    """
    Simple multimodal dataset wrapping multiple AnnData objects.

    Assumes that, when `paired=True`, all modalities have the same number
    of cells and matching order of cells (same obs_names).

    Parameters
    ----------
    adata_dict : dict
        Dict mapping modality name -> AnnData (e.g. {"rna": rna_adata, "adt": adt_adata}).
    layer : str or None
        If not None, use this .layers[layer] for all modalities (typical for raw counts).
        If None, use `.X` or `.obsm[X_key]` depending on X_key.
    X_key : str
        If "X", use `.X`; otherwise use `.obsm[X_key]` (e.g. for precomputed embeddings).
    paired : bool
        If True, all modalities must have the same n_obs; __len__ returns that.
        (Unpaired mode can be extended later if you want.)
    device : str or torch.device or None
        Optional device to move tensors to. If None, leave on CPU.
    """

    def __init__(
        self,
        adata_dict: Dict[str, AnnData],
        layer: Optional[str] = None,
        X_key: str = "X",
        paired: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.adata_dict = adata_dict
        self.modalities = list(adata_dict.keys())
        self.layer = layer
        self.X_key = X_key
        self.paired = paired
        self.device = device

        # basic checks
        if len(self.modalities) == 0:
            raise ValueError("adata_dict must contain at least one modality.")

        if paired:
            n_cells = None
            ref_obs = None
            for name, adata in adata_dict.items():
                if n_cells is None:
                    n_cells = adata.n_obs
                    ref_obs = adata.obs_names
                else:
                    if adata.n_obs != n_cells:
                        raise ValueError(
                            f"All modalities must have same n_obs when paired=True; "
                            f"got {name} with n_obs={adata.n_obs} vs {n_cells}."
                        )
                    # optional: enforce matching obs_names
                    if not np.array_equal(adata.obs_names, ref_obs):
                        raise ValueError(
                            f"obs_names for modality '{name}' do not match reference."
                        )
            self.n_cells = n_cells
        else:
            # you could sum lengths or do something fancier for unpaired mode
            # For now we still require paired=True in the training code.
            raise NotImplementedError("Unpaired mode is not yet implemented.")

    def __len__(self) -> int:
        return self.n_cells

    def _get_X_row(self, adata: AnnData, idx: int) -> np.ndarray:
        """
        Extract one row as a dense float32 numpy array from AnnData.
        Respects `self.layer` and `self.X_key`.
        """
        if self.layer is not None:
            if self.layer not in adata.layers:
                raise KeyError(f"Layer '{self.layer}' not found in adata.layers.")
            X = adata.layers[self.layer]
        else:
            if self.X_key == "X":
                X = adata.X
            else:
                if self.X_key not in adata.obsm:
                    raise KeyError(f"X_key '{self.X_key}' not found in adata.obsm.")
                X = adata.obsm[self.X_key]

        if sp.issparse(X):
            row = X[idx].toarray().ravel()
        else:
            row = np.asarray(X[idx]).ravel()

        return row.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict mapping modality -> tensor of shape [n_features].
        """
        out: Dict[str, torch.Tensor] = {}
        for name, adata in self.adata_dict.items():
            row_np = self._get_X_row(adata, idx)
            t = torch.from_numpy(row_np)
            if self.device is not None:
                t = t.to(self.device)
            out[name] = t
        return out
