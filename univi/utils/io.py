# univi/utils/io.py

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union, Mapping

import os
import json

import numpy as np
import torch
import scipy.sparse as sp
import anndata as ad


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {"model_state": model_state}
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def save_config_json(config: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = json.load(f)

    cfg.setdefault("model", {})
    model_cfg = cfg["model"]

    # Backwards-compat defaults
    model_cfg.setdefault("loss_mode", "v1")
    model_cfg.setdefault("v1_recon", "cross")
    model_cfg.setdefault("v1_recon_mix", 0.0)
    model_cfg.setdefault("normalize_v1_terms", True)

    # Old configs used a single dropout
    base_do = model_cfg.get("dropout", 0.0)
    model_cfg.setdefault("encoder_dropout", base_do)
    model_cfg.setdefault("decoder_dropout", base_do)

    model_cfg.setdefault("beta", 5.0)
    model_cfg.setdefault("gamma", 60.0)

    if "training" in cfg:
        cfg["training"].setdefault("device", "cuda")

    return cfg


def save_anndata_splits(
    adata: ad.AnnData,
    outdir: str,
    prefix: str = "dataset",
    split_key: Optional[str] = "split",
    split_map: Optional[Dict[str, Union[Sequence[int], Sequence[str]]]] = None,
    train_label: str = "train",
    val_label: str = "val",
    test_label: str = "test",
    copy: bool = True,
    write_backed: bool = False,
) -> Dict[str, ad.AnnData]:
    os.makedirs(outdir, exist_ok=True)

    def subset_from_idx(idx_list: Sequence[Union[int, str]]) -> ad.AnnData:
        if len(idx_list) == 0:
            return adata[:0].copy() if copy else adata[:0]
        first = idx_list[0]
        if isinstance(first, str):
            return adata[list(idx_list)]
        return adata[list(idx_list), :]

    if split_map is not None:
        train = subset_from_idx(list(split_map.get("train", [])))
        val = subset_from_idx(list(split_map.get("val", [])))
        test = subset_from_idx(list(split_map.get("test", [])))
    else:
        if split_key is None or split_key not in adata.obs:
            raise ValueError(
                "Expected split labels in adata.obs[%r], or provide split_map={'train':...,'val':...,'test':...}."
                % split_key
            )
        s = adata.obs[split_key].astype(str)
        train = adata[s == train_label]
        val = adata[s == val_label]
        test = adata[s == test_label]

    if copy:
        train = train.copy()
        val = val.copy()
        test = test.copy()

    paths = {
        "train": os.path.join(outdir, "%s_train.h5ad" % prefix),
        "val": os.path.join(outdir, "%s_val.h5ad" % prefix),
        "test": os.path.join(outdir, "%s_test.h5ad" % prefix),
    }

    train.write_h5ad(paths["train"])
    val.write_h5ad(paths["val"])
    test.write_h5ad(paths["test"])

    return {"train": train, "val": val, "test": test}


def _select_X(
    adata_obj: ad.AnnData,
    layer: Optional[str],
    X_key: str,
):
    # X_key != "X" => obsm
    if X_key != "X":
        if X_key not in adata_obj.obsm:
            raise KeyError("X_key=%r not found in adata.obsm. Keys=%s" % (X_key, list(adata_obj.obsm.keys())))
        return adata_obj.obsm[X_key]

    # X_key == "X" => layers[layer] or X
    if layer is not None:
        if layer not in adata_obj.layers:
            raise KeyError("layer=%r not found in adata.layers. Keys=%s" % (layer, list(adata_obj.layers.keys())))
        return adata_obj.layers[layer]
    return adata_obj.X


@torch.no_grad()
def write_univi_latent(
    model,
    adata_dict: Dict[str, ad.AnnData],
    *,
    obsm_key: str = "X_univi",
    batch_size: int = 512,
    device: Optional[Union[str, torch.device]] = None,
    use_mean: bool = False,
    # NEW: support layer / X_key (global or per-modality)
    layer: Union[None, str, Mapping[str, Optional[str]]] = None,
    X_key: Union[str, Mapping[str, str]] = "X",
    require_paired: bool = True,
) -> np.ndarray:
    """
    Encode paired modalities and write a *shared/fused* latent Z to each adata.obsm[obsm_key].

    Notes
    -----
    - This writes THE SAME Z into every modality AnnData (because it's fused).
    - If you want modality-specific means (mu_modality), use evaluation.encode_adata(..., latent="modality_mean").
    """
    model.eval()

    names = list(adata_dict.keys())
    if len(names) == 0:
        raise ValueError("adata_dict is empty.")

    n = int(adata_dict[names[0]].n_obs)

    if require_paired:
        ref = adata_dict[names[0]].obs_names.values
        for nm in names[1:]:
            if adata_dict[nm].n_obs != n:
                raise ValueError("n_obs mismatch: %s has %d vs %d" % (nm, adata_dict[nm].n_obs, n))
            if not np.array_equal(adata_dict[nm].obs_names.values, ref):
                raise ValueError("obs_names mismatch between %s and %s" % (names[0], nm))

    # device default: model device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    # normalize layer/X_key to per-modality mappings
    if isinstance(layer, dict):
        layer_by_mod = dict(layer)
    else:
        layer_by_mod = {nm: layer for nm in names}

    if isinstance(X_key, dict):
        xkey_by_mod = dict(X_key)
    else:
        xkey_by_mod = {nm: X_key for nm in names}

    zs = []
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        x_dict = {}

        for nm in names:
            adata_obj = adata_dict[nm]
            X = _select_X(adata_obj, layer_by_mod.get(nm, None), xkey_by_mod.get(nm, "X"))
            xb = X[start:end]
            if sp.issparse(xb):
                xb = xb.toarray()
            else:
                xb = np.asarray(xb)
            x_dict[nm] = torch.as_tensor(xb, dtype=torch.float32, device=device)

        # forward compat: some forwards accept epoch or y; safest is plain model(x_dict)
        out = model(x_dict)
        z = out["mu_z"] if (use_mean and ("mu_z" in out)) else out["z"]
        zs.append(z.detach().cpu().numpy())

    Z = np.vstack(zs)

    for nm in names:
        adata_dict[nm].obsm[obsm_key] = Z

    return Z

