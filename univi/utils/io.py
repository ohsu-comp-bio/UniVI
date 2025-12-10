# univi/utils/io.py

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union, Mapping

import os
import json

import numpy as np
import torch
import scipy.sparse as sp
import anndata as ad

SplitMap = Dict[str, Any]


# ---------------------------------------------------------------------
# Checkpointing (backwards compatible, plus model/optimizer convenience)
# ---------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model_state: Optional[Dict[str, Any]] = None,
    optimizer_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    trainer_state: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    strict_label_compat: bool = True,
) -> None:
    if model_state is None and model is not None:
        model_state = model.state_dict()
    if optimizer_state is None and optimizer is not None:
        optimizer_state = optimizer.state_dict()

    if model_state is None:
        raise ValueError("save_checkpoint requires either model_state=... or model=...")

    payload: Dict[str, Any] = {
        "format_version": 3,
        "model_state": model_state,
    }

    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if trainer_state is not None:
        payload["trainer_state"] = dict(trainer_state)
    if scaler_state is not None:
        payload["scaler_state"] = dict(scaler_state)
    if config is not None:
        payload["config"] = dict(config)
    if extra is not None:
        payload["extra"] = dict(extra)

    # --- classification metadata (legacy + multi-head) ---
    if model is not None:
        meta: Dict[str, Any] = {}

        # legacy
        n_label_classes = getattr(model, "n_label_classes", None)
        label_names = getattr(model, "label_names", None)
        label_head_name = getattr(model, "label_head_name", None)

        if n_label_classes is not None:
            meta.setdefault("legacy", {})["n_label_classes"] = int(n_label_classes)
        if label_names is not None:
            meta.setdefault("legacy", {})["label_names"] = list(label_names)
        if label_head_name is not None:
            meta["label_head_name"] = str(label_head_name)

        # multi-head (if available)
        class_heads_cfg = getattr(model, "class_heads_cfg", None)
        head_label_names = getattr(model, "head_label_names", None)

        if isinstance(class_heads_cfg, dict) and len(class_heads_cfg) > 0:
            meta.setdefault("multi", {})["heads"] = {k: dict(v) for k, v in class_heads_cfg.items()}
        if isinstance(head_label_names, dict) and len(head_label_names) > 0:
            meta.setdefault("multi", {})["label_names"] = {k: list(v) for k, v in head_label_names.items()}

        # alternate API if you prefer
        if hasattr(model, "get_classification_meta"):
            try:
                meta = dict(model.get_classification_meta())
            except Exception:
                pass

        if meta:
            payload["label_meta"] = meta
            payload["strict_label_compat"] = bool(strict_label_compat)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, *, map_location: Union[str, torch.device, None] = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def restore_checkpoint(
    payload_or_path: Union[str, Dict[str, Any]],
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: Union[str, torch.device, None] = "cpu",
    strict: bool = True,
    restore_label_names: bool = True,
    enforce_label_compat: bool = True,
) -> Dict[str, Any]:
    payload = load_checkpoint(payload_or_path, map_location=map_location) if isinstance(payload_or_path, str) else payload_or_path

    # compatibility check
    if enforce_label_compat:
        meta = payload.get("label_meta", {}) or {}

        # legacy
        legacy = meta.get("legacy", {}) if isinstance(meta, dict) else {}
        ckpt_n = legacy.get("n_label_classes", None)
        model_n = getattr(model, "n_label_classes", None)
        if ckpt_n is not None and model_n is not None and int(ckpt_n) != int(model_n):
            raise ValueError(
                f"Checkpoint n_label_classes={ckpt_n} does not match model n_label_classes={model_n}. "
                "Rebuild the model with the same n_label_classes."
            )

        # multi-head
        multi = meta.get("multi", {}) if isinstance(meta, dict) else {}
        ckpt_heads = multi.get("heads", None)
        model_heads = getattr(model, "class_heads_cfg", None)
        if isinstance(ckpt_heads, dict) and isinstance(model_heads, dict):
            for hk, hcfg in ckpt_heads.items():
                if hk not in model_heads:
                    raise ValueError(
                        f"Checkpoint contains head {hk!r} but model does not. "
                        f"Model heads: {list(model_heads.keys())}"
                    )
                ckpt_c = int(hcfg.get("n_classes", -1))
                model_c = int(model_heads[hk].get("n_classes", -1))
                if ckpt_c != model_c:
                    raise ValueError(
                        f"Head {hk!r} n_classes mismatch: checkpoint={ckpt_c}, model={model_c}."
                    )

    model.load_state_dict(payload["model_state"], strict=bool(strict))

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    if scaler is not None and "scaler_state" in payload and payload["scaler_state"] is not None:
        try:
            scaler.load_state_dict(payload["scaler_state"])
        except Exception:
            pass

    if restore_label_names:
        meta = payload.get("label_meta", {}) or {}

        legacy = meta.get("legacy", {}) if isinstance(meta, dict) else {}
        label_names = legacy.get("label_names", None)
        if label_names is not None and hasattr(model, "set_label_names"):
            try:
                model.set_label_names(list(label_names))
            except Exception:
                pass

        multi = meta.get("multi", {}) if isinstance(meta, dict) else {}
        head_names = multi.get("label_names", None)
        if isinstance(head_names, dict) and hasattr(model, "set_head_label_names"):
            for hk, names in head_names.items():
                try:
                    model.set_head_label_names(str(hk), list(names))
                except Exception:
                    pass

    return payload


# ---------------------------------------------------------------------
# JSON config helpers
# ---------------------------------------------------------------------

def save_config_json(config: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg


# ---------------------------------------------------------------------
# AnnData split helpers
# ---------------------------------------------------------------------

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
    *,
    save_split_map: bool = True,
    split_map_name: Optional[str] = None,
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
        "train": os.path.join(outdir, f"{prefix}_train.h5ad"),
        "val": os.path.join(outdir, f"{prefix}_val.h5ad"),
        "test": os.path.join(outdir, f"{prefix}_test.h5ad"),
    }

    train.write_h5ad(paths["train"])
    val.write_h5ad(paths["val"])
    test.write_h5ad(paths["test"])

    if save_split_map:
        sm = {
            "train": train.obs_names.tolist(),
            "val": val.obs_names.tolist(),
            "test": test.obs_names.tolist(),
            "split_key": split_key,
            "train_label": train_label,
            "val_label": val_label,
            "test_label": test_label,
        }
        fn = split_map_name or f"{prefix}_split_map.json"
        with open(os.path.join(outdir, fn), "w") as f:
            json.dump(sm, f, indent=2)

    return {"train": train, "val": val, "test": test}


def load_split_map(outdir: str, prefix: str = "dataset", split_map_name: Optional[str] = None) -> Dict[str, Any]:
    fn = split_map_name or f"{prefix}_split_map.json"
    path = os.path.join(outdir, fn)
    with open(path) as f:
        return json.load(f)


def load_anndata_splits(outdir: str, prefix: str = "dataset") -> Dict[str, ad.AnnData]:
    paths = {
        "train": os.path.join(outdir, f"{prefix}_train.h5ad"),
        "val": os.path.join(outdir, f"{prefix}_val.h5ad"),
        "test": os.path.join(outdir, f"{prefix}_test.h5ad"),
    }
    return {k: ad.read_h5ad(p) for k, p in paths.items()}


# ---------------------------------------------------------------------
# Latent writer (uses encode_fused if available, supports epoch/y dict)
# ---------------------------------------------------------------------

def _select_X(
    adata_obj: ad.AnnData,
    layer: Optional[str],
    X_key: str,
):
    if X_key != "X":
        if X_key not in adata_obj.obsm:
            raise KeyError("X_key=%r not found in adata.obsm. Keys=%s" % (X_key, list(adata_obj.obsm.keys())))
        return adata_obj.obsm[X_key]

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
    epoch: int = 0,
    y: Optional[Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]] = None,
    layer: Union[None, str, Mapping[str, Optional[str]]] = None,
    X_key: Union[str, Mapping[str, str]] = "X",
    require_paired: bool = True,
) -> np.ndarray:
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

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    layer_by_mod = dict(layer) if isinstance(layer, dict) else {nm: layer for nm in names}
    xkey_by_mod = dict(X_key) if isinstance(X_key, dict) else {nm: X_key for nm in names}

    # normalize y (optional): tensor or dict of tensors
    y_t = None
    if y is not None:
        if isinstance(y, dict):
            y_t = {}
            for k, v in y.items():
                tt = v if torch.is_tensor(v) else torch.as_tensor(v)
                y_t[str(k)] = tt.long()
        else:
            tt = y if torch.is_tensor(y) else torch.as_tensor(y)
            y_t = tt.long()

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
            xb = np.asarray(xb)
            x_dict[nm] = torch.as_tensor(xb, dtype=torch.float32, device=device)

        yb = None
        if isinstance(y_t, dict):
            yb = {k: v[start:end].to(device) for k, v in y_t.items()}
        elif torch.is_tensor(y_t):
            yb = y_t[start:end].to(device)

        if hasattr(model, "encode_fused"):
            mu_z, logvar_z, z = model.encode_fused(x_dict, epoch=int(epoch), y=yb, use_mean=bool(use_mean))
            z_use = z
        else:
            out = model(x_dict)  # fallback
            z_use = out["mu_z"] if (use_mean and ("mu_z" in out)) else out["z"]

        zs.append(z_use.detach().cpu().numpy())

    Z = np.vstack(zs)

    for nm in names:
        adata_dict[nm].obsm[obsm_key] = Z

    return Z


