# univi/utils/io.py

from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Union
import os
import json
import torch
import scanpy as sc
import anndata as ad


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    payload: Dict[str, Any] = {
        "model_state": model_state,
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")


def save_config_json(config: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

def load_config(config_path: str):
    import json
    with open(config_path) as f:
        cfg = json.load(f)

    # Ensure model section exists
    if "model" not in cfg:
        cfg["model"] = {}

    model_cfg = cfg["model"]

    # Default to "v1" if old JSON missing these
    model_cfg.setdefault("loss_mode", "v1")
    model_cfg.setdefault("v1_recon", "cross")
    model_cfg.setdefault("v1_recon_mix", 0.0)
    model_cfg.setdefault("normalize_v1_terms", True)

    # --- Dropout split (backwards compatible) ---
    # Old configs used a single "dropout". New configs can specify both.
    base_do = model_cfg.get("dropout", 0.0)
    model_cfg.setdefault("encoder_dropout", base_do)
    model_cfg.setdefault("decoder_dropout", base_do)

    # Default align/beta/gamma values
    model_cfg.setdefault("beta", 5.0)
    model_cfg.setdefault("gamma", 60.0)

    # Optional backwards compat for missing training keys
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
    """
    Save train/val/test splits as separate AnnData objects for downstream use.

    You can provide splits in one of two ways:

    1) split_key mode (recommended):
        - adata.obs[split_key] contains labels like {"train","val","test"} (customizable)

    2) split_map mode:
        - split_map is a dict like:
          {"train": train_indices, "val": val_indices, "test": test_indices}
          where indices can be obs_names (strings) or integer positions.

    Writes:
        {outdir}/{prefix}_train.h5ad
        {outdir}/{prefix}_val.h5ad
        {outdir}/{prefix}_test.h5ad

    Returns:
        dict with keys {"train","val","test"} to AnnData views/copies.
    """
    os.makedirs(outdir, exist_ok=True)

    # -------------------------
    # Build index masks
    # -------------------------
    if split_map is not None:
        # indices can be obs_names or integer positions
        def subset_from_idx(idx):
            if len(idx) == 0:
                return adata[:0].copy() if copy else adata[:0]
            first = idx[0]
            if isinstance(first, str):
                return adata[idx]
            else:
                return adata[idx, :]
        train = subset_from_idx(list(split_map.get("train", [])))
        val   = subset_from_idx(list(split_map.get("val", [])))
        test  = subset_from_idx(list(split_map.get("test", [])))

    else:
        if split_key is None or split_key not in adata.obs:
            raise ValueError(
                f"Expected split labels in adata.obs['{split_key}'], "
                f"or provide split_map={{'train':..., 'val':..., 'test':...}}."
            )

        s = adata.obs[split_key].astype(str)
        train = adata[s == train_label]
        val   = adata[s == val_label]
        test  = adata[s == test_label]

    # Optionally materialize (views -> real objects)
    if copy:
        train = train.copy()
        val   = val.copy()
        test  = test.copy()

    # -------------------------
    # Write to disk
    # -------------------------
    paths = {
        "train": os.path.join(outdir, f"{prefix}_train.h5ad"),
        "val":   os.path.join(outdir, f"{prefix}_val.h5ad"),
        "test":  os.path.join(outdir, f"{prefix}_test.h5ad"),
    }

    if write_backed:
        # backed writing avoids loading full X into memory in some cases
        train.write_h5ad(paths["train"])
        val.write_h5ad(paths["val"])
        test.write_h5ad(paths["test"])
    else:
        # normal write
        train.write_h5ad(paths["train"])
        val.write_h5ad(paths["val"])
        test.write_h5ad(paths["test"])

    return {"train": train, "val": val, "test": test}

