#!/usr/bin/env python

"""
Train UniVI from a JSON config.

Usage:
    python scripts/train_univi.py --config parameter_files/defaults_cite_seq.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import anndata as ad

from univi import UniVIMultiModalVAE, UniVIConfig, ModalityConfig, UniVITrainer, TrainingConfig
from univi.data import dataset_from_anndata_dict
from univi.utils.seed import set_seed
from univi.utils.io import save_checkpoint, save_config_json
from univi.utils.logging import get_logger


logger = get_logger("univi.train")


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def load_anndata_dict(modality_cfgs: List[Dict[str, Any]]) -> Dict[str, ad.AnnData]:
    """
    Read one AnnData per modality from config list.
    Each modality entry should have:
      - name
      - h5ad_path
      - (optional) layer
      - (optional) X_key ("X" by default)
    """
    adatas: Dict[str, ad.AnnData] = {}
    for m in modality_cfgs:
        name = m["name"]
        path = m["h5ad_path"]
        layer = m.get("layer", None)
        X_key = m.get("X_key", "X")

        if not os.path.exists(path):
            raise FileNotFoundError(f"AnnData file for modality '{name}' not found: {path}")

        logger.info(f"Loading modality '{name}' from {path}")
        a = ad.read_h5ad(path)

        # Attach metadata for later
        a.uns.setdefault("_univi", {})
        a.uns["_univi"]["layer"] = layer
        a.uns["_univi"]["X_key"] = X_key

        adatas[name] = a

    return adatas


def build_dataset_and_modality_configs(
    cfg: Dict[str, Any]
) -> Tuple[Dict[str, ad.AnnData], UniVIConfig]:
    """
    - Load AnnData for each modality
    - Build MultiModalDataset
    - Build UniVIConfig + ModalityConfig list
    """
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    modalities_cfg = data_cfg["modalities"]
    adata_dict = load_anndata_dict(modalities_cfg)

    # Build dataset (intersects obs_names)
    # We assume each AnnData knows which layer or X_key to use via uns
    first_adata = next(iter(adata_dict.values()))
    default_layer = first_adata.uns["_univi"]["layer"]
    default_X_key = first_adata.uns["_univi"]["X_key"]

    dataset = dataset_from_anndata_dict(
        adata_dict,
        X_key=default_X_key,
        layer=default_layer,
    )

    # Build modality configs (input_dims inferred from AnnData)
    modality_configs: List[ModalityConfig] = []
    for m in modalities_cfg:
        name = m["name"]
        a = adata_dict[name]

        # Determine feature dimension using same layer/X_key as dataset
        layer = a.uns["_univi"]["layer"]
        X_key = a.uns["_univi"]["X_key"]
        if layer is not None:
            X = a.layers[layer]
        else:
            if hasattr(a, X_key):
                X = getattr(a, X_key)
            else:
                X = a.X
        if hasattr(X, "shape"):
            input_dim = X.shape[1]
        else:
            X = np.asarray(X)
            input_dim = X.shape[1]

        hidden_dims = m.get("hidden_dims", model_cfg.get("hidden_dims_default", [256, 128]))

        modality_configs.append(
            ModalityConfig(
                name=name,
                input_dim=int(input_dim),
                hidden_dims=hidden_dims,
                dropout=model_cfg.get("dropout", 0.1),
                batchnorm=model_cfg.get("batchnorm", True),
                likelihood=m.get("likelihood", "gaussian"),
            )
        )

        logger.info(
            f"Modality '{name}': input_dim={input_dim}, hidden_dims={hidden_dims}, "
            f"likelihood={m.get('likelihood', 'gaussian')}"
        )

    univi_cfg = UniVIConfig(
        latent_dim=model_cfg["latent_dim"],
        modalities=modality_configs,
        beta=model_cfg.get("beta", 1.0),
        gamma=model_cfg.get("gamma", 1.0),
        kl_anneal_start=model_cfg.get("kl_anneal_start", 0),
        kl_anneal_end=model_cfg.get("kl_anneal_end", 0),
        align_anneal_start=model_cfg.get("align_anneal_start", 0),
        align_anneal_end=model_cfg.get("align_anneal_end", 0),
    )

    return {"adata_dict": adata_dict, "dataset": dataset}, univi_cfg


def split_train_val(
    dataset,
    train_fraction: float,
    seed: int = 0,
):
    n = len(dataset)
    n_train = int(train_fraction * n)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist()) if len(val_idx) > 0 else None
    return train_ds, val_ds


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train UniVI from config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file (e.g., parameter_files/defaults_cite_seq.json).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get("output", {}).get("dir", "univi_runs")
    os.makedirs(out_dir, exist_ok=True)

    seed = cfg.get("training", {}).get("seed", 0)
    set_seed(seed, deterministic=False)
    logger.info(f"Using seed = {seed}")

    # Build data + model config
    data_model, univi_cfg = build_dataset_and_modality_configs(cfg)
    dataset = data_model["dataset"]

    # Train/val split
    train_fraction = cfg["training"].get("train_fraction", 0.9)
    train_ds, val_ds = split_train_val(dataset, train_fraction, seed=seed)

    batch_size = cfg["training"].get("batch_size", 256)
    num_workers = cfg["training"].get("num_workers", 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if val_ds is not None
        else None
    )

    # Instantiate trainer
    model = UniVIMultiModalVAE(univi_cfg)

    train_cfg = TrainingConfig(
        n_epochs=cfg["training"].get("n_epochs", 200),
        lr=cfg["training"].get("lr", 1e-3),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
        grad_clip=cfg["training"].get("grad_clip", None),
        device=cfg["training"].get("device", "cuda"),
        log_every=cfg["training"].get("log_every", 10),
    )

    trainer = UniVITrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=train_cfg,
    )

    # Instantiate model
    loss_mode = cfg["model"].get("loss_mode", "lite")  # or "v1"
    v1_recon = cfg["model"].get("v1_recon", "cross")
    v1_recon_mix = cfg["model"].get("v1_recon_mix", 0.0)
    normalize_v1_terms = cfg["model"].get("normalize_v1_terms", True)

    model = UniVIMultiModalVAE(
        univi_cfg,
        loss_mode=loss_mode,
        v1_recon=v1_recon,
        v1_recon_mix=v1_recon_mix,
        normalize_v1_terms=normalize_v1_terms,
    ).to(train_cfg.device)

    logger.info("Starting training...")
    trainer.fit()
    logger.info("Training finished.")

    # Save checkpoint + config
    ckpt_path = os.path.join(out_dir, "univi_checkpoint.pt")
    save_checkpoint(
        ckpt_path,
        model_state=model.state_dict(),
        optimizer_state=trainer.optimizer.state_dict(),
        extra={
            "config_path": os.path.abspath(args.config),
            "loss_mode": loss_mode,
            "v1_recon": v1_recon,
            "v1_recon_mix": v1_recon_mix,
            "normalize_v1_terms": normalize_v1_terms,
        },
    )
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # Save a copy of the config used
    cfg_out_path = os.path.join(out_dir, "config_used.json")
    save_config_json(cfg, cfg_out_path)
    logger.info(f"Saved config snapshot to {cfg_out_path}")


if __name__ == "__main__":
    main()
