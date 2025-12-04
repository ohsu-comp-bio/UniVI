#!/usr/bin/env python
"""Train UniVI from a JSON config.

This is the manuscript/revision entry-point.

Key properties:
- Supports per-modality (layer, X_key) selection:
    * X_key == "X" uses adata.X or adata.layers[layer]
    * otherwise uses adata.obsm[X_key] (e.g. ATAC LSI in .obsm["X_lsi"])
- Does NOT rely on adata.uns["_univi"] (no hidden state)
- Passes objective controls (loss_mode, v1_recon, etc.) through to the model constructor
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import anndata as ad

from univi import UniVIMultiModalVAE, UniVIConfig, ModalityConfig, UniVITrainer, TrainingConfig
from univi.data import dataset_from_anndata_dict, infer_input_dim, align_paired_obs_names
from univi.utils.seed import set_seed
from univi.utils.io import save_checkpoint, save_config_json, load_config
from univi.utils.logging import get_logger

logger = get_logger("univi.train")


def split_train_val(dataset, train_fraction: float, seed: int = 0):
    n = len(dataset)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(train_fraction * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:] if n_train < n else None
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx) if val_idx is not None and len(val_idx) > 0 else None
    return train_ds, val_ds


def load_anndata_dict(modality_cfgs: List[Dict[str, Any]], data_root: str | None) -> Dict[str, ad.AnnData]:
    out: Dict[str, ad.AnnData] = {}
    for m in modality_cfgs:
        name = m["name"]
        path = m["h5ad_path"]
        if data_root is not None and not os.path.isabs(path):
            path = os.path.join(data_root, path)
        logger.info(f"Loading {name} from: {path}")
        out[name] = ad.read_h5ad(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--outdir", required=True, help="Output directory for checkpoints/logs")
    ap.add_argument("--data-root", default=None, help="Optional root to prepend to relative h5ad_path entries")
    ap.add_argument("--device", default=None, help="Override training.device (e.g. cuda, cuda:0, cpu)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.setdefault("training", {})
        cfg["training"]["device"] = args.device

    seed = int(cfg.get("training", {}).get("seed", 0))
    set_seed(seed, deterministic=False)
    logger.info(f"Seed = {seed}")

    modalities_cfg = cfg["data"]["modalities"]
    adata_dict = load_anndata_dict(modalities_cfg, data_root=args.data_root)

    # Enforce paired order on intersection (robust to filtering differences)
    adata_dict = align_paired_obs_names(adata_dict, how="intersection")

    layer_by = {m["name"]: m.get("layer", None) for m in modalities_cfg}
    xkey_by = {m["name"]: m.get("X_key", "X") for m in modalities_cfg}

    dataset, adata_dict = dataset_from_anndata_dict(
        adata_dict,
        layer=layer_by,
        X_key=xkey_by,
        paired=True,
        align_obs=False,
    )

    model_cfg = cfg.get("model", {})

    mod_cfgs: List[ModalityConfig] = []
    for m in modalities_cfg:
        name = m["name"]
        a = adata_dict[name]
        input_dim = infer_input_dim(a, layer=layer_by[name], X_key=xkey_by[name])

        hidden_default = model_cfg.get("hidden_dims_default", [256, 128])
        enc = list(m.get("encoder_hidden", m.get("hidden_dims", hidden_default)))
        dec = list(m.get("decoder_hidden", enc[::-1]))

        mod_cfgs.append(
            ModalityConfig(
                name=name,
                input_dim=int(input_dim),
                encoder_hidden=enc,
                decoder_hidden=dec,
                likelihood=m.get("likelihood", "gaussian"),
            )
        )

    univi_cfg = UniVIConfig(
        latent_dim=int(model_cfg.get("latent_dim", 32)),
        modalities=mod_cfgs,
        beta=float(model_cfg.get("beta", 1.0)),
        gamma=float(model_cfg.get("gamma", 1.0)),
        encoder_dropout=float(model_cfg.get("encoder_dropout", model_cfg.get("dropout", 0.0))),
        decoder_dropout=float(model_cfg.get("decoder_dropout", model_cfg.get("dropout", 0.0))),
        encoder_batchnorm=bool(model_cfg.get("encoder_batchnorm", model_cfg.get("batchnorm", True))),
        decoder_batchnorm=bool(model_cfg.get("decoder_batchnorm", False)),
        kl_anneal_start=int(model_cfg.get("kl_anneal_start", 0)),
        kl_anneal_end=int(model_cfg.get("kl_anneal_end", 0)),
        align_anneal_start=int(model_cfg.get("align_anneal_start", 0)),
        align_anneal_end=int(model_cfg.get("align_anneal_end", 0)),
    )

    model = UniVIMultiModalVAE(
        univi_cfg,
        loss_mode=model_cfg.get("loss_mode", "v2"),
        v1_recon=model_cfg.get("v1_recon", "cross"),
        v1_recon_mix=float(model_cfg.get("v1_recon_mix", 0.0)),
        normalize_v1_terms=bool(model_cfg.get("normalize_v1_terms", True)),
    )

    train_fraction = float(cfg.get("training", {}).get("train_fraction", 0.9))
    train_ds, val_ds = split_train_val(dataset, train_fraction, seed=seed)

    train_cfg = TrainingConfig(
        n_epochs=int(cfg.get("training", {}).get("n_epochs", 200)),
        batch_size=int(cfg.get("training", {}).get("batch_size", 256)),
        lr=float(cfg.get("training", {}).get("lr", 1e-3)),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 0.0)),
        grad_clip=cfg.get("training", {}).get("grad_clip", None),
        num_workers=int(cfg.get("training", {}).get("num_workers", 0)),
        device=str(cfg.get("training", {}).get("device", "cpu")),
        seed=seed,
        early_stopping=bool(cfg.get("training", {}).get("early_stopping", False)),
        patience=int(cfg.get("training", {}).get("patience", 20)),
        min_delta=float(cfg.get("training", {}).get("min_delta", 0.0)),
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers) if val_ds is not None else None

    trainer = UniVITrainer(model=model, train_loader=train_loader, val_loader=val_loader, train_cfg=train_cfg)

    hist = trainer.fit()

    ckpt_path = os.path.join(args.outdir, "univi_checkpoint.pt")
    opt_state = trainer.optimizer.state_dict() if hasattr(trainer, "optimizer") and trainer.optimizer is not None else None
    save_checkpoint(
        ckpt_path,
        model_state=trainer.model.state_dict(),
        optimizer_state=opt_state,
        extra={"history": hist, "univi_cfg": univi_cfg.__dict__},
    )
    logger.info(f"Saved checkpoint: {ckpt_path}")

    cfg_out = os.path.join(args.outdir, "config_resolved.json")
    save_config_json(cfg, cfg_out)
    logger.info(f"Saved config: {cfg_out}")


if __name__ == "__main__":
    main()
