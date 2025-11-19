#!/usr/bin/env python

"""
Evaluate a trained UniVI model on alignment + label transfer.

Usage:
    python scripts/evaluate_univi.py \
        --config parameter_files/defaults_cite_seq.json \
        --checkpoint path/to/univi_checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import torch
import anndata as ad

from torch.utils.data import DataLoader

from univi import UniVIMultiModalVAE, UniVIConfig, ModalityConfig
from univi.data import dataset_from_anndata_dict
from univi.evaluation import foscttm, knn_label_transfer_accuracy
from univi.utils.io import load_checkpoint
from univi.utils.seed import set_seed
from univi.utils.logging import get_logger
from univi.utils.torch_utils import to_numpy


logger = get_logger("univi.eval")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_anndata_dict(modality_cfgs):
    adatas = {}
    for m in modality_cfgs:
        name = m["name"]
        path = m["h5ad_path"]
        layer = m.get("layer", None)
        X_key = m.get("X_key", "X")
        a = ad.read_h5ad(path)
        a.uns.setdefault("_univi", {})
        a.uns["_univi"]["layer"] = layer
        a.uns["_univi"]["X_key"] = X_key
        adatas[name] = a
    return adatas


def build_univi_config(cfg: Dict[str, Any], adata_dict: Dict[str, ad.AnnData]) -> UniVIConfig:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    modalities_cfg = data_cfg["modalities"]

    modality_configs = []
    for m in modalities_cfg:
        name = m["name"]
        a = adata_dict[name]
        layer = a.uns["_univi"]["layer"]
        X_key = a.uns["_univi"]["X_key"]
        if layer is not None:
            X = a.layers[layer]
        else:
            X = getattr(a, X_key) if hasattr(a, X_key) else a.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
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

    return UniVIConfig(
        latent_dim=model_cfg["latent_dim"],
        modalities=modality_configs,
        beta=model_cfg.get("beta", 1.0),
        gamma=model_cfg.get("gamma", 1.0),
        kl_anneal_start=model_cfg.get("kl_anneal_start", 0),
        kl_anneal_end=model_cfg.get("kl_anneal_end", 0),
        align_anneal_start=model_cfg.get("align_anneal_start", 0),
        align_anneal_end=model_cfg.get("align_anneal_end", 0),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate UniVI model.")
    parser.add_argument("--config", type=str, required=True, help="Config JSON used for training.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--modality_pair", nargs=2, metavar=("MOD1", "MOD2"),
                        help="Two modality names to evaluate (e.g., rna adt).")
    parser.add_argument("--label_key", type=str, default=None,
                        help="obs column in AnnData for label-transfer evaluation (must exist in both modalities).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("training", {}).get("seed", 0)
    set_seed(seed)

    adata_dict = load_anndata_dict(cfg["data"]["modalities"])
    univi_cfg = build_univi_config(cfg, adata_dict)

    # Build dataset and loader
    first = next(iter(adata_dict.values()))
    dataset = dataset_from_anndata_dict(
        adata_dict,
        X_key=first.uns["_univi"]["X_key"],
        layer=first.uns["_univi"]["layer"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate model and load checkpoint
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = UniVIMultiModalVAE(univi_cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Collect latent embeddings per modality
    z_per_mod = {name: [] for name in model.modalities}

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Use deterministic means (mu) as embedding
            z_mu = model.encode_modalities(batch)
            for name, z in z_mu.items():
                z_per_mod[name].append(to_numpy(z))

    z_per_mod = {k: np.concatenate(v, axis=0) for k, v in z_per_mod.items()}

    # FOSCTTM between two modalities (if requested)
    if args.modality_pair is not None:
        m1, m2 = args.modality_pair
        if m1 not in z_per_mod or m2 not in z_per_mod:
            raise ValueError(f"Unknown modalities for evaluation: {m1}, {m2}")
        f = foscttm(z_per_mod[m1], z_per_mod[m2])
        logger.info(f"FOSCTTM({m1}, {m2}) = {f:.4f}")

    # Label-transfer accuracy (if label_key provided)
    if args.label_key is not None and args.modality_pair is not None:
        m1, m2 = args.modality_pair
        a1 = adata_dict[m1]
        a2 = adata_dict[m2]
        if args.label_key not in a1.obs.columns or args.label_key not in a2.obs.columns:
            raise ValueError(f"label_key '{args.label_key}' not found in both modalities.")

        # intersect obs_names ordering with dataset (dataset_from_anndata_dict already intersected)
        common = np.intersect1d(a1.obs_names, a2.obs_names)
        # reorder embeddings to match obs_names order (assuming dataset order == common)
        # if not, you might want to store the index mapping when building the dataset
        y1 = a1.obs.loc[common, args.label_key].values
        y2 = a2.obs.loc[common, args.label_key].values

        # For simplicity assume embeddings are ordered consistently; if not, adjust accordingly.
        z1 = z_per_mod[m1][: len(common)]
        z2 = z_per_mod[m2][: len(common)]

        acc_1to2 = knn_label_transfer_accuracy(z1, y1, z2, y2, k=15)
        acc_2to1 = knn_label_transfer_accuracy(z2, y2, z1, y1, k=15)
        logger.info(f"kNN label-transfer {m1}→{m2}: acc = {acc_1to2:.3f}")
        logger.info(f"kNN label-transfer {m2}→{m1}: acc = {acc_2to1:.3f}")


if __name__ == "__main__":
    main()
