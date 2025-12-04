#!/usr/bin/env python
"""Unified benchmarking wrapper (revision helper).

This script provides a consistent CLI to run:
- UniVI (paired cross-modality metrics: FOSCTTM + label transfer)
- Harmony baseline (within-modality *batch correction* on RNA PCs)

Note:
Harmony is not a cross-modality method. For transparency, the Harmony baseline
here corresponds to the common use case: batch correction inside one modality.

Outputs a metrics.json + metrics.csv in the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from univi.pipeline import load_model_and_data, encode_latents_paired
from univi.evaluation import compute_foscttm, label_transfer_knn, compute_modality_mixing


def run_univi(args) -> Dict[str, Any]:
    cfg, adata_dict, model, layer_by, xkey_by = load_model_and_data(
        args.config,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        align_obs=True,
    )
    m1, m2 = args.m1, args.m2
    Z = encode_latents_paired(
        model,
        {m1: adata_dict[m1], m2: adata_dict[m2]},
        layer_by={m1: layer_by.get(m1), m2: layer_by.get(m2)},
        xkey_by={m1: xkey_by.get(m1, "X"), m2: xkey_by.get(m2, "X")},
        batch_size=args.batch_size,
        device=args.device,
        fused=True,
    )
    Z1, Z2 = Z[m1], Z[m2]
    fos = compute_foscttm(Z1, Z2, metric=args.metric)

    # label transfer
    if args.label_key not in adata_dict[m1].obs:
        raise KeyError(f"label_key={args.label_key!r} missing from {m1}.obs")
    y1 = np.asarray(adata_dict[m1].obs[args.label_key].values)
    y2 = np.asarray(adata_dict[m2].obs[args.label_key].values) if args.label_key in adata_dict[m2].obs else y1

    _, acc12, _ = label_transfer_knn(Z1, y1, Z2, y2, k=args.k, metric=args.metric)
    _, acc21, _ = label_transfer_knn(Z2, y2, Z1, y1, k=args.k, metric=args.metric)

    # modality mixing on fused latent
    fused = Z.get("fused", None)
    mix = np.nan
    if fused is not None:
        modality_labels = np.array([m1] * fused.shape[0] + [m2] * fused.shape[0])
        Zmix = np.vstack([fused, fused])  # same coords, used only for API convenience
        mix = compute_modality_mixing(Zmix, modality_labels, k=min(20, fused.shape[0]-1 if fused.shape[0]>1 else 1), metric=args.metric)

    return {
        "method": "univi",
        "foscttm": float(fos),
        f"label_transfer_acc_{m1}_to_{m2}": float(acc12) if acc12 is not None else np.nan,
        f"label_transfer_acc_{m2}_to_{m1}": float(acc21) if acc21 is not None else np.nan,
        "modality_mixing_proxy": float(mix),
        "n_cells": int(Z1.shape[0]),
        "latent_dim": int(Z1.shape[1]),
    }


def run_harmony(args) -> Dict[str, Any]:
    try:
        import harmonypy as hm
    except Exception as e:
        raise SystemExit("Harmony baseline requested but 'harmonypy' is not installed. Try: pip install harmonypy") from e

    import scanpy as sc

    cfg, adata_dict, model, layer_by, xkey_by = load_model_and_data(
        args.config,
        checkpoint_path=None,
        data_root=args.data_root,
        device="cpu",
        align_obs=False,
    )
    if args.batch_key is None:
        raise SystemExit("--batch-key is required for --method harmony")

    # Use RNA only
    adata = adata_dict[args.m1].copy()
    if args.batch_key not in adata.obs:
        raise SystemExit(f"batch_key {args.batch_key!r} not found in adata.obs")

    # Basic preprocessing: PCA on X (or layer) described in config
    layer = None
    xkey = "X"
    for m in cfg["data"]["modalities"]:
        if m["name"] == args.m1:
            layer = m.get("layer", None)
            xkey = m.get("X_key", "X")
            break

    # ensure PCA basis exists
    sc.pp.pca(adata, n_comps=args.n_pcs, use_highly_variable=False, svd_solver="arpack", layer=layer if xkey=="X" else None)
    pcs = adata.obsm["X_pca"]
    meta = adata.obs.copy()

    ho = hm.run_harmony(pcs, meta_data=meta, vars_use=[args.batch_key])
    Zcorr = ho.Z_corr.T  # (cells, pcs)

    mixing = compute_modality_mixing(Zcorr, np.asarray(meta[args.batch_key].values), k=min(20, Zcorr.shape[0]-1 if Zcorr.shape[0]>1 else 1), metric=args.metric)

    return {
        "method": "harmony",
        "task": "batch_correction_rna",
        "batch_key": args.batch_key,
        "batch_mixing_score": float(mixing),
        "n_cells": int(Zcorr.shape[0]),
        "n_pcs": int(Zcorr.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["univi", "harmony"], default="univi")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="Required for method=univi")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data-root", default=None)

    ap.add_argument("--m1", default="rna")
    ap.add_argument("--m2", default="adt")
    ap.add_argument("--label-key", default="cell_type")
    ap.add_argument("--batch-key", default=None, help="For harmony baseline only (within-modality batch correction).")
    ap.add_argument("--n-pcs", type=int, default=50)

    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--metric", default="euclidean")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.method == "univi":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --method univi")
        metrics = run_univi(args)
    else:
        metrics = run_harmony(args)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
