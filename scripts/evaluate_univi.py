#!/usr/bin/env python
"""Evaluate a trained UniVI model on paired CITE-seq (RNA/ADT) style configs.

Outputs:
- metrics.json
- metrics.csv (single row, easy to stack)
- optional plots
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, Optional

import numpy as np

from univi.pipeline import load_model_and_data, encode_latents_paired
from univi.evaluation import compute_foscttm, label_transfer_knn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--modality-pair", nargs=2, default=["rna", "adt"], help="e.g. rna adt")
    ap.add_argument("--label-key", default="cell_type")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--metric", default="euclidean")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg, adata_dict, model, layer_by, xkey_by = load_model_and_data(
        args.config,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        align_obs=True,
    )

    m1, m2 = args.modality_pair
    Z = encode_latents_paired(
        model,
        {m1: adata_dict[m1], m2: adata_dict[m2]},
        layer_by={m1: layer_by.get(m1), m2: layer_by.get(m2)},
        xkey_by={m1: xkey_by.get(m1, "X"), m2: xkey_by.get(m2, "X")},
        batch_size=args.batch_size,
        device=args.device,
        fused=False,
    )
    Z1 = Z[m1]
    Z2 = Z[m2]

    fos = compute_foscttm(Z1, Z2, metric=args.metric)

    # Labels: prefer modality-specific; fall back to m1
    if args.label_key in adata_dict[m1].obs:
        y1 = np.asarray(adata_dict[m1].obs[args.label_key].values)
    else:
        raise KeyError(f"label_key={args.label_key!r} not found in adata_dict[{m1}].obs")

    if args.label_key in adata_dict[m2].obs:
        y2 = np.asarray(adata_dict[m2].obs[args.label_key].values)
    else:
        y2 = y1  # paired CITE-seq often shares labels

    pred_12, acc_12, _ = label_transfer_knn(Z1, y1, Z2, y2, k=15, metric=args.metric)
    pred_21, acc_21, _ = label_transfer_knn(Z2, y2, Z1, y1, k=15, metric=args.metric)

    metrics: Dict[str, Any] = {
        "foscttm": float(fos),
        "label_transfer_acc_%s_to_%s" % (m1, m2): None if acc_12 is None else float(acc_12),
        "label_transfer_acc_%s_to_%s" % (m2, m1): None if acc_21 is None else float(acc_21),
        "n_cells": int(Z1.shape[0]),
        "latent_dim": int(Z1.shape[1]),
        "modality_1": m1,
        "modality_2": m2,
    }

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    # also save a one-row CSV for stacking
    import pandas as pd
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
