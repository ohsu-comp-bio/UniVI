# univi/evaluation.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union, Mapping, Sequence, List

import numpy as np
import scipy.sparse as sp
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    roc_auc_score,
)
from sklearn.cluster import KMeans


# =============================================================================
# Helpers
# =============================================================================
def to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def _mean_sem(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0, 0.0
    if x.size == 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars/arrays into JSON-safe python types."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _is_tensor(x: Any) -> bool:
    return torch.is_tensor(x)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


def _extract_tensor_from_decoder_output(
    out: Any,
    *,
    prefer: str = "auto",
    apply_link: Optional[str] = None,
) -> torch.Tensor:
    """
    Turn a decoder output into a tensor suitable for returning/storing.

    out:
      - torch.Tensor OR a dict produced by a decoder.

    prefer:
      - "auto" : pick sensible default by keys
      - "mean" : pick mean-like for Gaussian/GaussianDiag
      - "mu"   : pick mu-like for NB/ZINB
      - "rate" : pick rate for Poisson
      - "logits": pick logits for Bernoulli/Categorical etc.
      - "probs": pick probs if available, otherwise compute if possible

    apply_link:
      - None
      - "sigmoid": logits -> probs
      - "softmax": logits -> probs
      - "softplus": log_rate -> rate
    """
    if _is_tensor(out):
        x = out
    elif isinstance(out, dict):
        p = str(prefer).lower().strip()

        if p in out and _is_tensor(out[p]):
            x = out[p]
        else:
            key_order_auto = [
                "mean", "mu", "rate", "probs", "logits", "log_rate", "logvar",
            ]
            if p == "probs":
                key_order = ["probs", "logits", "mean", "mu", "rate", "log_rate"]
            elif p == "logits":
                key_order = ["logits", "mean", "mu", "rate", "log_rate", "probs"]
            elif p == "mean":
                key_order = ["mean", "mu", "rate", "probs", "logits", "log_rate"]
            elif p == "mu":
                key_order = ["mu", "mean", "rate", "probs", "logits", "log_rate"]
            elif p == "rate":
                key_order = ["rate", "log_rate", "mu", "mean", "probs", "logits"]
            else:
                key_order = key_order_auto

            x = None
            for k in key_order:
                if k in out and _is_tensor(out[k]):
                    x = out[k]
                    break

            if x is None:
                if len(out) == 1:
                    x = next(iter(out.values()))
                    if not _is_tensor(x):
                        raise TypeError(f"Decoder output dict has non-tensor leaf: {type(x)}")
                else:
                    raise TypeError(f"Could not extract tensor from decoder output keys={list(out.keys())}")
    else:
        raise TypeError(f"Unsupported decoder output type: {type(out)}")

    if apply_link is not None:
        link = str(apply_link).lower().strip()
        if link == "sigmoid":
            x = _sigmoid(x)
        elif link == "softmax":
            x = _softmax(x, dim=-1)
        elif link == "softplus":
            x = torch.nn.functional.softplus(x)
        else:
            raise ValueError(f"Unknown apply_link={apply_link!r}")

    return x


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# =============================================================================
# Manuscript helpers (LSC17 score)
# =============================================================================
LSC17_GENES = [
    "DNMT3B", "ZBTB46", "NYNRIN", "ARHGAP22", "LAPTM4B", "MMRN1", "DPYSL3", "FAM30A",
    "CDK6", "CPXM1", "SOCS2", "SMIM24", "EMP1", "BEX3", "CD34", "AKR1C3", "ADGRG1",
]


def add_lsc17_scores(
    adata,
    layer: Optional[str] = None,
    gene_key: Optional[str] = None,
    prefix: str = "LSC17",
) -> Any:
    var_names = np.asarray(adata.var_names if gene_key is None else adata.var[gene_key])
    gene_to_idx = {g: i for i, g in enumerate(var_names)}
    present = [g for g in LSC17_GENES if g in gene_to_idx]
    idx = [gene_to_idx[g] for g in present]

    X = adata.X if layer is None else (adata.layers[layer] if layer in adata.layers else adata.X)
    X = X[:, idx]
    X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float32)

    score = X.mean(axis=1)
    mu = float(np.mean(score))
    sd = float(np.std(score, ddof=0))
    z = (score - mu) / (sd + 1e-8)

    adata.obs[f"{prefix}_score"] = np.asarray(score).ravel()
    adata.obs[f"{prefix}_z"] = np.asarray(z).ravel()
    adata.obs[f"{prefix}_n_genes"] = int(len(present))
    return adata


# =============================================================================
# FOSCTTM + Recall@k
# =============================================================================
def compute_foscttm(
    Z1: np.ndarray,
    Z2: np.ndarray,
    metric: str = "euclidean",
    block_size: int = 512,
    return_sem: bool = False,
    return_per_cell: bool = False,
) -> Union[float, Tuple[float, float], Tuple[float, np.ndarray], Tuple[float, float, np.ndarray]]:
    Z1 = np.asarray(Z1, dtype=np.float32)
    Z2 = np.asarray(Z2, dtype=np.float32)

    if Z1.shape != Z2.shape:
        raise ValueError(f"Z1/Z2 must have same shape for FOSCTTM. Got {Z1.shape} vs {Z2.shape}")

    n = int(Z1.shape[0])
    if n <= 1:
        if return_sem and return_per_cell:
            return 0.0, 0.0, np.zeros(n, dtype=np.float32)
        if return_sem:
            return 0.0, 0.0
        if return_per_cell:
            return 0.0, np.zeros(n, dtype=np.float32)
        return 0.0

    metric = str(metric).lower().strip()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("compute_foscttm currently supports metric in {'euclidean','cosine'}.")

    fos = np.empty(n, dtype=np.float32)

    if metric == "euclidean":
        Z2_T = Z2.T
        n2 = np.sum(Z2 * Z2, axis=1)
        for i0 in range(0, n, int(block_size)):
            i1 = min(i0 + int(block_size), n)
            A = Z1[i0:i1]
            n1 = np.sum(A * A, axis=1)[:, None]
            d2 = n1 + n2[None, :] - 2.0 * (A @ Z2_T)
            true = d2[np.arange(i1 - i0), np.arange(i0, i1)]
            fos[i0:i1] = (d2 < true[:, None]).sum(axis=1) / (n - 1)
    else:
        Z2_T = Z2.T
        n2 = np.linalg.norm(Z2, axis=1) + 1e-8
        for i0 in range(0, n, int(block_size)):
            i1 = min(i0 + int(block_size), n)
            A = Z1[i0:i1]
            n1 = np.linalg.norm(A, axis=1) + 1e-8
            sim = (A @ Z2_T) / (n1[:, None] * n2[None, :])
            d = 1.0 - sim
            true = d[np.arange(i1 - i0), np.arange(i0, i1)]
            fos[i0:i1] = (d < true[:, None]).sum(axis=1) / (n - 1)

    m, s = _mean_sem(fos.astype(float))
    if return_sem and return_per_cell:
        return float(m), float(s), fos
    if return_sem:
        return float(m), float(s)
    if return_per_cell:
        return float(m), fos
    return float(m)


def compute_match_recall_at_k(
    Z1: np.ndarray,
    Z2: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    block_size: int = 512,
    return_sem: bool = False,
    return_per_cell: bool = False,
) -> Union[float, Tuple[float, float], Tuple[float, np.ndarray], Tuple[float, float, np.ndarray]]:
    Z1 = np.asarray(Z1, dtype=np.float32)
    Z2 = np.asarray(Z2, dtype=np.float32)

    if Z1.shape != Z2.shape:
        raise ValueError(f"Z1/Z2 must have same shape. Got {Z1.shape} vs {Z2.shape}")

    n = int(Z1.shape[0])
    if n == 0:
        raise ValueError("Empty inputs.")
    if n == 1:
        hits = np.array([1.0], dtype=np.float32)
        if return_sem and return_per_cell:
            return 1.0, 0.0, hits
        if return_sem:
            return 1.0, 0.0
        if return_per_cell:
            return 1.0, hits
        return 1.0

    k = int(max(1, min(int(k), n)))
    metric = str(metric).lower().strip()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("compute_match_recall_at_k currently supports metric in {'euclidean','cosine'}.")

    hits = np.empty(n, dtype=np.float32)

    if metric == "euclidean":
        Z2_T = Z2.T
        n2 = np.sum(Z2 * Z2, axis=1)
        for i0 in range(0, n, int(block_size)):
            i1 = min(i0 + int(block_size), n)
            A = Z1[i0:i1]
            n1 = np.sum(A * A, axis=1)[:, None]
            d2 = n1 + n2[None, :] - 2.0 * (A @ Z2_T)
            topk = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            for r in range(i1 - i0):
                hits[i0 + r] = 1.0 if (i0 + r) in topk[r] else 0.0
    else:
        Z2_T = Z2.T
        n2 = np.linalg.norm(Z2, axis=1) + 1e-8
        for i0 in range(0, n, int(block_size)):
            i1 = min(i0 + int(block_size), n)
            A = Z1[i0:i1]
            n1 = np.linalg.norm(A, axis=1) + 1e-8
            sim = (A @ Z2_T) / (n1[:, None] * n2[None, :])
            d = 1.0 - sim
            topk = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
            for r in range(i1 - i0):
                hits[i0 + r] = 1.0 if (i0 + r) in topk[r] else 0.0

    m, s = _mean_sem(hits.astype(float))
    if return_sem and return_per_cell:
        return float(m), float(s), hits
    if return_sem:
        return float(m), float(s)
    if return_per_cell:
        return float(m), hits
    return float(m)


# =============================================================================
# Modality mixing / entropy / distances
# =============================================================================
def compute_modality_mixing(
    Z: np.ndarray,
    modality_labels: np.ndarray,
    k: int = 20,
    metric: str = "euclidean",
    return_sem: bool = False,
    return_per_cell: bool = False,
) -> Union[float, Tuple[float, float], Tuple[float, np.ndarray], Tuple[float, float, np.ndarray]]:
    Z = np.asarray(Z, dtype=np.float32)
    modality_labels = np.asarray(modality_labels)
    if Z.shape[0] != modality_labels.shape[0]:
        raise ValueError("Z and modality_labels must align on n_cells.")

    n = int(Z.shape[0])
    if n <= 1:
        if return_sem and return_per_cell:
            return 0.0, 0.0, np.zeros(n, dtype=np.float32)
        if return_sem:
            return 0.0, 0.0
        if return_per_cell:
            return 0.0, np.zeros(n, dtype=np.float32)
        return 0.0

    metric = str(metric).lower().strip()
    k_eff = int(min(max(int(k), 1), n - 1))

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nn.fit(Z)
    neigh_idx = nn.kneighbors(Z, return_distance=False)[:, 1:]

    neigh_mods = modality_labels[neigh_idx]
    frac_other = (neigh_mods != modality_labels[:, None]).mean(axis=1).astype(np.float32)

    m, s = _mean_sem(frac_other.astype(float))
    if return_sem and return_per_cell:
        return float(m), float(s), frac_other
    if return_sem:
        return float(m), float(s)
    if return_per_cell:
        return float(m), frac_other
    return float(m)


def compute_modality_entropy(
    Z: np.ndarray,
    modality_labels: np.ndarray,
    k: int = 30,
    metric: str = "euclidean",
    return_sem: bool = False,
    return_per_cell: bool = False,
) -> Union[float, Tuple[float, float], Tuple[float, np.ndarray], Tuple[float, float, np.ndarray]]:
    Z = np.asarray(Z, dtype=np.float32)
    modality_labels = np.asarray(modality_labels)

    n = int(Z.shape[0])
    if n <= 1:
        ent = np.zeros(n, dtype=np.float32)
        m, s = _mean_sem(ent.astype(float))
        if return_sem and return_per_cell:
            return float(m), float(s), ent
        if return_sem:
            return float(m), float(s)
        if return_per_cell:
            return float(m), ent
        return float(m)

    uniq_mods = np.unique(modality_labels)
    mcount = int(uniq_mods.size)
    if mcount <= 1:
        ent = np.zeros(n, dtype=np.float32)
        m, s = _mean_sem(ent.astype(float))
        if return_sem and return_per_cell:
            return float(m), float(s), ent
        if return_sem:
            return float(m), float(s)
        if return_per_cell:
            return float(m), ent
        return float(m)

    k_eff = int(min(max(int(k), 1), n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=str(metric))
    nn.fit(Z)
    neigh_idx = nn.kneighbors(Z, return_distance=False)[:, 1:]

    mod_to_code = {m: i for i, m in enumerate(uniq_mods)}
    codes = np.vectorize(mod_to_code.get)(modality_labels)

    ent = np.empty(n, dtype=np.float32)
    denom = np.log(float(mcount) + 1e-12)

    for i in range(n):
        neigh_codes = codes[neigh_idx[i]]
        p = np.bincount(neigh_codes, minlength=mcount).astype(np.float32)
        p = p / (p.sum() + 1e-8)
        h = -(p * np.log(p + 1e-12)).sum()
        ent[i] = float(h / (denom + 1e-12))

    m, s = _mean_sem(ent.astype(float))
    if return_sem and return_per_cell:
        return float(m), float(s), ent
    if return_sem:
        return float(m), float(s)
    if return_per_cell:
        return float(m), ent
    return float(m)


def compute_same_vs_diff_neighbor_distances(
    Z: np.ndarray,
    modality_labels: np.ndarray,
    k: int = 30,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    Z = np.asarray(Z, dtype=np.float32)
    modality_labels = np.asarray(modality_labels)

    n = int(Z.shape[0])
    if n <= 1:
        return {"same_mean": np.nan, "same_median": np.nan, "diff_mean": np.nan, "diff_median": np.nan, "k": int(k)}

    k_eff = int(min(max(int(k), 1), n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=str(metric))
    nn.fit(Z)
    dists, idx = nn.kneighbors(Z, return_distance=True)
    dists, idx = dists[:, 1:], idx[:, 1:]

    same = []
    diff = []
    for i in range(n):
        neigh = idx[i]
        dm = dists[i]
        mask_same = modality_labels[neigh] == modality_labels[i]
        same.append(dm[mask_same])
        diff.append(dm[~mask_same])

    same = np.concatenate(same) if len(same) else np.array([], dtype=np.float32)
    diff = np.concatenate(diff) if len(diff) else np.array([], dtype=np.float32)

    return {
        "same_mean": float(np.mean(same)) if same.size else np.nan,
        "same_median": float(np.median(same)) if same.size else np.nan,
        "diff_mean": float(np.mean(diff)) if diff.size else np.nan,
        "diff_median": float(np.median(diff)) if diff.size else np.nan,
        "k": int(k_eff),
    }


# =============================================================================
# Label transfer (kNN)
# =============================================================================
def label_transfer_knn(
    Z_source: np.ndarray,
    labels_source: np.ndarray,
    Z_target: np.ndarray,
    labels_target: Optional[np.ndarray] = None,
    k: int = 15,
    metric: str = "euclidean",
    return_label_order: bool = False,
    return_f1: bool = False,
):
    Z_source = np.asarray(Z_source, dtype=np.float32)
    Z_target = np.asarray(Z_target, dtype=np.float32)
    labels_source = np.asarray(labels_source)
    if labels_target is not None:
        labels_target = np.asarray(labels_target)

    n_source = int(Z_source.shape[0])
    if n_source == 0:
        raise ValueError("Z_source is empty.")

    k_eff = int(min(max(int(k), 1), n_source))
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(Z_source)
    neigh_idx = nn.kneighbors(Z_target, return_distance=False)

    uniq_src, src_codes = np.unique(labels_source, return_inverse=True)

    pred_codes = np.empty(Z_target.shape[0], dtype=np.int64)
    for i in range(Z_target.shape[0]):
        votes = src_codes[neigh_idx[i]]
        bc = np.bincount(votes, minlength=len(uniq_src))
        pred_codes[i] = int(bc.argmax())

    pred_labels = uniq_src[pred_codes]

    if labels_target is None:
        return pred_labels, None, np.array([])

    label_order = np.unique(np.concatenate([labels_target, pred_labels]))
    acc = float(accuracy_score(labels_target, pred_labels))
    cm = confusion_matrix(labels_target, pred_labels, labels=label_order)

    extras = []
    if return_label_order:
        extras.append(label_order)
    if return_f1:
        extras.append({
            "macro_f1": float(f1_score(labels_target, pred_labels, average="macro")),
            "weighted_f1": float(f1_score(labels_target, pred_labels, average="weighted")),
        })

    if not extras:
        return pred_labels, acc, cm
    return (pred_labels, acc, cm, *extras)


def summarize_bidirectional_transfer(
    Z_a: np.ndarray,
    y_a: np.ndarray,
    Z_b: np.ndarray,
    y_b: np.ndarray,
    k: int = 15,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    pred_ab, acc_ab, cm_ab, order_ab, f1_ab = label_transfer_knn(
        Z_source=Z_a, labels_source=y_a,
        Z_target=Z_b, labels_target=y_b,
        k=k, metric=metric,
        return_label_order=True, return_f1=True,
    )
    pred_ba, acc_ba, cm_ba, order_ba, f1_ba = label_transfer_knn(
        Z_source=Z_b, labels_source=y_b,
        Z_target=Z_a, labels_target=y_a,
        k=k, metric=metric,
        return_label_order=True, return_f1=True,
    )

    worst_macro = float(min(f1_ab["macro_f1"], f1_ba["macro_f1"]))
    worst_weighted = float(min(f1_ab["weighted_f1"], f1_ba["weighted_f1"]))

    return {
        "ab": {"acc": acc_ab, "f1": f1_ab, "cm": cm_ab, "order": order_ab, "pred": pred_ab},
        "ba": {"acc": acc_ba, "f1": f1_ba, "cm": cm_ba, "order": order_ba, "pred": pred_ba},
        "worst_direction_macro_f1": worst_macro,
        "worst_direction_weighted_f1": worst_weighted,
        "k": int(k),
        "metric": str(metric),
    }


# =============================================================================
# Reconstruction metrics (continuous + binary)
# =============================================================================
def mse_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)
    if x_true.shape != x_pred.shape:
        raise ValueError("x_true and x_pred must have same shape.")
    return np.mean((x_true - x_pred) ** 2, axis=0)


def pearson_corr_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    x_true = np.asarray(x_true, dtype=np.float32)
    x_pred = np.asarray(x_pred, dtype=np.float32)
    if x_true.shape != x_pred.shape:
        raise ValueError("x_true and x_pred must have same shape.")

    x_true_c = x_true - x_true.mean(axis=0, keepdims=True)
    x_pred_c = x_pred - x_pred.mean(axis=0, keepdims=True)

    num = (x_true_c * x_pred_c).sum(axis=0)
    denom = np.sqrt((x_true_c ** 2).sum(axis=0) * (x_pred_c ** 2).sum(axis=0)) + 1e-8
    return num / denom


def auc_per_feature_binary(x_true01: np.ndarray, x_score: np.ndarray) -> np.ndarray:
    """
    Per-feature AUC for binary ground-truth.
    x_true01: 0/1
    x_score: probabilities or logits (both ok for AUC)
    """
    x_true01 = np.asarray(x_true01)
    x_score = np.asarray(x_score)
    if x_true01.shape != x_score.shape:
        raise ValueError("x_true01 and x_score must have same shape.")
    aucs = np.full(x_true01.shape[1], np.nan, dtype=np.float32)
    for j in range(x_true01.shape[1]):
        y = x_true01[:, j]
        if y.min() == y.max():
            continue
        aucs[j] = float(roc_auc_score(y, x_score[:, j]))
    return aucs


def reconstruction_metrics(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    *,
    kind: str = "continuous",   # "continuous" | "binary"
) -> Dict[str, Any]:
    kind = str(kind).lower().strip()
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)

    if kind == "binary":
        aucs = auc_per_feature_binary(x_true, x_pred)
        return {
            "auc_mean": float(np.nanmean(aucs)),
            "auc_median": float(np.nanmedian(aucs)),
            "auc_per_feature": aucs,
        }

    pf_mse = mse_per_feature(x_true, x_pred)
    pf_r = pearson_corr_per_feature(x_true, x_pred)
    return {
        "mse_mean": float(np.mean(pf_mse)),
        "mse_median": float(np.median(pf_mse)),
        "pearson_mean": float(np.mean(pf_r)),
        "pearson_median": float(np.median(pf_r)),
        "mse_per_feature": pf_mse,
        "pearson_per_feature": pf_r,
    }


# =============================================================================
# Encoding + cross-modal prediction (robust decoder outputs)
# =============================================================================
def encode_adata(
    model,
    adata,
    modality: str,
    device: str = "cpu",
    layer: Optional[str] = None,
    X_key: str = "X",
    batch_size: int = 1024,
    latent: str = "moe_mean",
    random_state: int = 0,
) -> np.ndarray:
    """
    Encode a *single* modality into z.

    latent:
      - "moe_mean" / "moe_sample": fused MoE/PoE posterior
      - "modality_mean" / "modality_sample": that modality posterior only
    """
    from .data import _get_matrix

    latent = str(latent).lower().strip()
    valid = {"moe_mean", "moe_sample", "modality_mean", "modality_sample"}
    if latent not in valid:
        raise ValueError("latent must be one of %s; got %r" % (sorted(valid), latent))

    def _sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        eps = torch.randn(mu.shape, device=mu.device, generator=gen, dtype=mu.dtype)
        return mu + eps * torch.exp(0.5 * logvar)

    model.eval()
    X = _get_matrix(adata, layer=layer, X_key=X_key)
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(random_state))

    zs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], int(batch_size)):
            end = min(start + int(batch_size), X.shape[0])
            xb = torch.as_tensor(np.asarray(X[start:end]), dtype=torch.float32, device=dev)

            mu_dict, logvar_dict = model.encode_modalities({modality: xb})

            if "modality" in latent:
                mu = mu_dict[modality]
                lv = logvar_dict[modality]
                z = mu if latent.endswith("_mean") else _sample_gaussian(mu, lv, gen)
            else:
                if hasattr(model, "mixture_of_experts"):
                    mu_z, logvar_z = model.mixture_of_experts(mu_dict, logvar_dict)
                elif hasattr(model, "fuse_posteriors"):
                    mu_z, logvar_z = model.fuse_posteriors(mu_dict, logvar_dict)
                else:
                    mu_z, logvar_z = mu_dict[modality], logvar_dict[modality]

                z = mu_z if latent.endswith("_mean") else _sample_gaussian(mu_z, logvar_z, gen)

            zs.append(_to_numpy(z))

    return np.vstack(zs) if zs else np.zeros((0, 0), dtype=np.float32)


def cross_modal_predict(
    model,
    adata_src,
    src_mod: str,
    tgt_mod: str,
    device: str = "cpu",
    layer: Optional[str] = None,
    X_key: str = "X",
    batch_size: int = 512,
    use_moe: bool = True,
    decoder_prefer: str = "auto",
    decoder_link: Optional[str] = None,
) -> np.ndarray:
    """
    Encode src_mod then decode tgt_mod.
    Robust to decoder outputs that are dicts.
    """
    from .data import _get_matrix

    model.eval()
    X = _get_matrix(adata_src, layer=layer, X_key=X_key)
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    dev = torch.device(device)

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], int(batch_size)):
            end = min(start + int(batch_size), X.shape[0])
            xb = torch.as_tensor(np.asarray(X[start:end]), dtype=torch.float32, device=dev)

            mu_dict, logvar_dict = model.encode_modalities({src_mod: xb})

            if use_moe and hasattr(model, "mixture_of_experts"):
                mu_z, _ = model.mixture_of_experts(mu_dict, logvar_dict)
            else:
                mu_z = mu_dict[src_mod]

            xhat_dict = model.decode_modalities(mu_z)
            if tgt_mod not in xhat_dict:
                raise KeyError(f"Target modality {tgt_mod!r} not found. Available: {list(xhat_dict.keys())}")

            xhat_t = _extract_tensor_from_decoder_output(
                xhat_dict[tgt_mod],
                prefer=decoder_prefer,
                apply_link=decoder_link,
            )
            preds.append(_to_numpy(xhat_t))

    return np.vstack(preds) if preds else np.zeros((0, 0), dtype=np.float32)


# =============================================================================
# NEW: Latent sampling / generation utilities (README-compatible)
# =============================================================================
def fit_latent_gaussians_by_label(
    Z: np.ndarray,
    labels: Sequence[Any],
    *,
    min_n: int = 20,
    shrink: float = 1e-3,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit diagonal Gaussians per label:
      mean_c, var_c (per dim)

    shrink: add to var for stability.
    """
    Z = np.asarray(Z, dtype=np.float32)
    y = np.asarray(labels).astype(str)

    out: Dict[str, Dict[str, Any]] = {}
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        if idx.size < int(min_n):
            continue
        Zc = Z[idx]
        mu = Zc.mean(axis=0)
        var = Zc.var(axis=0) + float(shrink)
        out[str(lab)] = {"n": int(idx.size), "mean": mu.astype(np.float32), "var": var.astype(np.float32)}
    return out


def sample_latent_from_label_gaussians(
    gauss: Dict[str, Dict[str, Any]],
    label: str,
    n: int,
    *,
    random_state: int = 0,
) -> np.ndarray:
    if label not in gauss:
        raise KeyError(f"label {label!r} not present. Available: {list(gauss.keys())[:20]}")
    rng = np.random.default_rng(int(random_state))
    mu = gauss[label]["mean"]
    var = gauss[label]["var"]
    eps = rng.standard_normal((int(n), mu.shape[0]), dtype=np.float32)
    return mu[None, :] + eps * np.sqrt(var[None, :]).astype(np.float32)


def decode_from_latent(
    model,
    Z: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 512,
    decoder_prefer: str = "auto",
    decoder_link: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Decode latents into *all* modalities returned by model.decode_modalities.
    Returns modality -> numpy array
    """
    model.eval()
    dev = torch.device(device)
    Z = np.asarray(Z, dtype=np.float32)

    outs: Dict[str, List[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, Z.shape[0], int(batch_size)):
            end = min(start + int(batch_size), Z.shape[0])
            zb = torch.as_tensor(Z[start:end], dtype=torch.float32, device=dev)

            dec = model.decode_modalities(zb)
            for mod, v in dec.items():
                x = _extract_tensor_from_decoder_output(v, prefer=decoder_prefer, apply_link=decoder_link)
                outs.setdefault(mod, []).append(_to_numpy(x))

    return {m: np.vstack(chunks) for m, chunks in outs.items()}


def generate_from_latent(
    model,
    n: Optional[int] = None,
    *,
    target_mod: Optional[str] = None,
    device: str = "cpu",
    z_source: str = "prior",
    z: Optional[np.ndarray] = None,
    batch_size: int = 512,
    return_mean: bool = True,
    sample_likelihood: bool = False,
    decoder_prefer: str = "auto",
    decoder_link: Optional[str] = None,
    random_state: int = 0,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    README-compatible generator.

    Examples
    --------
    Xgen = generate_from_latent(model, n=5000, target_mod="rna", z_source="prior")

    Notes
    -----
    - If z is None, samples z ~ N(0, I) when z_source == "prior".
    - return_mean is kept for API parity (decoders return mean-like outputs).
    - sample_likelihood is accepted but not implemented here.
    """
    if sample_likelihood:
        raise NotImplementedError("sample_likelihood=True is not implemented yet in generate_from_latent().")

    if z is None:
        if n is None:
            raise ValueError("Provide either `z` or `n`.")
        if str(z_source).lower().strip() != "prior":
            raise ValueError("Only z_source='prior' is supported when `z` is not provided.")

        latent_dim = getattr(getattr(model, "cfg", None), "latent_dim", None)
        if latent_dim is None:
            latent_dim = getattr(model, "latent_dim", None)
        if latent_dim is None:
            raise ValueError("Could not infer latent_dim from model; provide `z` explicitly.")

        rng = np.random.default_rng(int(random_state))
        z = rng.standard_normal((int(n), int(latent_dim)), dtype=np.float32)

    dec = decode_from_latent(
        model,
        np.asarray(z, dtype=np.float32),
        device=device,
        batch_size=batch_size,
        decoder_prefer=decoder_prefer,
        decoder_link=decoder_link,
    )

    if target_mod is None:
        return dec
    if target_mod not in dec:
        raise KeyError(f"target_mod {target_mod!r} not found. Available: {list(dec.keys())}")
    return dec[target_mod]


# =============================================================================
# NEW: Evaluate reconstruction / imputation error against ground truth (README-compatible)
# =============================================================================
def evaluate_prediction_against_adata(
    adata_true,
    X_pred: np.ndarray,
    *,
    layer_true: Optional[str] = None,
    X_key_true: str = "X",
    kind: str = "continuous",     # "continuous" | "binary"
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compare X_pred (n_cells, n_features) to adata_true ground truth matrix.

    If feature_names is provided, it subsets BOTH true and pred to those features
    (by matching adata_true.var_names).
    """
    from .data import _get_matrix

    X_true = _get_matrix(adata_true, layer=layer_true, X_key=X_key_true)
    X_true = to_dense(X_true).astype(np.float32, copy=False)

    X_pred = np.asarray(X_pred, dtype=np.float32)
    if X_true.shape != X_pred.shape:
        raise ValueError(f"Shape mismatch: true {X_true.shape} vs pred {X_pred.shape}")

    feat_names_out: Optional[List[str]] = None
    if feature_names is not None:
        wanted = [str(x) for x in list(feature_names)]
        name_to_j = {str(n): j for j, n in enumerate(map(str, list(adata_true.var_names)))}
        cols = []
        for f in wanted:
            if f not in name_to_j:
                raise KeyError(f"Feature {f!r} not found in adata_true.var_names.")
            cols.append(int(name_to_j[f]))
        cols = np.asarray(cols, dtype=int)
        X_true = X_true[:, cols]
        X_pred = X_pred[:, cols]
        feat_names_out = wanted
    else:
        feat_names_out = [str(x) for x in list(map(str, list(adata_true.var_names)))]

    met = reconstruction_metrics(X_true, X_pred, kind=kind)

    per_cell = np.mean((X_true - X_pred) ** 2, axis=1)
    met["per_cell_mse"] = per_cell.astype(np.float32)
    met["per_cell_mean"] = float(np.mean(per_cell))

    # For plotting + downstream use
    met["X_true"] = X_true
    met["X_pred"] = X_pred
    met["feature_names"] = feat_names_out

    # README convenience: "summary" + "per_feature"
    if kind == "binary":
        met["summary"] = {
            "auc_mean": float(met["auc_mean"]),
            "auc_median": float(met["auc_median"]),
            "per_cell_mean": float(met["per_cell_mean"]),
        }
        met["per_feature"] = {
            "auc": np.asarray(met["auc_per_feature"]),
        }
    else:
        met["summary"] = {
            "mse_mean": float(met["mse_mean"]),
            "mse_median": float(met["mse_median"]),
            "pearson_mean": float(met["pearson_mean"]),
            "pearson_median": float(met["pearson_median"]),
            "per_cell_mean": float(met["per_cell_mean"]),
        }
        met["per_feature"] = {
            "mse": np.asarray(met["mse_per_feature"]),
            "pearson": np.asarray(met["pearson_per_feature"]),
        }

    return met


def evaluate_cross_reconstruction(
    model,
    adata_src,
    adata_tgt,
    *,
    src_mod: str,
    tgt_mod: str,
    device: str = "cpu",
    src_layer: Optional[str] = None,
    src_X_key: str = "X",
    tgt_layer: Optional[str] = None,
    tgt_X_key: str = "X",
    batch_size: int = 512,
    kind: str = "continuous",
    feature_names: Optional[Sequence[str]] = None,
    use_moe: bool = True,
    decoder_prefer: str = "auto",
    decoder_link: Optional[str] = None,
) -> Dict[str, Any]:
    """
    README-style one-call cross-reconstruction evaluation:

      1) X_pred = cross_modal_predict(model, adata_src, src_mod -> tgt_mod)
      2) compare X_pred against adata_tgt truth (layer/X_key)
      3) return dict with:
          - "summary"
          - "per_feature"
          - "X_true", "X_pred", "feature_names" (for plotting)

    This is the function README calls.
    """
    X_pred = cross_modal_predict(
        model,
        adata_src=adata_src,
        src_mod=src_mod,
        tgt_mod=tgt_mod,
        device=device,
        layer=src_layer,
        X_key=src_X_key,
        batch_size=batch_size,
        use_moe=use_moe,
        decoder_prefer=decoder_prefer,
        decoder_link=decoder_link,
    )

    rep = evaluate_prediction_against_adata(
        adata_tgt,
        X_pred,
        layer_true=tgt_layer,
        X_key_true=tgt_X_key,
        kind=kind,
        feature_names=feature_names,
    )

    # Add a bit of metadata to help users/debugging
    rep["src_mod"] = str(src_mod)
    rep["tgt_mod"] = str(tgt_mod)
    rep["kind"] = str(kind)
    rep["device"] = str(device)
    rep["batch_size"] = int(batch_size)
    rep["decoder_prefer"] = str(decoder_prefer)
    rep["decoder_link"] = decoder_link

    return rep


# =============================================================================
# Backwards/README-compatible aliases you referenced
# =============================================================================
def fit_label_latent_gaussians(
    Z: np.ndarray,
    labels: Sequence[Any],
    *,
    min_n: int = 20,
    shrink: float = 1e-3,
) -> Dict[str, Dict[str, Any]]:
    return fit_latent_gaussians_by_label(Z, labels, min_n=min_n, shrink=shrink)


def sample_latent_by_label(
    gauss: Dict[str, Dict[str, Any]],
    label: str,
    n: int,
    *,
    random_state: int = 0,
) -> np.ndarray:
    return sample_latent_from_label_gaussians(gauss, label, n, random_state=random_state)

