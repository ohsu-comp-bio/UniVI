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
)
from sklearn.cluster import KMeans


# =============================================================================
# Helpers
# =============================================================================

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
    """
    Compute LSC17 mean expression (per cell) and z-score it across cells.

    Stores:
      - adata.obs[f"{prefix}_score"]
      - adata.obs[f"{prefix}_z"]
      - adata.obs[f"{prefix}_n_genes"]

    Notes:
    - Expects expression to already be in the layer you want (often log1p normalized RNA).
    - If gene symbols aren't in adata.var_names, pass gene_key (column in adata.var).
    """
    var_names = np.asarray(adata.var_names if gene_key is None else adata.var[gene_key])
    gene_to_idx = {g: i for i, g in enumerate(var_names)}

    present = [g for g in LSC17_GENES if g in gene_to_idx]
    idx = [gene_to_idx[g] for g in present]

    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer] if layer in adata.layers else adata.X

    X = X[:, idx]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    score = X.mean(axis=1)
    mu = float(np.mean(score))
    sd = float(np.std(score, ddof=0))
    z = (score - mu) / (sd + 1e-8)

    adata.obs[f"{prefix}_score"] = np.asarray(score).ravel()
    adata.obs[f"{prefix}_z"] = np.asarray(z).ravel()
    adata.obs[f"{prefix}_n_genes"] = int(len(present))
    return adata


# =============================================================================
# FOSCTTM (exact, blockwise) + Recall@k (exact)
# =============================================================================

def compute_foscttm(
    Z1: np.ndarray,
    Z2: np.ndarray,
    metric: str = "euclidean",
    block_size: int = 512,
    return_sem: bool = False,
    return_per_cell: bool = False,
) -> Union[float, Tuple[float, float], Tuple[float, np.ndarray], Tuple[float, float, np.ndarray]]:
    """
    Compute FOSCTTM assuming 1:1 pairing between rows of Z1 and Z2.

    Definition used:
      For each i:
        frac_i = #{j: d(Z1[i], Z2[j]) < d(Z1[i], Z2[i])} / (N-1)
      FOSCTTM = mean_i frac_i

    Computed EXACTLY using blockwise pairwise distance computation.
    Supports metric in {"euclidean", "cosine"}.
    """
    Z1 = np.asarray(Z1, dtype=np.float32)
    Z2 = np.asarray(Z2, dtype=np.float32)

    if Z1.shape != Z2.shape:
        raise ValueError(f"Z1/Z2 must have same shape for FOSCTTM. Got {Z1.shape} vs {Z2.shape}")

    n = int(Z1.shape[0])
    if n <= 1:
        out0: Any = 0.0
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
        # squared Euclidean: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        Z2_T = Z2.T
        n2 = np.sum(Z2 * Z2, axis=1)  # (n,)
        for i0 in range(0, n, int(block_size)):
            i1 = min(i0 + int(block_size), n)
            A = Z1[i0:i1]
            n1 = np.sum(A * A, axis=1)[:, None]  # (b,1)
            d2 = n1 + n2[None, :] - 2.0 * (A @ Z2_T)  # (b,n)
            true = d2[np.arange(i1 - i0), np.arange(i0, i1)]
            fos[i0:i1] = (d2 < true[:, None]).sum(axis=1) / (n - 1)

    else:  # cosine distance = 1 - cosine_similarity
        Z2_T = Z2.T
        n2 = np.linalg.norm(Z2, axis=1) + 1e-8  # (n,)
        for i0 in range(0, n, int(block_size)):
            i1 = min(i0 + int(block_size), n)
            A = Z1[i0:i1]
            n1 = np.linalg.norm(A, axis=1) + 1e-8  # (b,)
            sim = (A @ Z2_T) / (n1[:, None] * n2[None, :])  # (b,n)
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
    """
    Recall@k for paired matching:
      hit_i = 1 if true match (i) is among k nearest neighbors of Z1[i] in Z2
      recall@k = mean_i hit_i

    Computed exactly blockwise for metric in {"euclidean","cosine"}.
    """
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
        n2 = np.sum(Z2 * Z2, axis=1)  # (n,)
        for i0 in range(0, n, int(block_size)):
            i1 = min(i0 + int(block_size), n)
            A = Z1[i0:i1]
            n1 = np.sum(A * A, axis=1)[:, None]  # (b,1)
            d2 = n1 + n2[None, :] - 2.0 * (A @ Z2_T)  # (b,n)
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
# Modality mixing + entropy + same-vs-diff neighbor distances
# =============================================================================

def compute_modality_mixing(
    Z: np.ndarray,
    modality_labels: np.ndarray,
    k: int = 20,
    metric: str = "euclidean",
    return_sem: bool = False,
    return_per_cell: bool = False,
) -> Union[float, Tuple[float, float], Tuple[float, np.ndarray], Tuple[float, float, np.ndarray]]:
    """
    Mean fraction of kNN neighbors that are from a different modality.
    """
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
    neigh_idx = nn.kneighbors(Z, return_distance=False)[:, 1:]  # drop self

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
    """
    Local modality entropy in kNN neighborhoods, normalized to [0,1] by log(#modalities).
    """
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

    # map modality labels to contiguous codes
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
    """
    Summarize distances to same-modality vs different-modality neighbors.
    """
    Z = np.asarray(Z, dtype=np.float32)
    modality_labels = np.asarray(modality_labels)

    n = int(Z.shape[0])
    if n <= 1:
        return {
            "same_mean": np.nan,
            "same_median": np.nan,
            "diff_mean": np.nan,
            "diff_median": np.nan,
            "k": int(k),
        }

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
# Label transfer (kNN) with extra stats (macro/weighted F1)
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
    """
    Backwards-compatible returns:
      - if labels_target is None: (pred_labels, None, empty_cm)
      - if labels_target provided and both flags False: (pred_labels, acc, cm)
      - if return_label_order True: add label_order
      - if return_f1 True: add f1_dict
      - if both True: add both (label_order, f1_dict) in that order
    """
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
    """
    Compute A->B and B->A transfers and return worst-direction macro-F1.
    Handy for dropout/robustness plots where you track "worse direction".
    """
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
# Reconstruction metrics (continuous)
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


def reconstruction_metrics(x_true: np.ndarray, x_pred: np.ndarray) -> Dict[str, Any]:
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
# Encoding + cross-modal prediction + MoE gate extraction (FIXED)
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
    Encode a *single* modality (one observed modality at a time) into z.

    latent:
      - "moe_mean" / "moe_sample": uses fused MoE/PoE posterior
      - "modality_mean" / "modality_sample": uses that modality's posterior only
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
    if sp.issparse(X):
        X = X.toarray()

    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(random_state))

    zs = []
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

            zs.append(z.detach().cpu().numpy())

    return np.vstack(zs)


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
) -> np.ndarray:
    """
    Encode src_mod then decode tgt_mod.
    """
    from .data import _get_matrix

    model.eval()
    X = _get_matrix(adata_src, layer=layer, X_key=X_key)
    if sp.issparse(X):
        X = X.toarray()

    dev = torch.device(device)

    preds = []
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
            preds.append(xhat_dict[tgt_mod].detach().cpu().numpy())

    return np.vstack(preds) if preds else np.zeros((0, 0), dtype=float)


def denoise_from_multimodal(
    model,
    x_dict: Dict[str, np.ndarray],
    target_mod: str,
    device: str = "cpu",
    batch_size: int = 512,
    use_mean: bool = True,
    attn_bias_cfg: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """
    True multimodal denoising:
      (observed modalities) -> fused latent -> decode target_mod

    x_dict: modality -> array (n_cells, d_mod)
    """
    model.eval()
    dev = torch.device(device)

    mods = list(x_dict.keys())
    if not mods:
        raise ValueError("x_dict is empty.")
    n = int(np.asarray(x_dict[mods[0]]).shape[0])

    out = []
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            end = min(start + int(batch_size), n)

            xb = {
                m: torch.as_tensor(np.asarray(x_dict[m][start:end]), dtype=torch.float32, device=dev)
                for m in mods
            }

            # encode fused
            mu_z, logvar_z, z = model.encode_fused(
                xb,
                use_mean=bool(use_mean),
                inject_label_expert=False,
                attn_bias_cfg=attn_bias_cfg,
            )

            # decode
            dec = model.decode_modalities(z)
            dec_out = dec[target_mod]

            # unwrap mean-like output
            if isinstance(dec_out, dict) and ("mean" in dec_out):
                xhat = dec_out["mean"]
            elif isinstance(dec_out, dict) and ("mu" in dec_out):
                xhat = dec_out["mu"]
            elif isinstance(dec_out, dict) and ("logits" in dec_out):
                xhat = dec_out["logits"]
            else:
                xhat = dec_out

            out.append(xhat.detach().cpu().numpy())

    return np.vstack(out)


def denoise_adata(
    model,
    adata,
    modality: str,
    device: str = "cpu",
    layer: Optional[str] = None,
    X_key: str = "X",
    batch_size: int = 512,
    out_layer: Optional[str] = None,
    overwrite_X: bool = False,
    dtype: Optional[np.dtype] = np.float32,
    *,
    # NEW: if provided, do fused multimodal denoising
    adata_by_mod: Optional[Dict[str, Any]] = None,
    layer_by_mod: Optional[Dict[str, Optional[str]]] = None,
    X_key_by_mod: Optional[Dict[str, str]] = None,
    use_mean: bool = True,
    attn_bias_cfg: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """
    Backwards-compatible:
      - If adata_by_mod is None: old behavior (single-modality self-recon)
      - If adata_by_mod is provided: true multimodal denoising via fused latent
    """
    if adata_by_mod is None:
        # OLD behavior (single-modality self-recon)
        X_hat = cross_modal_predict(
            model,
            adata_src=adata,
            src_mod=modality,
            tgt_mod=modality,
            device=device,
            layer=layer,
            X_key=X_key,
            batch_size=batch_size,
            use_moe=True,
        )
        if dtype is not None:
            X_hat = np.asarray(X_hat, dtype=dtype)

        if overwrite_X:
            adata.X = X_hat
        elif out_layer is not None:
            adata.layers[out_layer] = X_hat

        return X_hat

    # NEW: multimodal denoise
    from .data import _get_matrix  # to avoid hard dependency if not used

    # ensure target present
    if modality not in adata_by_mod:
        adata_by_mod = dict(adata_by_mod)
        adata_by_mod[modality] = adata

    layer_by_mod = layer_by_mod or {}
    X_key_by_mod = X_key_by_mod or {}

    mods = list(adata_by_mod.keys())
    if not mods:
        raise ValueError("adata_by_mod is empty.")

    # Load all observed modalities into numpy matrices
    x_dict_np: Dict[str, np.ndarray] = {}
    n = None
    for m, a in adata_by_mod.items():
        lay = layer_by_mod.get(m, None)
        xk = X_key_by_mod.get(m, "X")
        X = _get_matrix(a, layer=lay, X_key=xk)
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        if n is None:
            n = int(X.shape[0])
        elif int(X.shape[0]) != int(n):
            raise ValueError("All adata objects in adata_by_mod must have same n_obs.")
        x_dict_np[m] = X

    X_hat = denoise_from_multimodal(
        model,
        x_dict=x_dict_np,
        target_mod=modality,
        device=device,
        batch_size=batch_size,
        use_mean=use_mean,
        attn_bias_cfg=attn_bias_cfg,
    )

    if dtype is not None:
        X_hat = np.asarray(X_hat, dtype=dtype)

    if overwrite_X:
        adata.X = X_hat
    elif out_layer is not None:
        adata.layers[out_layer] = X_hat

    return X_hat


# =============================================================================
# MoE gate extraction (FIXED + EXPANDED)
# =============================================================================

def _precision_gates_from_logvar_dict(
    logvar_dict: Mapping[str, torch.Tensor],
    modality_order: Sequence[str],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Notebook-style gates derived from per-modality posterior uncertainty.

    Returns:
      W: (B, M) where
        W[b,m] = mean_d  prec_m[b,d] / sum_k prec_k[b,d]
      with prec = exp(-logvar)
    """
    precisions = [torch.exp(-logvar_dict[m]) for m in modality_order]  # (B,D) each
    P = torch.stack(precisions, dim=1)                                # (B,M,D)
    denom = P.sum(dim=1, keepdim=True).clamp_min(eps)                 # (B,1,D)
    frac = P / denom                                                  # (B,M,D)
    return frac.mean(dim=2)                                           # (B,M)


def _effective_precision_gates(
    logvar_dict: Mapping[str, torch.Tensor],
    router_weights: torch.Tensor,
    modality_order: Sequence[str],
    gate_eps: float = 1e-6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Gates reflecting actual contribution to fused posterior when you gate precisions:
      precision'_m = (w_m + gate_eps) * exp(-logvar_m)

    We renormalize across modalities per latent-dim, then mean over dims.
    """
    precisions = [torch.exp(-logvar_dict[m]) for m in modality_order]  # list (B,D)
    P = torch.stack(precisions, dim=1)                                # (B,M,D)

    w = router_weights
    if w.dim() != 2 or w.shape[1] != len(modality_order):
        raise ValueError(f"router_weights must be (B,M) with M={len(modality_order)}. Got {tuple(w.shape)}")

    P_eff = P * (w + float(gate_eps)).unsqueeze(-1)                   # (B,M,D)
    denom = P_eff.sum(dim=1, keepdim=True).clamp_min(eps)             # (B,1,D)
    frac = P_eff / denom                                              # (B,M,D)
    return frac.mean(dim=2)                                           # (B,M)


def encode_moe_gates_from_tensors(
    model,
    x_dict: Mapping[str, np.ndarray],
    device: str = "cpu",
    batch_size: int = 1024,
    modality_order: Optional[Sequence[str]] = None,
    *,
    kind: str = "precision",   # "precision" | "router" | "effective_precision"
    return_logits: bool = False,
) -> Dict[str, Any]:
    """
    Extract modality contribution weights for multi-observed cells.

    Parameters
    ----------
    kind:
      - "precision": notebook-style from logvar_dict (works regardless of router)
      - "router": learned router softmax weights from model.mixture_of_experts(..., return_weights=True)
      - "effective_precision": (router weights * precisions) renormalized like precision
                              (best for 'who drove the fused posterior')

    return_logits:
      - only meaningful for "router" / "effective_precision" and only if model supports return_logits

    Returns
    -------
      dict with:
        - weights: (n_cells, n_modalities)
        - modality_order: list[str]
        - per_modality_mean/median
        - (optional) logits: (n_cells, n_modalities) for router kinds (if available)
        - kind: str
    """
    model.eval()
    dev = torch.device(device)

    mods = list(modality_order) if modality_order is not None else list(x_dict.keys())
    if not mods:
        raise ValueError("Empty x_dict / modality_order.")

    n = int(np.asarray(x_dict[mods[0]]).shape[0])
    for m in mods[1:]:
        if int(np.asarray(x_dict[m]).shape[0]) != n:
            raise ValueError("All modalities in x_dict must have the same n_cells.")

    kind = str(kind).lower().strip()
    if kind not in {"precision", "router", "effective_precision"}:
        raise ValueError("kind must be one of {'precision','router','effective_precision'}.")

    gate_eps = float(getattr(model, "moe_gate_eps", 1e-6))

    W_chunks: List[np.ndarray] = []
    logits_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            end = min(start + int(batch_size), n)

            xb_dict = {
                m: torch.as_tensor(np.asarray(x_dict[m][start:end]), dtype=torch.float32, device=dev)
                for m in mods
            }

            mu_dict, logvar_dict = model.encode_modalities(xb_dict)

            if kind == "precision":
                w = _precision_gates_from_logvar_dict(logvar_dict, mods)
                W_chunks.append(w.detach().cpu().numpy())
                continue

            # router / effective_precision require mixture_of_experts
            if not hasattr(model, "mixture_of_experts"):
                raise AttributeError("Model has no mixture_of_experts method; cannot extract router gating weights.")

            # ask for weights (and optionally logits) from the model
            try:
                if return_logits:
                    _mu_z, _lv_z, w_router, logits = model.mixture_of_experts(
                        mu_dict, logvar_dict, return_weights=True, return_logits=True, modality_order=list(mods)
                    )
                else:
                    _mu_z, _lv_z, w_router = model.mixture_of_experts(
                        mu_dict, logvar_dict, return_weights=True, return_logits=False, modality_order=list(mods)
                    )
                    logits = None
            except TypeError:
                # fallback if the model doesn't accept modality_order / return_logits
                _mu_z, _lv_z, w_router = model.mixture_of_experts(mu_dict, logvar_dict, return_weights=True)
                logits = None

            if kind == "router":
                W_chunks.append(w_router.detach().cpu().numpy())
                if logits is not None:
                    logits_chunks.append(logits.detach().cpu().numpy())
            else:
                w_eff = _effective_precision_gates(logvar_dict, w_router, mods, gate_eps=gate_eps)
                W_chunks.append(w_eff.detach().cpu().numpy())
                if logits is not None:
                    logits_chunks.append(logits.detach().cpu().numpy())

    W = np.vstack(W_chunks) if W_chunks else np.zeros((0, len(mods)), dtype=np.float32)

    per_mean = {m: float(np.mean(W[:, i])) for i, m in enumerate(mods)} if W.size else {m: np.nan for m in mods}
    per_med = {m: float(np.median(W[:, i])) for i, m in enumerate(mods)} if W.size else {m: np.nan for m in mods}

    out: Dict[str, Any] = {
        "weights": W,
        "modality_order": list(mods),
        "per_modality_mean": per_mean,
        "per_modality_median": per_med,
        "kind": kind,
    }

    if logits_chunks:
        out["logits"] = np.vstack(logits_chunks)
    else:
        out["logits"] = None

    return out


def summarize_gate_contrasts(W: np.ndarray, modality_order: Sequence[str]) -> Dict[str, Any]:
    """
    Pairwise contrasts like wRNA - wATAC to assess modality contribution.
    """
    W = np.asarray(W, dtype=np.float32)
    mods = list(modality_order)
    out: Dict[str, Any] = {}
    for i, mi in enumerate(mods):
        for j, mj in enumerate(mods):
            if i >= j:
                continue
            diff = W[:, i] - W[:, j]
            out[f"gate_diff_{mi}_minus_{mj}_mean"] = float(np.mean(diff)) if diff.size else np.nan
            out[f"gate_diff_{mi}_minus_{mj}_median"] = float(np.median(diff)) if diff.size else np.nan
    return out


# =============================================================================
# Fused-space metrics (kmeans ARI/NMI + silhouette)
# =============================================================================

def compute_kmeans_ari_nmi(
    Z: np.ndarray,
    labels: np.ndarray,
    n_clusters: Optional[int] = None,
    random_state: int = 0,
) -> Dict[str, float]:
    Z = np.asarray(Z, dtype=np.float32)
    labels = np.asarray(labels)

    uniq = np.unique(labels)
    k = int(n_clusters) if n_clusters is not None else int(len(uniq))
    if k <= 1 or Z.shape[0] == 0:
        return {"kmeans_ari": float("nan"), "kmeans_nmi": float("nan"), "kmeans_k": float(k)}

    km = KMeans(n_clusters=k, random_state=int(random_state), n_init="auto")
    pred = km.fit_predict(Z)
    return {
        "kmeans_ari": float(adjusted_rand_score(labels, pred)),
        "kmeans_nmi": float(normalized_mutual_info_score(labels, pred)),
        "kmeans_k": float(k),
    }


def compute_silhouette(
    Z: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    Z = np.asarray(Z, dtype=np.float32)
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    if uniq.size < 2 or np.any(counts < 2) or Z.shape[0] < 3:
        return float("nan")
    return float(silhouette_score(Z, labels, metric=str(metric)))


# =============================================================================
# Tri-modal pairwise correspondence helper
# =============================================================================

def compute_pairwise_foscttm_grid(
    Z_by_mod: Dict[str, np.ndarray],
    metric: str = "euclidean",
    block_size: int = 512,
    return_sem: bool = True,
) -> Dict[str, Any]:
    mods = list(Z_by_mod.keys())
    out: Dict[str, Any] = {"metric": str(metric), "mods": mods}

    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            mi, mj = mods[i], mods[j]
            Zi, Zj = Z_by_mod[mi], Z_by_mod[mj]
            key = f"foscttm_{mi}_vs_{mj}"
            if Zi.shape == Zj.shape and Zi.shape[0] > 1:
                if return_sem:
                    m, s = compute_foscttm(Zi, Zj, metric=metric, block_size=block_size, return_sem=True)
                    out[key] = float(m)
                    out[key + "_sem"] = float(s)
                else:
                    out[key] = float(compute_foscttm(Zi, Zj, metric=metric, block_size=block_size))
            else:
                out[key] = None
                if return_sem:
                    out[key + "_sem"] = None

    return out


# =============================================================================
# High-level alignment eval (Figure-ready)
# =============================================================================

def evaluate_alignment(
    Z1: Optional[np.ndarray] = None,
    Z2: Optional[np.ndarray] = None,
    model=None,
    adata1=None,
    adata2=None,
    mod1: Optional[str] = None,
    mod2: Optional[str] = None,
    device: str = "cpu",
    layer1: Optional[str] = None,
    layer2: Optional[str] = None,
    X_key1: str = "X",
    X_key2: str = "X",
    batch_size: int = 1024,
    latent: str = "moe_mean",
    latent1: Optional[str] = None,
    latent2: Optional[str] = None,
    random_state: int = 0,
    metric: str = "euclidean",
    k_mixing: int = 20,
    k_entropy: int = 30,
    k_transfer: int = 15,
    modality_labels: Optional[np.ndarray] = None,
    labels_source: Optional[np.ndarray] = None,
    labels_target: Optional[np.ndarray] = None,
    compute_bidirectional_transfer: bool = False,
    recall_ks: Tuple[int, ...] = (1, 5, 10),
    foscttm_block_size: int = 512,
    # fused-space metrics
    Z_fused: Optional[np.ndarray] = None,
    fused_labels: Optional[np.ndarray] = None,
    fused_kmeans: bool = True,
    fused_silhouette: bool = True,
    fused_random_state: int = 0,
    # gating
    gate_weights: Optional[np.ndarray] = None,
    gate_modality_order: Optional[Sequence[str]] = None,
    gate_kind: Optional[str] = None,
    json_safe: bool = True,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - cross-latent: foscttm + sem, recall@k + sem, modality mixing + sem
      - label transfer: acc + cm + label_order + f1s
      - optional bidirectional transfer summary
      - optional fused-space: kmeans ARI/NMI, silhouette
      - optional gating summaries if gate_weights provided
    """
    out: Dict[str, Any] = {}

    lat1 = latent if latent1 is None else latent1
    lat2 = latent if latent2 is None else latent2

    if Z1 is None or Z2 is None:
        if model is None or adata1 is None or adata2 is None or mod1 is None or mod2 is None:
            raise ValueError("Provide either (Z1, Z2) or (model, adata1, adata2, mod1, mod2).")

        Z1 = encode_adata(
            model, adata1, modality=mod1, device=device, layer=layer1, X_key=X_key1,
            batch_size=batch_size, latent=lat1, random_state=random_state
        )
        Z2 = encode_adata(
            model, adata2, modality=mod2, device=device, layer=layer2, X_key=X_key2,
            batch_size=batch_size, latent=lat2, random_state=random_state
        )

    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    out["n1"] = int(Z1.shape[0])
    out["n2"] = int(Z2.shape[0])
    out["dim"] = int(Z1.shape[1]) if Z1.ndim == 2 else None
    out["latent1"] = lat1
    out["latent2"] = lat2
    out["metric"] = str(metric)

    # FOSCTTM + SEM
    if Z1.shape == Z2.shape and Z1.shape[0] > 1:
        fos_mean, fos_sem = compute_foscttm(
            Z1, Z2, metric=metric, block_size=foscttm_block_size, return_sem=True, return_per_cell=False
        )
        out["foscttm"] = fos_mean
        out["foscttm_sem"] = fos_sem
    else:
        out["foscttm"] = None
        out["foscttm_sem"] = None

    # Recall@k
    if Z1.shape == Z2.shape and Z1.shape[0] > 1:
        for k in recall_ks:
            r_mean, r_sem = compute_match_recall_at_k(
                Z1, Z2, k=int(k), metric=metric, block_size=foscttm_block_size, return_sem=True, return_per_cell=False
            )
            out[f"recall_at_{int(k)}"] = r_mean
            out[f"recall_at_{int(k)}_sem"] = r_sem
    else:
        for k in recall_ks:
            out[f"recall_at_{int(k)}"] = None
            out[f"recall_at_{int(k)}_sem"] = None

    # Modality mixing + entropy computed on concatenated embeddings
    Z_concat = None
    if (Z1.ndim == 2 and Z2.ndim == 2 and Z1.shape[1] == Z2.shape[1]):
        Z_concat = np.vstack([Z1, Z2])

    if Z_concat is not None and Z_concat.shape[0] > 1:
        if modality_labels is None:
            modality_labels = np.concatenate([np.repeat("mod1", Z1.shape[0]), np.repeat("mod2", Z2.shape[0])])

        mix_mean, mix_sem = compute_modality_mixing(
            Z_concat, modality_labels=np.asarray(modality_labels),
            k=k_mixing, metric=metric, return_sem=True, return_per_cell=False
        )
        out["modality_mixing"] = mix_mean
        out["modality_mixing_sem"] = mix_sem
        out["k_mixing"] = int(k_mixing)

        ent_mean, ent_sem = compute_modality_entropy(
            Z_concat, modality_labels=np.asarray(modality_labels),
            k=k_entropy, metric=metric, return_sem=True, return_per_cell=False
        )
        out["modality_entropy"] = ent_mean
        out["modality_entropy_sem"] = ent_sem
        out["k_entropy"] = int(k_entropy)

        out["same_vs_diff_neighbor_distances"] = compute_same_vs_diff_neighbor_distances(
            Z_concat, modality_labels=np.asarray(modality_labels), k=k_entropy, metric=metric
        )
    else:
        out["modality_mixing"] = None
        out["modality_mixing_sem"] = None
        out["k_mixing"] = int(k_mixing)
        out["modality_entropy"] = None
        out["modality_entropy_sem"] = None
        out["k_entropy"] = int(k_entropy)
        out["same_vs_diff_neighbor_distances"] = None

    # Label transfer (Z1->Z2)
    if labels_source is not None:
        pred, acc, cm, order, f1d = label_transfer_knn(
            Z_source=Z1,
            labels_source=np.asarray(labels_source),
            Z_target=Z2,
            labels_target=np.asarray(labels_target) if labels_target is not None else None,
            k=k_transfer,
            metric=metric,
            return_label_order=True,
            return_f1=True,
        )
        out["label_transfer_pred"] = pred
        out["label_transfer_acc"] = acc
        out["label_transfer_cm"] = cm
        out["label_transfer_label_order"] = order
        out["label_transfer_f1"] = f1d
        out["k_transfer"] = int(k_transfer)

        if compute_bidirectional_transfer and (labels_target is not None):
            out["bidirectional_transfer"] = summarize_bidirectional_transfer(
                Z_a=Z1, y_a=np.asarray(labels_source),
                Z_b=Z2, y_b=np.asarray(labels_target),
                k=k_transfer, metric=metric,
            )
        else:
            out["bidirectional_transfer"] = None
    else:
        out["label_transfer_pred"] = None
        out["label_transfer_acc"] = None
        out["label_transfer_cm"] = None
        out["label_transfer_label_order"] = None
        out["label_transfer_f1"] = None
        out["k_transfer"] = int(k_transfer)
        out["bidirectional_transfer"] = None

    # Fused-space metrics (optional)
    if Z_fused is not None and fused_labels is not None:
        Zf = np.asarray(Z_fused, dtype=np.float32)
        yl = np.asarray(fused_labels)
        out["fused_n"] = int(Zf.shape[0])
        out["fused_dim"] = int(Zf.shape[1]) if Zf.ndim == 2 else None

        if fused_kmeans:
            out["fused_kmeans"] = compute_kmeans_ari_nmi(Zf, yl, random_state=fused_random_state)
        else:
            out["fused_kmeans"] = None

        if fused_silhouette:
            out["fused_silhouette"] = compute_silhouette(Zf, yl, metric=metric)
        else:
            out["fused_silhouette"] = None
    else:
        out["fused_kmeans"] = None
        out["fused_silhouette"] = None

    # Gating summaries (optional)
    if gate_weights is not None:
        W = np.asarray(gate_weights, dtype=np.float32)
        out["gate_kind"] = gate_kind
        out["gate_weights_mean"] = float(np.mean(W)) if W.size else np.nan
        out["gate_weights_per_mod_mean"] = (
            {m: float(np.mean(W[:, i])) for i, m in enumerate(gate_modality_order or range(W.shape[1]))}
            if W.size else None
        )
        out["gate_contrasts"] = summarize_gate_contrasts(W, gate_modality_order or [str(i) for i in range(W.shape[1])])
    else:
        out["gate_kind"] = None
        out["gate_weights_mean"] = None
        out["gate_weights_per_mod_mean"] = None
        out["gate_contrasts"] = None

    if json_safe:
        out = {k: _json_safe(v) for k, v in out.items()}

    return out

