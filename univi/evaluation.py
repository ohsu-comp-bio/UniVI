# univi/evaluation.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# ----------------------------
# Small helpers
# ----------------------------
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


# ------------------------------------------------------------------
# 1. FOSCTTM (exact, blockwise) + Recall@k (top-k match rate)
# ------------------------------------------------------------------
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

    This is computed EXACTLY using blockwise pairwise distance computation to avoid NxN kneighbors storage.

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
            # indices of k smallest (unordered), then check membership
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


# ------------------------------------------------------------------
# 2. Modality mixing
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 3. Label transfer (kNN) with extra stats (macro/weighted F1)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 4. Reconstruction metrics (continuous; useful for CITE-seq CLR/gaussian)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 5. Encoding + cross-modal prediction
# ------------------------------------------------------------------
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
                # robust fallback in case a future refactor renames MoE fuser
                if hasattr(model, "mixture_of_experts"):
                    mu_z, logvar_z = model.mixture_of_experts(mu_dict, logvar_dict)
                elif hasattr(model, "fuse_posteriors"):
                    mu_z, logvar_z = model.fuse_posteriors(mu_dict, logvar_dict)
                else:
                    # single-modality fallback
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

    For paired data with ONLY src_mod observed, MoE fusion == src posterior.
    Still, use_moe=False can be handy if you want to force src-only even if model changes.
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
) -> np.ndarray:
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


# ------------------------------------------------------------------
# 6. High-level alignment eval (Figure-ready)
# ------------------------------------------------------------------
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
    k_transfer: int = 15,
    modality_labels: Optional[np.ndarray] = None,
    labels_source: Optional[np.ndarray] = None,
    labels_target: Optional[np.ndarray] = None,
    recall_ks: Tuple[int, ...] = (1, 5, 10),
    foscttm_block_size: int = 512,
    json_safe: bool = True,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - foscttm (mean), foscttm_sem
      - recall@k + sem for each k in recall_ks
      - modality_mixing (mean), modality_mixing_sem
      - label transfer: acc, macro/weighted f1 (optional), confusion matrix + label order
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

    # Modality mixing computed on concatenated embeddings
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
    else:
        out["modality_mixing"] = None
        out["modality_mixing_sem"] = None
        out["k_mixing"] = int(k_mixing)

    # Label transfer
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
    else:
        out["label_transfer_pred"] = None
        out["label_transfer_acc"] = None
        out["label_transfer_cm"] = None
        out["label_transfer_label_order"] = None
        out["label_transfer_f1"] = None
        out["k_transfer"] = int(k_transfer)

    if json_safe:
        out = {k: _json_safe(v) for k, v in out.items()}

    return out

