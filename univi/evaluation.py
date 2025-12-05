# univi/evaluation.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix


def compute_foscttm(Z1: np.ndarray, Z2: np.ndarray, metric: str = "euclidean") -> float:
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    if Z1.shape != Z2.shape:
        raise ValueError("Z1/Z2 must have same shape for FOSCTTM. Got %r vs %r" % (Z1.shape, Z2.shape))

    n = int(Z1.shape[0])
    if n <= 1:
        return 0.0

    nn = NearestNeighbors(n_neighbors=n, metric=metric)
    nn.fit(Z2)
    _, idx = nn.kneighbors(Z1, return_distance=True)

    ranks = np.empty(n, dtype=float)
    for i in range(n):
        pos = np.where(idx[i] == i)[0]
        ranks[i] = (int(pos[0]) / (n - 1)) if pos.size else 1.0

    return float(ranks.mean())


def compute_modality_mixing(
    Z: np.ndarray,
    modality_labels: np.ndarray,
    k: int = 20,
    metric: str = "euclidean",
) -> float:
    Z = np.asarray(Z)
    modality_labels = np.asarray(modality_labels)
    if Z.shape[0] != modality_labels.shape[0]:
        raise ValueError("Z and modality_labels must align on n_cells.")

    n = int(Z.shape[0])
    if n <= 1:
        return 0.0

    k_eff = int(min(max(int(k), 1), n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nn.fit(Z)
    _, neigh_idx = nn.kneighbors(Z)

    neigh_idx = neigh_idx[:, 1:]
    neigh_mods = modality_labels[neigh_idx]
    frac_other = (neigh_mods != modality_labels[:, None]).mean(axis=1)
    return float(frac_other.mean())


def label_transfer_knn(
    Z_source: np.ndarray,
    labels_source: np.ndarray,
    Z_target: np.ndarray,
    labels_target: Optional[np.ndarray] = None,
    k: int = 15,
    metric: str = "euclidean",
    return_label_order: bool = False,  # <-- NEW (default keeps your old scripts working)
) -> Tuple[np.ndarray, Optional[float], np.ndarray]:
    """
    Majority-vote kNN label transfer from source → target.

    Default return (backwards compatible):
        pred_labels, accuracy (or None), confusion_matrix (or empty)

    If return_label_order=True and labels_target is not None:
        pred_labels, accuracy, confusion_matrix, label_order
    """
    Z_source = np.asarray(Z_source)
    Z_target = np.asarray(Z_target)
    labels_source = np.asarray(labels_source)

    if labels_target is not None:
        labels_target = np.asarray(labels_target)

    n_source = int(Z_source.shape[0])
    if n_source == 0:
        raise ValueError("Z_source is empty.")

    k_eff = int(min(max(int(k), 1), n_source))
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(Z_source)
    _, neigh_idx = nn.kneighbors(Z_target)

    # Fast majority vote that supports string labels:
    # map labels_source -> integer codes
    uniq_src, src_codes = np.unique(labels_source, return_inverse=True)

    pred_codes = np.empty(Z_target.shape[0], dtype=np.int64)
    for i in range(Z_target.shape[0]):
        votes = src_codes[neigh_idx[i]]
        # bincount on codes
        bc = np.bincount(votes, minlength=len(uniq_src))
        pred_codes[i] = int(bc.argmax())

    pred_labels = uniq_src[pred_codes]

    if labels_target is None:
        empty = np.array([])
        return pred_labels, None, empty

    label_order = np.unique(np.concatenate([labels_target, pred_labels]))
    acc = float(accuracy_score(labels_target, pred_labels))
    cm = confusion_matrix(labels_target, pred_labels, labels=label_order)

    if return_label_order:
        return pred_labels, acc, cm, label_order  # type: ignore[return-value]

    return pred_labels, acc, cm


def mse_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)
    if x_true.shape != x_pred.shape:
        raise ValueError("x_true and x_pred must have same shape.")
    return np.mean((x_true - x_pred) ** 2, axis=0)


def pearson_corr_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)
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

    gen = torch.Generator(device=device)
    gen.manual_seed(int(random_state))

    zs = []
    with torch.no_grad():
        for start in range(0, X.shape[0], int(batch_size)):
            end = min(start + int(batch_size), X.shape[0])
            xb = torch.as_tensor(np.asarray(X[start:end]), dtype=torch.float32, device=device)

            mu_dict, logvar_dict = model.encode_modalities({modality: xb})

            if "modality" in latent:
                mu = mu_dict[modality]
                lv = logvar_dict[modality]
                z = mu if latent.endswith("_mean") else _sample_gaussian(mu, lv, gen)
            else:
                moe_out = model.mixture_of_experts(mu_dict, logvar_dict)
                if not isinstance(moe_out, (tuple, list)) or len(moe_out) < 2:
                    raise RuntimeError("mixture_of_experts must return (mu, logvar).")
                mu_z, logvar_z = moe_out[0], moe_out[1]
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
) -> np.ndarray:
    from .data import _get_matrix

    model.eval()
    X = _get_matrix(adata_src, layer=layer, X_key=X_key)
    if sp.issparse(X):
        X = X.toarray()

    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], int(batch_size)):
            end = min(start + int(batch_size), X.shape[0])
            xb = torch.as_tensor(np.asarray(X[start:end]), dtype=torch.float32, device=device)

            mu_dict, logvar_dict = model.encode_modalities({src_mod: xb})
            mu_z, _ = model.mixture_of_experts(mu_dict, logvar_dict)
            xhat_dict = model.decode_modalities(mu_z)
            if tgt_mod not in xhat_dict:
                raise KeyError("Target modality %r not found. Available: %s" % (tgt_mod, list(xhat_dict.keys())))
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
    )
    if dtype is not None:
        X_hat = np.asarray(X_hat, dtype=dtype)

    if overwrite_X:
        adata.X = X_hat
    elif out_layer is not None:
        adata.layers[out_layer] = X_hat

    return X_hat


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
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    lat1 = latent if latent1 is None else latent1
    lat2 = latent if latent2 is None else latent2

    if Z1 is None or Z2 is None:
        if model is None or adata1 is None or adata2 is None or mod1 is None or mod2 is None:
            raise ValueError("Provide either (Z1, Z2) or (model, adata1, adata2, mod1, mod2).")

        Z1 = encode_adata(model, adata1, modality=mod1, device=device, layer=layer1, X_key=X_key1,
                         batch_size=batch_size, latent=lat1, random_state=random_state)
        Z2 = encode_adata(model, adata2, modality=mod2, device=device, layer=layer2, X_key=X_key2,
                         batch_size=batch_size, latent=lat2, random_state=random_state)

    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    out["n1"] = int(Z1.shape[0])
    out["n2"] = int(Z2.shape[0])
    out["dim"] = int(Z1.shape[1]) if Z1.ndim == 2 else None
    out["latent1"] = lat1
    out["latent2"] = lat2

    if Z1.shape == Z2.shape and Z1.shape[0] > 1:
        out["foscttm"] = compute_foscttm(Z1, Z2, metric=metric)
    else:
        out["foscttm"] = None

    Z_concat = None
    if (Z1.ndim == 2 and Z2.ndim == 2 and Z1.shape[1] == Z2.shape[1]):
        Z_concat = np.vstack([Z1, Z2])
    if Z_concat is not None and Z_concat.shape[0] > 1:
        if modality_labels is None:
            modality_labels = np.concatenate([np.repeat("mod1", Z1.shape[0]), np.repeat("mod2", Z2.shape[0])])
        out["modality_mixing"] = compute_modality_mixing(Z_concat, modality_labels=np.asarray(modality_labels),
                                                         k=k_mixing, metric=metric)
    else:
        out["modality_mixing"] = None

    if labels_source is not None:
        if labels_target is not None:
            pred, acc, cm, order = label_transfer_knn(
                Z_source=Z1,
                labels_source=np.asarray(labels_source),
                Z_target=Z2,
                labels_target=np.asarray(labels_target),
                k=k_transfer,
                metric=metric,
                return_label_order=True,
            )
            out["label_transfer_label_order"] = order
        else:
            pred, acc, cm = label_transfer_knn(
                Z_source=Z1,
                labels_source=np.asarray(labels_source),
                Z_target=Z2,
                labels_target=None,
                k=k_transfer,
                metric=metric,
                return_label_order=False,
            )
            out["label_transfer_label_order"] = None

        out["label_transfer_pred"] = pred
        out["label_transfer_acc"] = acc
        out["label_transfer_cm"] = cm
    else:
        out["label_transfer_pred"] = None
        out["label_transfer_acc"] = None
        out["label_transfer_cm"] = None
        out["label_transfer_label_order"] = None

    return out

