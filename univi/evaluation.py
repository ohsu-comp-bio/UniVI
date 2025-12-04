# univi/evaluation.py

from __future__ import annotations
from typing import Optional, Tuple, Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix

import scipy.sparse as sp
import torch


# ------------------------------------------------------------------
# 1. FOSCTTM (Fraction of Samples Closer Than the True Match)
# ------------------------------------------------------------------

def compute_foscttm(
    Z1: np.ndarray,
    Z2: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """
    Compute FOSCTTM assuming 1:1 pairing between rows of Z1 and Z2.

    For each i:
      - compute distance from Z1[i] to all Z2[j]
      - count how many j have dist(Z1[i], Z2[j]) < dist(Z1[i], Z2[i])
      - divide by (N-1)
    Return the mean over i.
    """
    assert Z1.shape == Z2.shape, "Z1/Z2 must have same shape for FOSCTTM."
    n = Z1.shape[0]

    nn = NearestNeighbors(n_neighbors=n, metric=metric)
    nn.fit(Z2)
    dists, indices = nn.kneighbors(Z1, return_distance=True)

    ranks = np.zeros(n, dtype=float)
    for i in range(n):
        # distance to true match:
        true_dist = np.linalg.norm(Z1[i] - Z2[i])
        # how many closer than the true match?
        closer = np.sum(dists[i] < true_dist) - 1  # minus self if counted
        ranks[i] = closer / max(n - 1, 1)

    return float(ranks.mean())


# ------------------------------------------------------------------
# 2. Modality mixing (how well modalities mix in kNN graph)
# ------------------------------------------------------------------

def compute_modality_mixing(
    Z: np.ndarray,
    modality_labels: np.ndarray,
    k: int = 20,
    metric: str = "euclidean",
) -> float:
    """
    For each cell, compute fraction of its k nearest neighbors
    that come from *other* modalities. Then average over cells.

    Returns value in [0, 1]: higher ~ better mixing (if modalities
    should be aligned).
    """
    modality_labels = np.asarray(modality_labels)
    assert Z.shape[0] == modality_labels.shape[0]

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(Z)
    neigh_dist, neigh_idx = nn.kneighbors(Z)

    # drop self (first neighbor)
    neigh_idx = neigh_idx[:, 1:]

    frac_other = []
    for i in range(Z.shape[0]):
        my_mod = modality_labels[i]
        neigh_mods = modality_labels[neigh_idx[i]]
        frac = np.mean(neigh_mods != my_mod)
        frac_other.append(frac)

    return float(np.mean(frac_other))


# ------------------------------------------------------------------
# 3. Label transfer accuracy (e.g. ADT → RNA)
# ------------------------------------------------------------------

def label_transfer_knn(
    Z_source: np.ndarray,
    labels_source: np.ndarray,
    Z_target: np.ndarray,
    labels_target: Optional[np.ndarray] = None,
    k: int = 15,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, Optional[float], np.ndarray]:
    """
    Majority-vote kNN label transfer from source → target.

    Parameters
    ----------
    Z_source : (n_source, d)
        Embeddings for source cells (with labels_source).
    labels_source : (n_source,)
        Source labels (e.g. ADT-derived annotations).
    Z_target : (n_target, d)
        Embeddings for target cells.
    labels_target : (n_target,) or None
        True labels for target, if available (for accuracy).
    k : int
        kNN for label transfer.
    metric : str
        Distance metric for kNN.

    Returns
    -------
    pred_labels : (n_target,)
        Predicted labels for target cells.
    accuracy : float or None
        Accuracy if labels_target provided; else None.
    conf_mat : np.ndarray
        Confusion matrix if labels_target provided, else empty array.
    """
    labels_source = np.asarray(labels_source)
    if labels_target is not None:
        labels_target = np.asarray(labels_target)

    # Build kNN over source
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(Z_source)
    neigh_dist, neigh_idx = nn.kneighbors(Z_target)

    # majority vote
    pred_labels = []
    for i in range(Z_target.shape[0]):
        neigh_labels = labels_source[neigh_idx[i]]
        # simple majority vote
        vals, counts = np.unique(neigh_labels, return_counts=True)
        pred_labels.append(vals[np.argmax(counts)])

    pred_labels = np.array(pred_labels, dtype=labels_source.dtype)

    if labels_target is None:
        return pred_labels, None, np.array([])

    acc = accuracy_score(labels_target, pred_labels)
    cm = confusion_matrix(labels_target, pred_labels, labels=np.unique(labels_target))

    return pred_labels, float(acc), cm


# ------------------------------------------------------------------
# 4. Simple reconstruction metrics (optional cross-modal check)
# ------------------------------------------------------------------

def mse_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """
    Mean squared error per feature (dimension).
    """
    assert x_true.shape == x_pred.shape
    return np.mean((x_true - x_pred) ** 2, axis=0)


def pearson_corr_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """
    Pearson correlation per feature, computed column-wise.
    """
    assert x_true.shape == x_pred.shape
    # center
    x_true_c = x_true - x_true.mean(axis=0, keepdims=True)
    x_pred_c = x_pred - x_pred.mean(axis=0, keepdims=True)

    num = (x_true_c * x_pred_c).sum(axis=0)
    denom = np.sqrt((x_true_c ** 2).sum(axis=0) * (x_pred_c ** 2).sum(axis=0)) + 1e-8
    return num / denom

# ------------------------------------------------------------------
# 5. Use a trained model to perform different tasks on a test set
# ------------------------------------------------------------------

def encode_adata(
    model,
    adata,
    modality: str,
    device: str = "cpu",
    layer: Optional[str] = None,
    X_key: str = "X",
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Push an AnnData through the UniVI encoder for a single modality.

    Selection rules:
      - if X_key != "X": uses adata.obsm[X_key]
      - else: uses adata.layers[layer] if layer is not None, else adata.X

    Returns a (n_cells, latent_dim) numpy array of the *fused* mean latent.
    """
    from .data import _get_matrix
    model.eval()
    X = _get_matrix(adata, layer=layer, X_key=X_key)
    if sp.issparse(X):
        X = X.toarray()

    zs = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            xb = torch.as_tensor(np.asarray(X[start:end]), dtype=torch.float32, device=device)
            mu_dict, logvar_dict = model.encode_modalities({modality: xb})
            mu_z, _ = model.mixture_of_experts(mu_dict, logvar_dict)
            zs.append(mu_z.cpu().numpy())

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
    """
    Given source modality data, encode to z and decode into target modality.

    Selection rules:
      - if X_key != "X": uses adata_src.obsm[X_key]
      - else: uses adata_src.layers[layer] if layer is not None, else adata_src.X

    Returns reconstructed X_hat for the target modality as a numpy array.
    """
    from .data import _get_matrix
    model.eval()
    X = _get_matrix(adata_src, layer=layer, X_key=X_key)
    if sp.issparse(X):
        X = X.toarray()

    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            xb = torch.as_tensor(np.asarray(X[start:end]), dtype=torch.float32, device=device)

            mu_dict, logvar_dict = model.encode_modalities({src_mod: xb})
            mu_z, _ = model.mixture_of_experts(mu_dict, logvar_dict)
            xhat_dict = model.decode_modalities(mu_z)
            preds.append(xhat_dict[tgt_mod].cpu().numpy())

    return np.vstack(preds)
def denoise_adata(
    model,
    adata,
    modality: str,
    device: str = "cpu",
    layer: Optional[str] = None,
    X_key: str = "X",
    batch_size: int = 512,
) -> np.ndarray:
    """
    Denoise (reconstruct) a modality by encoding and decoding within the same modality.
    """
    return cross_modal_predict(
        model,
        adata_src=adata,
        src_mod=modality,
        tgt_mod=modality,
        device=device,
        layer=layer,
        X_key=X_key,
        batch_size=batch_size,
    )
def fit_latent_gaussians_by_label(
    Z: np.ndarray,
    labels: np.ndarray,
    min_cells: int = 20,
    eps: float = 1e-3,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Given latent Z [n_cells, d] and string labels, fit a Gaussian per label.

    Returns:
        {label: {"mu": mu, "cov": cov}}
    """
    from collections import defaultdict

    out: Dict[str, Dict[str, np.ndarray]] = {}
    uniq = np.unique(labels)
    for lab in uniq:
        idx = np.where(labels == lab)[0]
        if len(idx) < min_cells:
            continue
        Z_lab = Z[idx]
        mu = Z_lab.mean(axis=0)
        # empirical covariance with a small diagonal jitter
        cov = np.cov(Z_lab, rowvar=False)
        cov = cov + eps * np.eye(Z_lab.shape[1])
        out[lab] = {"mu": mu, "cov": cov}
    return out

def sample_from_latent_gaussians(
    gaussians: Dict[str, Dict[str, np.ndarray]],
    sample_spec: Dict[str, int],
    random_state: int = 0,
) -> Dict[str, np.ndarray]:
    """
    sample_spec: dict {label: n_samples}
    Returns dict {label: Z_samples}
    """
    rng = np.random.default_rng(random_state)
    samples: Dict[str, np.ndarray] = {}
    for lab, n in sample_spec.items():
        if lab not in gaussians:
            continue
        mu = gaussians[lab]["mu"]
        cov = gaussians[lab]["cov"]
        z_samp = rng.multivariate_normal(mu, cov, size=n)
        samples[lab] = z_samp
    return samples

# ------------------------------------------------------------------
# 5. Helper: encode an AnnData with a trained UniVI model
# ------------------------------------------------------------------

import scipy.sparse as sp
import torch
