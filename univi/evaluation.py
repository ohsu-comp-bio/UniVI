# univi/evaluation.py

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import numpy as np
import scipy.sparse as sp
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix


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
      - compute distances from Z1[i] to all Z2[j]
      - count how many j have dist(Z1[i], Z2[j]) < dist(Z1[i], Z2[i])
      - divide by (N-1)
    Return the mean over i.

    Notes
    -----
    - Uses the same distance metric for both neighbor distances and true-match distance.
    - Returns 0.0 if N <= 1.
    """
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    assert Z1.shape == Z2.shape, "Z1/Z2 must have same shape for FOSCTTM."

    n = Z1.shape[0]
    if n <= 1:
        return 0.0

    # full neighbor list so we can compare rank vs true match
    nn = NearestNeighbors(n_neighbors=n, metric=metric)
    nn.fit(Z2)
    dists, idx = nn.kneighbors(Z1, return_distance=True)

    # find where the true match (index i) appears in each row ordering
    # and compute fraction closer than true match: rank / (n-1)
    ranks = np.empty(n, dtype=float)
    for i in range(n):
        pos = np.where(idx[i] == i)[0]
        if pos.size == 0:
            # should never happen, but keep robust
            ranks[i] = 1.0
        else:
            rank = int(pos[0])  # number of neighbors *before* the true match
            ranks[i] = rank / (n - 1)

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
    Z = np.asarray(Z)
    modality_labels = np.asarray(modality_labels)
    assert Z.shape[0] == modality_labels.shape[0], "Z and modality_labels must align on n_cells."
    n = Z.shape[0]
    if n <= 1:
        return 0.0

    k_eff = int(min(max(k, 1), n - 1))

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nn.fit(Z)
    _, neigh_idx = nn.kneighbors(Z)

    # drop self (first neighbor)
    neigh_idx = neigh_idx[:, 1:]

    neigh_mods = modality_labels[neigh_idx]  # (n, k_eff)
    frac_other = (neigh_mods != modality_labels[:, None]).mean(axis=1)

    return float(frac_other.mean())


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
) -> Tuple[np.ndarray, Optional[float], np.ndarray, np.ndarray]:
    """
    Majority-vote kNN label transfer from source → target.

    Returns
    -------
    pred_labels : (n_target,)
    accuracy : float or None
    conf_mat : (n_labels, n_labels) or empty
    label_order : (n_labels,) label order used for confusion matrix
    """
    Z_source = np.asarray(Z_source)
    Z_target = np.asarray(Z_target)
    labels_source = np.asarray(labels_source)

    if labels_target is not None:
        labels_target = np.asarray(labels_target)

    n_source = Z_source.shape[0]
    if n_source == 0:
        raise ValueError("Z_source is empty.")
    k_eff = int(min(max(k, 1), n_source))

    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(Z_source)
    _, neigh_idx = nn.kneighbors(Z_target)

    # majority vote
    pred_labels = np.empty(Z_target.shape[0], dtype=labels_source.dtype)
    for i in range(Z_target.shape[0]):
        neigh_labels = labels_source[neigh_idx[i]]
        vals, counts = np.unique(neigh_labels, return_counts=True)
        pred_labels[i] = vals[np.argmax(counts)]

    if labels_target is None:
        return pred_labels, None, np.array([]), np.array([])

    # stable label order = union of target + predicted (so CM is interpretable)
    label_order = np.unique(np.concatenate([labels_target, pred_labels]))
    acc = accuracy_score(labels_target, pred_labels)
    cm = confusion_matrix(labels_target, pred_labels, labels=label_order)

    return pred_labels, float(acc), cm, label_order


# ------------------------------------------------------------------
# 4. Simple reconstruction metrics
# ------------------------------------------------------------------

def mse_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """Mean squared error per feature (column)."""
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)
    assert x_true.shape == x_pred.shape
    return np.mean((x_true - x_pred) ** 2, axis=0)


def pearson_corr_per_feature(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    """Pearson correlation per feature, computed column-wise."""
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)
    assert x_true.shape == x_pred.shape

    x_true_c = x_true - x_true.mean(axis=0, keepdims=True)
    x_pred_c = x_pred - x_pred.mean(axis=0, keepdims=True)

    num = (x_true_c * x_pred_c).sum(axis=0)
    denom = np.sqrt((x_true_c ** 2).sum(axis=0) * (x_pred_c ** 2).sum(axis=0)) + 1e-8
    return num / denom


def reconstruction_metrics(
    x_true: np.ndarray,
    x_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Convenience wrapper returning a small metrics dict.
    """
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
    latent: str = "moe_mean",         # <- NEW
    random_state: int = 0,            # <- NEW (for sampling)
) -> np.ndarray:
    """
    Push an AnnData through the UniVI encoder for a single modality.

    Selection rules:
      - if X_key != "X": uses adata.obsm[X_key]
      - else: uses adata.layers[layer] if layer is not None, else adata.X

    Parameters
    ----------
    latent : {"moe_mean","moe_sample","modality_mean","modality_sample"}
        Which latent representation to return.
    random_state : int
        RNG seed used only for "*_sample" modes.

    Returns
    -------
    Z : (n_cells, latent_dim)
        Chosen embedding.
    """
    from .data import _get_matrix

    latent = str(latent).lower().strip()
    valid = {"moe_mean", "moe_sample", "modality_mean", "modality_sample"}
    if latent not in valid:
        raise ValueError(f"latent must be one of {sorted(valid)}; got {latent!r}")

    def _sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        eps = torch.randn(mu.shape, device=mu.device, generator=gen, dtype=mu.dtype)
        return mu + eps * torch.exp(0.5 * logvar)

    model.eval()
    X = _get_matrix(adata, layer=layer, X_key=X_key)
    if sp.issparse(X):
        X = X.toarray()

    # deterministic sampling across batches (if you choose *_sample)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(random_state))

    zs = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            xb = torch.as_tensor(np.asarray(X[start:end]), dtype=torch.float32, device=device)

            mu_dict, logvar_dict = model.encode_modalities({modality: xb})

            if "modality" in latent:
                mu = mu_dict[modality]
                lv = logvar_dict[modality]
                z = mu if latent.endswith("_mean") else _sample_gaussian(mu, lv, gen)
            else:
                moe_out = model.mixture_of_experts(mu_dict, logvar_dict)
                # allow either (mu, logvar) or longer tuples
                if isinstance(moe_out, (tuple, list)) and len(moe_out) >= 2:
                    mu_z, logvar_z = moe_out[0], moe_out[1]
                else:
                    raise RuntimeError("model.mixture_of_experts(...) must return (mu, logvar) for moe_* modes.")
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

            if tgt_mod not in xhat_dict:
                raise KeyError(
                    f"Target modality '{tgt_mod}' not found in decoder outputs. "
                    f"Available: {list(xhat_dict.keys())}"
                )

            preds.append(xhat_dict[tgt_mod].detach().cpu().numpy())

    return np.vstack(preds) if len(preds) else np.zeros((0, 0), dtype=float)


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
    """
    Denoise (reconstruct) a modality by encoding and decoding within the same modality.

    If out_layer is provided, writes the result to `adata.layers[out_layer]`.
    If overwrite_X is True, writes the result to `adata.X` (and ignores out_layer).
    Always returns the denoised matrix.
    """
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


# ------------------------------------------------------------------
# 6. Fit per-label latent Gaussians (simple generative utility)
# ------------------------------------------------------------------

def fit_latent_gaussians_by_label(
    Z: np.ndarray,
    labels: np.ndarray,
    min_cells: int = 20,
    eps: float = 1e-3,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Given latent Z [n_cells, d] and string (or categorical) labels, fit a Gaussian per label.

    Returns:
        {label: {"mu": mu, "cov": cov}}
    """
    Z = np.asarray(Z)
    labels = np.asarray(labels)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    uniq = np.unique(labels)
    d = Z.shape[1] if Z.ndim == 2 else 0

    for lab in uniq:
        idx = np.where(labels == lab)[0]
        if idx.size < min_cells:
            continue

        Z_lab = Z[idx]
        mu = Z_lab.mean(axis=0)

        # empirical covariance + jitter
        cov = np.cov(Z_lab, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov = cov + eps * np.eye(d)

        out[str(lab)] = {"mu": mu, "cov": cov}

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
        samples[lab] = rng.multivariate_normal(mu, cov, size=int(n))

    return samples


def evaluate_alignment(
    *,
    # Option A: provide embeddings directly
    Z1: Optional[np.ndarray] = None,
    Z2: Optional[np.ndarray] = None,

    # Option B: or compute embeddings from AnnData
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

    # NEW: latent selection
    latent: str = "moe_mean",
    latent1: Optional[str] = None,
    latent2: Optional[str] = None,
    random_state: int = 0,

    # Metrics config
    metric: str = "euclidean",
    k_mixing: int = 20,
    k_transfer: int = 15,

    # Mixing labels
    modality_labels: Optional[np.ndarray] = None,

    # Label transfer inputs
    labels_source: Optional[np.ndarray] = None,
    labels_target: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    One-stop evaluation for alignment between two modalities.

    latent / latent1 / latent2:
      - "moe_mean", "moe_sample", "modality_mean", "modality_sample"
      - latent1/latent2 override latent if provided
    """
    out: Dict[str, Any] = {}

    lat1 = latent if latent1 is None else latent1
    lat2 = latent if latent2 is None else latent2

    # --- get embeddings ---
    if Z1 is None or Z2 is None:
        if model is None or adata1 is None or adata2 is None or mod1 is None or mod2 is None:
            raise ValueError("Provide either (Z1, Z2) or (model, adata1, adata2, mod1, mod2).")

        Z1 = encode_adata(
            model, adata1, modality=mod1, device=device,
            layer=layer1, X_key=X_key1, batch_size=batch_size,
            latent=lat1, random_state=random_state,
        )
        Z2 = encode_adata(
            model, adata2, modality=mod2, device=device,
            layer=layer2, X_key=X_key2, batch_size=batch_size,
            latent=lat2, random_state=random_state,
        )

    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    out["n1"] = int(Z1.shape[0])
    out["n2"] = int(Z2.shape[0])
    out["dim"] = int(Z1.shape[1]) if (Z1.ndim == 2) else None
    out["latent1"] = lat1
    out["latent2"] = lat2

    # --- FOSCTTM (requires 1:1 pairing) ---
    if Z1.shape == Z2.shape and Z1.shape[0] > 1:
        out["foscttm"] = compute_foscttm(Z1, Z2, metric=metric)
    else:
        out["foscttm"] = None

    # --- modality mixing ---
    Z_concat = None
    if (Z1.ndim == 2 and Z2.ndim == 2 and Z1.shape[1] == Z2.shape[1]):
        Z_concat = np.vstack([Z1, Z2])

    if Z_concat is not None and Z_concat.shape[0] > 1:
        if modality_labels is None:
            modality_labels = np.concatenate(
                [np.repeat("mod1", Z1.shape[0]), np.repeat("mod2", Z2.shape[0])]
            )
        out["modality_mixing"] = compute_modality_mixing(
            Z_concat, modality_labels=np.asarray(modality_labels), k=k_mixing, metric=metric
        )
    else:
        out["modality_mixing"] = None

    # --- label transfer ---
    if labels_source is not None:
        pred, acc, cm, order = label_transfer_knn(
            Z_source=Z1,
            labels_source=np.asarray(labels_source),
            Z_target=Z2,
            labels_target=None if labels_target is None else np.asarray(labels_target),
            k=k_transfer,
            metric=metric,
        )
        out["label_transfer_pred"] = pred
        out["label_transfer_acc"] = acc
        out["label_transfer_cm"] = cm
        out["label_transfer_label_order"] = order
    else:
        out["label_transfer_pred"] = None
        out["label_transfer_acc"] = None
        out["label_transfer_cm"] = None
        out["label_transfer_label_order"] = None

    return out


