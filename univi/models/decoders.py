# univi/models/decoders.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

from .mlp import build_mlp


# -------------------------------------------------------------------------
# Generic decoder config
# -------------------------------------------------------------------------


@dataclass
class DecoderConfig:
    """
    Generic configuration for feed-forward decoders.

    Parameters
    ----------
    output_dim:
        Number of output features (e.g. n_genes, n_peaks, n_proteins).
    hidden_dims:
        Hidden layer sizes for the MLP trunk.
    dropout:
        Dropout probability between hidden layers.
    batchnorm:
        Whether to apply BatchNorm1d between hidden layers.
    """

    output_dim: int
    hidden_dims: List[int]
    dropout: float = 0.0
    batchnorm: bool = False


# -------------------------------------------------------------------------
# Gaussian decoders (for continuous / normalized data)
# -------------------------------------------------------------------------


class GaussianDecoder(nn.Module):
    """
    Simple decoder: z -> reconstruction mean (for MSE / Gaussian-like losses).

    Useful for:
      - log-normalized RNA
      - library-size-normalized ATAC topic scores
      - continuous embeddings / latent factors
    """

    def __init__(self, cfg: DecoderConfig, latent_dim: int):
        super().__init__()
        self.cfg = cfg
        self.net = build_mlp(
            in_dim=latent_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=cfg.output_dim,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        mean : (batch, output_dim)
            Mean of a unit-variance Gaussian, typically paired with MSE loss.
        """
        return self.net(z)


class GaussianDiagDecoder(nn.Module):
    """
    Decoder that outputs mean and log-variance for each feature.

    Useful if you want a full Gaussian likelihood with learned per-feature
    uncertainty.
    """

    def __init__(self, cfg: DecoderConfig, latent_dim: int):
        super().__init__()
        self.cfg = cfg

        self.backbone = build_mlp(
            in_dim=latent_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=2 * cfg.output_dim,  # mean + logvar
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            'mean'   : (batch, output_dim)
            'logvar' : (batch, output_dim)
        """
        out = self.backbone(z)
        mean, logvar = out.chunk(2, dim=-1)
        return {"mean": mean, "logvar": logvar}


# -------------------------------------------------------------------------
# Bernoulli / binary decoders (e.g. binarized ATAC, presence/absence)
# -------------------------------------------------------------------------


class BernoulliDecoder(nn.Module):
    """
    Decoder for Bernoulli likelihoods: z -> logits.

    Useful for:
      - Binarized accessibility (ATAC)
      - Presence/absence features
      - Any 0/1 indicator data
    """

    def __init__(self, cfg: DecoderConfig, latent_dim: int):
        super().__init__()
        self.cfg = cfg
        self.net = build_mlp(
            in_dim=latent_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=cfg.output_dim,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            'logits' : (batch, output_dim)
        """
        logits = self.net(z)
        return {"logits": logits}


# -------------------------------------------------------------------------
# Poisson decoders (simple count model)
# -------------------------------------------------------------------------


class PoissonDecoder(nn.Module):
    """
    Decoder for Poisson likelihoods: z -> log-rate.

    Often OK for simple count data, but for RNA/ADT, NB/ZINB is usually better.
    """

    def __init__(self, cfg: DecoderConfig, latent_dim: int):
        super().__init__()
        self.cfg = cfg
        self.net = build_mlp(
            in_dim=latent_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=cfg.output_dim,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            'log_rate' : (batch, output_dim)
            'rate'     : (batch, output_dim), softplus(log_rate)
        """
        log_rate = self.net(z)
        rate = F.softplus(log_rate)
        return {"log_rate": log_rate, "rate": rate}


# -------------------------------------------------------------------------
# Negative Binomial / Zero-Inflated Negative Binomial decoders
# (standard for scRNA & ADT)
# -------------------------------------------------------------------------


class NegativeBinomialDecoder(nn.Module):
    """
    Negative Binomial decoder for overdispersed counts (RNA, ADT).

    Parameterization:
      - mean mu  > 0  (via softplus)
      - dispersion theta > 0 (can be global or gene-wise)

    The actual NB log-likelihood will be implemented in the loss / objectives.
    """

    def __init__(
        self,
        cfg: DecoderConfig,
        latent_dim: int,
        dispersion: str = "gene",  # 'global' or 'gene'
        init_log_theta: float = 0.0,
    ):
        """
        Parameters
        ----------
        dispersion:
            'global' : single dispersion shared across all features
            'gene'   : one dispersion per feature
        """
        super().__init__()
        self.cfg = cfg
        self.dispersion = dispersion

        self.mu_net = build_mlp(
            in_dim=latent_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=cfg.output_dim,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )

        if dispersion == "global":
            self.log_theta = nn.Parameter(torch.full((1,), float(init_log_theta)))
        elif dispersion == "gene":
            self.log_theta = nn.Parameter(
                torch.full((cfg.output_dim,), float(init_log_theta))
            )
        else:
            raise ValueError(f"Unknown dispersion mode: {dispersion}")

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            'mu'        : (batch, output_dim), NB mean
            'log_theta' : (1,) or (output_dim,)
        """
        mu = F.softplus(self.mu_net(z))  # strictly positive
        return {"mu": mu, "log_theta": self.log_theta}


class ZeroInflatedNegativeBinomialDecoder(nn.Module):
    """
    ZINB decoder: NB mean + dispersion + zero-inflation logits.

    Common for:
      - raw scRNA counts
      - ADT counts with extra dropout
    """

    def __init__(
        self,
        cfg: DecoderConfig,
        latent_dim: int,
        dispersion: str = "gene",
        init_log_theta: float = 0.0,
    ):
        super().__init__()
        self.cfg = cfg
        self.dispersion = dispersion

        # backbone -> split into mu + dropout logits
        self.backbone = build_mlp(
            in_dim=latent_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=2 * cfg.output_dim,  # mu_logits + dropout_logits
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )

        if dispersion == "global":
            self.log_theta = nn.Parameter(torch.full((1,), float(init_log_theta)))
        elif dispersion == "gene":
            self.log_theta = nn.Parameter(
                torch.full((cfg.output_dim,), float(init_log_theta))
            )
        else:
            raise ValueError(f"Unknown dispersion mode: {dispersion}")

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            'mu'         : (batch, output_dim), NB mean
            'log_theta'  : (1,) or (output_dim,)
            'logit_pi'   : (batch, output_dim), logit of zero-inflation probability
        """
        out = self.backbone(z)
        mu_logits, logit_pi = out.chunk(2, dim=-1)

        # Softplus to ensure positivity of mu
        mu = F.softplus(mu_logits)

        return {
            "mu": mu,
            "log_theta": self.log_theta,
            "logit_pi": logit_pi,
        }


# -------------------------------------------------------------------------
# Logistic-normal decoders (for compositions / probabilities)
# -------------------------------------------------------------------------


class LogisticNormalDecoder(nn.Module):
    """
    Decoder that outputs logits for a logistic-normal / softmax distribution.

    Useful for:
      - compositional data (e.g., topic proportions)
      - probability vectors (e.g., normalized protein fractions)
    """

    def __init__(self, cfg: DecoderConfig, latent_dim: int):
        super().__init__()
        self.cfg = cfg
        self.net = build_mlp(
            in_dim=latent_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=cfg.output_dim,
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            'logits' : (batch, output_dim)
            'probs'  : (batch, output_dim), softmax(logits)
        """
        logits = self.net(z)
        probs = F.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}


# -------------------------------------------------------------------------
# Categorical decoder (for discrete labels)
# -------------------------------------------------------------------------


class CategoricalDecoder(nn.Module):
    """
    Decoder that maps z -> categorical logits.

    Typically used if you have a discrete outcome or want to predict a
    label distribution from the latent space.
    """

    def __init__(self, cfg: DecoderConfig, latent_dim: int):
        super().__init__()
        self.cfg = cfg
        self.net = build_mlp(
            in_dim=latent_dim,
            hidden_dims=cfg.hidden_dims,
            out_dim=cfg.output_dim,  # number of categories
            dropout=cfg.dropout,
            batchnorm=cfg.batchnorm,
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            'logits' : (batch, n_categories)
            'probs'  : (batch, n_categories)
        """
        logits = self.net(z)
        probs = F.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}


# -------------------------------------------------------------------------
# Registry / factory
# -------------------------------------------------------------------------


DECODER_REGISTRY = {
    "gaussian": GaussianDecoder,
    "gaussian_diag": GaussianDiagDecoder,
    "bernoulli": BernoulliDecoder,
    "poisson": PoissonDecoder,
    "nb": NegativeBinomialDecoder,
    "negative_binomial": NegativeBinomialDecoder,
    "zinb": ZeroInflatedNegativeBinomialDecoder,
    "zero_inflated_negative_binomial": ZeroInflatedNegativeBinomialDecoder,
    "logistic_normal": LogisticNormalDecoder,
    "categorical": CategoricalDecoder,
}


def build_decoder(
    kind: str,
    cfg: DecoderConfig,
    latent_dim: int,
    **kwargs: Any,
) -> nn.Module:
    """
    Convenience factory: pick a decoder class by name.

    Examples
    --------
    RNA ~ ZINB:
        build_decoder(
            kind="zinb",
            cfg=DecoderConfig(output_dim=n_genes, hidden_dims=[256, 128]),
            latent_dim=z_dim,
            dispersion="gene"
        )

    ATAC (binarized) ~ Bernoulli:
        build_decoder(
            kind="bernoulli",
            cfg=DecoderConfig(output_dim=n_peaks, hidden_dims=[256]),
            latent_dim=z_dim
        )

    ADT ~ NB:
        build_decoder(
            kind="nb",
            cfg=DecoderConfig(output_dim=n_proteins, hidden_dims=[128]),
            latent_dim=z_dim,
            dispersion="global"
        )
    """
    key = kind.lower()
    if key not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder kind: {kind!r}. "
                         f"Available: {list(DECODER_REGISTRY.keys())}")

    cls = DECODER_REGISTRY[key]
    return cls(cfg=cfg, latent_dim=latent_dim, **kwargs)
