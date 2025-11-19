# univi/models/univi.py

from __future__ import annotations
from typing import Dict, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F

from univi.config import UniVIConfig, ModalityConfig
from .mlp import build_mlp


class UniVIMultiModalVAE(nn.Module):
    """
    Generic multi-modal mixture-of-experts β-VAE.

    - Encoders: x_m -> (mu_m, logvar_m)
    - Mixture-of-experts to get shared q(z|x)
    - Decoders: z -> parameters of p(x_m | z) with
        * Gaussian
        * MSE (Gaussian with fixed variance)
        * Negative Binomial (NB)
        * Zero-Inflated NB (ZINB)
    """

    LOGVAR_MIN = -10.0
    LOGVAR_MAX = 10.0
    EPS = 1e-8

    def __init__(self, cfg: UniVIConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim

        # Max weights for annealing
        self.beta_max = cfg.beta
        self.gamma_max = cfg.gamma

        self.modality_names = [m.name for m in cfg.modalities]

        # Per-modality modules
        self.encoders = nn.ModuleDict()
        self.encoder_heads = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        # Distribution-specific parameters
        self.nb_log_theta = nn.ParameterDict()   # per-feature log(θ)
        self.zinb_logit_pi = nn.ParameterDict()  # per-feature logit(π) for ZINB

        # Build encoders/decoders and extra params per modality
        for m in cfg.modalities:
            assert isinstance(m, ModalityConfig), (
                f"Each entry in cfg.modalities must be ModalityConfig, got {type(m)}"
            )

            # ---------- encoder: x_m -> [μ_m, log σ_m^2] ----------
            enc = build_mlp(
                in_dim=m.input_dim,
                hidden_dims=m.encoder_hidden,
                out_dim=self.latent_dim * 2,
                activation=nn.ReLU(),
                dropout=cfg.encoder_dropout,
                batchnorm=cfg.encoder_batchnorm,
            )
            self.encoders[m.name] = enc
            self.encoder_heads[m.name] = nn.Identity()

            # ---------- decoder: z -> raw pre-activation ----------
            dec_hidden = (
                m.decoder_hidden if m.decoder_hidden
                else [max(64, self.latent_dim)]
            )
            dec = build_mlp(
                in_dim=self.latent_dim,
                hidden_dims=dec_hidden,
                out_dim=m.input_dim,  # one parameter per feature
                activation=nn.ReLU(),
                dropout=cfg.decoder_dropout,
                batchnorm=cfg.decoder_batchnorm,
            )
            self.decoders[m.name] = dec

            # ---------- distribution-specific parameters ----------
            likelihood = (m.likelihood or "gaussian").lower()

            if likelihood in ("nb", "negative_binomial", "zinb"):
                # gene-wise inverse dispersion (θ > 0)
                init_log_theta = math.log(1.0)
                param = nn.Parameter(
                    torch.full((m.input_dim,), init_log_theta, dtype=torch.float32)
                )
                self.nb_log_theta[m.name] = param

            if likelihood == "zinb":
                # gene-wise dropout logits (π in (0, 1))
                init_logit_pi = 0.0
                param = nn.Parameter(
                    torch.full((m.input_dim,), init_logit_pi, dtype=torch.float32)
                )
                self.zinb_logit_pi[m.name] = param

        # shared prior N(0, I)
        self.register_buffer("prior_mu", torch.zeros(self.latent_dim))
        self.register_buffer("prior_logvar", torch.zeros(self.latent_dim))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _split_mu_logvar(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = torch.chunk(h, 2, dim=-1)
        logvar = torch.clamp(logvar, self.LOGVAR_MIN, self.LOGVAR_MAX)
        return mu, logvar

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_gaussian(
        self,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(q || p) for diagonal Gaussians, per-sample.
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = (
            logvar_p
            - logvar_q
            + (var_q + (mu_q - mu_p) ** 2) / var_p
            - 1.0
        )
        return 0.5 * kl.sum(dim=-1)

    # ---------- NB / ZINB log-likelihoods ----------

    @staticmethod
    def _nb_nll(
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Negative Binomial negative log-likelihood per entry.

        x: counts >= 0
        mu: mean > 0
        theta: inverse dispersion > 0 (larger => closer to Poisson)
        """
        mu = mu.clamp(min=eps)
        theta = theta.clamp(min=eps)

        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        t2 = theta * (torch.log(theta) - torch.log(theta + mu))
        t3 = x * (torch.log(mu) - torch.log(theta + mu))
        log_prob = t1 + t2 + t3
        return -log_prob  # same shape as x

    @staticmethod
    def _zinb_nll(
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        logit_pi: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Zero-Inflated NB negative log-likelihood per entry.
        """
        mu = mu.clamp(min=eps)
        theta = theta.clamp(min=eps)
        pi = torch.sigmoid(logit_pi)  # dropout probability

        # NB log pmf
        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        t2 = theta * (torch.log(theta) - torch.log(theta + mu))
        t3 = x * (torch.log(mu) - torch.log(theta + mu))
        log_nb = t1 + t2 + t3

        is_zero = (x < eps)

        # x > 0: log(1 - pi) + log NB(x)
        log_prob_pos = torch.log1p(-pi + eps) + log_nb

        # x == 0: log(pi + (1 - pi)*NB(0))
        log_nb_zero = theta * (torch.log(theta) - torch.log(theta + mu))
        log_prob_zero = torch.log(pi + (1.0 - pi) * torch.exp(log_nb_zero) + eps)

        log_prob = torch.where(is_zero, log_prob_zero, log_prob_pos)
        return -log_prob

    def _recon_loss(
        self,
        x: torch.Tensor,
        raw_dec_out: torch.Tensor,
        likelihood: str,
        mod_name: str,
    ) -> torch.Tensor:
        """
        Per-cell reconstruction loss.

        Returns a tensor of shape (batch,).
        """
        likelihood = (likelihood or "gaussian").lower()

        if likelihood in ("gaussian", "normal"):
            # treat raw_dec_out as mean
            recon = raw_dec_out
            return ((x - recon) ** 2).sum(dim=-1)

        if likelihood == "mse":
            recon = raw_dec_out
            return ((x - recon) ** 2).mean(dim=-1)

        if likelihood in ("nb", "negative_binomial"):
            # MLP output -> log-mean -> mean via softplus
            log_mu = raw_dec_out
            mu = F.softplus(log_mu) + self.EPS

            log_theta = self.nb_log_theta[mod_name]  # (n_features,)
            theta = torch.exp(log_theta).unsqueeze(0).expand_as(mu)

            nll = self._nb_nll(x, mu, theta, eps=self.EPS)
            return nll.sum(dim=-1)

        if likelihood == "zinb":
            log_mu = raw_dec_out
            mu = F.softplus(log_mu) + self.EPS

            log_theta = self.nb_log_theta[mod_name]
            theta = torch.exp(log_theta).unsqueeze(0).expand_as(mu)

            logit_pi = self.zinb_logit_pi[mod_name].unsqueeze(0).expand_as(mu)

            nll = self._zinb_nll(x, mu, theta, logit_pi, eps=self.EPS)
            return nll.sum(dim=-1)

        # fallback: squared error
        recon = raw_dec_out
        return ((x - recon) ** 2).sum(dim=-1)

    def _alignment_loss(
        self,
        mu_per_mod: Dict[str, torch.Tensor],
        logvar_per_mod: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Symmetric cross-modal alignment:
        average pairwise L2 between means across modalities.
        """
        names = list(mu_per_mod.keys())
        if len(names) < 2:
            # no alignment penalty if < 2 modalities present
            return torch.zeros(
                mu_per_mod[names[0]].size(0),
                device=mu_per_mod[names[0]].device,
            )

        losses = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                mu_i, mu_j = mu_per_mod[names[i]], mu_per_mod[names[j]]
                losses.append(((mu_i - mu_j) ** 2).sum(dim=-1))

        if not losses:
            return torch.zeros(
                mu_per_mod[names[0]].size(0),
                device=mu_per_mod[names[0]].device,
            )
        stacked = torch.stack(losses, dim=0)
        return stacked.mean(dim=0)

    # ------------------------------------------------------------------
    # encode / decode / MoE
    # ------------------------------------------------------------------

    def encode_modalities(
        self,
        x_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mu_dict: Dict[str, torch.Tensor] = {}
        logvar_dict: Dict[str, torch.Tensor] = {}

        for m in self.modality_names:
            if m not in x_dict or x_dict[m] is None:
                continue
            h = self.encoders[m](x_dict[m])
            h = self.encoder_heads[m](h)
            mu, logvar = self._split_mu_logvar(h)
            mu_dict[m] = mu
            logvar_dict[m] = logvar
        return mu_dict, logvar_dict

    def mixture_of_experts(
        self,
        mu_dict: Dict[str, torch.Tensor],
        logvar_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Product-of-experts–style combination in precision space.
        """
        mus = list(mu_dict.values())
        logvars = list(logvar_dict.values())
        precisions = [torch.exp(-lv) for lv in logvars]  # 1/var
        precision_sum = torch.stack(precisions, dim=0).sum(dim=0) + 1e-8
        mu_weighted = torch.stack(
            [m * p for m, p in zip(mus, precisions)],
            dim=0,
        ).sum(dim=0)
        mu_comb = mu_weighted / precision_sum
        var_comb = 1.0 / precision_sum
        logvar_comb = torch.log(var_comb + 1e-8)
        return mu_comb, logvar_comb

    def decode_modalities(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        xhat_dict: Dict[str, torch.Tensor] = {}
        for m in self.modality_names:
            xhat_dict[m] = self.decoders[m](z)
        return xhat_dict

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        x_dict: modality -> [batch, features] tensor
                missing modalities can be omitted or set to None.
        epoch: current epoch (for KL / alignment annealing)
        """
        # 1) encode per modality
        mu_dict, logvar_dict = self.encode_modalities(x_dict)
        assert len(mu_dict) > 0, "At least one modality must be present."

        # 2) mixture-of-experts to get shared q(z|x)
        mu_z, logvar_z = self.mixture_of_experts(mu_dict, logvar_dict)
        z = self._reparameterize(mu_z, logvar_z)

        # 3) decode
        xhat_dict = self.decode_modalities(z)

        # 4) reconstruction losses
        recon_losses: Dict[str, torch.Tensor] = {}
        recon_total = 0.0
        for m_cfg in self.cfg.modalities:
            name = m_cfg.name
            if name not in x_dict or x_dict[name] is None:
                continue
            loss_m = self._recon_loss(
                x=x_dict[name],
                raw_dec_out=xhat_dict[name],
                likelihood=m_cfg.likelihood,
                mod_name=name,
            )
            recon_losses[name] = loss_m
            recon_total = recon_total + loss_m  # tensor broadcast

        # 5) KL to prior (per-sample)
        mu_p = self.prior_mu.expand_as(mu_z)
        logvar_p = self.prior_logvar.expand_as(logvar_z)
        kl = self._kl_gaussian(mu_z, logvar_z, mu_p, logvar_p)

        # 6) cross-modal alignment loss
        align_loss = self._alignment_loss(mu_dict, logvar_dict)

        # 7) annealing schedules
        beta = self._anneal_weight(
            epoch,
            self.cfg.kl_anneal_start,
            self.cfg.kl_anneal_end,
            self.beta_max,
        )
        gamma = self._anneal_weight(
            epoch,
            self.cfg.align_anneal_start,
            self.cfg.align_anneal_end,
            self.gamma_max,
        )

        # 8) total loss (per-sample)
        loss = recon_total + beta * kl + gamma * align_loss

        # 9) aggregate and return
        loss_mean = loss.mean()
        recon_mean = recon_total.mean()
        kl_mean = kl.mean()
        align_mean = align_loss.mean()

        beta_t = torch.tensor(beta, device=loss_mean.device)
        gamma_t = torch.tensor(gamma, device=loss_mean.device)

        return {
            "loss": loss_mean,
            "recon_total": recon_mean,
            "kl": kl_mean,
            "align": align_mean,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "z": z,
            "xhat": xhat_dict,
            "mu_dict": mu_dict,
            "logvar_dict": logvar_dict,
            "recon_per_modality": {k: v.mean() for k, v in recon_losses.items()},
            "beta": beta_t,
            "gamma": gamma_t,
        }

    # ------------------------------------------------------------------
    # annealing helper
    # ------------------------------------------------------------------

    def _anneal_weight(self, epoch: int, start: int, end: int, max_val: float) -> float:
        if end <= start:
            return max_val
        if epoch <= start:
            return 0.0
        if epoch >= end:
            return max_val
        frac = (epoch - start) / float(end - start)
        return max_val * frac
