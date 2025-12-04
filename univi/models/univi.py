# univi/models/univi.py

from __future__ import annotations

from typing import Dict, Tuple, Optional
import math

import torch
from torch import nn
import torch.nn.functional as F

from univi.config import UniVIConfig, ModalityConfig
from .mlp import build_mlp


class UniVIMultiModalVAE(nn.Module):
    """
    Generic multi-modal mixture-of-experts β-VAE.

    Objectives controlled by `loss_mode`:
      - "v2" / "lite" / "light": fused posterior (precision-weighted PoE-style) +
          per-mod recon + β·KL(q||p) + γ·mean pairwise L2(μ_i, μ_j)
      - "v1" / "paper": paper-style cross-reconstruction + cross-posterior KL alignment
          (requires paired / pseudo-paired samples).
    """

    LOGVAR_MIN = -10.0
    LOGVAR_MAX = 10.0
    EPS = 1e-8

    def __init__(
        self,
        cfg: UniVIConfig,
        *,
        loss_mode: str = "v2",
        v1_recon: str = "cross",
        v1_recon_mix: float = 0.0,
        normalize_v1_terms: bool = True,
    ):
        super().__init__()
        self.cfg = cfg

        # Objective controls
        self.loss_mode = str(loss_mode).lower()
        self.v1_recon = str(v1_recon).lower()
        self.v1_recon_mix = float(v1_recon_mix)
        self.normalize_v1_terms = bool(normalize_v1_terms)

        self.latent_dim = cfg.latent_dim
        self.beta_max = cfg.beta
        self.gamma_max = cfg.gamma

        self.modality_names = [m.name for m in cfg.modalities]
        self.mod_cfg_by_name: Dict[str, ModalityConfig] = {m.name: m for m in cfg.modalities}

        # Per-modality modules
        self.encoders = nn.ModuleDict()
        self.encoder_heads = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        # Likelihood-specific parameters
        self.nb_log_theta = nn.ParameterDict()   # per-feature log(theta)
        self.zinb_logit_pi = nn.ParameterDict()  # per-feature logit(pi)

        for m in cfg.modalities:
            assert isinstance(m, ModalityConfig), (
                f"Each entry in cfg.modalities must be ModalityConfig, got {type(m)}"
            )

            # Encoder: x_m -> (mu_m, logvar_m)
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

            # Decoder: z -> raw outputs (interpreted by likelihood in _recon_loss)
            dec_hidden = m.decoder_hidden if m.decoder_hidden else [max(64, self.latent_dim)]
            dec = build_mlp(
                in_dim=self.latent_dim,
                hidden_dims=dec_hidden,
                out_dim=m.input_dim,
                activation=nn.ReLU(),
                dropout=cfg.decoder_dropout,
                batchnorm=cfg.decoder_batchnorm,
            )
            self.decoders[m.name] = dec

            likelihood = (m.likelihood or "gaussian").lower()
            if likelihood in ("nb", "negative_binomial", "zinb"):
                init_log_theta = math.log(1.0)
                self.nb_log_theta[m.name] = nn.Parameter(
                    torch.full((m.input_dim,), init_log_theta, dtype=torch.float32)
                )

            if likelihood == "zinb":
                self.zinb_logit_pi[m.name] = nn.Parameter(
                    torch.full((m.input_dim,), 0.0, dtype=torch.float32)
                )

        # Shared prior N(0, I)
        self.register_buffer("prior_mu", torch.zeros(self.latent_dim))
        self.register_buffer("prior_logvar", torch.zeros(self.latent_dim))

    # ----------------------------- helpers -----------------------------

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
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
        return 0.5 * kl.sum(dim=-1)

    @staticmethod
    def _nb_nll(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mu = mu.clamp(min=eps)
        theta = theta.clamp(min=eps)
        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        t2 = theta * (torch.log(theta) - torch.log(theta + mu))
        t3 = x * (torch.log(mu) - torch.log(theta + mu))
        return -(t1 + t2 + t3)

    @staticmethod
    def _zinb_nll(
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        logit_pi: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        mu = mu.clamp(min=eps)
        theta = theta.clamp(min=eps)
        pi = torch.sigmoid(logit_pi)

        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        t2 = theta * (torch.log(theta) - torch.log(theta + mu))
        t3 = x * (torch.log(mu) - torch.log(theta + mu))
        log_nb = t1 + t2 + t3

        is_zero = (x < eps)

        log_prob_pos = torch.log1p(-pi + eps) + log_nb
        log_nb_zero = theta * (torch.log(theta) - torch.log(theta + mu))
        log_prob_zero = torch.log(pi + (1.0 - pi) * torch.exp(log_nb_zero) + eps)

        log_prob = torch.where(is_zero, log_prob_zero, log_prob_pos)
        return -log_prob

    def _recon_loss(self, x: torch.Tensor, raw_dec_out: torch.Tensor, likelihood: str, mod_name: str) -> torch.Tensor:
        likelihood = (likelihood or "gaussian").lower()

        if likelihood in ("gaussian", "normal"):
            recon = raw_dec_out
            return ((x - recon) ** 2).sum(dim=-1)

        if likelihood == "mse":
            recon = raw_dec_out
            return ((x - recon) ** 2).mean(dim=-1)

        if likelihood in ("nb", "negative_binomial"):
            mu = F.softplus(raw_dec_out) + self.EPS
            theta = torch.exp(self.nb_log_theta[mod_name]).unsqueeze(0).expand_as(mu)
            return self._nb_nll(x, mu, theta, eps=self.EPS).sum(dim=-1)

        if likelihood == "zinb":
            mu = F.softplus(raw_dec_out) + self.EPS
            theta = torch.exp(self.nb_log_theta[mod_name]).unsqueeze(0).expand_as(mu)
            logit_pi = self.zinb_logit_pi[mod_name].unsqueeze(0).expand_as(mu)
            return self._zinb_nll(x, mu, theta, logit_pi, eps=self.EPS).sum(dim=-1)

        # fallback
        recon = raw_dec_out
        return ((x - recon) ** 2).sum(dim=-1)

    def _alignment_loss(self, mu_per_mod: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(mu_per_mod.keys())
        if len(names) < 2:
            return torch.zeros(mu_per_mod[names[0]].size(0), device=mu_per_mod[names[0]].device)

        losses = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                losses.append(((mu_per_mod[names[i]] - mu_per_mod[names[j]]) ** 2).sum(dim=-1))
        return torch.stack(losses, dim=0).mean(dim=0)

    # -------------------------- encode/decode/PoE --------------------------

    def encode_modalities(self, x_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mu_dict: Dict[str, torch.Tensor] = {}
        logvar_dict: Dict[str, torch.Tensor] = {}
        for m in self.modality_names:
            if m not in x_dict or x_dict[m] is None:
                continue
            h = self.encoder_heads[m](self.encoders[m](x_dict[m]))
            mu, logvar = self._split_mu_logvar(h)
            mu_dict[m] = mu
            logvar_dict[m] = logvar
        return mu_dict, logvar_dict

    def mixture_of_experts(
        self,
        mu_dict: Dict[str, torch.Tensor],
        logvar_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # precision-weighted fusion (PoE-style in diagonal Gaussian case)
        mus = list(mu_dict.values())
        logvars = list(logvar_dict.values())
        precisions = [torch.exp(-lv) for lv in logvars]
        precision_sum = torch.stack(precisions, dim=0).sum(dim=0) + 1e-8
        mu_weighted = torch.stack([m * p for m, p in zip(mus, precisions)], dim=0).sum(dim=0)
        mu_comb = mu_weighted / precision_sum
        var_comb = 1.0 / precision_sum
        logvar_comb = torch.log(var_comb + 1e-8)
        return mu_comb, logvar_comb

    def decode_modalities(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {m: self.decoders[m](z) for m in self.modality_names}

    # ------------------------------ forward ------------------------------

    def forward(self, x_dict: Dict[str, torch.Tensor], epoch: int = 0) -> Dict[str, torch.Tensor]:
        mode = (self.loss_mode or "v2").lower()

        if mode in ("v1", "paper", "cross"):
            return self._forward_v1(x_dict=x_dict, epoch=epoch)

        if mode in ("v2", "lite", "light", "moe", "poe", "fused"):
            return self._forward_v2(x_dict=x_dict, epoch=epoch)

        raise ValueError(f"Unknown loss_mode={self.loss_mode!r}. Expected 'v1' or 'v2'/'lite'.")

    def _forward_v2(self, x_dict: Dict[str, torch.Tensor], epoch: int = 0) -> Dict[str, torch.Tensor]:
        mu_dict, logvar_dict = self.encode_modalities(x_dict)
        assert len(mu_dict) > 0, "At least one modality must be present."

        mu_z, logvar_z = self.mixture_of_experts(mu_dict, logvar_dict)
        z = self._reparameterize(mu_z, logvar_z)
        xhat_dict = self.decode_modalities(z)

        recon_losses: Dict[str, torch.Tensor] = {}
        recon_total: Optional[torch.Tensor] = None

        for name, m_cfg in self.mod_cfg_by_name.items():
            if name not in x_dict or x_dict[name] is None:
                continue
            loss_m = self._recon_loss(x=x_dict[name], raw_dec_out=xhat_dict[name], likelihood=m_cfg.likelihood, mod_name=name)
            recon_losses[name] = loss_m
            recon_total = loss_m if recon_total is None else (recon_total + loss_m)

        assert recon_total is not None, "No modalities in x_dict produced recon loss."

        mu_p = self.prior_mu.expand_as(mu_z)
        logvar_p = self.prior_logvar.expand_as(logvar_z)
        kl = self._kl_gaussian(mu_z, logvar_z, mu_p, logvar_p)

        align_loss = self._alignment_loss(mu_dict)

        beta = self._anneal_weight(epoch, self.cfg.kl_anneal_start, self.cfg.kl_anneal_end, self.beta_max)
        gamma = self._anneal_weight(epoch, self.cfg.align_anneal_start, self.cfg.align_anneal_end, self.gamma_max)

        loss = recon_total + beta * kl + gamma * align_loss

        return {
            "loss": loss.mean(),
            "recon_total": recon_total.mean(),
            "kl": kl.mean(),
            "align": align_loss.mean(),
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "z": z,
            "xhat": xhat_dict,
            "mu_dict": mu_dict,
            "logvar_dict": logvar_dict,
            "recon_per_modality": {k: v.mean() for k, v in recon_losses.items()},
            "beta": torch.tensor(beta, device=loss.device),
            "gamma": torch.tensor(gamma, device=loss.device),
        }

    def _forward_v1(self, x_dict: Dict[str, torch.Tensor], epoch: int = 0) -> Dict[str, torch.Tensor]:
        mu_dict, logvar_dict = self.encode_modalities(x_dict)
        assert len(mu_dict) > 0, "At least one modality must be present."

        present = list(mu_dict.keys())
        K = len(present)

        # reporting latent (for outputs)
        mu_moe, logvar_moe = self.mixture_of_experts(mu_dict, logvar_dict)
        z_moe = self._reparameterize(mu_moe, logvar_moe)

        # source z per modality
        z_src = {m: self._reparameterize(mu_dict[m], logvar_dict[m]) for m in present}
        v1_recon = (self.v1_recon or "cross").lower()

        def recon_target_from_z(z: torch.Tensor, target_mod: str) -> torch.Tensor:
            raw = self.decoders[target_mod](z)
            m_cfg = self.mod_cfg_by_name[target_mod]
            return self._recon_loss(x=x_dict[target_mod], raw_dec_out=raw, likelihood=m_cfg.likelihood, mod_name=target_mod)

        recon_per_target = {m: torch.zeros(mu_dict[present[0]].size(0), device=mu_dict[present[0]].device) for m in present}
        recon_total: Optional[torch.Tensor] = None

        def add_loss(t: torch.Tensor):
            nonlocal recon_total
            recon_total = t if recon_total is None else (recon_total + t)

        if v1_recon.startswith("src:"):
            src_name = v1_recon.split("src:", 1)[1].strip()
            if src_name not in z_src:
                raise ValueError(f"v1_recon={self.v1_recon!r} but '{src_name}' not present. Present={present}")
            z_use = z_src[src_name]
            for j in present:
                lj = recon_target_from_z(z_use, j)
                recon_per_target[j] += lj
                add_loss(lj)

        elif v1_recon == "self":
            for j in present:
                lj = recon_target_from_z(z_src[j], j)
                recon_per_target[j] += lj
                add_loss(lj)

        elif v1_recon in ("avg", "average"):
            z_avg = torch.stack([z_src[m] for m in present], dim=0).mean(dim=0)
            for j in present:
                lj = recon_target_from_z(z_avg, j)
                recon_per_target[j] += lj
                add_loss(lj)
            z_moe = z_avg

        elif v1_recon in ("moe", "poe", "fused"):
            for j in present:
                lj = recon_target_from_z(z_moe, j)
                recon_per_target[j] += lj
                add_loss(lj)

        else:
            for k in present:
                zk = z_src[k]
                for j in present:
                    lj = recon_target_from_z(zk, j)
                    recon_per_target[j] += lj
                    add_loss(lj)

            if self.v1_recon_mix > 0.0 and K > 1:
                z_avg = torch.stack([z_src[m] for m in present], dim=0).mean(dim=0)
                mix = float(self.v1_recon_mix)
                for j in present:
                    lj = recon_target_from_z(z_avg, j)
                    recon_per_target[j] += mix * lj
                    add_loss(mix * lj)

        assert recon_total is not None, "v1 recon produced no loss."

        # normalize recon so it doesn't scale with #modalities
        if self.normalize_v1_terms:
            if v1_recon == "self":
                denom = max(K, 1)
            elif v1_recon.startswith("src:") or v1_recon in ("avg", "average", "moe", "poe", "fused"):
                denom = max(K, 1)
            else:
                denom = max(K * K, 1)

            recon_total = recon_total / float(denom)
            recon_per_target = {k: v / float(denom) for k, v in recon_per_target.items()}

        # KL(q_k || p)
        mu_p = self.prior_mu.expand_as(mu_dict[present[0]])
        logvar_p = self.prior_logvar.expand_as(logvar_dict[present[0]])
        kl = torch.stack([self._kl_gaussian(mu_dict[k], logvar_dict[k], mu_p, logvar_p) for k in present], dim=0).sum(dim=0)
        if self.normalize_v1_terms:
            kl = kl / float(max(K, 1))

        # cross KL(q_k || q_j), ordered pairs
        if K < 2:
            cross_kl = torch.zeros(mu_dict[present[0]].size(0), device=mu_dict[present[0]].device)
        else:
            cross_terms = []
            for i in range(K):
                for j in range(K):
                    if i == j:
                        continue
                    mi, lvi = mu_dict[present[i]], logvar_dict[present[i]]
                    mj, lvj = mu_dict[present[j]], logvar_dict[present[j]]
                    cross_terms.append(self._kl_gaussian(mi, lvi, mj, lvj))
            cross_kl = torch.stack(cross_terms, dim=0).sum(dim=0)
            if self.normalize_v1_terms:
                cross_kl = cross_kl / float(K * (K - 1))

        beta = self._anneal_weight(epoch, self.cfg.kl_anneal_start, self.cfg.kl_anneal_end, self.beta_max)
        gamma = self._anneal_weight(epoch, self.cfg.align_anneal_start, self.cfg.align_anneal_end, self.gamma_max)

        loss = recon_total + beta * kl + gamma * cross_kl
        xhat_dict = self.decode_modalities(z_moe)

        return {
            "loss": loss.mean(),
            "recon_total": recon_total.mean(),
            "kl": kl.mean(),
            "align": cross_kl.mean(),
            "cross_kl": cross_kl.mean(),
            "mu_z": mu_moe,
            "logvar_z": logvar_moe,
            "z": z_moe,
            "xhat": xhat_dict,
            "mu_dict": mu_dict,
            "logvar_dict": logvar_dict,
            "recon_per_modality": {k: v.mean() for k, v in recon_per_target.items()},
            "beta": torch.tensor(beta, device=loss.device),
            "gamma": torch.tensor(gamma, device=loss.device),
        }

    def _anneal_weight(self, epoch: int, start: int, end: int, max_val: float) -> float:
        if end <= start:
            return float(max_val)
        if epoch <= start:
            return 0.0
        if epoch >= end:
            return float(max_val)
        frac = (epoch - start) / float(end - start)
        return float(max_val) * float(frac)
