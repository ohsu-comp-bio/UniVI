# univi/models/univi.py

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..config import UniVIConfig, ModalityConfig
from .mlp import build_mlp
from .decoders import DecoderConfig, build_decoder  # optional categorical head


class UniVIMultiModalVAE(nn.Module):
    """
    Multi-modal β-VAE with per-modality encoders and decoders.

    loss_mode:
      - "v2" / "lite": fused posterior (precision-weighted fusion) +
          per-mod recon + β·KL(q_fused||p) + γ·mean pairwise L2(μ_i, μ_j)
      - "v1": paper-style:
          * per-modality posteriors q_m(z|x_m) and samples z_m
          * reconstruction terms controlled by v1_recon
          * KL(q_m || p) + γ·(mean over ordered pairs i!=j KL(q_i || q_j))

    v1_recon options:
      - "cross": all ordered pairs src!=tgt (default)
      - "self":  self recon only (tgt->tgt)
      - "avg" (aliases: "both", "paper", "self+cross"):
            equal total weight on self and cross recon (50/50), with weights
            automatically adjusted for any number of modalities K:
              self term weight  = 0.5 / K
              cross term weight = 0.5 / (K*(K-1))
      - "avg_z" / "mean_z": decode from the *averaged* modality latents z̄ into each target
      - "moe": decode from a fused latent sample (from mixture_of_experts) into each target
      - "src:<name>": use one chosen source modality latent to reconstruct all targets
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

        # optional label head (decoder-only)
        n_label_classes: int = 0,
        label_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.cfg = cfg

        self.loss_mode = str(loss_mode).lower().strip()
        self.v1_recon = str(v1_recon).lower().strip()
        self.v1_recon_mix = float(v1_recon_mix)
        self.normalize_v1_terms = bool(normalize_v1_terms)

        self.latent_dim = int(cfg.latent_dim)
        self.beta_max = float(cfg.beta)
        self.gamma_max = float(cfg.gamma)

        self.modality_names: List[str] = [m.name for m in cfg.modalities]
        self.mod_cfg_by_name: Dict[str, ModalityConfig] = {m.name: m for m in cfg.modalities}

        # Per-modality modules
        self.encoders = nn.ModuleDict()
        self.encoder_heads = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        # Likelihood-specific parameters (simple, per-feature)
        self.nb_log_theta = nn.ParameterDict()   # per-feature log(theta)
        self.zinb_logit_pi = nn.ParameterDict()  # per-feature logit(pi)

        for m in cfg.modalities:
            if not isinstance(m, ModalityConfig):
                raise TypeError(f"cfg.modalities must contain ModalityConfig, got {type(m)}")

            # Encoder: x_m -> (mu_m, logvar_m)
            enc = build_mlp(
                in_dim=m.input_dim,
                hidden_dims=m.encoder_hidden,
                out_dim=self.latent_dim * 2,
                activation=nn.ReLU(),
                dropout=float(cfg.encoder_dropout),
                batchnorm=bool(cfg.encoder_batchnorm),
            )
            self.encoders[m.name] = enc
            self.encoder_heads[m.name] = nn.Identity()

            # Decoder: z -> raw outputs (interpreted by _recon_loss)
            dec_hidden = m.decoder_hidden if m.decoder_hidden else [max(64, self.latent_dim)]
            dec = build_mlp(
                in_dim=self.latent_dim,
                hidden_dims=dec_hidden,
                out_dim=m.input_dim,
                activation=nn.ReLU(),
                dropout=float(cfg.decoder_dropout),
                batchnorm=bool(cfg.decoder_batchnorm),
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

        # Optional label head: categorical decoder on z only (decoder-only supervision)
        self.n_label_classes = int(n_label_classes) if n_label_classes is not None else 0
        self.label_loss_weight = float(label_loss_weight)
        if self.n_label_classes > 0:
            dec_cfg = DecoderConfig(
                output_dim=self.n_label_classes,
                hidden_dims=[max(64, self.latent_dim)],
                dropout=float(cfg.decoder_dropout),
                batchnorm=bool(cfg.decoder_batchnorm),
            )
            self.label_decoder = build_decoder("categorical", cfg=dec_cfg, latent_dim=self.latent_dim)
        else:
            self.label_decoder = None

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
            return ((x - raw_dec_out) ** 2).sum(dim=-1)

        if likelihood == "mse":
            return ((x - raw_dec_out) ** 2).mean(dim=-1)

        if likelihood in ("nb", "negative_binomial"):
            mu = F.softplus(raw_dec_out) + self.EPS
            theta = torch.exp(self.nb_log_theta[mod_name]).unsqueeze(0).expand_as(mu)
            return self._nb_nll(x, mu, theta, eps=self.EPS).sum(dim=-1)

        if likelihood == "zinb":
            mu = F.softplus(raw_dec_out) + self.EPS
            theta = torch.exp(self.nb_log_theta[mod_name]).unsqueeze(0).expand_as(mu)
            logit_pi = self.zinb_logit_pi[mod_name].unsqueeze(0).expand_as(mu)
            return self._zinb_nll(x, mu, theta, logit_pi, eps=self.EPS).sum(dim=-1)

        return ((x - raw_dec_out) ** 2).sum(dim=-1)

    def _alignment_loss_l2mu(self, mu_per_mod: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(mu_per_mod.keys())
        if len(names) < 2:
            # safe even if empty (though callers ensure non-empty)
            if len(names) == 0:
                return torch.tensor(0.0, device=next(self.parameters()).device).expand(1)
            k = names[0]
            return torch.zeros(mu_per_mod[k].size(0), device=mu_per_mod[k].device)
        losses = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                losses.append(((mu_per_mod[names[i]] - mu_per_mod[names[j]]) ** 2).sum(dim=-1))
        return torch.stack(losses, dim=0).mean(dim=0)

    # -------------------------- encode/decode/fuse --------------------------

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

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        epoch: int = 0,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        mode = (self.loss_mode or "v2").lower()
        if mode in ("v1", "paper", "cross"):
            return self._forward_v1(x_dict=x_dict, epoch=epoch, y=y)
        if mode in ("v2", "lite", "light", "moe", "poe", "fused"):
            return self._forward_v2(x_dict=x_dict, epoch=epoch, y=y)
        raise ValueError(f"Unknown loss_mode={self.loss_mode!r}.")

    def _forward_v2(self, x_dict: Dict[str, torch.Tensor], epoch: int = 0, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mu_dict, logvar_dict = self.encode_modalities(x_dict)
        if len(mu_dict) == 0:
            raise ValueError("At least one modality must be present in x_dict.")

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

        if recon_total is None:
            raise RuntimeError("No modalities in x_dict produced recon loss.")

        mu_p = self.prior_mu.expand_as(mu_z)
        logvar_p = self.prior_logvar.expand_as(logvar_z)
        kl = self._kl_gaussian(mu_z, logvar_z, mu_p, logvar_p)

        align_loss = self._alignment_loss_l2mu(mu_dict)

        beta = self._anneal_weight(epoch, self.cfg.kl_anneal_start, self.cfg.kl_anneal_end, self.beta_max)
        gamma = self._anneal_weight(epoch, self.cfg.align_anneal_start, self.cfg.align_anneal_end, self.gamma_max)

        loss = recon_total + beta * kl + gamma * align_loss

        # optional label head
        class_loss = None
        class_logits = None
        if self.label_decoder is not None and y is not None:
            y = y.long()
            dec_out = self.label_decoder(z)
            class_logits = dec_out["logits"] if isinstance(dec_out, dict) and "logits" in dec_out else dec_out
            class_loss = F.cross_entropy(class_logits, y, reduction="none")
            loss = loss + self.label_loss_weight * class_loss

        out: Dict[str, Any] = {
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
        if class_loss is not None:
            out["class_loss"] = class_loss.mean()
        if class_logits is not None:
            out["class_logits"] = class_logits
        return out

    def _forward_v1(self, x_dict: Dict[str, torch.Tensor], epoch: int = 0, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        TRUE v1:
          - one posterior per modality: q_m(z|x_m)
          - one latent sample per modality: z_m
          - recon terms depend on v1_recon (see class docstring)
          - KL(q_m||p) and cross_kl = mean_{i!=j} KL(q_i||q_j) (ordered pairs)
        """
        mu_dict, logvar_dict = self.encode_modalities(x_dict)
        if len(mu_dict) == 0:
            raise ValueError("At least one modality must be present in x_dict.")

        present = list(mu_dict.keys())
        K = len(present)

        # for reporting only (fused latent)
        mu_moe, logvar_moe = self.mixture_of_experts(mu_dict, logvar_dict)
        z_moe = self._reparameterize(mu_moe, logvar_moe)

        # modality-specific latents
        z_mod = {m: self._reparameterize(mu_dict[m], logvar_dict[m]) for m in present}

        def recon_from_z_to_target(z_src: torch.Tensor, target_mod: str) -> torch.Tensor:
            raw = self.decoders[target_mod](z_src)
            m_cfg = self.mod_cfg_by_name[target_mod]
            return self._recon_loss(
                x=x_dict[target_mod],
                raw_dec_out=raw,
                likelihood=m_cfg.likelihood,
                mod_name=target_mod,
            )

        device = next(iter(mu_dict.values())).device
        B = next(iter(mu_dict.values())).shape[0]

        recon_per_target: Dict[str, torch.Tensor] = {m: torch.zeros(B, device=device) for m in present}
        recon_counts: Dict[str, int] = {m: 0 for m in present}

        recon_total: Optional[torch.Tensor] = None
        n_terms = 0

        def add_term(t: torch.Tensor, tgt: str):
            nonlocal recon_total, n_terms
            recon_per_target[tgt] = recon_per_target[tgt] + t
            recon_counts[tgt] += 1
            recon_total = t if recon_total is None else (recon_total + t)
            n_terms += 1

        v1_recon = self.v1_recon

        # aliases for "equal parts self + cross"
        if v1_recon in ("avg", "both", "paper", "self+cross", "self_cross", "hybrid"):
            if K <= 1:
                tgt = present[0]
                add_term(recon_from_z_to_target(z_mod[tgt], tgt), tgt=tgt)
            else:
                w_self = 0.5 / float(K)
                w_cross = 0.5 / float(K * (K - 1))

                # self terms
                for tgt in present:
                    add_term(w_self * recon_from_z_to_target(z_mod[tgt], tgt), tgt=tgt)

                # ordered cross terms
                for src in present:
                    for tgt in present:
                        if src == tgt:
                            continue
                        add_term(w_cross * recon_from_z_to_target(z_mod[src], tgt), tgt=tgt)

        elif v1_recon.startswith("src:"):
            src_name = v1_recon.split("src:", 1)[1].strip()
            if src_name not in z_mod:
                raise ValueError(f"v1_recon={self.v1_recon!r} but '{src_name}' not present. Present={present}")
            for tgt in present:
                add_term(recon_from_z_to_target(z_mod[src_name], tgt), tgt=tgt)

        elif v1_recon == "self":
            for tgt in present:
                add_term(recon_from_z_to_target(z_mod[tgt], tgt), tgt=tgt)

        elif v1_recon in ("avg_z", "mean_z", "average_z"):
            z_avg = torch.stack([z_mod[m] for m in present], dim=0).mean(dim=0)
            for tgt in present:
                add_term(recon_from_z_to_target(z_avg, tgt), tgt=tgt)
            z_moe = z_avg  # for reporting

        elif v1_recon in ("moe", "poe", "fused"):
            for tgt in present:
                add_term(recon_from_z_to_target(z_moe, tgt), tgt=tgt)

        else:
            # DEFAULT: FULL CROSS for any K>=2 (ordered pairs src!=tgt)
            if K >= 2:
                for src in present:
                    for tgt in present:
                        if src == tgt:
                            continue
                        add_term(recon_from_z_to_target(z_mod[src], tgt), tgt=tgt)
            else:
                tgt = present[0]
                add_term(recon_from_z_to_target(z_mod[tgt], tgt), tgt=tgt)

            # optional extra "mix" term from average latent (small stabilizer)
            if self.v1_recon_mix > 0.0 and K >= 2:
                mix = float(self.v1_recon_mix)
                z_avg = torch.stack([z_mod[m] for m in present], dim=0).mean(dim=0)
                for tgt in present:
                    add_term(mix * recon_from_z_to_target(z_avg, tgt), tgt=tgt)

        if recon_total is None or n_terms == 0:
            raise RuntimeError("v1 reconstruction produced no loss terms.")

        # normalize recon_total by number of terms ONLY for the unweighted modes.
        # For the "avg/both/paper" weighted mode, recon_total is already scale-stable.
        weighted_mode = v1_recon in ("avg", "both", "paper", "self+cross", "self_cross", "hybrid")
        if self.normalize_v1_terms and (not weighted_mode):
            recon_total = recon_total / float(n_terms)

        # recon_per_target: average *per target* over its own term count (more interpretable)
        recon_per_target_mean: Dict[str, torch.Tensor] = {}
        for tgt in present:
            ct = max(int(recon_counts[tgt]), 1)
            recon_per_target_mean[tgt] = recon_per_target[tgt] / float(ct)

        # KL(q_m || p)
        mu_p = self.prior_mu.expand_as(mu_dict[present[0]])
        logvar_p = self.prior_logvar.expand_as(logvar_dict[present[0]])
        kl_terms = [self._kl_gaussian(mu_dict[m], logvar_dict[m], mu_p, logvar_p) for m in present]
        kl = torch.stack(kl_terms, dim=0).sum(dim=0)
        if self.normalize_v1_terms:
            kl = kl / float(max(K, 1))

        # cross KL(q_i || q_j) over ordered pairs i!=j
        if K < 2:
            cross_kl = torch.zeros_like(kl)
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

        # optional label head (decoder-only)
        class_loss = None
        class_logits = None
        if self.label_decoder is not None and y is not None:
            y = y.long()
            dec_out = self.label_decoder(z_moe)
            class_logits = dec_out["logits"] if isinstance(dec_out, dict) and "logits" in dec_out else dec_out
            class_loss = F.cross_entropy(class_logits, y, reduction="none")
            loss = loss + self.label_loss_weight * class_loss

        # for convenience, decode from fused z for "xhat" visualization
        xhat_dict = self.decode_modalities(z_moe)

        out: Dict[str, Any] = {
            "loss": loss.mean(),
            "recon_total": recon_total.mean(),
            "kl": kl.mean(),
            "align": cross_kl.mean(),          # keep legacy key name
            "cross_kl": cross_kl.mean(),
            "mu_z": mu_moe,
            "logvar_z": logvar_moe,
            "z": z_moe,
            "xhat": xhat_dict,
            "mu_dict": mu_dict,
            "logvar_dict": logvar_dict,
            "recon_per_modality": {k: v.mean() for k, v in recon_per_target_mean.items()},
            "beta": torch.tensor(beta, device=loss.device),
            "gamma": torch.tensor(gamma, device=loss.device),
            "v1_recon_terms": torch.tensor(float(n_terms), device=loss.device),
        }
        if class_loss is not None:
            out["class_loss"] = class_loss.mean()
        if class_logits is not None:
            out["class_logits"] = class_logits
        return out

    def _anneal_weight(self, epoch: int, start: int, end: int, max_val: float) -> float:
        start = int(start)
        end = int(end)
        if end <= start:
            return float(max_val)
        if epoch <= start:
            return 0.0
        if epoch >= end:
            return float(max_val)
        frac = (epoch - start) / float(end - start)
        return float(max_val) * float(frac)

