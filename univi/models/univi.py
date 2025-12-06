# univi/models/univi.py

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..config import UniVIConfig, ModalityConfig
from .mlp import build_mlp
from .decoders import DecoderConfig, build_decoder


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

    Optional supervised shaping:
      - label_decoder: p(y|z) categorical head (always available if n_label_classes>0)
      - label_encoder: q(z|y) label expert (optional) that can be injected into v2 MoE fusion
        (masked for unlabeled; ignored via huge logvar).
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
        # ---- label head (decoder-side): p(y|z) ----
        n_label_classes: int = 0,
        label_loss_weight: float = 1.0,
        # ---- label expert (encoder-side): q(z|y) injected into v2 fusion ----
        use_label_encoder: bool = False,
        label_moe_weight: float = 1.0,
        unlabeled_logvar: float = 20.0,
        label_encoder_warmup: int = 0,
        label_ignore_index: int = -1,
        classify_from_mu: bool = True,
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

        for m in cfg.modalities:
            if not isinstance(m, ModalityConfig):
                raise TypeError(f"cfg.modalities must contain ModalityConfig, got {type(m)}")

            self.encoders[m.name] = build_mlp(
                in_dim=m.input_dim,
                hidden_dims=m.encoder_hidden,
                out_dim=self.latent_dim * 2,
                activation=nn.ReLU(),
                dropout=float(cfg.encoder_dropout),
                batchnorm=bool(cfg.encoder_batchnorm),
            )
            self.encoder_heads[m.name] = nn.Identity()

            # Decoder via registry
            dec_hidden = list(m.decoder_hidden) if m.decoder_hidden else [max(64, self.latent_dim)]
            dec_cfg = DecoderConfig(
                output_dim=int(m.input_dim),
                hidden_dims=dec_hidden,
                dropout=float(cfg.decoder_dropout),
                batchnorm=bool(cfg.decoder_batchnorm),
            )

            lk = (m.likelihood or "gaussian").lower().strip()

            # Only pass NB/ZINB kwargs to the NB/ZINB decoders (others won't accept them)
            decoder_kwargs: Dict[str, Any] = {}
            if lk in ("nb", "negative_binomial", "zinb", "zero_inflated_negative_binomial"):
                dispersion = getattr(m, "dispersion", "gene")
                init_log_theta = float(getattr(m, "init_log_theta", 0.0))
                decoder_kwargs = dict(
                    dispersion=dispersion,
                    init_log_theta=init_log_theta,
                    eps=self.EPS,
                )

            self.decoders[m.name] = build_decoder(
                lk,
                cfg=dec_cfg,
                latent_dim=self.latent_dim,
                **decoder_kwargs,
            )

        # Shared prior N(0, I)
        self.register_buffer("prior_mu", torch.zeros(self.latent_dim))
        self.register_buffer("prior_logvar", torch.zeros(self.latent_dim))

        # ---- label decoder head: p(y|z) ----
        self.n_label_classes = int(n_label_classes) if n_label_classes is not None else 0
        self.label_loss_weight = float(label_loss_weight)
        self.label_ignore_index = int(label_ignore_index)
        self.classify_from_mu = bool(classify_from_mu)

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

        # ---- label encoder expert: q(z|y) ----
        self.use_label_encoder = bool(use_label_encoder)
        self.label_moe_weight = float(label_moe_weight)
        self.unlabeled_logvar = float(unlabeled_logvar)
        self.label_encoder_warmup = int(label_encoder_warmup)

        if self.n_label_classes > 0 and self.use_label_encoder:
            self.label_encoder = build_mlp(
                in_dim=self.n_label_classes,
                hidden_dims=[max(64, self.latent_dim)],
                out_dim=self.latent_dim * 2,
                activation=nn.ReLU(),
                dropout=float(cfg.encoder_dropout),
                batchnorm=bool(cfg.encoder_batchnorm),
            )
        else:
            self.label_encoder = None

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

    # -------------------- decoder-output unwrapping --------------------

    def _unwrap_decoder_out(self, dec_out: Any) -> Any:
        """
        Normalize decoder outputs so recon loss can handle common conventions.

        Accepts:
          - Tensor: returned as-is
          - dict: returned as-is (but lightly normalized for common aliases)
          - (dict,) / [dict] or (tensor,) / [tensor]: unwrap singleton containers
        """
        if isinstance(dec_out, (tuple, list)):
            if len(dec_out) != 1:
                raise TypeError(
                    f"Unsupported decoder output container of length {len(dec_out)}: {type(dec_out)!r}"
                )
            dec_out = dec_out[0]

        if torch.is_tensor(dec_out):
            return dec_out

        if isinstance(dec_out, dict):
            out = dict(dec_out)
            # gentle alias support
            if "mu" not in out and "mean" in out:
                out["mu"] = out["mean"]
            return out

        raise TypeError(f"Unsupported decoder output type: {type(dec_out)!r}")

    # -------------------- categorical modality helpers --------------------

    @staticmethod
    def _is_categorical_likelihood(likelihood: Optional[str]) -> bool:
        lk = (likelihood or "").lower()
        return lk in ("categorical", "cat", "ce", "cross_entropy")

    def _categorical_targets_and_mask(
        self,
        x: torch.Tensor,
        *,
        n_classes: int,
        ignore_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2 and x.size(1) == 1:
            x = x[:, 0]

        if x.dim() == 2:
            mask = x.sum(dim=-1) > 0.5
            y = x.argmax(dim=-1).long()
            if mask.any() and ((y[mask] < 0) | (y[mask] >= n_classes)).any():
                bad = y[mask][((y[mask] < 0) | (y[mask] >= n_classes))][:10].detach().cpu().tolist()
                raise ValueError(f"Categorical targets out of range [0,{n_classes-1}]: e.g. {bad}")
            return y, mask

        y = x.view(-1)
        if y.dtype.is_floating_point:
            y = y.round()
        y = y.long()

        invalid = (y != int(ignore_index)) & ((y < 0) | (y >= n_classes))
        if invalid.any():
            bad = y[invalid][:10].detach().cpu().tolist()
            raise ValueError(f"Categorical targets out of range [0,{n_classes-1}] (non-ignore): e.g. {bad}")

        mask = (y != int(ignore_index))
        return y, mask

    def _encode_categorical_input_if_needed(self, mod_name: str, x: torch.Tensor) -> torch.Tensor:
        m_cfg = self.mod_cfg_by_name[mod_name]
        if not self._is_categorical_likelihood(m_cfg.likelihood):
            return x

        C = int(m_cfg.input_dim)
        ignore_index = int(getattr(m_cfg, "ignore_index", self.label_ignore_index))

        if x.dim() == 2 and x.size(1) == 1:
            x = x[:, 0]

        if x.dim() == 2:
            return x.float()

        y, mask = self._categorical_targets_and_mask(x, n_classes=C, ignore_index=ignore_index)
        B = y.shape[0]
        x_oh = torch.zeros((B, C), device=y.device, dtype=torch.float32)
        if mask.any():
            x_oh[mask] = F.one_hot(y[mask], num_classes=C).float()
        return x_oh

    # -------------------- reconstruction loss --------------------

    def _recon_loss(self, x: torch.Tensor, raw_dec_out: Any, likelihood: str, mod_name: str) -> torch.Tensor:
        likelihood = (likelihood or "gaussian").lower().strip()
        dec_out = self._unwrap_decoder_out(raw_dec_out)

        # categorical
        if self._is_categorical_likelihood(likelihood):
            m_cfg = self.mod_cfg_by_name[mod_name]
            C = int(m_cfg.input_dim)
            ignore_index = int(getattr(m_cfg, "ignore_index", self.label_ignore_index))

            logits = dec_out["logits"] if isinstance(dec_out, dict) else dec_out
            y, mask = self._categorical_targets_and_mask(x, n_classes=C, ignore_index=ignore_index)

            nll = torch.zeros(y.shape[0], device=logits.device)
            if mask.any():
                nll[mask] = F.cross_entropy(logits[mask], y[mask], reduction="none")
            return nll

        # gaussian / gaussian_diag / mse
        if likelihood in ("gaussian", "normal", "mse", "gaussian_diag"):
            if isinstance(dec_out, dict) and ("mean" in dec_out) and ("logvar" in dec_out):
                mean = dec_out["mean"]
                logvar = dec_out["logvar"].clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)
                var = torch.exp(logvar)
                nll = 0.5 * (logvar + (x - mean) ** 2 / (var + self.EPS))
                return nll.sum(dim=-1)

            pred = dec_out["mean"] if isinstance(dec_out, dict) and ("mean" in dec_out) else dec_out
            if likelihood == "mse":
                return ((x - pred) ** 2).mean(dim=-1)
            return ((x - pred) ** 2).sum(dim=-1)

        # bernoulli
        if likelihood == "bernoulli":
            logits = dec_out["logits"] if isinstance(dec_out, dict) else dec_out
            nll = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
            return nll.sum(dim=-1)

        # poisson
        if likelihood == "poisson":
            log_rate = dec_out["log_rate"] if isinstance(dec_out, dict) and "log_rate" in dec_out else dec_out
            nll = F.poisson_nll_loss(log_rate, x, log_input=True, full=False, reduction="none")
            return nll.sum(dim=-1)

        # negative binomial
        if likelihood in ("nb", "negative_binomial"):
            if not isinstance(dec_out, dict) or ("mu" not in dec_out) or ("log_theta" not in dec_out):
                raise ValueError(f"NB recon expects dict with keys ('mu','log_theta'); got {type(dec_out)}")
            mu = dec_out["mu"]
            theta = torch.exp(dec_out["log_theta"])
            if theta.dim() == 1:
                theta = theta.unsqueeze(0).expand_as(mu)
            return self._nb_nll(x, mu, theta, eps=self.EPS).sum(dim=-1)

        # ZINB
        if likelihood in ("zinb", "zero_inflated_negative_binomial"):
            if not isinstance(dec_out, dict) or ("mu" not in dec_out) or ("log_theta" not in dec_out) or ("logit_pi" not in dec_out):
                raise ValueError(f"ZINB recon expects dict with keys ('mu','log_theta','logit_pi'); got {type(dec_out)}")
            mu = dec_out["mu"]
            theta = torch.exp(dec_out["log_theta"])
            logit_pi = dec_out["logit_pi"]

            if theta.dim() == 1:
                theta = theta.unsqueeze(0).expand_as(mu)
            if logit_pi.dim() == 1:
                logit_pi = logit_pi.unsqueeze(0).expand_as(mu)

            return self._zinb_nll(x, mu, theta, logit_pi, eps=self.EPS).sum(dim=-1)

        # fallback
        pred = dec_out["mean"] if isinstance(dec_out, dict) and ("mean" in dec_out) else dec_out
        return ((x - pred) ** 2).sum(dim=-1)

    # -------------------- alignment loss --------------------

    def _alignment_loss_l2mu(self, mu_per_mod: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(mu_per_mod.keys())
        if len(names) < 2:
            if len(names) == 0:
                return torch.tensor(0.0, device=next(self.parameters()).device).expand(1)
            k = names[0]
            return torch.zeros(mu_per_mod[k].size(0), device=mu_per_mod[k].device)

        losses = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                losses.append(((mu_per_mod[names[i]] - mu_per_mod[names[j]]) ** 2).sum(dim=-1))
        return torch.stack(losses, dim=0).mean(dim=0)

    # -------------------- label expert --------------------

    def _encode_labels_as_expert(
        self,
        y: torch.Tensor,
        B: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_y = torch.zeros(B, self.latent_dim, device=device)
        logvar_y = torch.full((B, self.latent_dim), float(self.unlabeled_logvar), device=device)

        if self.label_encoder is None or y is None:
            return mu_y, logvar_y

        y = y.long()
        mask = (y >= 0) & (y != self.label_ignore_index)
        if not mask.any():
            return mu_y, logvar_y

        y_oh = F.one_hot(y[mask], num_classes=self.n_label_classes).float()
        h = self.label_encoder(y_oh)
        mu_l, logvar_l = self._split_mu_logvar(h)

        w = float(self.label_moe_weight)
        if w != 1.0:
            logvar_l = logvar_l - math.log(max(w, 1e-8))

        mu_y[mask] = mu_l
        logvar_y[mask] = logvar_l
        return mu_y, logvar_y

    # -------------------------- encode/decode/fuse --------------------------

    def encode_modalities(self, x_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mu_dict: Dict[str, torch.Tensor] = {}
        logvar_dict: Dict[str, torch.Tensor] = {}

        for m in self.modality_names:
            if m not in x_dict or x_dict[m] is None:
                continue

            x_in = self._encode_categorical_input_if_needed(m, x_dict[m])

            h = self.encoder_heads[m](self.encoders[m](x_in))
            mu, logvar = self._split_mu_logvar(h)
            mu_dict[m] = mu
            logvar_dict[m] = logvar

        return mu_dict, logvar_dict

    def mixture_of_experts(self, mu_dict: Dict[str, torch.Tensor], logvar_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        mus = list(mu_dict.values())
        logvars = list(logvar_dict.values())
        precisions = [torch.exp(-lv) for lv in logvars]
        precision_sum = torch.stack(precisions, dim=0).sum(dim=0) + 1e-8
        mu_weighted = torch.stack([m * p for m, p in zip(mus, precisions)], dim=0).sum(dim=0)
        mu_comb = mu_weighted / precision_sum
        var_comb = 1.0 / precision_sum
        logvar_comb = torch.log(var_comb + 1e-8)
        return mu_comb, logvar_comb

    def decode_modalities(self, z: torch.Tensor) -> Dict[str, Any]:
        return {m: self.decoders[m](z) for m in self.modality_names}

    # ------------------------------ forward ------------------------------

    def forward(self, x_dict: Dict[str, torch.Tensor], epoch: int = 0, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
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

        # Inject label expert into fusion (optional)
        if self.label_encoder is not None and y is not None and epoch >= self.label_encoder_warmup:
            B = next(iter(mu_dict.values())).shape[0]
            device = next(iter(mu_dict.values())).device
            mu_y, logvar_y = self._encode_labels_as_expert(y=y, B=B, device=device)
            mu_dict["__label__"] = mu_y
            logvar_dict["__label__"] = logvar_y

        mu_z, logvar_z = self.mixture_of_experts(mu_dict, logvar_dict)
        z = self._reparameterize(mu_z, logvar_z)
        xhat_dict = self.decode_modalities(z)

        recon_total: Optional[torch.Tensor] = None
        recon_losses: Dict[str, torch.Tensor] = {}

        for name, m_cfg in self.mod_cfg_by_name.items():
            if name not in x_dict or x_dict[name] is None:
                continue
            loss_m = self._recon_loss(
                x=x_dict[name],
                raw_dec_out=xhat_dict[name],
                likelihood=m_cfg.likelihood,
                mod_name=name,
            )
            recon_losses[name] = loss_m
            recon_total = loss_m if recon_total is None else (recon_total + loss_m)

        if recon_total is None:
            raise RuntimeError("No modalities in x_dict produced recon loss.")

        mu_p = self.prior_mu.expand_as(mu_z)
        logvar_p = self.prior_logvar.expand_as(logvar_z)
        kl = self._kl_gaussian(mu_z, logvar_z, mu_p, logvar_p)

        align_loss = self._alignment_loss_l2mu({k: v for k, v in mu_dict.items() if not k.startswith("__")})

        beta = self._anneal_weight(epoch, self.cfg.kl_anneal_start, self.cfg.kl_anneal_end, self.beta_max)
        gamma = self._anneal_weight(epoch, self.cfg.align_anneal_start, self.cfg.align_anneal_end, self.gamma_max)

        loss = recon_total + beta * kl + gamma * align_loss

        # label decoder head (optional)
        class_loss = None
        class_logits = None
        if self.label_decoder is not None:
            z_for_cls = mu_z if self.classify_from_mu else z
            dec_out = self.label_decoder(z_for_cls)
            class_logits = dec_out["logits"] if isinstance(dec_out, dict) and "logits" in dec_out else dec_out

            if y is not None:
                B = loss.shape[0]
                y = y.long()
                mask = (y >= 0) & (y != self.label_ignore_index)
                per_cell = torch.zeros(B, device=loss.device)
                if mask.any():
                    per_cell[mask] = F.cross_entropy(class_logits[mask], y[mask], reduction="none")
                class_loss = per_cell
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
        mu_dict, logvar_dict = self.encode_modalities(x_dict)
        if len(mu_dict) == 0:
            raise ValueError("At least one modality must be present in x_dict.")

        present = list(mu_dict.keys())
        K = len(present)

        mu_moe, logvar_moe = self.mixture_of_experts(mu_dict, logvar_dict)
        z_moe = self._reparameterize(mu_moe, logvar_moe)

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

        if v1_recon in ("avg", "both", "paper", "self+cross", "self_cross", "hybrid"):
            if K <= 1:
                tgt = present[0]
                add_term(recon_from_z_to_target(z_mod[tgt], tgt), tgt=tgt)
            else:
                w_self = 0.5 / float(K)
                w_cross = 0.5 / float(K * (K - 1))

                for tgt in present:
                    add_term(w_self * recon_from_z_to_target(z_mod[tgt], tgt), tgt=tgt)

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
            z_moe = z_avg

        elif v1_recon in ("moe", "poe", "fused"):
            for tgt in present:
                add_term(recon_from_z_to_target(z_moe, tgt), tgt=tgt)

        else:
            if K >= 2:
                for src in present:
                    for tgt in present:
                        if src == tgt:
                            continue
                        add_term(recon_from_z_to_target(z_mod[src], tgt), tgt=tgt)
            else:
                tgt = present[0]
                add_term(recon_from_z_to_target(z_mod[tgt], tgt), tgt=tgt)

            if self.v1_recon_mix > 0.0 and K >= 2:
                mix = float(self.v1_recon_mix)
                z_avg = torch.stack([z_mod[m] for m in present], dim=0).mean(dim=0)
                for tgt in present:
                    add_term(mix * recon_from_z_to_target(z_avg, tgt), tgt=tgt)

        if recon_total is None or n_terms == 0:
            raise RuntimeError("v1 reconstruction produced no loss terms.")

        weighted_mode = v1_recon in ("avg", "both", "paper", "self+cross", "self_cross", "hybrid")
        if self.normalize_v1_terms and (not weighted_mode):
            recon_total = recon_total / float(n_terms)

        recon_per_target_mean: Dict[str, torch.Tensor] = {}
        for tgt in present:
            ct = max(int(recon_counts[tgt]), 1)
            recon_per_target_mean[tgt] = recon_per_target[tgt] / float(ct)

        mu_p = self.prior_mu.expand_as(mu_dict[present[0]])
        logvar_p = self.prior_logvar.expand_as(logvar_dict[present[0]])
        kl_terms = [self._kl_gaussian(mu_dict[m], logvar_dict[m], mu_p, logvar_p) for m in present]
        kl = torch.stack(kl_terms, dim=0).sum(dim=0)
        if self.normalize_v1_terms:
            kl = kl / float(max(K, 1))

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

        class_loss = None
        class_logits = None
        if self.label_decoder is not None:
            z_for_cls = mu_moe if self.classify_from_mu else z_moe
            dec_out = self.label_decoder(z_for_cls)
            class_logits = dec_out["logits"] if isinstance(dec_out, dict) and "logits" in dec_out else dec_out

            if y is not None:
                y = y.long()
                mask = (y >= 0) & (y != self.label_ignore_index)
                per_cell = torch.zeros(B, device=loss.device)
                if mask.any():
                    per_cell[mask] = F.cross_entropy(class_logits[mask], y[mask], reduction="none")
                class_loss = per_cell
                loss = loss + self.label_loss_weight * class_loss

        xhat_dict = self.decode_modalities(z_moe)

        out: Dict[str, Any] = {
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

    # --------------------------- convenience API ---------------------------

    def _infer_batch_size(self, x_dict: Dict[str, torch.Tensor]) -> int:
        for v in x_dict.values():
            if v is not None:
                return int(v.shape[0])
        raise ValueError("x_dict has no non-None tensors; cannot infer batch size.")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_fused(
        self,
        x_dict: Dict[str, torch.Tensor],
        *,
        epoch: int = 0,
        y: Optional[torch.Tensor] = None,
        use_mean: bool = True,
        inject_label_expert: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_dict, logvar_dict = self.encode_modalities(x_dict)
        if len(mu_dict) == 0:
            raise ValueError("At least one modality must be present in x_dict.")

        if (
            inject_label_expert
            and (self.label_encoder is not None)
            and (y is not None)
            and (epoch >= self.label_encoder_warmup)
        ):
            B = next(iter(mu_dict.values())).shape[0]
            dev = next(iter(mu_dict.values())).device
            mu_y, logvar_y = self._encode_labels_as_expert(y=y, B=B, device=dev)
            mu_dict["__label__"] = mu_y
            logvar_dict["__label__"] = logvar_y

        mu_z, logvar_z = self.mixture_of_experts(mu_dict, logvar_dict)
        z = mu_z if use_mean else self._reparameterize(mu_z, logvar_z)
        return mu_z, logvar_z, z

    @torch.no_grad()
    def encode_per_modality(self, x_dict: Dict[str, torch.Tensor], *, use_mean: bool = True) -> Dict[str, torch.Tensor]:
        mu_dict, logvar_dict = self.encode_modalities(x_dict)
        z_mod: Dict[str, torch.Tensor] = {}
        for m, mu in mu_dict.items():
            z_mod[m] = mu if use_mean else self._reparameterize(mu, logvar_dict[m])
        return z_mod

    @torch.no_grad()
    def reconstruct_from_z(self, z: torch.Tensor) -> Dict[str, Any]:
        return self.decode_modalities(z)

    @torch.no_grad()
    def reconstruct(
        self,
        x_dict: Dict[str, torch.Tensor],
        *,
        epoch: int = 0,
        y: Optional[torch.Tensor] = None,
        use_mean: bool = True,
        inject_label_expert: bool = True,
    ) -> Dict[str, Any]:
        mu_z, logvar_z, z = self.encode_fused(
            x_dict,
            epoch=epoch,
            y=y,
            use_mean=use_mean,
            inject_label_expert=inject_label_expert,
        )
        xhat = self.decode_modalities(z)
        return {"mu_z": mu_z, "logvar_z": logvar_z, "z": z, "xhat": xhat}

    @torch.no_grad()
    def predict_labels(
        self,
        x_dict: Dict[str, torch.Tensor],
        *,
        epoch: int = 0,
        y: Optional[torch.Tensor] = None,
        use_mean: bool = True,
        inject_label_expert: bool = True,
        return_probs: bool = True,
    ) -> Optional[torch.Tensor]:
        if self.label_decoder is None:
            return None

        mu_z, logvar_z, z = self.encode_fused(
            x_dict,
            epoch=epoch,
            y=y,
            use_mean=use_mean,
            inject_label_expert=inject_label_expert,
        )
        z_for_cls = mu_z if self.classify_from_mu else z
        dec_out = self.label_decoder(z_for_cls)
        logits = dec_out["logits"] if isinstance(dec_out, dict) and "logits" in dec_out else dec_out
        return logits if not return_probs else F.softmax(logits, dim=-1)

    @torch.no_grad()
    def sample_prior(self, n: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
        dev = device if device is not None else self.device
        return torch.randn(int(n), self.latent_dim, device=dev)

    @torch.no_grad()
    def generate(
        self,
        n: int,
        *,
        device: Optional[torch.device] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if z is None:
            z = self.sample_prior(n, device=device)
        xhat = self.decode_modalities(z)
        return {"z": z, "xhat": xhat}

    def freeze_modality(self, modality: str, freeze: bool = True) -> None:
        if modality not in self.encoders or modality not in self.decoders:
            raise ValueError(f"Unknown modality {modality!r}. Known={list(self.encoders.keys())}")

        for p in self.encoders[modality].parameters():
            p.requires_grad = not freeze
        for p in self.encoder_heads[modality].parameters():
            p.requires_grad = not freeze
        for p in self.decoders[modality].parameters():
            p.requires_grad = not freeze

