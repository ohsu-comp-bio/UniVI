# univi/models/univi.py
from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List, Union, Mapping, Callable
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..config import UniVIConfig, ModalityConfig, ClassHeadConfig
from .mlp import build_mlp
from .decoders import DecoderConfig, build_decoder
from .encoders import build_gaussian_encoder, build_multimodal_transformer_encoder


YType = Union[torch.Tensor, Dict[str, torch.Tensor]]


class _GradReverseFn(torch.autograd.Function):
    """Gradient reversal layer: forward identity, backward -lambda * grad."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class UniVIMultiModalVAE(nn.Module):
    """
    Multi-modal β-VAE with per-modality encoders and decoders.

    Defaults (backwards-compatible)
    -------------------------------
    - Per-modality encoders: MLP (or transformer if set on that modality)
    - Fused posterior: MoE/PoE-style precision fusion over per-modality posteriors

    Optional fused multimodal transformer posterior
    ----------------------------------------------
    If cfg.fused_encoder_type == "multimodal_transformer", build a fused encoder that:
      x_dict -> concat(tokens_rna, tokens_adt, tokens_atac, ...) -> transformer -> (mu_fused, logvar_fused)

    In that mode:
      - v2/lite can use mu_fused/logvar_fused for z
      - alignment term becomes mean_i ||mu_i - mu_fused||^2 (instead of pairwise)
      - if required fused modalities are missing (cfg.fused_require_all_modalities=True), fall back to MoE fusion

    Optional attention bias (safe no-op)
    ------------------------------------
    You can pass `attn_bias_cfg` into forward/encode_fused/predict_heads:

      attn_bias_cfg = {
        "atac": {"type": "distance", "lengthscale_bp": 50000, "same_chrom_only": True},
      }

    - Per-modality transformer encoders: will try to build a distance attention bias if the tokenizer
      supports topk indices + coords and exposes build_distance_attn_bias().
    - Fused multimodal transformer encoder: will fill within-modality blocks (e.g. ATAC slice) and
      keep cross-modality blocks neutral (0).

    Recon scaling (IMPORTANT)
    ------------------------
    Recon losses in _recon_loss are typically sums over features (dim=-1). In multi-modal settings
    with very different feature dims (RNA >> ADT/ATAC-LSI), RNA can dominate gradients.

    This class supports:
      - recon_normalize_by_dim: divide per-cell recon by D**recon_dim_power
      - per-modality weights via ModalityConfig.recon_weight (default 1.0)
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

        # ---- recon scaling controls ----
        recon_normalize_by_dim: bool = True,
        recon_dim_power: float = 1.0,  # 1.0 => divide by D; 0.5 => divide by sqrt(D)

        # ---- legacy single label head (kept) ----
        n_label_classes: int = 0,
        label_loss_weight: float = 1.0,
        use_label_encoder: bool = False,
        label_moe_weight: float = 1.0,
        unlabeled_logvar: float = 20.0,
        label_encoder_warmup: int = 0,
        label_ignore_index: int = -1,
        classify_from_mu: bool = True,
        label_head_name: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.cfg.validate()

        self.loss_mode = str(loss_mode).lower().strip()
        self.v1_recon = str(v1_recon).lower().strip()
        self.v1_recon_mix = float(v1_recon_mix)
        self.normalize_v1_terms = bool(normalize_v1_terms)

        # recon scaling
        self.recon_normalize_by_dim = bool(recon_normalize_by_dim)
        self.recon_dim_power = float(recon_dim_power)

        self.latent_dim = int(cfg.latent_dim)
        self.beta_max = float(cfg.beta)
        self.gamma_max = float(cfg.gamma)

        self.modality_names: List[str] = [m.name for m in cfg.modalities]
        self.mod_cfg_by_name: Dict[str, ModalityConfig] = {m.name: m for m in cfg.modalities}

        # ------------------------------------------------------------
        # Per-modality modules
        # ------------------------------------------------------------
        self.encoders = nn.ModuleDict()
        self.encoder_heads = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        for m in cfg.modalities:
            if not isinstance(m, ModalityConfig):
                raise TypeError(f"cfg.modalities must contain ModalityConfig, got {type(m)}")

            self.encoders[m.name] = build_gaussian_encoder(uni_cfg=cfg, mod_cfg=m)
            self.encoder_heads[m.name] = nn.Identity()

            dec_hidden = list(m.decoder_hidden) if m.decoder_hidden else [max(64, self.latent_dim)]
            dec_cfg = DecoderConfig(
                output_dim=int(m.input_dim),
                hidden_dims=dec_hidden,
                dropout=float(cfg.decoder_dropout),
                batchnorm=bool(cfg.decoder_batchnorm),
            )

            lk = (m.likelihood or "gaussian").lower().strip()
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

        # ------------------------------------------------------------
        # Optional fused multimodal encoder
        # ------------------------------------------------------------
        self.fused_encoder_type = (getattr(cfg, "fused_encoder_type", "moe") or "moe").lower().strip()
        self.fused_require_all = bool(getattr(cfg, "fused_require_all_modalities", True))
        self.fused_modalities = list(getattr(cfg, "fused_modalities", None) or self.modality_names)

        if self.fused_encoder_type == "multimodal_transformer":
            self.fused_encoder = build_multimodal_transformer_encoder(
                uni_cfg=cfg,
                modalities=cfg.modalities,
                fused_modalities=self.fused_modalities,
            )
        else:
            self.fused_encoder = None

        # ------------------------------------------------------------
        # Legacy single label head (kept)
        # ------------------------------------------------------------
        self.n_label_classes = int(n_label_classes) if n_label_classes is not None else 0
        self.label_loss_weight = float(label_loss_weight)
        self.label_ignore_index = int(label_ignore_index)
        self.classify_from_mu = bool(classify_from_mu)
        self.label_head_name = str(label_head_name or getattr(cfg, "label_head_name", "label"))

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

        self.label_names: Optional[List[str]] = None
        self.label_name_to_id: Optional[Dict[str, int]] = None

        # Legacy label encoder expert (optional)
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

        # Multi-head supervised decoders (incl. adversarial heads)
        self.class_heads = nn.ModuleDict()
        self.class_heads_cfg: Dict[str, Dict[str, Any]] = {}
        self.head_label_names: Dict[str, List[str]] = {}
        self.head_label_name_to_id: Dict[str, Dict[str, int]] = {}

        if cfg.class_heads:
            for h in cfg.class_heads:
                if not isinstance(h, ClassHeadConfig):
                    raise TypeError(f"cfg.class_heads must contain ClassHeadConfig, got {type(h)}")

                name = str(h.name)
                n_classes = int(h.n_classes)

                dec_cfg = DecoderConfig(
                    output_dim=n_classes,
                    hidden_dims=[max(64, self.latent_dim)],
                    dropout=float(cfg.decoder_dropout),
                    batchnorm=bool(cfg.decoder_batchnorm),
                )
                self.class_heads[name] = build_decoder("categorical", cfg=dec_cfg, latent_dim=self.latent_dim)
                self.class_heads_cfg[name] = {
                    "n_classes": n_classes,
                    "loss_weight": float(h.loss_weight),
                    "ignore_index": int(h.ignore_index),
                    "from_mu": bool(h.from_mu),
                    "warmup": int(h.warmup),
                    "adversarial": bool(getattr(h, "adversarial", False)),
                    "adv_lambda": float(getattr(h, "adv_lambda", 1.0)),
                }

    # ----------------------------- label name utilities -----------------------------

    def set_label_names(self, label_names: List[str]) -> None:
        if self.n_label_classes <= 0:
            raise ValueError("n_label_classes=0; cannot set label names.")
        if len(label_names) != int(self.n_label_classes):
            raise ValueError(f"label_names length {len(label_names)} != n_label_classes {self.n_label_classes}")

        self.label_names = [str(x) for x in label_names]

        def norm(s: str) -> str:
            return " ".join(str(s).strip().lower().split())

        m: Dict[str, int] = {}
        for i, name in enumerate(self.label_names):
            m[name] = i
            m[norm(name)] = i
        self.label_name_to_id = m

    def set_head_label_names(self, head: str, label_names: List[str]) -> None:
        head = str(head)
        if head not in self.class_heads_cfg:
            raise KeyError(f"Unknown head {head!r}. Known heads: {list(self.class_heads_cfg)}")

        n = int(self.class_heads_cfg[head]["n_classes"])
        if len(label_names) != n:
            raise ValueError(f"Head {head!r}: label_names length {len(label_names)} != n_classes {n}")

        names = [str(x) for x in label_names]
        self.head_label_names[head] = names

        def norm(s: str) -> str:
            return " ".join(str(s).strip().lower().split())

        m: Dict[str, int] = {}
        for i, name in enumerate(names):
            m[name] = i
            m[norm(name)] = i
        self.head_label_name_to_id[head] = m

    # ----------------------------- helpers -----------------------------

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

    def _alignment_loss_l2mu_pairwise(self, mu_per_mod: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = list(mu_per_mod.keys())
        if len(names) < 2:
            # return per-cell zeros if we can infer B, else scalar 0
            if len(names) == 1:
                k = names[0]
                return torch.zeros(mu_per_mod[k].size(0), device=mu_per_mod[k].device)
            return torch.tensor(0.0, device=next(self.parameters()).device)

        losses = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                losses.append(((mu_per_mod[names[i]] - mu_per_mod[names[j]]) ** 2).sum(dim=-1))
        return torch.stack(losses, dim=0).mean(dim=0)

    def _alignment_loss_to_fused(self, mu_per_mod: Dict[str, torch.Tensor], mu_fused: torch.Tensor) -> torch.Tensor:
        if len(mu_per_mod) == 0:
            return torch.zeros(mu_fused.size(0), device=mu_fused.device)
        losses = [((mu - mu_fused) ** 2).sum(dim=-1) for mu in mu_per_mod.values()]
        return torch.stack(losses, dim=0).mean(dim=0)

    @staticmethod
    def _is_categorical_likelihood(likelihood: Optional[str]) -> bool:
        lk = (likelihood or "").lower().strip()
        return lk in ("categorical", "cat", "ce", "cross_entropy", "multinomial", "softmax")

    # -------------------------- recon scaling --------------------------

    def _recon_scale(self, mod_name: str, x: torch.Tensor, likelihood: Optional[str]) -> float:
        """
        Scalar multiplier applied to per-cell recon loss for this modality.

        - Optional per-modality config: ModalityConfig.recon_weight (default 1.0)
        - Optional dim normalization: divide by D**recon_dim_power
        - Avoid double-normalizing for likelihood == 'mse' when _recon_loss already uses mean(dim=-1)
        """
        m_cfg = self.mod_cfg_by_name[mod_name]
        w = float(getattr(m_cfg, "recon_weight", 1.0))

        if not self.recon_normalize_by_dim:
            return w

        lk = (likelihood or "gaussian").lower().strip()
        if lk == "mse":
            return w

        D = 0
        try:
            D = int(getattr(m_cfg, "input_dim", 0))
        except Exception:
            D = 0
        if D <= 0 and x is not None and x.dim() == 2:
            D = int(x.shape[1])
        if D <= 0:
            D = 1

        denom = float(D) ** float(self.recon_dim_power)
        denom = max(denom, 1.0)
        return w / denom

    # ------------------------------ categorical utilities ------------------------------

    def _categorical_targets_and_mask(
        self,
        x: torch.Tensor,
        *,
        n_classes: int,
        ignore_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2 and x.size(1) == 1:
            x = x[:, 0]

        if x.dim() == 2:
            mask = x.sum(dim=-1) > 0.5
            y = x.argmax(dim=-1).long()
            return y, mask

        y = x.view(-1)
        if y.dtype.is_floating_point:
            y = y.round()
        y = y.long()

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

    # -------------------------- attention-bias helpers (optional, safe no-op) --------------------------

    def _build_distance_bias_for_permod_transformer(
        self,
        enc: nn.Module,
        x_in: torch.Tensor,
        cfg_m: Mapping[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Attempt to build (tokens, key_padding_mask, attn_bias) for a TransformerGaussianEncoder-like module.

        Returns (None, None, None) if:
          - encoder isn't transformer-style,
          - tokenizer doesn't expose topk indices,
          - coords aren't attached,
          - config isn't distance type, etc.
        """
        typ = str(cfg_m.get("type", "")).lower().strip()
        if typ != "distance":
            return None, None, None

        vec2tok = getattr(enc, "vec2tok", None)
        core = getattr(enc, "encoder", None)
        if vec2tok is None or core is None:
            return None, None, None

        if not hasattr(vec2tok, "build_distance_attn_bias"):
            return None, None, None

        try:
            tokens, key_padding_mask, meta = vec2tok(x_in, return_indices=True)
        except Exception:
            return None, None, None

        topk_idx = None if meta is None else meta.get("topk_idx", None)
        if topk_idx is None:
            return None, None, None

        lengthscale_bp = float(cfg_m.get("lengthscale_bp", 50_000.0))
        same_chrom_only = bool(cfg_m.get("same_chrom_only", True))
        include_cls = bool(getattr(vec2tok, "add_cls_token", False))

        try:
            attn_bias = vec2tok.build_distance_attn_bias(
                topk_idx,
                lengthscale_bp=lengthscale_bp,
                same_chrom_only=same_chrom_only,
                include_cls=include_cls,
            )
        except Exception:
            return None, None, None

        return tokens, key_padding_mask, attn_bias

    def _build_fused_attn_bias_fn(
        self,
        attn_bias_cfg: Mapping[str, Any],
        sub_x_dict: Dict[str, torch.Tensor],
    ) -> Optional[Callable[[Dict[str, Any]], Optional[torch.Tensor]]]:
        """
        Build a callable(meta)->(B,T,T) for MultiModalTransformerGaussianEncoder.

        Behavior:
          - fills ONLY within-modality blocks for modalities configured as distance bias
          - keeps cross-modality logits neutral (0)
        """
        fused = self.fused_encoder
        if fused is None:
            return None

        vec2tok_map = getattr(fused, "vec2tok", None)
        if vec2tok_map is None:
            return None

        want_any = any(
            str(v.get("type", "")).lower().strip() == "distance"
            for v in attn_bias_cfg.values()
            if isinstance(v, Mapping)
        )
        if not want_any:
            return None

        def fn(meta: Dict[str, Any]) -> Optional[torch.Tensor]:
            if not meta:
                return None

            any_x = next(iter(sub_x_dict.values()))
            B = int(any_x.shape[0])

            slices = meta.get("slices_with_cls", meta.get("slices", {}))
            if not slices:
                return None

            T = 0
            for _, (a, b) in slices.items():
                T = max(T, int(b))
            if bool(meta.get("has_global_cls", False)):
                T = max(T, 1)

            bias_full = torch.zeros((B, T, T), device=any_x.device, dtype=torch.float32)

            for m, cfg_m in attn_bias_cfg.items():
                if not isinstance(cfg_m, Mapping):
                    continue
                if str(cfg_m.get("type", "")).lower().strip() != "distance":
                    continue
                if m not in slices:
                    continue
                if m not in vec2tok_map:
                    continue

                tok = vec2tok_map[m]
                if not hasattr(tok, "build_distance_attn_bias"):
                    continue

                mmeta = meta.get(m, {}) or {}
                topk_idx = mmeta.get("topk_idx", None)
                if topk_idx is None:
                    continue

                lengthscale_bp = float(cfg_m.get("lengthscale_bp", 50_000.0))
                same_chrom_only = bool(cfg_m.get("same_chrom_only", True))

                try:
                    local = tok.build_distance_attn_bias(
                        topk_idx,
                        lengthscale_bp=lengthscale_bp,
                        same_chrom_only=same_chrom_only,
                        include_cls=False,
                    )
                except Exception:
                    continue

                a, b = slices[m]
                a = int(a)
                b = int(b)
                if (b - a) != int(local.shape[1]):
                    continue

                bias_full[:, a:b, a:b] = local

            return bias_full

        return fn

    # -------------------------- encode/decode/fuse --------------------------

    def encode_modalities(
        self,
        x_dict: Dict[str, torch.Tensor],
        *,
        attn_bias_cfg: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mu_dict: Dict[str, torch.Tensor] = {}
        logvar_dict: Dict[str, torch.Tensor] = {}

        for m in self.modality_names:
            if m not in x_dict or x_dict[m] is None:
                continue

            x_in = self._encode_categorical_input_if_needed(m, x_dict[m])
            enc = self.encoders[m]

            tokens = key_padding_mask = attn_bias = None
            if attn_bias_cfg is not None and isinstance(attn_bias_cfg, Mapping):
                cfg_m = attn_bias_cfg.get(m, None)
                if isinstance(cfg_m, Mapping):
                    tokens, key_padding_mask, attn_bias = self._build_distance_bias_for_permod_transformer(enc, x_in, cfg_m)

            if tokens is not None and getattr(enc, "encoder", None) is not None:
                h = enc.encoder(
                    tokens,
                    key_padding_mask=key_padding_mask,
                    attn_bias=attn_bias,
                    return_attn=False,
                )
                mu, logvar = torch.chunk(h, 2, dim=-1)
            else:
                try:
                    mu, logvar = enc(x_in, attn_bias=attn_bias)
                except TypeError:
                    mu, logvar = enc(x_in)

            mu = self.encoder_heads[m](mu)
            logvar = torch.clamp(logvar, self.LOGVAR_MIN, self.LOGVAR_MAX)

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
        precision_sum = torch.stack(precisions, dim=0).sum(dim=0).clamp_min(self.EPS)

        mu_weighted = torch.stack([m * p for m, p in zip(mus, precisions)], dim=0).sum(dim=0)
        mu_comb = mu_weighted / precision_sum

        var_comb = 1.0 / precision_sum
        logvar_comb = torch.log(var_comb.clamp_min(self.EPS))
        return mu_comb, logvar_comb

    def decode_modalities(self, z: torch.Tensor) -> Dict[str, Any]:
        return {m: self.decoders[m](z) for m in self.modality_names}

    # -------------------- fused posterior --------------------

    def _present_fused_modalities(self, x_dict: Dict[str, torch.Tensor]) -> List[str]:
        return [m for m in self.fused_modalities if (m in x_dict) and (x_dict[m] is not None)]

    def _can_use_fused_encoder(self, x_dict: Dict[str, torch.Tensor]) -> bool:
        if self.fused_encoder is None:
            return False
        if not self.fused_modalities:
            return False

        present = self._present_fused_modalities(x_dict)
        if self.fused_require_all:
            return len(present) == len(self.fused_modalities)
        return len(present) >= 1

    def _compute_fused_posterior(
        self,
        x_dict: Dict[str, torch.Tensor],
        *,
        attn_bias_cfg: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.fused_encoder is not None
        present = self._present_fused_modalities(x_dict)
        sub = {m: x_dict[m] for m in present}

        attn_bias_fn = None
        want_meta = False
        if attn_bias_cfg is not None and isinstance(attn_bias_cfg, Mapping):
            attn_bias_fn = self._build_fused_attn_bias_fn(attn_bias_cfg, sub)
            want_meta = attn_bias_fn is not None

        out = self.fused_encoder(sub, return_token_meta=want_meta, attn_bias_fn=attn_bias_fn)
        if want_meta:
            mu_f, logvar_f, _meta = out  # type: ignore[misc]
        else:
            mu_f, logvar_f = out  # type: ignore[misc]

        logvar_f = torch.clamp(logvar_f, self.LOGVAR_MIN, self.LOGVAR_MAX)
        return mu_f, logvar_f

    # -------------------- label expert --------------------

    def _encode_labels_as_expert(self, y: torch.Tensor, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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
        mu_l, logvar_l = torch.chunk(h, 2, dim=-1)
        logvar_l = torch.clamp(logvar_l, self.LOGVAR_MIN, self.LOGVAR_MAX)

        w = float(self.label_moe_weight)
        if w != 1.0:
            logvar_l = logvar_l - math.log(max(w, 1e-8))

        mu_y[mask] = mu_l
        logvar_y[mask] = logvar_l
        return mu_y, logvar_y

    def _extract_legacy_y(self, y: Optional[YType]) -> Optional[torch.Tensor]:
        if y is None:
            return None
        if isinstance(y, Mapping):
            v = y.get(self.label_head_name, None)
            return v.long() if v is not None else None
        return y.long()

    def _grad_reverse(self, x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return _GradReverseFn.apply(x, float(lambd))

    def _apply_multihead_losses(
        self,
        *,
        mu_z: torch.Tensor,
        z: torch.Tensor,
        y: Optional[YType],
        epoch: int,
        loss: torch.Tensor,
        loss_annealed: torch.Tensor,
        loss_fixed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        head_logits: Dict[str, torch.Tensor] = {}
        head_loss_means: Dict[str, torch.Tensor] = {}

        if y is None or not isinstance(y, Mapping) or len(self.class_heads) == 0:
            return loss, loss_annealed, loss_fixed, head_logits, head_loss_means

        B = int(mu_z.size(0))
        device = mu_z.device

        for name, head in self.class_heads.items():
            cfg_h = self.class_heads_cfg[name]
            if int(epoch) < int(cfg_h["warmup"]):
                continue

            y_h = y.get(name, None)
            if y_h is None:
                continue

            y_h = y_h.long()
            z_in = mu_z if bool(cfg_h["from_mu"]) else z

            if bool(cfg_h.get("adversarial", False)):
                z_in = self._grad_reverse(z_in, cfg_h.get("adv_lambda", 1.0))

            dec_out = head(z_in)
            logits = dec_out["logits"] if isinstance(dec_out, dict) and "logits" in dec_out else dec_out
            head_logits[name] = logits

            ignore_index = int(cfg_h["ignore_index"])
            mask = (y_h != ignore_index) & (y_h >= 0)
            per_cell = torch.zeros(B, device=device)
            if mask.any():
                per_cell[mask] = F.cross_entropy(logits[mask], y_h[mask], reduction="none")

            w = float(cfg_h["loss_weight"])
            if w != 0.0:
                loss = loss + w * per_cell
                loss_annealed = loss_annealed + w * per_cell
                loss_fixed = loss_fixed + w * per_cell

            head_loss_means[name] = per_cell.mean()

        return loss, loss_annealed, loss_fixed, head_logits, head_loss_means

    # ------------------------------ forward dispatcher ------------------------------

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        epoch: int = 0,
        y: Optional[YType] = None,
        attn_bias_cfg: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,  # keep older wrappers from breaking if they pass extra args
    ) -> Dict[str, torch.Tensor]:
        mode = (self.loss_mode or "v2").lower()
        if mode in ("v1", "paper", "cross"):
            return self._forward_v1(x_dict=x_dict, epoch=epoch, y=y, attn_bias_cfg=attn_bias_cfg)
        if mode in ("v2", "lite", "light", "moe", "poe", "fused"):
            return self._forward_v2(x_dict=x_dict, epoch=epoch, y=y, attn_bias_cfg=attn_bias_cfg)
        raise ValueError(f"Unknown loss_mode={self.loss_mode!r}.")

    # ------------------------------ v2 / lite ------------------------------

    def _forward_v2(
        self,
        x_dict: Dict[str, torch.Tensor],
        epoch: int = 0,
        y: Optional[YType] = None,
        attn_bias_cfg: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        mu_dict, logvar_dict = self.encode_modalities(x_dict, attn_bias_cfg=attn_bias_cfg)
        if len(mu_dict) == 0:
            raise ValueError("At least one modality must be present in x_dict.")

        use_fused = (self.fused_encoder_type == "multimodal_transformer") and self._can_use_fused_encoder(x_dict)
        mu_fused: Optional[torch.Tensor] = None
        logvar_fused: Optional[torch.Tensor] = None

        if use_fused:
            mu_fused, logvar_fused = self._compute_fused_posterior(x_dict, attn_bias_cfg=attn_bias_cfg)
            mu_z, logvar_z = mu_fused, logvar_fused
        else:
            mu_z, logvar_z = self.mixture_of_experts(mu_dict, logvar_dict)

        y_legacy = self._extract_legacy_y(y)
        if self.label_encoder is not None and y_legacy is not None and epoch >= self.label_encoder_warmup:
            B = mu_z.shape[0]
            device = mu_z.device
            mu_y, logvar_y = self._encode_labels_as_expert(y=y_legacy, B=B, device=device)

            base_mu_dict = {"__base__": mu_z, "__label__": mu_y}
            base_lv_dict = {"__base__": logvar_z, "__label__": logvar_y}
            mu_z, logvar_z = self.mixture_of_experts(base_mu_dict, base_lv_dict)

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

            # apply recon scaling (dim normalization + optional per-modality recon_weight)
            s = self._recon_scale(name, x_dict[name], m_cfg.likelihood)
            if s != 1.0:
                loss_m = loss_m * float(s)

            recon_losses[name] = loss_m
            recon_total = loss_m if recon_total is None else (recon_total + loss_m)

        if recon_total is None:
            raise RuntimeError("No modalities in x_dict produced recon loss.")

        mu_p = self.prior_mu.expand_as(mu_z)
        logvar_p = self.prior_logvar.expand_as(logvar_z)
        kl = self._kl_gaussian(mu_z, logvar_z, mu_p, logvar_p)

        real_mu = {k: v for k, v in mu_dict.items() if not k.startswith("__")}
        if use_fused and (mu_fused is not None):
            align_loss = self._alignment_loss_to_fused(real_mu, mu_fused)
        else:
            align_loss = self._alignment_loss_l2mu_pairwise(real_mu)

        beta_t = self._anneal_weight(epoch, self.cfg.kl_anneal_start, self.cfg.kl_anneal_end, self.beta_max)
        gamma_t = self._anneal_weight(epoch, self.cfg.align_anneal_start, self.cfg.align_anneal_end, self.gamma_max)

        beta_used = beta_t if self.training else self.beta_max
        gamma_used = gamma_t if self.training else self.gamma_max

        loss_annealed = recon_total + beta_t * kl + gamma_t * align_loss
        loss_fixed = recon_total + self.beta_max * kl + self.gamma_max * align_loss
        loss = recon_total + beta_used * kl + gamma_used * align_loss

        class_loss = None
        class_logits = None
        if self.label_decoder is not None:
            z_for_cls = mu_z if self.classify_from_mu else z
            dec_out = self.label_decoder(z_for_cls)
            class_logits = dec_out["logits"] if isinstance(dec_out, dict) and "logits" in dec_out else dec_out

            if y_legacy is not None:
                B = loss.shape[0]
                yy = y_legacy.long()
                mask = (yy >= 0) & (yy != self.label_ignore_index)
                per_cell = torch.zeros(B, device=loss.device)
                if mask.any():
                    per_cell[mask] = F.cross_entropy(class_logits[mask], yy[mask], reduction="none")
                class_loss = per_cell

                loss = loss + self.label_loss_weight * class_loss
                loss_annealed = loss_annealed + self.label_loss_weight * class_loss
                loss_fixed = loss_fixed + self.label_loss_weight * class_loss

        loss, loss_annealed, loss_fixed, head_logits, head_loss_means = self._apply_multihead_losses(
            mu_z=mu_z,
            z=z,
            y=y,
            epoch=epoch,
            loss=loss,
            loss_annealed=loss_annealed,
            loss_fixed=loss_fixed,
        )

        out: Dict[str, Any] = {
            "loss": loss.mean(),
            "recon_total": recon_total.mean(),
            "kl": kl.mean(),
            "align": align_loss.mean() if torch.is_tensor(align_loss) else torch.tensor(float(align_loss), device=loss.device),
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "z": z,
            "xhat": xhat_dict,
            "mu_dict": mu_dict,
            "logvar_dict": logvar_dict,
            # these are after scaling
            "recon_per_modality": {k: v.mean() for k, v in recon_losses.items()},
            "beta": torch.tensor(beta_t, device=loss.device),
            "gamma": torch.tensor(gamma_t, device=loss.device),
            "beta_used": torch.tensor(beta_used, device=loss.device),
            "gamma_used": torch.tensor(gamma_used, device=loss.device),
            "loss_annealed": loss_annealed.mean(),
            "loss_fixed": loss_fixed.mean(),
            "used_fused_encoder": torch.tensor(1.0 if use_fused else 0.0, device=loss.device),
        }
        if class_loss is not None:
            out["class_loss"] = class_loss.mean()
        if class_logits is not None:
            out["class_logits"] = class_logits
        if head_logits:
            out["head_logits"] = head_logits
        if head_loss_means:
            out["head_losses"] = head_loss_means
        return out

    # ------------------------------ v1 ------------------------------

    def _forward_v1(
        self,
        x_dict: Dict[str, torch.Tensor],
        epoch: int = 0,
        y: Optional[YType] = None,
        attn_bias_cfg: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        mu_dict, logvar_dict = self.encode_modalities(x_dict, attn_bias_cfg=attn_bias_cfg)
        if len(mu_dict) == 0:
            raise ValueError("At least one modality must be present in x_dict.")

        present = list(mu_dict.keys())
        K = len(present)

        use_fused = (self.fused_encoder_type == "multimodal_transformer") and self._can_use_fused_encoder(x_dict)
        if use_fused:
            mu_moe, logvar_moe = self._compute_fused_posterior(x_dict, attn_bias_cfg=attn_bias_cfg)
        else:
            mu_moe, logvar_moe = self.mixture_of_experts(mu_dict, logvar_dict)

        z_moe = self._reparameterize(mu_moe, logvar_moe)
        z_mod = {m: self._reparameterize(mu_dict[m], logvar_dict[m]) for m in present}

        def recon_from_z_to_target(z_src: torch.Tensor, target_mod: str) -> torch.Tensor:
            raw = self.decoders[target_mod](z_src)
            m_cfg = self.mod_cfg_by_name[target_mod]
            loss_t = self._recon_loss(
                x=x_dict[target_mod],
                raw_dec_out=raw,
                likelihood=m_cfg.likelihood,
                mod_name=target_mod,
            )
            s = self._recon_scale(target_mod, x_dict[target_mod], m_cfg.likelihood)
            if s != 1.0:
                loss_t = loss_t * float(s)
            return loss_t

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

        beta_t = self._anneal_weight(epoch, self.cfg.kl_anneal_start, self.cfg.kl_anneal_end, self.beta_max)
        gamma_t = self._anneal_weight(epoch, self.cfg.align_anneal_start, self.cfg.align_anneal_end, self.gamma_max)

        beta_used = beta_t if self.training else self.beta_max
        gamma_used = gamma_t if self.training else self.gamma_max

        loss_annealed = recon_total + beta_t * kl + gamma_t * cross_kl
        loss_fixed = recon_total + self.beta_max * kl + self.gamma_max * cross_kl
        loss = recon_total + beta_used * kl + gamma_used * cross_kl

        y_legacy = self._extract_legacy_y(y)
        class_loss = None
        class_logits = None
        if self.label_decoder is not None:
            z_for_cls = mu_moe if self.classify_from_mu else z_moe
            dec_out = self.label_decoder(z_for_cls)
            class_logits = dec_out["logits"] if isinstance(dec_out, dict) and "logits" in dec_out else dec_out

            if y_legacy is not None:
                yy = y_legacy.long()
                mask = (yy >= 0) & (yy != self.label_ignore_index)
                per_cell = torch.zeros(B, device=loss.device)
                if mask.any():
                    per_cell[mask] = F.cross_entropy(class_logits[mask], yy[mask], reduction="none")
                class_loss = per_cell

                loss = loss + self.label_loss_weight * class_loss
                loss_annealed = loss_annealed + self.label_loss_weight * class_loss
                loss_fixed = loss_fixed + self.label_loss_weight * class_loss

        loss, loss_annealed, loss_fixed, head_logits, head_loss_means = self._apply_multihead_losses(
            mu_z=mu_moe,
            z=z_moe,
            y=y,
            epoch=epoch,
            loss=loss,
            loss_annealed=loss_annealed,
            loss_fixed=loss_fixed,
        )

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
            # these are after scaling
            "recon_per_modality": {k: v.mean() for k, v in recon_per_target_mean.items()},
            "beta": torch.tensor(beta_t, device=loss.device),
            "gamma": torch.tensor(gamma_t, device=loss.device),
            "beta_used": torch.tensor(beta_used, device=loss.device),
            "gamma_used": torch.tensor(gamma_used, device=loss.device),
            "loss_annealed": loss_annealed.mean(),
            "loss_fixed": loss_fixed.mean(),
            "v1_recon_terms": torch.tensor(float(n_terms), device=loss.device),
            "used_fused_encoder": torch.tensor(1.0 if use_fused else 0.0, device=loss.device),
        }
        if class_loss is not None:
            out["class_loss"] = class_loss.mean()
        if class_logits is not None:
            out["class_logits"] = class_logits
        if head_logits:
            out["head_logits"] = head_logits
        if head_loss_means:
            out["head_losses"] = head_loss_means
        return out

    # ------------------------------ reconstruction loss ------------------------------

    def _unwrap_decoder_out(self, dec_out: Any) -> Any:
        if isinstance(dec_out, (tuple, list)):
            if len(dec_out) != 1:
                raise TypeError(f"Unsupported decoder output container of length {len(dec_out)}: {type(dec_out)!r}")
            dec_out = dec_out[0]
        if torch.is_tensor(dec_out):
            return dec_out
        if isinstance(dec_out, dict):
            out = dict(dec_out)
            if "mu" not in out and "mean" in out:
                out["mu"] = out["mean"]
            return out
        raise TypeError(f"Unsupported decoder output type: {type(dec_out)!r}")

    @staticmethod
    def _nb_nll(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mu = mu.clamp(min=eps)
        theta = theta.clamp(min=eps)
        t1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        t2 = theta * (torch.log(theta) - torch.log(theta + mu))
        t3 = x * (torch.log(mu) - torch.log(theta + mu))
        return -(t1 + t2 + t3)

    @staticmethod
    def _zinb_nll(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, logit_pi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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

    def _recon_loss(self, x: torch.Tensor, raw_dec_out: Any, likelihood: str, mod_name: str) -> torch.Tensor:
        likelihood = (likelihood or "gaussian").lower().strip()
        dec_out = self._unwrap_decoder_out(raw_dec_out)

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

        if likelihood == "bernoulli":
            logits = dec_out["logits"] if isinstance(dec_out, dict) else dec_out
            nll = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
            return nll.sum(dim=-1)

        if likelihood == "poisson":
            log_rate = dec_out["log_rate"] if isinstance(dec_out, dict) and "log_rate" in dec_out else dec_out
            nll = F.poisson_nll_loss(log_rate, x, log_input=True, full=False, reduction="none")
            return nll.sum(dim=-1)

        if likelihood in ("nb", "negative_binomial"):
            if not isinstance(dec_out, dict) or ("mu" not in dec_out) or ("log_theta" not in dec_out):
                raise ValueError(f"NB recon expects dict with keys ('mu','log_theta'); got {type(dec_out)}")
            mu = dec_out["mu"]
            theta = torch.exp(dec_out["log_theta"])
            if theta.dim() == 1:
                theta = theta.unsqueeze(0).expand_as(mu)
            return self._nb_nll(x, mu, theta, eps=self.EPS).sum(dim=-1)

        if likelihood in ("zinb", "zero_inflated_negative_binomial"):
            if (
                not isinstance(dec_out, dict)
                or ("mu" not in dec_out)
                or ("log_theta" not in dec_out)
                or ("logit_pi" not in dec_out)
            ):
                raise ValueError("ZINB recon expects dict with keys ('mu','log_theta','logit_pi').")
            mu = dec_out["mu"]
            theta = torch.exp(dec_out["log_theta"])
            logit_pi = dec_out["logit_pi"]
            if theta.dim() == 1:
                theta = theta.unsqueeze(0).expand_as(mu)
            if logit_pi.dim() == 1:
                logit_pi = logit_pi.unsqueeze(0).expand_as(mu)
            return self._zinb_nll(x, mu, theta, logit_pi, eps=self.EPS).sum(dim=-1)

        pred = dec_out["mean"] if isinstance(dec_out, dict) and ("mean" in dec_out) else dec_out
        return ((x - pred) ** 2).sum(dim=-1)

    # ------------------------------ annealing ------------------------------

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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_fused(
        self,
        x_dict: Dict[str, torch.Tensor],
        *,
        epoch: int = 0,
        y: Optional[YType] = None,
        use_mean: bool = True,
        inject_label_expert: bool = True,
        attn_bias_cfg: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_dict, logvar_dict = self.encode_modalities(x_dict, attn_bias_cfg=attn_bias_cfg)
        if len(mu_dict) == 0:
            raise ValueError("At least one modality must be present in x_dict.")

        use_fused = (self.fused_encoder_type == "multimodal_transformer") and self._can_use_fused_encoder(x_dict)
        if use_fused:
            mu_z, logvar_z = self._compute_fused_posterior(x_dict, attn_bias_cfg=attn_bias_cfg)
        else:
            mu_z, logvar_z = self.mixture_of_experts(mu_dict, logvar_dict)

        y_legacy = self._extract_legacy_y(y)
        if (
            inject_label_expert
            and (self.label_encoder is not None)
            and (y_legacy is not None)
            and (epoch >= self.label_encoder_warmup)
        ):
            B = mu_z.shape[0]
            dev = mu_z.device
            mu_y, logvar_y = self._encode_labels_as_expert(y=y_legacy, B=B, device=dev)
            base_mu_dict = {"__base__": mu_z, "__label__": mu_y}
            base_lv_dict = {"__base__": logvar_z, "__label__": logvar_y}
            mu_z, logvar_z = self.mixture_of_experts(base_mu_dict, base_lv_dict)

        z = mu_z if use_mean else self._reparameterize(mu_z, logvar_z)
        return mu_z, logvar_z, z

    @torch.no_grad()
    def predict_heads(
        self,
        x_dict: Dict[str, torch.Tensor],
        *,
        epoch: int = 0,
        y: Optional[YType] = None,
        use_mean: bool = True,
        inject_label_expert: bool = True,
        return_probs: bool = True,
        attn_bias_cfg: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        mu_z, logvar_z, z = self.encode_fused(
            x_dict,
            epoch=epoch,
            y=y,
            use_mean=use_mean,
            inject_label_expert=inject_label_expert,
            attn_bias_cfg=attn_bias_cfg,
        )
        out: Dict[str, torch.Tensor] = {}

        if self.label_decoder is not None:
            z_for_cls = mu_z if self.classify_from_mu else z
            dec_out = self.label_decoder(z_for_cls)
            logits = dec_out["logits"] if isinstance(dec_out, dict) and "logits" in dec_out else dec_out
            out[self.label_head_name] = logits if not return_probs else F.softmax(logits, dim=-1)

        for name, head in self.class_heads.items():
            cfg_h = self.class_heads_cfg[name]
            z_in = mu_z if bool(cfg_h["from_mu"]) else z
            dec_out = head(z_in)
            logits = dec_out["logits"] if isinstance(dec_out, dict) and "logits" in dec_out else dec_out
            out[name] = logits if not return_probs else F.softmax(logits, dim=-1)

        return out

    def get_classification_meta(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "label_head_name": self.label_head_name,
            "legacy": {
                "n_label_classes": int(self.n_label_classes),
                "label_ignore_index": int(self.label_ignore_index),
            },
            "multi": {
                "heads": {k: dict(v) for k, v in self.class_heads_cfg.items()},
            },
        }
        if self.label_names is not None:
            meta["legacy"]["label_names"] = list(self.label_names)
        if self.head_label_names:
            meta["multi"]["label_names"] = {k: list(v) for k, v in self.head_label_names.items()}
        return meta

