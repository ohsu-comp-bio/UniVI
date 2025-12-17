# univi/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Sequence, Any


# =============================================================================
# Transformer + tokenizer config
# =============================================================================

@dataclass
class TransformerConfig:
    """
    Configuration for transformer encoder backends.

    Notes
    -----
    - Mirrors fields expected by univi/models/transformer.py:TransformerConfig.
    - max_tokens is only needed if you enable learned positional embeddings.
    - Relative positional bias is optional and intended mainly for ATAC peaks
      when you provide token_pos (basepair midpoints) at runtime.
    """
    d_model: int
    num_heads: int
    num_layers: int
    dim_feedforward: int = 4096
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    pooling: Literal["cls", "mean"] = "mean"
    max_tokens: Optional[int] = None

    # Optional: binned relative-position attention bias (e.g., genomic distance)
    use_relpos_bias: bool = False
    relpos_num_bins: int = 32
    relpos_max_dist: float = 1e6  # basepairs


@dataclass
class TokenizerConfig:
    """
    Turns (B, F) into (B, T, D_in) + optional key_padding_mask.

    Modes
    -----
    - "topk_scalar":   top-k features per cell, scalar value only -> (B, K, 1)
    - "topk_channels": top-k features per cell, multiple channels -> (B, K, C)
                       channels from: "value", "rank", "dropout"
    - "patch":         split features into contiguous patches -> (B, T, patch_size)
                       OR project each patch -> (B, T, patch_proj_dim)
    - "topk_embed":    top-k features per cell with explicit feature identity:
                       token = Emb(feature_id) + MLP(channels)
                       -> (B, K, d_model)

                       Optionally add ATAC coordinate embeddings:
                       token += Emb(chrom_id) + MLP(midpoint_bp / coord_scale)

    Notes
    -----
    - topk_embed is the recommended way to use attention over sparse omics features
      without losing feature identity.
    - If you need relative bias, you should pass token_pos (bp midpoints) into the
      transformer at runtime. The tokenizer will stash it in tokenizer.last_meta["token_pos"].
    """
    mode: Literal["topk_scalar", "topk_channels", "patch", "topk_embed"] = "topk_scalar"

    # top-k settings
    n_tokens: int = 256
    channels: Sequence[Literal["value", "rank", "dropout"]] = ("value",)

    # patch settings
    patch_size: int = 32
    patch_proj_dim: Optional[int] = None

    # general
    add_cls_token: bool = False

    # ---- topk_embed settings ----
    # required for topk_embed
    n_features: Optional[int] = None
    d_model: Optional[int] = None
    value_mlp_hidden: int = 256

    # optional coord embeddings (mainly ATAC)
    use_coords: bool = False
    chrom_vocab_size: int = 0
    coord_scale: float = 1e6  # divide bp midpoints by this before coord MLP

    # Optional per-feature metadata for coords (set at runtime; not great for JSON)
    # Expected keys: {"chrom": ..., "start": ..., "end": ...}
    # Values can be lists/arrays/torch tensors; tokenizer will convert to tensors.
    feature_info: Optional[Dict[str, Any]] = None


# =============================================================================
# Core UniVI configs
# =============================================================================

@dataclass
class ModalityConfig:
    """
    Configuration for a single modality.

    Notes
    -----
    - For categorical modalities, set:
        likelihood="categorical"
        input_dim = n_classes (C)

      and optionally set:
        input_kind="obs"
        obs_key="your_obs_column"

      The dataset returns a (B,1) tensor of label codes; the model converts
      to one-hot for encoding and to class indices for CE.

    - ignore_index is used for unlabeled entries (masked in CE).
    """
    name: str
    input_dim: int
    encoder_hidden: List[int]
    decoder_hidden: List[int]
    likelihood: str = "gaussian"

    # categorical modality support
    ignore_index: int = -1
    input_kind: Literal["matrix", "obs"] = "matrix"
    obs_key: Optional[str] = None

    # encoder backend (per-modality only)
    encoder_type: Literal["mlp", "transformer"] = "mlp"
    transformer: Optional[TransformerConfig] = None
    tokenizer: Optional[TokenizerConfig] = None


@dataclass
class ClassHeadConfig:
    """
    Configuration for an auxiliary supervised classification head p(y_h | z).

    Notes
    -----
    - from_mu=True: classify from mu_z (more stable), else from sampled z.
    - warmup: epoch before enabling this head's loss.
    - adversarial=True: gradient reversal head (domain/tech confusion).
    """
    name: str
    n_classes: int
    loss_weight: float = 1.0
    ignore_index: int = -1
    from_mu: bool = True
    warmup: int = 0

    adversarial: bool = False
    adv_lambda: float = 1.0


@dataclass
class UniVIConfig:
    latent_dim: int
    modalities: List[ModalityConfig]

    beta: float = 1.0
    gamma: float = 1.0

    encoder_dropout: float = 0.0
    decoder_dropout: float = 0.0
    encoder_batchnorm: bool = True
    decoder_batchnorm: bool = False

    kl_anneal_start: int = 0
    kl_anneal_end: int = 0
    align_anneal_start: int = 0
    align_anneal_end: int = 0

    class_heads: Optional[List[ClassHeadConfig]] = None
    label_head_name: str = "label"

    # ---------------------------------------------------------------------
    # Optional fused multimodal encoder over concatenated tokens
    # ---------------------------------------------------------------------
    fused_encoder_type: Literal["moe", "multimodal_transformer"] = "moe"
    fused_transformer: Optional[TransformerConfig] = None
    fused_modalities: Optional[Sequence[str]] = None  # default: all modalities
    fused_add_modality_embeddings: bool = True
    fused_require_all_modalities: bool = True  # if True: fall back to MoE when missing

    def validate(self) -> None:
        if int(self.latent_dim) <= 0:
            raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")

        # modality name sanity
        names = [m.name for m in self.modalities]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate modality names in cfg.modalities: {dupes}")

        mod_by_name: Dict[str, ModalityConfig] = {m.name: m for m in self.modalities}

        for m in self.modalities:
            if int(m.input_dim) <= 0:
                raise ValueError(f"Modality {m.name!r}: input_dim must be > 0, got {m.input_dim}")

            lk = (m.likelihood or "").lower().strip()
            if lk in ("categorical", "cat", "ce", "cross_entropy", "multinomial", "softmax"):
                if int(m.input_dim) < 2:
                    raise ValueError(f"Categorical modality {m.name!r}: input_dim must be n_classes >= 2.")
                if m.input_kind == "obs" and not m.obs_key:
                    raise ValueError(f"Categorical modality {m.name!r}: input_kind='obs' requires obs_key.")

            enc_type = (m.encoder_type or "mlp").lower().strip()
            if enc_type not in ("mlp", "transformer"):
                raise ValueError(
                    f"Modality {m.name!r}: encoder_type must be 'mlp' or 'transformer', got {m.encoder_type!r}"
                )

            if enc_type == "transformer":
                if m.transformer is None:
                    raise ValueError(f"Modality {m.name!r}: encoder_type='transformer' requires transformer config.")
                if m.tokenizer is None:
                    raise ValueError(f"Modality {m.name!r}: encoder_type='transformer' requires tokenizer config.")
                _validate_tokenizer(m.name, m.tokenizer)

        # fused encoder sanity
        fe = (self.fused_encoder_type or "moe").lower().strip()
        if fe not in ("moe", "multimodal_transformer"):
            raise ValueError(
                f"fused_encoder_type must be 'moe' or 'multimodal_transformer', got {self.fused_encoder_type!r}"
            )

        if fe == "multimodal_transformer":
            if self.fused_transformer is None:
                raise ValueError("fused_encoder_type='multimodal_transformer' requires UniVIConfig.fused_transformer.")

            fused_names = list(self.fused_modalities) if self.fused_modalities is not None else list(mod_by_name.keys())
            if not fused_names:
                raise ValueError("fused_modalities is empty; expected at least one modality name.")

            missing = [n for n in fused_names if n not in mod_by_name]
            if missing:
                raise ValueError(f"fused_modalities contains unknown modalities: {missing}. Known: {list(mod_by_name)}")

            for n in fused_names:
                tok = mod_by_name[n].tokenizer
                if tok is None:
                    raise ValueError(
                        f"Fused multimodal transformer requires ModalityConfig.tokenizer for modality {n!r}."
                    )
                _validate_tokenizer(n, tok)

        # class head sanity
        if self.class_heads is not None:
            hn = [h.name for h in self.class_heads]
            if len(set(hn)) != len(hn):
                dupes = sorted({n for n in hn if hn.count(n) > 1})
                raise ValueError(f"Duplicate class head names in cfg.class_heads: {dupes}")
            for h in self.class_heads:
                if int(h.n_classes) < 2:
                    raise ValueError(f"Class head {h.name!r}: n_classes must be >= 2.")
                if float(h.loss_weight) < 0:
                    raise ValueError(f"Class head {h.name!r}: loss_weight must be >= 0.")
                if int(h.warmup) < 0:
                    raise ValueError(f"Class head {h.name!r}: warmup must be >= 0.")
                if float(getattr(h, "adv_lambda", 1.0)) < 0.0:
                    raise ValueError(f"Class head {h.name!r}: adv_lambda must be >= 0.")

        # anneal sanity
        for k in ("kl_anneal_start", "kl_anneal_end", "align_anneal_start", "align_anneal_end"):
            v = int(getattr(self, k))
            if v < 0:
                raise ValueError(f"{k} must be >= 0, got {v}")


def _validate_tokenizer(mod_name: str, tok: TokenizerConfig) -> None:
    mode = (tok.mode or "").lower().strip()
    if mode not in ("topk_scalar", "topk_channels", "patch", "topk_embed"):
        raise ValueError(
            f"Modality {mod_name!r}: tokenizer.mode must be one of "
            f"['topk_scalar','topk_channels','patch','topk_embed'], got {tok.mode!r}"
        )

    if mode in ("topk_scalar", "topk_channels", "topk_embed"):
        if int(tok.n_tokens) <= 0:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.n_tokens must be > 0 for topk_*")

    if mode == "topk_channels":
        if not tok.channels:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.channels must be non-empty for topk_channels")
        bad = [c for c in tok.channels if c not in ("value", "rank", "dropout")]
        if bad:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.channels has invalid entries: {bad}")

    if mode == "topk_embed":
        if tok.n_features is None or int(tok.n_features) <= 0:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.n_features must be set (>0) for topk_embed")
        if tok.d_model is None or int(tok.d_model) <= 0:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.d_model must be set (>0) for topk_embed")
        if not tok.channels:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.channels must be non-empty for topk_embed")
        bad = [c for c in tok.channels if c not in ("value", "rank", "dropout")]
        if bad:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.channels has invalid entries: {bad}")

        if tok.use_coords:
            if int(tok.chrom_vocab_size) <= 0:
                raise ValueError(
                    f"Modality {mod_name!r}: tokenizer.chrom_vocab_size must be > 0 when use_coords=True"
                )
            # feature_info may be injected at runtime; validate if present
            if tok.feature_info is not None:
                for k in ("chrom", "start", "end"):
                    if k not in tok.feature_info:
                        raise ValueError(
                            f"Modality {mod_name!r}: tokenizer.feature_info missing key {k!r} (required for coords)"
                        )

    if mode == "patch":
        if int(tok.patch_size) <= 0:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.patch_size must be > 0 for patch")
        if tok.patch_proj_dim is not None and int(tok.patch_proj_dim) <= 0:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.patch_proj_dim must be > 0 if set")


@dataclass
class TrainingConfig:
    n_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    log_every: int = 10
    grad_clip: Optional[float] = None
    num_workers: int = 0
    seed: int = 0

    early_stopping: bool = False
    patience: int = 20
    min_delta: float = 0.0

