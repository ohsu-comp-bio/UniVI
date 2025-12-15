# univi/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Sequence


# ----------------------------
# Transformer + tokenizer config
# ----------------------------

@dataclass
class TransformerConfig:
    """
    Configuration for the transformer encoder backend used inside modality encoders.

    Notes
    -----
    - This mirrors the fields expected by univi/models/transformer.py:TransformerConfig.
    - max_tokens is optional because token count usually comes from the tokenizer.
      If you use learned positional embeddings, we set max_tokens automatically
      based on the tokenizer output length (and +1 if add_cls_token=True).
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

    Notes
    -----
    - topk_* is good for large sparse inputs where you only want attention on the
      most informative features per cell.
    - patch is closer to "patch embeddings": groups of genes/features.
    - TransformerEncoder will map D_in -> d_model internally (input_proj).
    """
    mode: Literal["topk_scalar", "topk_channels", "patch"] = "topk_scalar"

    # top-k settings
    n_tokens: int = 256
    channels: Sequence[Literal["value", "rank", "dropout"]] = ("value",)

    # patch settings
    patch_size: int = 32
    patch_proj_dim: Optional[int] = None  # if set, patch vectors -> projection -> D_in=patch_proj_dim

    # general
    add_cls_token: bool = False


# ----------------------------
# Core UniVI configs
# ----------------------------

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

    # encoder backend
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

    def validate(self) -> None:
        if int(self.latent_dim) <= 0:
            raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")

        # modality name sanity
        names = [m.name for m in self.modalities]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate modality names in cfg.modalities: {dupes}")

        for m in self.modalities:
            if int(m.input_dim) <= 0:
                raise ValueError(f"Modality {m.name!r}: input_dim must be > 0, got {m.input_dim}")

            lk = (m.likelihood or "").lower().strip()
            if lk in ("categorical", "cat", "ce", "cross_entropy", "multinomial", "softmax"):
                if int(m.input_dim) < 2:
                    raise ValueError(f"Categorical modality {m.name!r}: input_dim must be n_classes >= 2.")
                if m.input_kind == "obs" and not m.obs_key:
                    raise ValueError(f"Categorical modality {m.name!r}: input_kind='obs' requires obs_key.")

            # encoder sanity
            enc_type = (m.encoder_type or "mlp").lower().strip()
            if enc_type not in ("mlp", "transformer"):
                raise ValueError(f"Modality {m.name!r}: encoder_type must be 'mlp' or 'transformer', got {m.encoder_type!r}")

            if enc_type == "transformer":
                if m.transformer is None:
                    raise ValueError(f"Modality {m.name!r}: encoder_type='transformer' requires transformer config.")
                if m.tokenizer is None:
                    raise ValueError(f"Modality {m.name!r}: encoder_type='transformer' requires tokenizer config.")

                # tokenizer sanity
                tok = m.tokenizer
                mode = (tok.mode or "").lower().strip()
                if mode not in ("topk_scalar", "topk_channels", "patch"):
                    raise ValueError(f"Modality {m.name!r}: tokenizer.mode must be one of "
                                     f"['topk_scalar','topk_channels','patch'], got {tok.mode!r}")

                if mode in ("topk_scalar", "topk_channels"):
                    if int(tok.n_tokens) <= 0:
                        raise ValueError(f"Modality {m.name!r}: tokenizer.n_tokens must be > 0 for topk_*")
                    if mode == "topk_channels":
                        if not tok.channels:
                            raise ValueError(f"Modality {m.name!r}: tokenizer.channels must be non-empty for topk_channels")
                        bad = [c for c in tok.channels if c not in ("value", "rank", "dropout")]
                        if bad:
                            raise ValueError(f"Modality {m.name!r}: tokenizer.channels has invalid entries: {bad}")

                if mode == "patch":
                    if int(tok.patch_size) <= 0:
                        raise ValueError(f"Modality {m.name!r}: tokenizer.patch_size must be > 0 for patch")
                    if tok.patch_proj_dim is not None and int(tok.patch_proj_dim) <= 0:
                        raise ValueError(f"Modality {m.name!r}: tokenizer.patch_proj_dim must be > 0 if set")

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


