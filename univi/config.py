# univi/config.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal


@dataclass
class ModalityConfig:
    """
    Configuration for a single modality.

    Notes
    -----
    - For categorical modalities, set:
        likelihood="categorical"
        input_dim = n_classes (C)
      and optionally set input_kind/input_key to control how the dataset reads inputs.

    - If you want categorical labels stored in adata.obs, set:
        input_kind="obs"
        obs_key="your_obs_column"
      The dataset will return a (B,1) tensor of label codes (float32), which the model
      converts to one-hot for encoding and to class indices for CE.

    - ignore_index is used for unlabeled entries (masked in CE).
    """
    name: str
    input_dim: int
    encoder_hidden: List[int]
    decoder_hidden: List[int]
    likelihood: str = "gaussian"

    # ---- categorical modality support ----
    ignore_index: int = -1
    input_kind: Literal["matrix", "obs"] = "matrix"
    obs_key: Optional[str] = None


@dataclass
class ClassHeadConfig:
    """
    Configuration for an auxiliary supervised classification head p(y_h | z).

    Notes
    -----
    - This is NOT a modality; it is a supervised head attached to the latent.
    - Use ignore_index to mask unknown labels (e.g. -1).
    - from_mu=True means classify from mu_z (more stable), else from sampled z.
    - warmup: epoch before enabling this head's loss.
    - adversarial=True turns this into a gradient-reversal domain/tech adversary.
      In that case, adv_lambda scales the reversed gradient.
    """
    name: str
    n_classes: int
    loss_weight: float = 1.0
    ignore_index: int = -1
    from_mu: bool = True
    warmup: int = 0

    # ---- NEW: adversarial / domain-confusion support ----
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

    # ---- NEW: multiple supervised heads (dict y) ----
    class_heads: Optional[List[ClassHeadConfig]] = None

    # If you use the built-in label_encoder expert injection, this indicates which y key to use
    # when y is a dict (default "label" for backwards compatibility).
    label_head_name: str = "label"

    def validate(self) -> None:
        """
        Validate common configuration mistakes early.
        Call this before constructing the model / starting training.
        """
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")

        names = [m.name for m in self.modalities]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate modality names in cfg.modalities: {dupes}")

        for m in self.modalities:
            lk = (m.likelihood or "").lower().strip()
            if m.input_dim <= 0:
                raise ValueError(f"Modality {m.name!r}: input_dim must be > 0, got {m.input_dim}")

            if lk in ("categorical", "cat", "ce", "cross_entropy", "multinomial", "softmax"):
                if m.input_dim < 2:
                    raise ValueError(f"Categorical modality {m.name!r}: input_dim must be n_classes >= 2.")
                if m.input_kind == "obs" and not m.obs_key:
                    raise ValueError(f"Categorical modality {m.name!r}: input_kind='obs' requires obs_key.")

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
                # adversarial sanity
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


