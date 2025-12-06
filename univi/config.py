# univi/config.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal


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
    # unlabeled sentinel (masked in CE and can map to all-zeros one-hot for encoders)
    ignore_index: int = -1

    # how to read this modality from AnnData in the dataset
    # - "matrix": read from X/layers/obsm (normal pathway)
    # - "obs": read from adata.obs[obs_key] as integer label codes (returned as (B,1))
    input_kind: Literal["matrix", "obs"] = "matrix"
    obs_key: Optional[str] = None


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

        # anneal sanity (not strict, just prevent obvious negatives)
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

